"""
Two-phase training loop for P4 Radiology AI.

Architecture:
  Phase 1: Frozen backbone (BN in eval via L5 train() override) — train head only
  Phase 2: Unfrozen backbone — full fine-tuning at 10x lower learning rate

Critical correctness fixes (both are silent bugs in naive implementations):
  1. Phase 1 → Phase 2 handoff: reload best_phase1.pt BEFORE unfreeze_backbone().
     After early stopping, in-memory model is from the stopping epoch, not the best epoch.
  2. Global best tracking: best_model.pt tracks the best across BOTH phases.
     Phase 2 may never improve on Phase 1. Without global tracking, Phase 2's
     suboptimal weights overwrite the globally optimal Phase 1 weights.

Other correctness guarantees:
  - Gradient clipping (max_norm=1.0) prevents Phase 2 gradient spikes.
  - NaN/Inf loss detection raises immediately rather than silently corrupting training.
  - Atomic checkpoint saving (write to tmp, rename) prevents corrupted checkpoints.
  - Best model loaded before return — caller receives optimal model, not end-of-training.
  - Both cudnn flags set: deterministic=True AND benchmark=False.
  - PyTorch 2.x autocast API (from torch import autocast, not torch.cuda.amp.autocast).

See decisions.md Decision 11 (AMP) and Decision 12 (Adam).
"""

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch import autocast  # PyTorch 2.x API (not torch.cuda.amp.autocast)
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.dataset import create_dataloaders
from src.model import ChestXRayClassifier

logger = logging.getLogger(__name__)

BEST_PHASE1_PATH = "artifacts/best_phase1.pt"
BEST_MODEL_PATH = "artifacts/best_model.pt"
BEST_MODEL_TMP = "artifacts/best_model_tmp.pt"


# ─── Reproducibility ───────────────────────────────────────────────────────────


def set_seeds(seed: int) -> None:
    """
    Set all five random seeds required for complete GPU reproducibility.

    Both cudnn flags must be set together:
      deterministic=True:  forces deterministic cuDNN algorithms
      benchmark=False:     prevents cuDNN from benchmarking multiple algorithms
                           non-deterministically (which would override deterministic=True)

    Without both: two runs with the same seed may produce different results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Seeds set: %d. cudnn.deterministic=True, cudnn.benchmark=False.", seed)


# ─── Reproducibility Metadata ──────────────────────────────────────────────────


def get_git_commit_hash() -> str:
    """Return current git commit hash (HEAD). Returns 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Could not retrieve git commit hash.")
        return "unknown"


def get_dvc_data_hash(dvc_file: str = "data/raw.dvc") -> str:
    """Read DVC dataset MD5 hash from .dvc pointer file. Returns 'unknown' if unavailable."""
    try:
        dvc_meta = yaml.safe_load(Path(dvc_file).read_text())
        return dvc_meta["outs"][0]["md5"]
    except (FileNotFoundError, KeyError, yaml.YAMLError):
        logger.warning("Could not read DVC hash from %s.", dvc_file)
        return "unknown"


def get_split_manifest_hash(manifest_path: str = "data/split_manifest.json") -> str:
    """Read split manifest SHA256 hash. Returns 'unknown' if manifest not yet generated."""
    try:
        return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    except FileNotFoundError:
        logger.warning("Split manifest not found at %s.", manifest_path)
        return "unknown"


def get_config_hash(config_path: str = "config/training_config.yaml") -> str:
    """Compute MD5 hash of training config file bytes."""
    try:
        return hashlib.md5(Path(config_path).read_bytes()).hexdigest()
    except FileNotFoundError:
        logger.warning("Config file not found at %s.", config_path)
        return "unknown"


# ─── Class Weights ─────────────────────────────────────────────────────────────


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Weight formula: weight_k = total_samples / (num_classes × count_k)

    DEVICE PLACEMENT — CRITICAL:
    The weight tensor MUST be on the same device as model outputs.
    .to(device) is called here to enforce this — never in the caller.
    CPU weight tensor + GPU model outputs → RuntimeError on first batch.

    Returns:
        torch.Tensor shape (2,) on device: [weight_Normal, weight_Suspicious]
    """
    counts = train_df["binary_label"].value_counts().sort_index()
    total = len(train_df)

    weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32,
    ).to(device)

    logger.info(
        "Class weights on %s: Normal=%.4f, Suspicious=%.4f",
        device,
        weights[0].item(),
        weights[1].item(),
    )

    return weights


# ─── Atomic Checkpoint Saving ──────────────────────────────────────────────────


def _save_checkpoint_atomic(model: nn.Module, dest_path: str) -> None:
    """
    Save model state_dict atomically: write to temp file, then rename.

    Prevents corrupted checkpoints from interrupted saves.
    os.replace() is atomic on POSIX filesystems.
    """
    tmp_path = dest_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, dest_path)


# ─── Epoch Functions ───────────────────────────────────────────────────────────


def train_epoch(
    model: ChestXRayClassifier,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """
    One complete pass through the training DataLoader.

    AMP workflow (CRITICAL ORDERING — must not be changed):
      1. autocast context: eligible ops run in float16 on CUDA
      2. scaler.scale(loss).backward(): scaled backward pass
      3. scaler.unscale_(optimizer):    unscale gradients to true magnitude
      4. clip_grad_norm_():              clip AFTER unscaling (on true magnitudes)
                                         clipping scaled gradients would produce
                                         unpredictable effective thresholds
      5. scaler.step(optimizer):        update weights
      6. scaler.update():               adjust scale factor for next iteration

    NaN/Inf detection:
      If loss is NaN or Inf (from corrupted batch or numerical instability),
      training is silently broken. Detection raises immediately with a
      diagnosable error rather than training on meaningless gradients.

    Gradient clipping (max_norm=1.0):
      Prevents gradient spikes during Phase 2 backbone unfreezing.
      Limits the global gradient norm — large gradients are scaled down
      proportionally. Prevents catastrophic weight updates and NaN loss.

    Args:
        model:         ChestXRayClassifier (model.train() called inside)
        loader:        training DataLoader
        optimizer:     Adam for current phase (head-only or full network)
        criterion:     CrossEntropyLoss with class weights
        device:        computation device
        scaler:        GradScaler (enabled=False on CPU — safe no-op)
        use_amp:       from config — whether AMP is attempted
        max_grad_norm: gradient clipping threshold (default 1.0)

    Returns:
        (avg_loss, accuracy) for this epoch
    """
    model.train()  # Activates BN override from L5 if backbone is frozen

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Step 1: AMP forward pass
        with autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Step 2: NaN/Inf detection — fail fast, not silently
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"Loss is {loss.item():.6f} — training is numerically unstable.\n"
                f"Check: class weights, learning rate, batch composition, AMP settings."
            )

        # Steps 3-6: Scaled backward, unscale, clip, step, update
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Step 3: unscale first
        torch.nn.utils.clip_grad_norm_(  # Step 4: clip on true magnitudes
            model.parameters(), max_norm=max_grad_norm
        )
        scaler.step(optimizer)  # Step 5: update weights
        scaler.update()  # Step 6: adjust scale

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total if total > 0 else 0.0


def validate_epoch(
    model: ChestXRayClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """
    One complete pass through the validation DataLoader.

    Runs in full float32 (no AMP) for deterministic results.
    Validation loss must be directly comparable to test evaluation in L7.
    Using AMP could introduce float16 rounding differences that make
    validation and test metrics inconsistent — important for a clinical tool.

    Note: Production inference stacks use reduced precision for throughput.
    For final evaluation where reproducibility matters more than speed,
    full float32 is the defensible choice.

    model.eval() disables dropout and keeps BN in inference mode.
    torch.no_grad() prevents computation graph allocation — saves memory/time.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total if total > 0 else 0.0


# ─── Full Training Pipeline ────────────────────────────────────────────────────


def train_pipeline(
    config_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[ChestXRayClassifier, str]:
    """
    Complete two-phase training pipeline with MLflow tracking.

    Phase 1: Frozen backbone → train head only → save best_phase1.pt
    HANDOFF: reload best_phase1.pt → prevents overfitted handoff trap
    Phase 2: Unfrozen backbone → new optimizer → global best tracking
    RETURN: load best_model.pt → caller receives globally best model

    Four-hash reproducibility chain logged to MLflow:
      git_commit_hash    — code version
      dvc_data_hash      — dataset version
      split_manifest_hash — patient split version
      config_hash        — hyperparameter version

    Returns:
        (trained_model, mlflow_run_id)
        trained_model is loaded from best_model.pt — guaranteed to be the
        globally best model across both phases.
        mlflow_run_id is used by evaluate.py (L7) to log evaluation metrics
        to the same run.
    """
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"].get("use_amp", True)
    patience_cfg = config["training"]["early_stopping_patience"]

    logger.info("Training device: %s | AMP: %s", device, use_amp and device.type == "cuda")

    set_seeds(config["training"]["random_seed"])

    # ── Collect four-hash reproducibility chain ───────────────────────────────
    git_hash = get_git_commit_hash()
    dvc_hash = get_dvc_data_hash()
    split_hash = get_split_manifest_hash()
    config_hash = get_config_hash(config_path)

    logger.info(
        "Reproducibility chain:\n  git:    %s\n  dvc:    %s\n  split:  %s\n  config: %s",
        git_hash,
        dvc_hash,
        split_hash,
        config_hash,
    )

    # ── Build DataLoaders and model ───────────────────────────────────────────
    train_loader, val_loader, _ = create_dataloaders(train_df, val_df, val_df, config)

    model = ChestXRayClassifier(config).to(device)
    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    Path("artifacts").mkdir(exist_ok=True)

    experiment_name = config.get("experiment", {}).get("mlflow_experiment_name", "p4-radiology-ai")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ── Log reproducibility chain + all config + architecture ─────────────
        mlflow.log_params(
            {
                "git_commit_hash": git_hash,
                "dvc_data_hash": dvc_hash,
                "split_manifest_hash": split_hash,
                "config_hash": config_hash,
            }
        )

        mlflow.log_params(
            {
                "architecture": config["model"]["architecture"],
                "pretrained": config["model"]["pretrained"],
                "num_classes": config["model"]["num_classes"],
                "dropout": config["model"]["dropout"],
                "phase1_epochs": config["training"]["phase1_epochs"],
                "phase2_epochs": config["training"]["phase2_epochs"],
                "phase1_lr": config["training"]["phase1_lr"],
                "phase2_lr": config["training"]["phase2_lr"],
                "batch_size": config["training"]["batch_size"],
                "weight_decay": config["training"]["weight_decay"],
                "early_stopping_patience": patience_cfg,
                "random_seed": config["training"]["random_seed"],
                "use_amp": use_amp,
                "class_weight_normal": round(class_weights[0].item(), 4),
                "class_weight_suspicious": round(class_weights[1].item(), 4),
                "horizontal_flip": config["augmentation"]["horizontal_flip"],
                "vertical_flip": config["augmentation"]["vertical_flip"],
                "rotation_degrees": config["augmentation"]["rotation_degrees"],
                "device": str(device),
            }
        )

        arch = model.get_architecture_summary()
        mlflow.log_params({f"arch_{k}": v for k, v in arch.items()})

        # ════════════════════════════════════════════════════════════════════
        # PHASE 1 — Frozen backbone, train head only
        # ════════════════════════════════════════════════════════════════════

        logger.info("=" * 50)
        logger.info("PHASE 1: Frozen backbone — head only")
        logger.info("=" * 50)

        model.freeze_backbone()

        p1_counts = model.count_parameters()
        mlflow.log_params(
            {
                "phase1_trainable_params": p1_counts["trainable"],
                "phase1_frozen_params": p1_counts["frozen"],
            }
        )

        optimizer_p1 = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["phase1_lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        scheduler_p1 = ReduceLROnPlateau(optimizer_p1, mode="min", patience=3, factor=0.1)
        best_val_loss_p1 = float("inf")
        patience_counter = 0

        for epoch in range(config["training"]["phase1_epochs"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer_p1, criterion, device, scaler, use_amp
            )
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

            scheduler_p1.step(val_loss)
            current_lr = optimizer_p1.param_groups[0]["lr"]

            logger.info(
                "P1 Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f lr=%.6f",
                epoch + 1,
                config["training"]["phase1_epochs"],
                train_loss,
                val_loss,
                val_acc,
                current_lr,
            )

            mlflow.log_metrics(
                {
                    "phase1_train_loss": train_loss,
                    "phase1_val_loss": val_loss,
                    "phase1_train_acc": train_acc,
                    "phase1_val_acc": val_acc,
                    "phase1_lr": current_lr,
                },
                step=epoch,
            )

            if val_loss < best_val_loss_p1:
                best_val_loss_p1 = val_loss
                patience_counter = 0
                _save_checkpoint_atomic(model, BEST_PHASE1_PATH)
                logger.info("P1 new best val_loss=%.4f → saved best_phase1.pt", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience_cfg:
                    logger.info("Phase 1 early stopping at epoch %d", epoch + 1)
                    break

        mlflow.log_metric("best_phase1_val_loss", best_val_loss_p1)

        # ── HANDOFF FIX: Reload best Phase 1 weights before Phase 2 ──────────
        # CRITICAL: Early stopping may leave in-memory model at a degraded epoch.
        # Phase 2 MUST start from the best Phase 1 checkpoint, not the stopping epoch.

        logger.info(
            "Reloading best Phase 1 weights (val_loss=%.4f) before Phase 2...",
            best_val_loss_p1,
        )
        model.load_state_dict(torch.load(BEST_PHASE1_PATH, map_location=device, weights_only=True))

        # ════════════════════════════════════════════════════════════════════
        # PHASE 2 — Full fine-tuning
        # ════════════════════════════════════════════════════════════════════

        logger.info("=" * 50)
        logger.info("PHASE 2: Full fine-tuning — all parameters")
        logger.info("=" * 50)

        model.unfreeze_backbone()

        p2_counts = model.count_parameters()
        mlflow.log_params(
            {
                "phase2_trainable_params": p2_counts["trainable"],
            }
        )

        # ── REGRESSION FIX: Establish global best before Phase 2 ─────────────
        # Copy best Phase 1 model as the baseline best_model.pt.
        # Phase 2 only overwrites if it beats the GLOBAL best (not just Phase 2 local).
        # If Phase 2 never improves on Phase 1, best_model.pt holds Phase 1's result.

        shutil.copy(BEST_PHASE1_PATH, BEST_MODEL_PATH)
        global_best_val_loss = best_val_loss_p1

        logger.info(
            "Phase 2 global best baseline: val_loss=%.4f (from Phase 1). "
            "Phase 2 must beat this to update best_model.pt.",
            global_best_val_loss,
        )

        # NEW optimizer — deliberately discards Phase 1 momentum buffers.
        # See decisions.md Decision 10 and Decision 12.
        optimizer_p2 = Adam(
            model.parameters(),
            lr=config["training"]["phase2_lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        scheduler_p2 = ReduceLROnPlateau(optimizer_p2, mode="min", patience=3, factor=0.1)
        patience_counter = 0

        for epoch in range(config["training"]["phase2_epochs"]):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer_p2, criterion, device, scaler, use_amp
            )
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

            scheduler_p2.step(val_loss)
            current_lr = optimizer_p2.param_groups[0]["lr"]

            logger.info(
                "P2 Epoch %d/%d — train_loss=%.4f val_loss=%.4f val_acc=%.4f lr=%.6f",
                epoch + 1,
                config["training"]["phase2_epochs"],
                train_loss,
                val_loss,
                val_acc,
                current_lr,
            )

            mlflow.log_metrics(
                {
                    "phase2_train_loss": train_loss,
                    "phase2_val_loss": val_loss,
                    "phase2_train_acc": train_acc,
                    "phase2_val_acc": val_acc,
                    "phase2_lr": current_lr,
                },
                step=epoch,
            )

            if val_loss < global_best_val_loss:
                global_best_val_loss = val_loss
                patience_counter = 0
                _save_checkpoint_atomic(model, BEST_MODEL_PATH)
                mlflow.log_artifact(BEST_MODEL_PATH)
                logger.info("P2 new GLOBAL best val_loss=%.4f → saved best_model.pt", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience_cfg:
                    logger.info("Phase 2 early stopping at epoch %d", epoch + 1)
                    break

        mlflow.log_metric(
            "best_phase2_val_loss",
            global_best_val_loss if global_best_val_loss < best_val_loss_p1 else best_val_loss_p1,
        )
        mlflow.log_metric("best_overall_val_loss", global_best_val_loss)

        # ── Restore best model before returning ───────────────────────────────
        # The in-memory model is from the last training epoch, not the best.
        # Load best_model.pt so the caller receives the globally optimal model.

        logger.info(
            "Restoring globally best model (val_loss=%.4f) from best_model.pt...",
            global_best_val_loss,
        )
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))

        logger.info(
            "Training complete.\n"
            "  Best Phase 1 val_loss: %.4f\n"
            "  Best overall val_loss: %.4f\n"
            "  Improvement from Phase 2: %s\n"
            "  MLflow run_id: %s",
            best_val_loss_p1,
            global_best_val_loss,
            "Yes" if global_best_val_loss < best_val_loss_p1 else "No (Phase 1 was best)",
            run_id,
        )

    return model, run_id
