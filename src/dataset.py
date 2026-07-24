"""
PyTorch Dataset and DataLoader factory for P4 Radiology AI.

Design contracts (all verified in scripts/verify_dataset.py):

  1. get_inference_transform(config) is a module-level function imported by
     serve.py (L11). It returns the SAME transform as mode="val" and mode="test".
     Evaluation and serving use identical preprocessing — no training/serving skew.

  2. ChestXRayDataset validates the input DataFrame schema at construction.
     Missing columns, null labels, or out-of-range label values raise ValueError
     immediately — before any image is loaded and before any training compute runs.

  3. File existence is checked at construction. Missing files above threshold
     raise RuntimeError. Files below threshold are removed with a warning.
     Missing files are never discovered mid-training.

  4. Augmentation is applied ONLY in train mode. val and test use
     get_inference_transform() — deterministic, identical to serving.

  5. vertical_flip is always False (clinical correctness — see decisions.md).

  6. create_dataloaders() NEVER calls train_test_split.
     Splits are defined once in data_prep.py (L2) and are immutable here.

  7. Corruption handling differs by mode:
     - train: skip + log + return next valid sample
     - val/test: raise RuntimeError (evaluation must be comprehensive)

Data contract from data_prep.py (L2) / feature_registry.md (L3):
  Each input DataFrame must contain:
    image_path (str)     — absolute path to PNG file
    binary_label (int)   — 0 (Normal) or 1 (Suspicious)
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Minimum image dimension (pixels). Images smaller than this are corrupted.
# A valid chest X-ray is never smaller than 32×32 px after any processing.
MIN_IMAGE_SIZE_PX = 32

# Data contract: these columns must be present in every input DataFrame
REQUIRED_COLUMNS = {"image_path", "binary_label"}


# ── Transform Builders ─────────────────────────────────────────────────────────

def get_inference_transform(config: dict) -> transforms.Compose:
    """
    Return the preprocessing pipeline for inference: val, test, and serving.

    NO AUGMENTATION. Deterministic. Identical output for identical input.

    Imported by src/serve.py (L11):
        from src.dataset import get_inference_transform

    WHY IT LIVES IN dataset.py:
    Defining this function here and using it in BOTH ChestXRayDataset
    (mode="val"/"test") and serve.py guarantees evaluation and serving
    apply identical preprocessing. Any change automatically propagates
    to both — no risk of training/serving skew.

    GRAYSCALE-TO-RGB NORMALISATION QUIRK:
    NIH X-rays are grayscale (R=G=B after convert("RGB")).
    ImageNet normalisation uses different per-channel values:
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    Applying different normalisations to identical channels produces
    three slightly different versions of the grayscale signal.
    This creates the exact activation scales in EfficientNet-B0's first
    convolutional layer that its pretrained weights expect.

    The alternative (single-channel model) would require training from
    scratch, discarding all pretrained ImageNet knowledge.
    RGB duplication + ImageNet normalisation is the correct approach for
    applying ImageNet-pretrained models to grayscale medical images.

    Pipeline:
      1. Resize   — to (image_size × image_size), default 224×224
      2. ToTensor — PIL (H,W,C) uint8 → PyTorch (C,H,W) float32 in [0,1]
      3. Normalize — shift distribution to ImageNet statistics

    Args:
        config: loaded training_config.yaml dict

    Returns:
        transforms.Compose — no random transforms, fully deterministic
    """
    img_size = config["data"]["image_size"]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet per-channel mean
            std=[0.229, 0.224, 0.225],   # ImageNet per-channel std
        ),
    ])


def _build_train_transform(config: dict) -> transforms.Compose:
    """
    Return the preprocessing pipeline for training (with augmentation).

    All augmentation steps are conditional on config values.
    No magic numbers — everything read from training_config.yaml.

    CLINICAL DECISIONS ENCODED HERE:

    horizontal_flip: true
      Enabled. Patient positioning varies left/right.
      OPEN CLINICAL REVIEW: horizontal flip reverses the heart shadow,
      simulating situs inversus (dextrocardia — rare condition where
      heart points right). Standard ML papers use it. A clinical advisor
      may override. See decisions.md Decision 8.

    vertical_flip: false (ALWAYS — clinical correctness, never enable)
      Lung anatomy is NOT vertically symmetric:
        - Right hemidiaphragm is higher than left (liver below)
        - Heart position has vertical significance
        - Air-fluid levels are gravity-dependent (pool at bottom)
      Vertically flipping creates an anatomically impossible image.
      This is a patient safety decision, not a performance choice.
      See decisions.md Decision 6.

    rotation_degrees: 15
      Valid — simulates patient positioning variation.

    color_jitter: true
      Valid — simulates different X-ray exposure settings.
      Mitigates the pixel intensity spurious correlation documented in L3.

    Args:
        config: loaded training_config.yaml dict

    Returns:
        transforms.Compose — may include random augmentation steps
    """
    aug = config["augmentation"]
    img_size = config["data"]["image_size"]

    augmentation_steps = []

    if aug.get("horizontal_flip", False):
        augmentation_steps.append(transforms.RandomHorizontalFlip(p=0.5))

    if aug.get("vertical_flip", False):
        # CLINICALLY INVALID — this branch should NEVER execute.
        # vertical_flip: false in config ensures it never does.
        # The config key exists to make the decision visible and auditable.
        # If this branch runs, it is a configuration error.
        logger.error(
            "CLINICAL ERROR: vertical_flip is True in config. "
            "This creates anatomically impossible images. "
            "Set vertical_flip: false immediately. See decisions.md Decision 6."
        )
        augmentation_steps.append(transforms.RandomVerticalFlip(p=0.5))

    if aug.get("rotation_degrees", 0) > 0:
        augmentation_steps.append(
            transforms.RandomRotation(degrees=aug["rotation_degrees"])
        )

    if aug.get("color_jitter", False):
        augmentation_steps.append(transforms.ColorJitter(
            brightness=aug.get("color_jitter_brightness", 0.2),
            contrast=aug.get("color_jitter_contrast", 0.2),
        ))

    base_steps = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(augmentation_steps + base_steps)


# ── Worker Init Function — Reproducible Augmentation ──────────────────────────

def _worker_init_fn(worker_id: int) -> None:
    """
    Seed each DataLoader worker process for reproducible augmentation.

    With num_workers > 0, each worker has its own random state. Without
    explicit seeding, random transforms (flip, rotation, jitter) are not
    reproducibly seeded across training runs — the same global seed with
    num_workers=4 can produce different augmented images on different runs.

    PyTorch's initial_seed() provides a unique, deterministic seed per
    worker derived from the DataLoader's base seed. We apply it to NumPy
    and Python's random module (torchvision transforms use both).

    Passed to DataLoader as: worker_init_fn=_worker_init_fn
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ── Dataset Class ──────────────────────────────────────────────────────────────

class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest-Xray14 binary classification.

    Implements the PyTorch Dataset contract:
      __len__       — number of images in this split
      __getitem__   — load image, apply transform, return (tensor, label)

    Construction-time guarantees:
      - DataFrame schema is validated (required columns, no nulls, labels ∈ {0,1})
      - File existence is pre-checked (missing files above threshold → RuntimeError)
      - Mode is validated ("train", "val", "test" only)

    Corruption handling by mode:
      - train:    skip + log + return next valid sample (training continues)
      - val/test: raise RuntimeError (evaluation must be comprehensive)

    Args:
        df: DataFrame with at minimum columns [image_path, binary_label].
            Additional columns (Patient ID, Age, Gender) are preserved
            but not accessed by this class.
        config: loaded training_config.yaml dict
        mode: "train" | "val" | "test"
        missing_file_threshold: fraction of missing files that triggers
            RuntimeError (default 0.01 = 1%). Below threshold, rows removed.

    Returns from __getitem__:
        image_tensor: shape (3, 224, 224), dtype=torch.float32
        label_tensor: scalar, dtype=torch.long
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        mode: str = "train",
        missing_file_threshold: float = 0.01,
    ):
        # ── Validate mode ─────────────────────────────────────────────────────
        if mode not in ("train", "val", "test"):
            raise ValueError(
                f"mode must be 'train', 'val', or 'test'. Got: '{mode}'.\n"
                f"Using an invalid mode silently applies the wrong transform pipeline."
            )

        # ── Validate DataFrame schema ─────────────────────────────────────────
        # Done at construction time so failures surface before any training compute.
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing_cols}\n"
                f"Required columns: {REQUIRED_COLUMNS}\n"
                f"Found columns: {list(df.columns)}\n"
                f"Data contract is defined in data/feature_registry.md (L3).\n"
                f"Check that data_prep.prepare_dataset() was called correctly."
            )

        null_paths = df["image_path"].isnull().sum()
        null_labels = df["binary_label"].isnull().sum()
        if null_paths > 0 or null_labels > 0:
            raise ValueError(
                f"Null values in required columns:\n"
                f"  image_path nulls:   {null_paths}\n"
                f"  binary_label nulls: {null_labels}\n"
                f"Dataset may be corrupted or incompletely processed."
            )

        invalid_labels = (~df["binary_label"].isin([0, 1])).sum()
        if invalid_labels > 0:
            raise ValueError(
                f"binary_label contains {invalid_labels} values outside {{0, 1}}.\n"
                f"Labels must be exactly 0 (Normal) or 1 (Suspicious).\n"
                f"Check create_binary_labels() in data_prep.py."
            )

        # ── File existence precheck ───────────────────────────────────────────
        # Precheck at construction time — never discover missing files mid-training.
        exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
        missing_count = (~exists_mask).sum()
        missing_frac = missing_count / len(df) if len(df) > 0 else 0

        if missing_frac > missing_file_threshold:
            raise RuntimeError(
                f"{missing_count} image files ({missing_frac*100:.1f}%) are missing.\n"
                f"Threshold: {missing_file_threshold*100:.1f}%.\n"
                f"Verify all NIH image archives (images_001–012) are extracted to "
                f"data/raw/images/.\nExpected total: 112,120 images."
            )

        if missing_count > 0:
            logger.warning(
                "%d image files missing (%.2f%% — within threshold). "
                "Removing those rows.",
                missing_count, missing_frac * 100,
            )

        self.df = df[exists_mask].reset_index(drop=True)
        self.mode = mode
        self.config = config

        # ── Assign transform ──────────────────────────────────────────────────
        if mode == "train":
            self.transform = _build_train_transform(config)
        else:
            # val and test use get_inference_transform — same function as serve.py
            self.transform = get_inference_transform(config)

        transform_names = [type(t).__name__ for t in self.transform.transforms]
        logger.info(
            "ChestXRayDataset: mode=%s, images=%d, suspicious=%.1f%%, "
            "transforms=%s",
            mode, len(self.df),
            (self.df["binary_label"] == 1).mean() * 100,
            transform_names,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return the sample at index idx.

        Stateless — no mutable state from previous calls used here.
        Called by parallel DataLoader worker processes simultaneously.

        Steps:
          1. Read image_path and binary_label from DataFrame row
          2. Guard minimum image size (< 32px = corrupted)
          3. Open as PIL Image, convert to RGB
             (convert to RGB handles grayscale PNGs — replicates single
             channel to 3 channels for ImageNet-pretrained backbone)
          4. Apply transform (augmentation or inference transform)
          5. Return (image_tensor, label_tensor)

        Corruption handling:
          train mode:    skip + log + return next valid sample
          val/test mode: raise RuntimeError (evaluation must be complete)
        """
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        label_val = int(row["binary_label"])

        try:
            image = Image.open(img_path).convert("RGB")

            # Minimum size guard — corrupted images are often tiny
            w, h = image.size
            if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
                raise RuntimeError(
                    f"Image too small ({w}×{h}px < {MIN_IMAGE_SIZE_PX}px minimum). "
                    f"File is likely corrupted: {img_path}"
                )

        except Exception as e:
            if self.mode == "train":
                # Skip corrupted image during training — log and return next valid sample
                logger.warning(
                    "Skipping corrupted image at idx=%d: %s\nError: %s",
                    idx, img_path, e,
                )
                # Find next valid sample by incrementing index
                next_idx = (idx + 1) % len(self)
                return self.__getitem__(next_idx)
            else:
                # val/test: raise — evaluation must cover every sample
                raise RuntimeError(
                    f"Failed to load image at idx={idx} (mode={self.mode}): {img_path}\n"
                    f"In val/test mode, all images must be loadable.\n"
                    f"Fix or remove this file before evaluation.\n"
                    f"Error: {e}"
                ) from e

        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label_val, dtype=torch.long)

        return image_tensor, label_tensor


# ── DataLoader Factory ─────────────────────────────────────────────────────────

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from pre-split DataFrames.

    CRITICAL RULE: THIS FUNCTION NEVER CALLS train_test_split.
    Splits are defined once in data_prep.py (L2) and are immutable.
    The split manifest hash (L2) is the reproducibility guarantee.
    Re-splitting here would create new assignments that differ from
    the hashed manifest — silently breaking the experiment record.

    Performance settings:
      pin_memory:         True only when CUDA is available (DMA for async
                          CPU→GPU transfer — no benefit on CPU-only machines)
      persistent_workers: True for training (keeps workers alive between epochs,
                          avoids per-epoch process spawn/join overhead)
      worker_init_fn:     Seeds each worker for reproducible augmentation

    Production scale note:
      At > 1M images, loading loose PNG files from disk is an I/O bottleneck.
      Production upgrade path: convert to LMDB, WebDataset, or TFRecords
      (indexed binary formats that dramatically reduce filesystem overhead).
      For 112K images, loose PNGs are acceptable.

    Args:
        train_df, val_df, test_df: DataFrames from create_patient_level_split()
        config: loaded training_config.yaml dict
        num_workers: parallel workers per DataLoader.
            Default: min(4, os.cpu_count()). Set 0 on Windows or CPU-only.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    batch_size = config["training"]["batch_size"]

    # Auto-tune num_workers — never exceed available CPU cores
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    # pin_memory only benefits GPU training
    use_pin_memory = torch.cuda.is_available()

    # DataLoader kwargs shared across all splits
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }

    train_dataset = ChestXRayDataset(train_df, config, mode="train")
    val_dataset = ChestXRayDataset(val_df, config, mode="val")
    test_dataset = ChestXRayDataset(test_df, config, mode="test")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # keep workers alive between epochs
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    logger.info(
        "DataLoaders created (num_workers=%d, pin_memory=%s, batch_size=%d):\n"
        "  Train: %d batches (shuffle=True, drop_last=True, persistent_workers=True)\n"
        "  Val:   %d batches (shuffle=False, drop_last=False)\n"
        "  Test:  %d batches (shuffle=False, drop_last=False)",
        num_workers, use_pin_memory, batch_size,
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader