"""
Grad-CAM explainability pipeline for P4 Radiology AI.

WHAT GRAD-CAM IS AND IS NOT:

  IS:  Gradient sensitivity map — which spatial regions influenced the class score
  IS:  Spatial localisation audit tool for clinical review

  IS NOT: Causal explanation of model reasoning
  IS NOT: Proof that the model "understands" pathology

A heatmap focused on the correct lung field is necessary but not sufficient
evidence of clinically sound reasoning. The model may activate on the correct
region using a shortcut (texture statistics, AP equipment signature) rather
than detecting the actual pathological finding.

CRITICAL IMPLEMENTATION FIXES:

1. requires_grad_(True) on input tensor:
   If model weights are frozen (freeze_backbone() or full inference freeze),
   no model parameter is a leaf requiring gradients. score.backward() crashes.
   Setting requires_grad_(True) on the input tensor guarantees the computational
   graph is built regardless of model weight state.

2. Batch size assertion:
   squeeze(0) is silent for batch=1, wrong for batch>=2.
   An assertion enforces the single-image contract explicitly.

3. Heatmap-as-mask overlay:
   Global alpha blending dims the entire X-ray by (1-alpha), obscuring
   diagnostic details in non-activated regions. Using the heatmap as a per-pixel
   alpha mask preserves 100% image quality where activation is zero.

4. target_class clarification:
   For FNs: target_class=1 shows ABSENT Suspicious signal (missing positive evidence)
   For FPs: target_class=1 shows ACTIVE spurious Suspicious signal
   These interpretations are different and documented in the summary.

See decisions.md Decision 15 for Grad-CAM vs SHAP rationale.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image

from src.dataset import get_inference_transform
from src.evaluate import load_model_and_config

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

GRADCAM_DIR     = Path("reports/gradcam")
GRADCAM_SUMMARY = GRADCAM_DIR / "summary.md"
THRESHOLD_PATH  = "artifacts/threshold.txt"


# ─── Target Layer Verification ────────────────────────────────────────────────

def verify_target_layer(
    model,
    device:  torch.device,
    config:  dict,
) -> bool:
    """
    Verify that backbone.features[-1] produces non-zero gradients.

    Mandatory pre-flight check before generating any heatmaps.
    A wrong target layer silently produces all-zero gradients — the resulting
    heatmap is noise that would mislead a clinical reviewer.

    Three checks:
      1. Forward hook fires (layer is in the computational graph)
      2. Feature maps are non-zero (layer produces activations)
      3. Gradients are non-zero (layer contributes to the class score)

    Uses requires_grad_(True) on input tensor to guarantee gradient flow
    even if model weights are frozen for inference optimisation.
    """
    target_layer = model.backbone.features[-1]
    transform    = get_inference_transform(config)
    img_size     = config["data"]["image_size"]

    dummy_pil    = Image.fromarray(
        np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
    )
    input_tensor = transform(dummy_pil).unsqueeze(0).to(device)

    # Must set requires_grad_(True) — see module docstring
    input_tensor = input_tensor.requires_grad_(True)

    captured = {"activations": None, "gradients": None}

    def fwd_hook(module, inp, out):
        captured["activations"] = out

    def bwd_hook(module, grad_in, grad_out):
        captured["gradients"] = grad_out[0]

    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    try:
        output = model(input_tensor)
        score  = output[0, 1]
        model.zero_grad()
        score.backward()

        if captured["activations"] is None:
            raise RuntimeError(
                f"Target layer forward hook did not fire. "
                f"Layer: {target_layer.__class__.__name__}"
            )

        if captured["activations"].abs().max() < 1e-8:
            raise RuntimeError(
                "Target layer feature maps are all zero. Model may not be processing input."
            )

        if captured["gradients"] is None or captured["gradients"].abs().max() < 1e-8:
            raise RuntimeError(
                "Target layer gradients are all zero. Heatmaps will be meaningless. "
                "Verify target layer is backbone.features[-1] for EfficientNet-B0."
            )

        feat_shape = tuple(captured["activations"].shape)
        grad_max   = float(captured["gradients"].abs().max())

        logger.info(
            "Target layer verified: %s\n"
            "  Feature map shape: %s\n"
            "  Max gradient: %.6f (> 0 ✓)",
            target_layer.__class__.__name__, feat_shape, grad_max,
        )

        return True

    finally:
        fwd_h.remove()
        bwd_h.remove()


# ─── Grad-CAM Computation ─────────────────────────────────────────────────────

def compute_gradcam(
    model:        nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: int,
    device:       torch.device,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a single image.

    Algorithm (Selvaraju et al. 2017):
      Step 1: α^c_k = (1/Z) Σ_ij (∂y^c / ∂A^k_ij)   importance weights
      Step 2: L = ReLU(Σ_k α^c_k · A^k)              weighted sum + ReLU
      Step 3: Upsample to input resolution, normalise to [0,1]

    CRITICAL: requires_grad_(True) on input tensor.
    Guarantees computational graph is built even if model weights are frozen.
    Without this, score.backward() raises RuntimeError if all weights have
    requires_grad=False (e.g., frozen for inference optimisation).

    CRITICAL: Do NOT use torch.no_grad() — gradients are required.

    CRITICAL: Call model.eval() before this function — BN must be in
    inference mode for consistent activations. Asserted here.

    GRADIENT SATURATION NOTE:
    For very high-confidence predictions (P near 0 or 1), softmax gradients
    are near zero — the heatmap may appear flat or diffuse. This is not a
    bug; it is a gradient saturation effect. A diffuse heatmap for a
    high-confidence FN means the model had no single localised signal driving
    its confident Normal prediction — the Normal confidence was distributed
    across many regions or spurious global features.

    TARGET CLASS INTERPRETATION:
    target_class=1 (Suspicious): shows which regions contributed to the
      Suspicious score. For FNs: shows absent Suspicious signal (missing
      positive evidence). For FPs/TPs: shows active Suspicious signal.
    target_class=0 (Normal): shows which regions supported Normal prediction.
      Used for TN cases.

    BATCH SIZE: exactly 1. Asserted. squeeze(0) is silent for batch>=2
    and produces wrong results without error.

    Args:
        model:         ChestXRayClassifier in eval mode (asserted)
        image_tensor:  (1, 3, H, W) float32 tensor on device
        target_layer:  model.backbone.features[-1] for EfficientNet-B0
        target_class:  0 (Normal) or 1 (Suspicious)
        device:        computation device

    Returns:
        heatmap (H, W) float32 ndarray in [0, 1]
    """
    assert not model.training, (
        "model.eval() must be called before compute_gradcam(). "
        "BatchNorm must be in inference mode for consistent activations."
    )

    assert image_tensor.size(0) == 1, (
        f"compute_gradcam() supports batch_size=1 only. "
        f"Got batch_size={image_tensor.size(0)}. "
        f"Call in a loop for multiple images."
    )

    captured = {"activations": None, "gradients": None}

    def save_activations(module, inp, out):
        captured["activations"] = out.detach()

    def save_gradients(module, grad_in, grad_out):
        captured["gradients"] = grad_out[0].detach()

    fwd_h = target_layer.register_forward_hook(save_activations)
    bwd_h = target_layer.register_full_backward_hook(save_gradients)

    try:
        # requires_grad_(True) guarantees graph is built even when weights are frozen
        # Do NOT wrap in torch.no_grad() — gradients are required for Grad-CAM
        image_tensor = image_tensor.requires_grad_(True)

        output       = model(image_tensor)
        score        = output[0, target_class]
        model.zero_grad()
        score.backward()

        activations = captured["activations"]   # (1, K, H_feat, W_feat)
        gradients   = captured["gradients"]     # (1, K, H_feat, W_feat)

        if activations is None or gradients is None:
            logger.warning("Hooks did not capture — returning blank heatmap.")
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]),
                            dtype=np.float32)

        # Step 1: importance weights = global average of gradients per feature map
        weights = gradients.squeeze(0).mean(dim=(1, 2))    # (K,)

        # Step 2: weighted sum of feature maps + ReLU
        feat    = activations.squeeze(0)                   # (K, H_feat, W_feat)
        cam     = (weights[:, None, None] * feat).sum(dim=0)   # (H_feat, W_feat)
        cam     = torch.clamp(cam, min=0.0)                # ReLU

        # Step 3: normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min).item() < 1e-8:
            # Flat heatmap — gradient saturation or uniform activation
            # This is a valid result (see gradient saturation note in docstring)
            cam_norm = torch.zeros_like(cam)
        else:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)

        # Upsample to input spatial dimensions
        h_out = image_tensor.shape[2]
        w_out = image_tensor.shape[3]

        cam_up = torch.nn.functional.interpolate(
            cam_norm.unsqueeze(0).unsqueeze(0),
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return cam_up.astype(np.float32)

    finally:
        # MANDATORY — hooks must be removed after every call.
        # Unremoved hooks accumulate in memory and corrupt subsequent gradients.
        fwd_h.remove()
        bwd_h.remove()


# ─── Heatmap Overlay ──────────────────────────────────────────────────────────

def overlay_heatmap(
    original_pil: Image.Image,
    heatmap:      np.ndarray,
    alpha:        float = 0.5,
) -> Image.Image:
    """
    Create a side-by-side image: original X-ray | heatmap overlay.

    HEATMAP-AS-MASK BLENDING (not global alpha):
    Global blending (1-alpha)*orig + alpha*heatmap dims the ENTIRE X-ray
    to (1-alpha) intensity, obscuring diagnostic details in non-activated
    regions. This is harmful for clinical review — radiologists rely on
    full dynamic range throughout the image.

    Correct approach: the heatmap VALUE acts as the per-pixel blend weight.
    Where heatmap=0.0: 100% original image, zero overlay.
    Where heatmap=1.0: (1-alpha) original + alpha overlay.
    Where heatmap=0.5: (1 - alpha*0.5) original + (alpha*0.5) overlay.

    This preserves full diagnostic quality in non-activated regions while
    highlighting activated regions proportionally to activation strength.

    COLORMAP: INFERNO (not JET).
    JET introduces false colour discontinuities at rainbow transitions
    (cyan↔green, yellow↔red) that create visually apparent "edges"
    corresponding to no actual activation boundary. This can mislead
    clinical reviewers into seeing anatomical structure that does not exist.

    INFERNO is perceptually uniform — colour transitions correspond linearly
    to activation magnitude. The correct choice for medical imaging.

    Args:
        original_pil: original PIL Image (any size)
        heatmap:      (H, W) float32 array in [0, 1]
        alpha:        maximum overlay strength at heatmap=1.0 (default 0.5)

    Returns:
        PIL Image (448, 224) — side-by-side: original | overlay
    """
    original_rgb = original_pil.convert("RGB").resize((224, 224))
    orig_arr     = np.array(original_rgb, dtype=np.float32) / 255.0

    # Apply INFERNO colormap
    cmap         = plt.cm.inferno
    heatmap_rgb  = cmap(heatmap)[:, :, :3]   # (H, W, 3), drop alpha

    # Heatmap-as-mask: per-pixel blend weight
    heatmap_mask = heatmap[..., None]         # (H, W, 1) broadcast to (H, W, 3)
    blended      = orig_arr * (1.0 - alpha * heatmap_mask) + \
                   heatmap_rgb * (alpha * heatmap_mask)

    blended      = np.clip(blended, 0.0, 1.0)
    blended_pil  = Image.fromarray((blended * 255).astype(np.uint8))

    # Side-by-side composite
    composite = Image.new("RGB", (448, 224))
    composite.paste(original_rgb, (0, 0))
    composite.paste(blended_pil,  (224, 0))

    return composite


# ─── Batch Heatmap Generation ─────────────────────────────────────────────────

def generate_case_heatmaps(
    model:        nn.Module,
    cases_df:     pd.DataFrame,
    label:        str,
    config:       dict,
    device:       torch.device,
    n_cases:      int,
    output_dir:   Path,
    target_class: int = 1,
) -> list[str]:
    """
    Generate Grad-CAM heatmaps for a set of cases, sorted by model_confidence.

    Sorted by model_confidence descending — most certain predictions first.
    For FNs: highest-confidence wrong predictions = most clinically important.
    For FPs: highest-confidence over-predictions = most likely systematic.
    For TPs: most confident correct predictions = best baseline for comparison.

    Args:
        model:        ChestXRayClassifier in eval mode
        cases_df:     DataFrame with image_path column and metadata
        label:        case type ("FN", "FP", "TP", "TN")
        config:       training config
        device:       computation device
        n_cases:      max heatmaps to generate
        output_dir:   save directory
        target_class: 1 for Suspicious (FN/FP/TP), 0 for Normal (TN)

    Returns:
        list of saved PNG file paths
    """
    if len(cases_df) == 0:
        logger.info("No %s cases to process.", label)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    transform    = get_inference_transform(config)
    target_layer = model.backbone.features[-1]

    sort_col     = "model_confidence" if "model_confidence" in cases_df.columns \
                   else "probability"

    sorted_cases = cases_df.sort_values(sort_col, ascending=False).head(n_cases)

    saved_paths = []

    for i, (_, row) in enumerate(sorted_cases.iterrows()):
        img_path = str(row["image_path"])

        try:
            with Image.open(img_path) as img:
                original_pil = img.convert("RGB")
                input_tensor = transform(original_pil).unsqueeze(0).to(device)
        except Exception as e:
            logger.warning("Cannot load image %s: %s — skipping.", img_path, e)
            continue

        try:
            heatmap = compute_gradcam(
                model, input_tensor, target_layer, target_class, device
            )
        except Exception as e:
            logger.warning("Grad-CAM failed for %s: %s — skipping.", img_path, e)
            continue

        overlay = overlay_heatmap(original_pil, heatmap)

        prob  = row.get("probability", 0.0)
        conf  = row.get("model_confidence", 0.0)

        fname = f"{label}_{i+1:02d}_prob{prob:.3f}_conf{conf:.3f}.png"
        fpath = output_dir / fname
        overlay.save(str(fpath))

        saved_paths.append(str(fpath))

    logger.info(
        "Generated %d/%d %s heatmaps → %s",
        len(saved_paths), min(n_cases, len(cases_df)), label, output_dir,
    )

    return saved_paths


# ─── Full Explainability Pipeline ─────────────────────────────────────────────

def run_explainability(
    config_path: str,
    fn_df:       pd.DataFrame,
    fp_df:       pd.DataFrame,
    all_test_df: pd.DataFrame,
    run_id:      str,
) -> dict:
    """
    Orchestrate Grad-CAM generation for all priority case types.

    Priority order (clinical importance):
      1. FN (target_class=1): shows ABSENT Suspicious signal in missed cases
         Sorted by model_confidence — most certain wrong predictions first
      2. FP (target_class=1): shows ACTIVE spurious Suspicious signal
         Reveals what drove false alarms — AP equipment, image artefacts?
      3. TP (target_class=1): correct Suspicious predictions (baseline)
         Heatmap should focus on clinically relevant anatomy
      4. TN (target_class=0): correct Normal predictions (reference)
         Heatmap for Normal class — should be diffuse or uninformative

    Note on FN interpretation: target_class=1 on a misclassified-as-Normal image
    shows which regions WOULD have needed to activate to push toward Suspicious.
    This is different from "what caused the Normal prediction." The correct reading
    is: "the model found insufficient Suspicious evidence in these spatial regions."
    """
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    model, _ = load_model_and_config(config_path, device)
    model.eval()

    # Mandatory pre-flight: verify target layer
    verify_target_layer(model, device, config)

    # Extract TP and TN from all_test_df
    tp_df = pd.DataFrame()
    tn_df = pd.DataFrame()

    if "error_type" in all_test_df.columns:
        tp_df = all_test_df[all_test_df["error_type"] == "TP"].reset_index(drop=True)
        tn_df = all_test_df[all_test_df["error_type"] == "TN"].reset_index(drop=True)

    results = {}

    fn_paths = generate_case_heatmaps(
        model, fn_df, "FN", config, device,
        n_cases=10, output_dir=GRADCAM_DIR / "fn_cases", target_class=1,
    )

    fp_paths = generate_case_heatmaps(
        model, fp_df, "FP", config, device,
        n_cases=10, output_dir=GRADCAM_DIR / "fp_cases", target_class=1,
    )

    tp_paths = generate_case_heatmaps(
        model, tp_df, "TP", config, device,
        n_cases=5, output_dir=GRADCAM_DIR / "tp_cases", target_class=1,
    )

    tn_paths = generate_case_heatmaps(
        model, tn_df, "TN", config, device,
        n_cases=3, output_dir=GRADCAM_DIR / "tn_cases", target_class=0,
    )

    results = {
        "fn_heatmaps": fn_paths,
        "fp_heatmaps": fp_paths,
        "tp_heatmaps": tp_paths,
        "tn_heatmaps": tn_paths,
    }

    total = sum(len(v) for v in results.values())

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "gradcam_fn_count": len(fn_paths),
            "gradcam_fp_count": len(fp_paths),
            "gradcam_tp_count": len(tp_paths),
            "gradcam_tn_count": len(tn_paths),
            "gradcam_total":    total,
        })

        if total > 0:
            mlflow.log_artifacts(str(GRADCAM_DIR), artifact_path="gradcam")

    _write_gradcam_summary(
        fn_paths=fn_paths, fp_paths=fp_paths,
        tp_paths=tp_paths, tn_paths=tn_paths,
        fn_df=fn_df, fp_df=fp_df,
    )

    logger.info(
        "Explainability complete: %d heatmaps total (FN=%d FP=%d TP=%d TN=%d)",
        total, len(fn_paths), len(fp_paths), len(tp_paths), len(tn_paths),
    )

    return results


# ─── Summary Writer ───────────────────────────────────────────────────────────

def _write_gradcam_summary(fn_paths, fp_paths, tp_paths, tn_paths, fn_df, fp_df):
    """Write reports/gradcam/summary.md — the gate artifact for L9."""

    # ── Ensure directory exists ──────────────────────────────────────────────
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    fn_high = 0
    if "conf_level" in fn_df.columns:
        fn_high = int((fn_df["conf_level"] == "High").sum())

    total = len(fn_paths) + len(fp_paths) + len(tp_paths) + len(tn_paths)

    gate_status = "✅ PASSED" if total >= 5 else f"❌ NEEDS MORE ({total}/5 minimum)"

    summary = f"""# Grad-CAM Explainability Summary — P4 Radiology AI

## 60 Seconds Academy — AI & ML

**Gate artifact status:** {gate_status}

**Method:** Grad-CAM (Gradient-weighted Class Activation Mapping)

**Target layer:** model.backbone.features[-1] (last MBConv, EfficientNet-B0)

**Feature map shape:** (batch, 1280, 7, 7) for 224×224 input

---

## FRAMING STATEMENT

Grad-CAM is a spatial localisation audit tool. It shows WHICH REGIONS influenced

the model's class score. It does NOT explain WHY the model reached its decision,

NOR does it prove the model understands the pathology.

A heatmap focused on the correct anatomy is necessary but not sufficient evidence

of clinically sound reasoning. The model may use a shortcut (local texture statistics,

AP equipment signature) that happens to overlap with the pathological region.

Use these heatmaps as supporting evidence — not as conclusive proof.

---

## Heatmaps Generated

| Case Type | Count | Location | Priority | target_class |

|---|---|---|---|---|

| False Negatives (FN) | {len(fn_paths)} | fn_cases/ | HIGHEST | 1 (Suspicious) |

| False Positives (FP) | {len(fp_paths)} | fp_cases/ | HIGH | 1 (Suspicious) |

| True Positives (TP) | {len(tp_paths)} | tp_cases/ | BASELINE | 1 (Suspicious) |

| True Negatives (TN) | {len(tn_paths)} | tn_cases/ | REFERENCE | 0 (Normal) |

FN high-confidence cases (model_confidence ≥ 0.80): {fn_high}

*These are the most dangerous — model was very certain about a wrong Normal prediction.*

---

## Interpreting FN Heatmaps (target_class=1)

**What these show:** Which regions WOULD HAVE needed to activate MORE to push the

model toward a Suspicious prediction. This is the ABSENT Suspicious signal.

**Different from "what caused the Normal prediction":** target_class=1 on a Normal-

predicted image shows missing positive evidence, not active negative evidence.

To see what drove the Normal prediction, you would need target_class=0.

**Clinical interpretation questions:**

- Does the activated region overlap with clinically relevant lung anatomy?

  (If yes: the model found insufficient signal in the right location)

- Is the heatmap flat/diffuse?

  (Possible gradient saturation — model was highly confident Normal with no

   localised Suspicious signal. Data gap or fundamental model limitation.)

- Does the activated region focus on the image border, clavicles, or labels?

  (Spurious correlation — the model is not attending to pathological anatomy)

---

## Interpreting FP Heatmaps (target_class=1)

**What these show:** Which regions drove the FALSE Suspicious prediction.

**Key question:** Do high-confidence FP heatmaps consistently activate in the same

non-pathological region across different patients?

- Consistent activation on AP equipment / image characteristics → confirms

  AP/PA spurious correlation hypothesis from L8 failure analysis

- Consistent activation on pacemaker leads → model learned device ≠ pathology

- Diffuse activation → model is uncertain (low-confidence FP, expected at low threshold)

---

## Heatmap Layout

Left panel: Original X-ray (100% opacity in non-activated regions)

Right panel: Original X-ray with INFERNO heatmap overlay

  - Heatmap is used as a per-pixel alpha mask (not global blend)

  - Zero-activation regions preserve full diagnostic image quality

  - Activated regions highlighted proportionally to activation magnitude

Colormap: INFERNO — perceptually uniform, no false colour discontinuities.

(JET colormap was rejected: introduces artefactual edges at rainbow transitions

 that can mislead clinical reviewers.)

---

## Clinical Observations

*Complete this section after reviewing with the clinical advisor.*

### FN Cases — Missing Suspicious Signal

**Heatmap pattern:** [populate — diffuse/localised? Anatomy region?]

**Clinical assessment:** [relevant anatomy focus? or spurious region?]

**Connects to L8 hypothesis:** [which root cause hypothesis does this support?]

### FP Cases — Spurious Suspicious Signal

**Heatmap pattern:** [populate — consistent region across cases? AP equipment?]

**Confirms/refutes AP/PA spurious correlation:** [Yes/No]

### TP Cases — Correct Suspicious Predictions (baseline)

**Heatmap pattern:** [populate — do correct predictions focus on appropriate anatomy?]

---

## Limitations

1. **Localisation ≠ reasoning.** Heatmaps reveal spatial sensitivity, not causal

   reasoning. The model may activate correctly while using a shortcut.

2. **Gradient saturation.** Very high-confidence predictions produce near-zero gradients

   — heatmaps appear flat. A flat heatmap for a high-confidence FN indicates the model

   had no localised Suspicious signal (distributed or spurious global features).

3. **7×7 spatial resolution.** Last convolutional layer produces 7×7 feature maps

   upsampled to 224×224. The heatmap has 49 effective resolution bins — coarse-grained.

   For finer attribution, Integrated Gradients is the documented upgrade path.

4. **No sanity baseline.** A proper interpretability validation would compare against

   Grad-CAM from a randomly-initialised model (which should produce noise). Without

   this comparison, there is no guarantee heatmaps reflect learned features rather

   than architectural priors.

5. **Model-specific.** These heatmaps reflect this specific trained model. Any retraining

   requires regenerating heatmaps.

6. **Confirmation bias risk.** Visually plausible heatmaps can reinforce clinical

   overconfidence in model predictions. Reviewers should be aware that the heatmap

   may look anatomically reasonable even when the model's reasoning is spurious.

---

## Gate Artifact Checklist

- [{"x" if total >= 5 else " "}] At least 5 heatmaps generated ({total} total)

- [ ] FN heatmap pattern documented in Clinical Observations above

- [ ] FP heatmap pattern documented and AP/PA correlation assessed

- [ ] TP baseline comparison completed

- [ ] Limitations acknowledged and included in model card (L10)

- [ ] Clinical advisor has reviewed FN and FP heatmaps

**This document must have all Clinical Observations sections completed

before L10 fairness evaluation and model card begins.**

"""

    GRADCAM_SUMMARY.write_text(summary)
    logger.info("Summary written to %s", GRADCAM_SUMMARY)