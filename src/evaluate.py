"""
Evaluation pipeline for P4 Radiology AI.

CORRECT EVALUATION SEQUENCE (order is critical):

  1. Inference on VALIDATION → compute ECE → decide Platt scaling
  2. Fit Platt scaling on VALIDATION raw scores (if ECE > 0.05)
  3. Calibrate VALIDATION probabilities
  4. Tune threshold on CALIBRATED VALIDATION probabilities
  5. Inference on TEST → calibrate with fitted Platt
  6. Apply LOCKED THRESHOLD to calibrated test probs
  7. Compute metrics, save visuals, log to MLflow

CRITICAL CORRECTNESS NOTES:

Threshold tuning on VALIDATION (not training):
  Training probabilities are overconfident/overfitted — skewed toward 0 and 1.
  A threshold tuned on this distribution does not generalise to deployment.
  Validation probabilities are representative of deployment-like uncertainty.
  Standard ML methodology: train→weights, validation→hyperparams/threshold/calibration,
  test→final frozen evaluation.

Calibration-threshold ordering:
  Threshold must be tuned on CALIBRATED validation probabilities, not raw.
  Platt scaling changes the probability space. A threshold tuned on raw
  probabilities is invalid when applied to calibrated probabilities.

McNemar's test:
  chi2_contingency() is Pearson's Chi-Square — NOT McNemar's test.
  McNemar uses ONLY discordant pairs: χ² = (|b-c|-1)² / (b+c)
  Using chi2_contingency on McNemar data produces wrong p-values.

Brier baseline:
  True naive baseline = prevalence × (1 - prevalence), computed dynamically.
  Hardcoding 0.25 assumes 50% prevalence — incorrect for NIH's ~46% Suspicious.

Stratified bootstrap:
  Simple resampling can distort class balance, destabilising minority-class CIs.
  Bootstrap samples must preserve the test set's Suspicious/Normal ratio.
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
from scipy.stats import chi2 as chi2_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from src.dataset import create_dataloaders
from src.model import ChestXRayClassifier

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

THRESHOLD_PATH   = "artifacts/threshold.txt"
EVAL_REPORT_PATH = "reports/evaluation_report.md"
BEST_MODEL_PATH  = "artifacts/best_model.pt"
REPORTS_DIR      = Path("reports/eval_artifacts")


# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_model_and_config(
    config_path: str,
    device: torch.device,
) -> tuple[ChestXRayClassifier, dict]:
    """Load best_model.pt and config. weights_only=True for PyTorch 2.x security."""
    config = yaml.safe_load(open(config_path))
    model  = ChestXRayClassifier(config)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    logger.info("Loaded best_model.pt on %s, eval mode.", device)
    return model, config


# ─── Inference ─────────────────────────────────────────────────────────────────

def get_predictions(
    model: ChestXRayClassifier,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference. Returns (labels, probs, raw_scores).
    probs:      softmax P(Suspicious) — for metric computation
    raw_scores: logit for Suspicious — input to Platt scaling
    Asserts model is in eval mode — catches accidental train-mode inference.
    """
    assert not model.training, (
        "get_predictions() called in training mode. Call model.eval() first."
    )

    all_labels, all_probs, all_raw = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            raw    = logits[:, 1]

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_raw.extend(raw.cpu().numpy())

    return (
        np.array(all_labels, dtype=np.int32),
        np.array(all_probs,  dtype=np.float32),
        np.array(all_raw,    dtype=np.float32),
    )


# ─── Calibration ──────────────────────────────────────────────────────────────

def compute_ece(
    labels: np.ndarray,
    probs:  np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    ECE = Σ_b (|bin_b| / N) × |accuracy(bin_b) - confidence(bin_b)|

    ECE directly measures the calibration gap: the average absolute difference
    between predicted probabilities and observed frequencies, weighted by bin size.

    WHY ECE over Brier score as calibration trigger:
    Brier score conflates accuracy and calibration. A model can have poor Brier
    score from poor accuracy, not poor calibration. ECE isolates the calibration
    component — the mismatch between confidence and actual accuracy.

    Threshold for Platt scaling: ECE > 0.05 (5% average calibration gap).

    Args:
        labels: true binary labels
        probs:  predicted probabilities
        n_bins: number of equal-width probability bins

    Returns:
        ECE as a float in [0, 1]
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(labels)

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue

        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def fit_platt_scaling(
    val_labels:     np.ndarray,
    val_raw_scores: np.ndarray,
) -> LogisticRegression:
    """
    Fit Platt scaling calibrator on VALIDATION raw scores.

    Platt scaling: P(y=1 | score) = sigmoid(a × score + b)
    The logistic regression learns a and b to map raw model scores to
    well-calibrated probabilities.

    MUST use validation set:
      Train set: model seen these → calibration overfits
      Test set:  using test labels = calibration leakage
      Validation: held-out from training, separate from test → correct

    Returns:
        Fitted LogisticRegression calibrator
    """
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(val_raw_scores.reshape(-1, 1), val_labels)

    logger.info(
        "Platt scaling fitted on validation. slope=%.4f, intercept=%.4f",
        calibrator.coef_[0][0], calibrator.intercept_[0],
    )

    return calibrator


def apply_platt_scaling(
    calibrator:  LogisticRegression,
    raw_scores:  np.ndarray,
) -> np.ndarray:
    """Apply fitted Platt calibrator to raw scores → calibrated probabilities."""
    return calibrator.predict_proba(
        raw_scores.reshape(-1, 1)
    )[:, 1].astype(np.float32)


# ─── Threshold Tuning ─────────────────────────────────────────────────────────

def tune_threshold(
    val_labels: np.ndarray,
    val_probs:  np.ndarray,
    min_precision: float = 0.60,
) -> tuple[float, float, float]:
    """
    Find optimal decision threshold using CALIBRATED VALIDATION probabilities.

    CORRECT METHODOLOGY:
      - Threshold is a hyperparameter → tuned on validation, not training
      - Must be tuned on CALIBRATED probabilities (after Platt scaling)
      - Training probabilities are overconfident (overfitted) and not
        representative of deployment probability distributions
      - Test set is never touched for threshold selection

    Search: grid [0.10, 0.90] step 0.01
    Objective: maximise recall subject to precision >= min_precision

    If no threshold satisfies the precision constraint, return the threshold
    with maximum recall and log a warning.

    Args:
        val_labels:    true labels from VALIDATION split
        val_probs:     calibrated P(Suspicious) from VALIDATION split
        min_precision: minimum acceptable precision (from config)

    Returns:
        (optimal_threshold, recall_at_threshold, precision_at_threshold)
    """
    best_threshold  = 0.5
    best_recall     = 0.0
    best_precision  = 0.0
    fallback_thresh = 0.5
    fallback_recall = 0.0

    for t in np.arange(0.10, 0.91, 0.01):
        preds     = (val_probs >= t).astype(int)
        recall    = recall_score(val_labels, preds, zero_division=0)
        precision = precision_score(val_labels, preds, zero_division=0)

        if recall > fallback_recall:
            fallback_recall = recall
            fallback_thresh = float(t)

        if precision >= min_precision and recall > best_recall:
            best_recall    = recall
            best_threshold = float(t)
            best_precision = precision

    if best_recall == 0.0:
        logger.warning(
            "No threshold satisfies precision >= %.2f on validation. "
            "Using best-recall fallback=%.2f. Consider reviewing model quality.",
            min_precision, fallback_thresh,
        )
        best_threshold = fallback_thresh
        best_recall    = fallback_recall

    logger.info(
        "Threshold tuned on CALIBRATED VALIDATION data:\n"
        "  threshold=%.3f  recall=%.4f  precision=%.4f",
        best_threshold, best_recall, best_precision,
    )

    return best_threshold, best_recall, best_precision


# ─── Stratified Bootstrap CI ──────────────────────────────────────────────────

def stratified_bootstrap_ci(
    labels:      np.ndarray,
    probs:       np.ndarray,
    threshold:   float,
    metric_fn,
    n_resamples: int = 1000,
    seed:        int = 42,
) -> tuple[float, float]:
    """
    Compute 95% bootstrap CI with STRATIFIED resampling.

    WHY STRATIFIED:
    Simple random resampling can distort class balance in bootstrap samples.
    A resample with 20% Suspicious (vs 46% in test) produces recall values
    not representative of the original distribution — destabilising CIs for
    minority-class metrics.

    Stratified resampling resamples Suspicious and Normal cases separately,
    preserving the class ratio in every bootstrap sample.

    Args:
        labels:      true binary labels from test split
        probs:       calibrated probabilities from test split
        threshold:   locked decision threshold
        metric_fn:   callable(labels, preds, zero_division=0) -> float
        n_resamples: bootstrap iterations
        seed:        random seed

    Returns:
        (ci_lower, ci_upper) — 2.5th and 97.5th percentiles
    """
    rng = np.random.default_rng(seed=seed)

    # Separate indices by class for stratified resampling
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    scores = []

    for _ in range(n_resamples):
        # Resample each class separately to preserve class ratio
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        boot_probs  = probs[boot_idx]
        boot_labels = labels[boot_idx]
        boot_preds  = (boot_probs >= threshold).astype(int)

        scores.append(metric_fn(boot_labels, boot_preds, zero_division=0))

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ─── Cost-Sensitive Evaluation ─────────────────────────────────────────────────

def compute_expected_cost(
    labels:    np.ndarray,
    preds:     np.ndarray,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> dict:
    """
    Compute expected clinical cost.
    Expected Cost = fn_weight × FN + fp_weight × FP
    Naive Cost    = fn_weight × total_Suspicious   (predict Normal always)

    Cost reduction % shows how much clinical harm the model prevents
    relative to having no screening tool at all.
    """
    cm              = confusion_matrix(labels, preds)
    tn, fp, fn, tp  = cm.ravel()

    expected_cost   = fn_weight * fn + fp_weight * fp
    naive_cost      = fn_weight * labels.sum()
    cost_reduction  = (1.0 - expected_cost / naive_cost) * 100 if naive_cost > 0 else 0.0

    logger.info(
        "Cost analysis: FN=%d FP=%d TP=%d TN=%d | "
        "cost=%.1f naive=%.1f reduction=%.1f%%",
        fn, fp, tp, tn, expected_cost, naive_cost, cost_reduction,
    )

    return {
        "fn": int(fn), "fp": int(fp), "tp": int(tp), "tn": int(tn),
        "expected_cost": float(expected_cost),
        "naive_cost":    float(naive_cost),
        "cost_reduction_pct": float(cost_reduction),
    }


# ─── McNemar's Test (Correct Formula) ─────────────────────────────────────────

def mcnemar_test_vs_naive(
    true_labels:  np.ndarray,
    model_preds:  np.ndarray,
) -> dict:
    """
    McNemar's test comparing trained model against naive baseline.

    CORRECT FORMULA — with Yates' continuity correction:
      b = cases where naive is wrong, model is right
      c = cases where naive is right, model is wrong
      χ² = (|b - c| - 1)² / (b + c)
      p-value = chi2.sf(χ², df=1)

    CRITICAL: DO NOT use chi2_contingency() for McNemar's test.
    chi2_contingency() computes Pearson's Chi-Square (tests independence).
    McNemar's tests marginal homogeneity in paired nominal data.
    They have different formulas, different assumptions, different p-values.
    Using chi2_contingency on McNemar data is a fundamental statistical error.

    The naive baseline predicts Normal (0) for every image.
    p < 0.05 means the trained model's improvement is statistically significant.

    Args:
        true_labels:  true binary labels (N,)
        model_preds:  binary predictions from trained model (N,)

    Returns:
        dict with chi2_stat, p_value, b, c, significant
    """
    naive_preds   = np.zeros_like(true_labels)  # always predict Normal
    naive_correct = (naive_preds == true_labels)
    model_correct = (model_preds == true_labels)

    b = int((~naive_correct &  model_correct).sum())  # naive wrong, model right
    c = int(( naive_correct & ~model_correct).sum())  # naive right, model wrong

    if (b + c) == 0:
        logger.warning("No discordant pairs for McNemar's test. Models make identical errors.")
        return {"chi2_stat": 0.0, "p_value": 1.0, "b": b, "c": c, "significant": False}

    # McNemar's Chi-square with Yates' continuity correction
    chi2_stat = float((abs(b - c) - 1) ** 2 / (b + c))
    p_value   = float(chi2_dist.sf(chi2_stat, df=1))  # survival function = 1 - CDF
    significant = p_value < 0.05

    logger.info(
        "McNemar's test vs naive: b=%d, c=%d, χ²=%.4f, p=%.6f, significant=%s",
        b, c, chi2_stat, p_value, significant,
    )

    return {
        "chi2_stat":   chi2_stat,
        "p_value":     p_value,
        "b":           b,
        "c":           c,
        "significant": significant,
    }


# ─── Visual Artifact Generation ───────────────────────────────────────────────

def _save_eval_artifacts(
    test_labels: np.ndarray,
    test_probs:  np.ndarray,
    threshold:   float,
    config:      dict,
) -> list[str]:
    """
    Save evaluation visual artifacts to reports/eval_artifacts/.
    Returns list of saved file paths for MLflow artifact logging.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    auc_roc_val  = roc_auc_score(test_labels, test_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"EfficientNet-B0 (AUC={auc_roc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    roc_path = str(REPORTS_DIR / "roc_curve.png")
    plt.savefig(roc_path, dpi=120)
    plt.close()
    saved.append(roc_path)

    # ── PR Curve ──────────────────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(test_labels, test_probs)
    auc_pr_val    = auc(rec, prec)
    prevalence    = test_labels.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#F44336", linewidth=2, label=f"EfficientNet-B0 (AUC-PR={auc_pr_val:.3f})")
    ax.axhline(y=prevalence, color="k", linestyle="--", linewidth=1,
               label=f"Naive baseline (precision={prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    pr_path = str(REPORTS_DIR / "pr_curve.png")
    plt.savefig(pr_path, dpi=120)
    plt.close()
    saved.append(pr_path)

    # ── Calibration Curve ─────────────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(test_labels, test_probs, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#9C27B0", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed Frequency)")
    ax.set_title("Calibration Curve (Reliability Diagram) — Test Set")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    cal_path = str(REPORTS_DIR / "calibration_curve.png")
    plt.savefig(cal_path, dpi=120)
    plt.close()
    saved.append(cal_path)

    # ── Threshold Sweep (Recall & Precision vs Threshold) ─────────────────────
    thresholds   = np.arange(0.10, 0.91, 0.01)
    recalls      = []
    precisions   = []

    for t in thresholds:
        preds = (test_probs >= t).astype(int)
        recalls.append(recall_score(test_labels, preds, zero_division=0))
        precisions.append(precision_score(test_labels, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, recalls,    color="#4CAF50", linewidth=2, label="Recall")
    ax.plot(thresholds, precisions, color="#F44336", linewidth=2, label="Precision")
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Locked threshold = {threshold:.2f}")
    ax.axhline(y=config["evaluation"]["recall_threshold"],    color="#4CAF50",
               linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(y=config["evaluation"]["precision_threshold"], color="#F44336",
               linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Threshold Sweep — Recall & Precision vs Threshold (Test Set)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    sweep_path = str(REPORTS_DIR / "threshold_sweep.png")
    plt.savefig(sweep_path, dpi=120)
    plt.close()
    saved.append(sweep_path)

    logger.info("Saved %d evaluation visual artifacts to %s", len(saved), REPORTS_DIR)
    return saved


# ─── Full Evaluation Pipeline ──────────────────────────────────────────────────

def full_evaluation(
    config_path: str,
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     pd.DataFrame,
    run_id:      str,
) -> dict:
    """
    Orchestrate all evaluation steps in the correct order.

    CORRECT SEQUENCE:
      1.  Load model
      2.  Inference on VALIDATION → raw probs + raw scores
      3.  Compute ECE on validation probs
      4.  If ECE > 0.05: fit Platt on val raw scores
      5.  Calibrate validation probs (Platt or identity)
      6.  Tune threshold on CALIBRATED VALIDATION probs
      7.  Inference on TEST → raw probs + raw scores
      8.  Calibrate test probs (same Platt or identity)
      9.  Apply locked threshold → binary test predictions
      10. Compute metrics (recall, precision, AUC, Brier, CI)
      11. Cost-sensitive evaluation
      12. McNemar's test
      13. Save visual artifacts
      14. Log to MLflow (same run as L6)
      15. Write evaluation_report.md
      16. Write threshold.txt

    NOTE: train_df is kept in signature for future use (e.g., computing
    training-set statistics for comparison) but is not used in threshold tuning.
    """
    config   = yaml.safe_load(open(config_path))
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = config["evaluation"]

    # ── Step 1: Load model ────────────────────────────────────────────────────
    model, _ = load_model_and_config(config_path, device)

    # ── Step 2: Build DataLoaders ─────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )

    # ── Steps 2-3: Inference on VALIDATION → ECE ─────────────────────────────
    logger.info("Inference on VALIDATION split...")
    val_labels, val_probs, val_raw = get_predictions(model, val_loader, device)

    val_ece_raw = compute_ece(val_labels, val_probs)
    val_brier_raw = brier_score_loss(val_labels, val_probs)

    logger.info(
        "Validation raw probs — ECE=%.4f, Brier=%.4f", val_ece_raw, val_brier_raw
    )

    # ── Steps 4-5: Platt scaling decision and calibration ─────────────────────
    ECE_THRESHOLD = 0.05
    apply_calibration = val_ece_raw > ECE_THRESHOLD

    calibrator = None

    if apply_calibration:
        logger.info(
            "Val ECE=%.4f > %.2f → fitting Platt scaling calibrator.",
            val_ece_raw, ECE_THRESHOLD,
        )
        calibrator = fit_platt_scaling(val_labels, val_raw)
        val_probs_calibrated = apply_platt_scaling(calibrator, val_raw)
        val_ece_calibrated   = compute_ece(val_labels, val_probs_calibrated)
        val_brier_calibrated = brier_score_loss(val_labels, val_probs_calibrated)

        logger.info(
            "After Platt scaling — ECE=%.4f, Brier=%.4f",
            val_ece_calibrated, val_brier_calibrated,
        )
    else:
        logger.info(
            "Val ECE=%.4f <= %.2f → model well-calibrated. Skipping Platt scaling.",
            val_ece_raw, ECE_THRESHOLD,
        )
        val_probs_calibrated = val_probs
        val_ece_calibrated   = val_ece_raw
        val_brier_calibrated = val_brier_raw

    # ── Step 6: Tune threshold on CALIBRATED VALIDATION probs ────────────────
    logger.info("Tuning threshold on calibrated VALIDATION probabilities...")
    threshold, val_recall, val_precision = tune_threshold(
        val_labels, val_probs_calibrated,
        min_precision=eval_cfg["precision_threshold"],
    )

    # ── Steps 7-8: Inference on TEST → calibrate ─────────────────────────────
    logger.info("Inference on TEST split...")
    test_labels, test_probs_raw, test_raw = get_predictions(model, test_loader, device)

    if calibrator is not None:
        test_probs = apply_platt_scaling(calibrator, test_raw)
        logger.info("Platt scaling applied to TEST probabilities.")
    else:
        test_probs = test_probs_raw

    # ── Step 9: Apply locked threshold ───────────────────────────────────────
    test_preds = (test_probs >= threshold).astype(int)

    # ── Step 10: Core metrics ─────────────────────────────────────────────────
    recall    = recall_score(test_labels, test_preds, zero_division=0)
    precision = precision_score(test_labels, test_preds, zero_division=0)
    auc_roc   = roc_auc_score(test_labels, test_probs)
    brier     = brier_score_loss(test_labels, test_probs)

    prec_curve, rec_curve, _ = precision_recall_curve(test_labels, test_probs)
    auc_pr    = auc(rec_curve, prec_curve)

    # Dynamic Brier naive baseline = prevalence × (1 - prevalence)
    prevalence   = float(test_labels.mean())
    brier_naive  = prevalence * (1.0 - prevalence)
    brier_passes = brier < brier_naive

    # Stratified bootstrap CIs
    recall_ci_lo, recall_ci_hi = stratified_bootstrap_ci(
        test_labels, test_probs, threshold, recall_score
    )
    prec_ci_lo, prec_ci_hi = stratified_bootstrap_ci(
        test_labels, test_probs, threshold, precision_score
    )

    # AUC-ROC bootstrap CI (using ranking metric — metric_fn is roc_auc_score without threshold)
    auc_rng = np.random.default_rng(seed=42)
    pos_idx = np.where(test_labels == 1)[0]
    neg_idx = np.where(test_labels == 0)[0]
    auc_scores = []

    for _ in range(1000):
        bp = auc_rng.choice(pos_idx, len(pos_idx), replace=True)
        bn = auc_rng.choice(neg_idx, len(neg_idx), replace=True)
        bi = np.concatenate([bp, bn])
        auc_scores.append(roc_auc_score(test_labels[bi], test_probs[bi]))

    auc_ci_lo = float(np.percentile(auc_scores, 2.5))
    auc_ci_hi = float(np.percentile(auc_scores, 97.5))

    # Quality gates
    recall_passes    = recall >= eval_cfg["recall_threshold"]
    precision_passes = precision >= eval_cfg["precision_threshold"]
    auc_passes       = auc_roc >= eval_cfg["auc_threshold"]
    all_pass         = all([recall_passes, precision_passes, auc_passes, brier_passes])

    logger.info(
        "Test evaluation:\n"
        "  Recall:    %.4f [%.4f, %.4f] → %s\n"
        "  Precision: %.4f [%.4f, %.4f] → %s\n"
        "  AUC-ROC:   %.4f [%.4f, %.4f] → %s\n"
        "  AUC-PR:    %.4f\n"
        "  Brier:     %.4f (naive=%.4f) → %s\n"
        "  Threshold: %.4f",
        recall, recall_ci_lo, recall_ci_hi,
        "PASS" if recall_passes else "FAIL",
        precision, prec_ci_lo, prec_ci_hi,
        "PASS" if precision_passes else "FAIL",
        auc_roc, auc_ci_lo, auc_ci_hi,
        "PASS" if auc_passes else "FAIL",
        auc_pr,
        brier, brier_naive, "PASS" if brier_passes else "FAIL",
        threshold,
    )

    # ── Step 11: Cost-sensitive ───────────────────────────────────────────────
    cost_stats = compute_expected_cost(
        test_labels, test_preds,
        fn_weight=eval_cfg.get("fn_cost_weight", 5.0),
        fp_weight=eval_cfg.get("fp_cost_weight", 1.0),
    )

    # ── Step 12: McNemar's test ───────────────────────────────────────────────
    mcnemar_stats = mcnemar_test_vs_naive(test_labels, test_preds)

    # ── Step 13: Save visual artifacts ───────────────────────────────────────
    artifact_paths = _save_eval_artifacts(test_labels, test_probs, threshold, config)

    # ── Step 14: Log to MLflow ────────────────────────────────────────────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            # Core quality gates
            "test_recall":              recall,
            "test_precision":           precision,
            "test_auc_roc":             auc_roc,
            "test_auc_pr":              auc_pr,
            "test_brier_score":         brier,
            "brier_naive_baseline":     brier_naive,
            "optimal_threshold":        threshold,
            # Confidence intervals
            "recall_ci_lower":          recall_ci_lo,
            "recall_ci_upper":          recall_ci_hi,
            "precision_ci_lower":       prec_ci_lo,
            "precision_ci_upper":       prec_ci_hi,
            "auc_roc_ci_lower":         auc_ci_lo,
            "auc_roc_ci_upper":         auc_ci_hi,
            # Quality gate results
            "quality_gate_recall":      float(recall_passes),
            "quality_gate_precision":   float(precision_passes),
            "quality_gate_auc":         float(auc_passes),
            "quality_gate_brier":       float(brier_passes),
            "all_quality_gates_pass":   float(all_pass),
            # Cost-sensitive
            "expected_cost":            cost_stats["expected_cost"],
            "naive_cost":               cost_stats["naive_cost"],
            "cost_reduction_pct":       cost_stats["cost_reduction_pct"],
            # McNemar
            "mcnemar_chi2":             mcnemar_stats["chi2_stat"],
            "mcnemar_p_value":          mcnemar_stats["p_value"],
            "mcnemar_significant":      float(mcnemar_stats["significant"]),
            # Calibration
            "val_ece_raw":              val_ece_raw,
            "val_ece_after_platt":      val_ece_calibrated if apply_calibration else val_ece_raw,
            "val_brier_raw":            val_brier_raw,
            "val_brier_after_platt":    val_brier_calibrated if apply_calibration else val_brier_raw,
            "platt_scaling_applied":    float(apply_calibration),
            "test_prevalence":          prevalence,
        })

        # Log visual artifacts
        for path in artifact_paths:
            mlflow.log_artifact(path)

    # ── Step 15: Write evaluation report ─────────────────────────────────────
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()

    _write_evaluation_report(
        threshold=threshold,
        recall=recall, recall_ci=(recall_ci_lo, recall_ci_hi),
        precision=precision, prec_ci=(prec_ci_lo, prec_ci_hi),
        auc_roc=auc_roc, auc_ci=(auc_ci_lo, auc_ci_hi),
        auc_pr=auc_pr,
        brier=brier, brier_naive=brier_naive,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        cost_stats=cost_stats,
        mcnemar_stats=mcnemar_stats,
        apply_calibration=apply_calibration,
        val_ece=val_ece_raw,
        config=config,
        all_pass=all_pass,
        prevalence=prevalence,
    )

    # ── Step 16: Write locked threshold ──────────────────────────────────────
    Path("artifacts").mkdir(exist_ok=True)
    Path(THRESHOLD_PATH).write_text(f"{threshold:.6f}")

    logger.info("Threshold %.6f locked in %s", threshold, THRESHOLD_PATH)

    results = {
        "threshold": threshold,
        "recall": recall, "precision": precision,
        "auc_roc": auc_roc, "auc_pr": auc_pr, "brier": brier,
        "brier_naive": brier_naive,
        "recall_ci": (recall_ci_lo, recall_ci_hi),
        "prec_ci": (prec_ci_lo, prec_ci_hi),
        "auc_ci": (auc_ci_lo, auc_ci_hi),
        "confusion_matrix": cm,
        "cost_stats": cost_stats,
        "mcnemar": mcnemar_stats,
        "quality_gates": {
            "recall_passes": recall_passes,
            "precision_passes": precision_passes,
            "auc_passes": auc_passes,
            "brier_passes": brier_passes,
            "all_pass": all_pass,
        },
        "platt_applied": apply_calibration,
    }

    status = "✅ ALL QUALITY GATES PASSED" if all_pass else "❌ QUALITY GATES FAILED"
    logger.info("Evaluation complete. %s", status)

    return results


def _write_evaluation_report(
    threshold, recall, recall_ci, precision, prec_ci,
    auc_roc, auc_ci, auc_pr, brier, brier_naive,
    tp, fp, fn, tn, cost_stats, mcnemar_stats,
    apply_calibration, val_ece, config, all_pass, prevalence,
) -> None:
    """Write evaluation_report.md."""
    eval_cfg = config["evaluation"]
    total    = tp + fp + fn + tn

    def gate(val, thresh, op=">="):
        met = val >= thresh if op == ">=" else val < thresh
        return f"{'✅ PASS' if met else '❌ FAIL'}"

    report = f"""# Evaluation Report — P4 Radiology AI

## 60 Seconds Academy — AI & ML

**Overall status:** {'✅ ALL QUALITY GATES PASSED' if all_pass else '❌ ONE OR MORE QUALITY GATES FAILED'}

---

## Methodology

- Threshold tuned on: **CALIBRATED VALIDATION split** (not training, not test)
- Platt scaling applied: **{'Yes (val ECE=' + f'{val_ece:.4f}' + ' > 0.05)' if apply_calibration else 'No (val ECE=' + f'{val_ece:.4f}' + ' <= 0.05, model well-calibrated)'}**
- Test prevalence: {prevalence:.3f} ({prevalence*100:.1f}% Suspicious)
- Dynamic Brier naive baseline: prevalence × (1-prevalence) = {brier_naive:.4f}

---

## Quality Gates

| Level | Metric | Threshold | Result | 95% CI | Status |
|-------|--------|-----------|--------|--------|--------|
| L1 Primary | Recall | ≥ {eval_cfg["recall_threshold"]:.2f} | {recall:.4f} | [{recall_ci[0]:.4f}, {recall_ci[1]:.4f}] | {gate(recall, eval_cfg["recall_threshold"])} |
| L2 Guard-rail | Precision | ≥ {eval_cfg["precision_threshold"]:.2f} | {precision:.4f} | [{prec_ci[0]:.4f}, {prec_ci[1]:.4f}] | {gate(precision, eval_cfg["precision_threshold"])} |
| L2 Guard-rail | AUC-ROC | ≥ {eval_cfg["auc_threshold"]:.2f} | {auc_roc:.4f} | [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}] | {gate(auc_roc, eval_cfg["auc_threshold"])} |
| L3 Calibration | Brier | < {brier_naive:.4f} (dynamic) | {brier:.4f} | — | {gate(brier, brier_naive, op="<")} |

**AUC-PR:** {auc_pr:.4f}

**Decision threshold:** {threshold:.4f} (tuned on calibrated validation, locked in artifacts/threshold.txt)

*Note: CIs computed using stratified bootstrap (1,000 resamples, preserves class ratio).*

---

## Confusion Matrix (Test Set — {total:,} images)

| | Predicted Normal | Predicted Suspicious |
|--|-----------------|---------------------|
| **Actually Normal** | TN = {tn:,} | FP = {fp:,} |
| **Actually Suspicious** | FN = {fn:,} | TP = {tp:,} |

- False Negative Rate: {fn/(fn+tp)*100:.1f}% of Suspicious images missed
- False Positive Rate: {fp/(fp+tn)*100:.1f}% of Normal images incorrectly flagged

---

## Cost-Sensitive Evaluation

FN weight = {eval_cfg.get("fn_cost_weight", 5.0):.1f}× (missed finding),  FP weight = {eval_cfg.get("fp_cost_weight", 1.0):.1f}× (unnecessary review)

| Measure | Value |
|---------|-------|
| Expected cost (model) | {cost_stats["expected_cost"]:.1f} |
| Naive baseline cost (predict Normal always) | {cost_stats["naive_cost"]:.1f} |
| **Cost reduction vs naive** | **{cost_stats["cost_reduction_pct"]:.1f}%** |

---

## Statistical Significance

McNemar's test vs naive baseline (correct formula: χ² = (|b-c|-1)²/(b+c)):

- b={mcnemar_stats["b"]} (naive wrong, model right), c={mcnemar_stats["c"]} (naive right, model wrong)
- χ² = {mcnemar_stats["chi2_stat"]:.4f}, p = {mcnemar_stats["p_value"]:.6f}
- Statistically significant (p < 0.05): **{'Yes' if mcnemar_stats["significant"] else 'No'}**

---

## Visual Artifacts

See `reports/eval_artifacts/` for:

- `roc_curve.png` — ROC curve with AUC
- `pr_curve.png` — Precision-Recall curve with AUC-PR
- `calibration_curve.png` — Reliability diagram
- `threshold_sweep.png` — Recall & Precision vs threshold

---

## Interview Answer This Enables

"The model achieved recall of {recall:.2f} (95% CI: {recall_ci[0]:.2f}–{recall_ci[1]:.2f}). The decision

threshold of {threshold:.2f} was tuned on calibrated validation probabilities — never on training data

(overconfident) or test data (leakage). Expected clinical cost was reduced by {cost_stats["cost_reduction_pct"]:.0f}%

vs the naive baseline (fn_weight=5, fp_weight=1). The improvement is statistically significant

(McNemar's χ²={mcnemar_stats["chi2_stat"]:.2f}, p={mcnemar_stats["p_value"]:.4f})."
"""

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(EVAL_REPORT_PATH).write_text(report)
    logger.info("Evaluation report written to %s", EVAL_REPORT_PATH)