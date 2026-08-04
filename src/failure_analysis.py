"""
Failure analysis pipeline for P4 Radiology AI.

CRITICAL CONCEPTUAL DISTINCTION (two separate concepts):

  Triage Tier (routing decision) = based on P(Suspicious)
    Determines which clinical queue the image enters.
    Tier1: P >= 0.80  → auto-priority Suspicious queue
    Tier2: 0.50 <= P < 0.80 → standard Suspicious queue
    Tier3: P < 0.50   → soft flag (if above threshold) or Normal

  Model Confidence (uncertainty measure) = max(P(Suspicious), 1 - P(Suspicious))
    Measures how CERTAIN the model is, regardless of direction.
    P=0.02 → confidence=0.98 (highly confident: Normal)
    P=0.52 → confidence=0.52 (uncertain: near 50/50 boundary)
    P=0.95 → confidence=0.95 (highly confident: Suspicious)

  Using P(Suspicious) alone as "confidence" is WRONG:
    P=0.02 is NOT low confidence — it is high confidence Normal.
    Labeling a confident-but-wrong FN as "low confidence" gives the
    clinical advisor a false interpretation of the failure mode.

Connections:
  Consumes: artifacts/best_model.pt (L6)
  Consumes: artifacts/threshold.txt (L7)
  Consumes: artifacts/test_df.parquet (L6 runner)
  Consumes: reports/eda/eda_summary.json (L3)
  Produces: reports/failure_report.md (gate artifact)
  Produces: artifacts/fp_cases.parquet (for L9 Grad-CAM, L10 fairness)
  Produces: artifacts/fn_cases.parquet (for L9 Grad-CAM, L10 fairness)
"""

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from src.dataset import create_dataloaders
from src.evaluate import get_predictions, load_model_and_config

logger = logging.getLogger(__name__)

FAILURE_REPORT_PATH = "reports/failure_report.md"
EDA_SUMMARY_PATH = "reports/eda/eda_summary.json"
THRESHOLD_PATH = "artifacts/threshold.txt"
FP_PARQUET_PATH = "artifacts/fp_cases.parquet"
FN_PARQUET_PATH = "artifacts/fn_cases.parquet"

# Triage tier thresholds (routing decision — based on P(Suspicious))
TRIAGE_TIER1_MIN = 0.80  # High Suspicious confidence → auto-priority
TRIAGE_TIER2_MIN = 0.50  # Moderate → standard Suspicious queue

# Model confidence thresholds (uncertainty measure — based on max(P, 1-P))
CONF_HIGH = 0.80  # Highly certain (either direction)
CONF_MODERATE = 0.65  # Moderately certain


# ─── Tier and Confidence Assignment ──────────────────────────────────────────


def assign_triage_and_confidence(
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Assign both the triage tier (routing) and model confidence (uncertainty).

    TRIAGE TIER (based on P(Suspicious) — determines clinical queue):
      Tier1: P >= 0.80            → auto-priority Suspicious
      Tier2: 0.50 <= P < 0.80    → standard Suspicious
      Tier3: threshold <= P < 0.50 → soft flag (if above threshold)
      Normal: P < threshold        → predicted Normal

    MODEL CONFIDENCE (based on max(P, 1-P) — measures certainty):
      High:     confidence >= 0.80   (very certain about the prediction)
      Moderate: 0.65 <= conf < 0.80  (reasonably certain)
      Low:      confidence < 0.65    (uncertain — near the 50/50 boundary)

    IMPORTANT: These are independent axes.
      P=0.02 → Triage="Normal", model_confidence=0.98, conf_level="High"
               (model is HIGHLY CONFIDENT it's Normal)
      P=0.52 → Triage="Tier2", model_confidence=0.52, conf_level="Low"
               (model is UNCERTAIN — just above the 50/50 boundary)
      P=0.95 → Triage="Tier1", model_confidence=0.95, conf_level="High"
               (model is HIGHLY CONFIDENT it's Suspicious)

    Args:
        probs:     P(Suspicious) array (N,)
        threshold: locked decision threshold from artifacts/threshold.txt

    Returns:
        DataFrame with columns: triage_tier, model_confidence, conf_level
    """
    # Triage tier: based on raw P(Suspicious)
    triage_tier = np.where(
        probs >= TRIAGE_TIER1_MIN,
        "Tier1",
        np.where(
            probs >= TRIAGE_TIER2_MIN, "Tier2", np.where(probs >= threshold, "Tier3", "Normal")
        ),
    )

    # Model confidence: distance from maximum uncertainty (0.50)
    confidence = np.maximum(probs, 1.0 - probs)

    # Confidence level label
    conf_level = np.where(
        confidence >= CONF_HIGH, "High", np.where(confidence >= CONF_MODERATE, "Moderate", "Low")
    )

    return pd.DataFrame(
        {
            "triage_tier": triage_tier,
            "model_confidence": confidence.round(4),
            "conf_level": conf_level,
        }
    )


# ─── Failure Case Extraction ──────────────────────────────────────────────────


def extract_failure_cases(
    model,
    test_loader,
    test_df: pd.DataFrame,
    threshold: float,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run inference on the test set and extract all FP and FN cases.

    Each returned row contains the original test_df metadata plus:
      probability:      P(Suspicious) from the model
      predicted_label:  0 or 1 (after applying threshold)
      triage_tier:      "Tier1"/"Tier2"/"Tier3"/"Normal" — routing decision
      model_confidence: max(P, 1-P) — uncertainty measure
      conf_level:       "High"/"Moderate"/"Low"
      error_type:       "FP"/"FN"/"TP"/"TN"

    Finding Labels is preserved from test_df for clinical advisor review.
    Without knowing what the NLP system flagged, the advisor cannot distinguish
    label noise errors from genuine model misses.

    Returns:
        (fp_df, fn_df, all_predictions_df)
    """
    labels, probs, _ = get_predictions(model, test_loader, device)
    preds = (probs >= threshold).astype(int)

    tier_conf_df = assign_triage_and_confidence(probs, threshold)

    result_df = test_df.copy().reset_index(drop=True)
    result_df["probability"] = probs
    result_df["predicted_label"] = preds
    result_df["triage_tier"] = tier_conf_df["triage_tier"].values
    result_df["model_confidence"] = tier_conf_df["model_confidence"].values
    result_df["conf_level"] = tier_conf_df["conf_level"].values

    def _error_type(row):
        if row["binary_label"] == 0 and row["predicted_label"] == 1:
            return "FP"
        elif row["binary_label"] == 1 and row["predicted_label"] == 0:
            return "FN"
        elif row["binary_label"] == 1 and row["predicted_label"] == 1:
            return "TP"
        else:
            return "TN"

    result_df["error_type"] = result_df.apply(_error_type, axis=1)

    fp_df = result_df[result_df["error_type"] == "FP"].reset_index(drop=True)
    fn_df = result_df[result_df["error_type"] == "FN"].reset_index(drop=True)

    logger.info(
        "Failure cases extracted:\n"
        "  Total: %d | TP: %d | TN: %d | FP: %d | FN: %d\n"
        "  FP rate: %.1f%% of Normal | FN rate: %.1f%% of Suspicious",
        len(result_df),
        (result_df["error_type"] == "TP").sum(),
        (result_df["error_type"] == "TN").sum(),
        len(fp_df),
        len(fn_df),
        len(fp_df) / max(1, (result_df["binary_label"] == 0).sum()) * 100,
        len(fn_df) / max(1, (result_df["binary_label"] == 1).sum()) * 100,
    )

    return fp_df, fn_df, result_df


# ─── Confidence Pattern Analysis ──────────────────────────────────────────────


def analyse_confidence_patterns(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    all_df: pd.DataFrame,
) -> dict:
    """
    Analyse how errors distribute across model confidence levels.

    Uses model_confidence = max(P, 1-P) — NOT triage tier.

    KEY QUESTIONS:
    1. High-confidence FNs (confidence >= 0.80, predicted Normal):
       Model was very certain it was Normal — catastrophically wrong.
       Most dangerous error type. Root cause: looks Normal by every feature
       the model learned, but pathology is present. NOT a threshold issue.

    2. Low-confidence FNs (confidence < 0.65):
       Model was uncertain — near the decision boundary.
       Expected at recall-optimised thresholds. Some recoverable by
       adjusting threshold (at precision cost).

    3. High-confidence FPs (confidence >= 0.80, predicted Suspicious):
       Systematic over-prediction. Model very confident about a wrong answer.
       Root cause: spurious correlation (AP positioning, normal anatomy
       patterns that trigger false alarms).

    4. Low-confidence FPs (confidence < 0.65):
       Expected at a low threshold. Not concerning unless dominant.
    """

    def conf_breakdown(df, name):
        if len(df) == 0:
            return {
                f"{name}_high_conf_count": 0,
                f"{name}_mod_conf_count": 0,
                f"{name}_low_conf_count": 0,
                f"{name}_mean_confidence": float("nan"),
            }

        counts = df["conf_level"].value_counts().to_dict()

        return {
            f"{name}_high_conf_count": int(counts.get("High", 0)),
            f"{name}_mod_conf_count": int(counts.get("Moderate", 0)),
            f"{name}_low_conf_count": int(counts.get("Low", 0)),
            f"{name}_mean_confidence": round(float(df["model_confidence"].mean()), 4),
            f"{name}_mean_probability": round(float(df["probability"].mean()), 4),
        }

    fp_stats = conf_breakdown(fp_df, "fp")
    fn_stats = conf_breakdown(fn_df, "fn")

    logger.info(
        "Confidence pattern analysis:\n"
        "  FN high-confidence: %d (most dangerous — model was certain, was wrong)\n"
        "  FN low-confidence:  %d (near-boundary — expected at low threshold)\n"
        "  FP high-confidence: %d (systematic over-prediction)\n"
        "  FP low-confidence:  %d (boundary FPs — expected)",
        fn_stats["fn_high_conf_count"],
        fn_stats["fn_low_conf_count"],
        fp_stats["fp_high_conf_count"],
        fp_stats["fp_low_conf_count"],
    )

    return {**fp_stats, **fn_stats}


# ─── Demographic Breakdown ────────────────────────────────────────────────────


def analyse_demographic_breakdown(
    all_df: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """
    Compute recall by gender and age group with bootstrap 95% CI.

    WHY BOOTSTRAP CI IS REQUIRED:
    Reporting a 4pp recall gap between gender groups without a confidence
    interval is statistically incomplete. With small subgroup N, a 4pp gap
    could be pure sampling noise. Bootstrap CI quantifies whether the gap
    is robust or coincidental.

    CONFOUNDING NOTE:
    Subgroup recall differences may not be independent. If elderly patients
    primarily receive AP-view X-rays (bedridden), age-based recall gaps may
    actually be view-position effects. The report explicitly flags this
    confounding — do not interpret subgroup gaps in isolation.

    Age groups (clinical convention):
      < 40:  younger adults — lower baseline disease prevalence
      40–60: middle-aged — rising prevalence
      > 60:  older adults — highest prevalence, highest-stakes detection

    Returns:
        dict with recall, CI, and count per subgroup
    """
    result = {}
    rng = np.random.default_rng(seed=seed)

    def _subgroup_recall_ci(subset_df, n_boot):
        """Compute recall and bootstrap CI for a subgroup DataFrame."""
        suspicious = subset_df[subset_df["binary_label"] == 1]
        if len(suspicious) == 0:
            return float("nan"), float("nan"), float("nan")

        recall = float((suspicious["predicted_label"] == 1).sum() / len(suspicious))

        # Bootstrap
        n = len(suspicious)
        boots = []
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            sample = suspicious.iloc[idx]
            boots.append(float((sample["predicted_label"] == 1).sum() / n))

        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        return recall, ci_lo, ci_hi

    # Gender breakdown
    if "Patient Gender" in all_df.columns:
        for gender in sorted(all_df["Patient Gender"].unique()):
            subset = all_df[all_df["Patient Gender"] == gender]
            recall, ci_lo, ci_hi = _subgroup_recall_ci(subset, n_bootstrap)

            result[f"recall_gender_{gender}"] = round(recall, 4) if not np.isnan(recall) else None
            result[f"recall_gender_{gender}_ci"] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
            result[f"count_gender_{gender}"] = int(len(subset))

            logger.info(
                "Gender %s: recall=%.4f CI=[%.4f, %.4f] n=%d",
                gender,
                recall if not np.isnan(recall) else -1,
                ci_lo,
                ci_hi,
                len(subset),
            )
    else:
        logger.warning("Patient Gender not in test_df — skipping gender breakdown.")

    # Age group breakdown
    if "Patient Age" in all_df.columns:
        age_groups = [
            ("under_40", all_df["Patient Age"] < 40),
            ("40_to_60", (all_df["Patient Age"] >= 40) & (all_df["Patient Age"] < 60)),
            ("over_60", all_df["Patient Age"] >= 60),
        ]

        for label, mask in age_groups:
            subset = all_df[mask]
            recall, ci_lo, ci_hi = _subgroup_recall_ci(subset, n_bootstrap)

            result[f"recall_age_{label}"] = round(recall, 4) if not np.isnan(recall) else None
            result[f"recall_age_{label}_ci"] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
            result[f"count_age_{label}"] = int(len(subset))

            logger.info(
                "Age %s: recall=%.4f CI=[%.4f, %.4f] n=%d",
                label,
                recall if not np.isnan(recall) else -1,
                ci_lo,
                ci_hi,
                len(subset),
            )
    else:
        logger.warning("Patient Age not in test_df — skipping age breakdown.")

    return result


# ─── View Position Error Analysis ─────────────────────────────────────────────


def analyse_view_position_errors(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    all_df: pd.DataFrame,
    eda_summary: dict,
) -> dict:
    """
    Measure recall AND precision by AP/PA, cross-referenced with L3 EDA.

    AP images: portable X-ray for bedridden/critical patients.
    PA images: standard upright for ambulatory outpatients.

    L3 EDA may have flagged a Suspicious rate gap between AP and PA.
    This function validates whether that risk flag manifested as errors.

    Beyond recall, precision by view position is also computed:
    - Low AP precision = model over-flags AP images (spurious correlation)
    - Low AP recall = model under-detects pathology in AP images

    These are different failure modes requiring different mitigations.

    Confidence distribution by view position is also analysed:
    - High-confidence AP FPs = systematic over-prediction (spurious correlation)
    - Low-confidence AP FPs = boundary cases (less concerning)
    """
    if "View Position" not in all_df.columns:
        logger.warning("View Position not in test_df — skipping AP/PA analysis.")
        return {}

    result = {}

    for pos in ["AP", "PA"]:
        subset = all_df[all_df["View Position"] == pos]
        suspicious = subset[subset["binary_label"] == 1]
        _ = subset[subset["binary_label"] == 0]

        # Recall
        recall = (
            float((suspicious["predicted_label"] == 1).sum() / len(suspicious))
            if len(suspicious) > 0
            else float("nan")
        )

        # Precision (among predicted Suspicious in this view)
        pred_susp = subset[subset["predicted_label"] == 1]
        precision = (
            float((pred_susp["binary_label"] == 1).sum() / len(pred_susp))
            if len(pred_susp) > 0
            else float("nan")
        )

        result[f"recall_{pos}"] = round(recall, 4)
        result[f"precision_{pos}"] = round(precision, 4)
        result[f"fn_{pos}_count"] = int(
            all_df[(all_df["View Position"] == pos) & (all_df["error_type"] == "FN")].shape[0]
        )
        result[f"fp_{pos}_count"] = int(
            all_df[(all_df["View Position"] == pos) & (all_df["error_type"] == "FP")].shape[0]
        )
        result[f"n_{pos}"] = int(len(subset))

        # Confidence distribution of FPs for this view
        pos_fp = (
            fp_df[fp_df["View Position"] == pos]
            if "View Position" in fp_df.columns
            else pd.DataFrame()
        )
        if len(pos_fp) > 0:
            high_conf_fp = int((pos_fp["conf_level"] == "High").sum())
            result[f"fp_{pos}_high_conf"] = high_conf_fp

        logger.info(
            "%s: recall=%.4f  precision=%.4f  FN=%d  FP=%d  n=%d",
            pos,
            recall,
            precision,
            result.get(f"fn_{pos}_count", 0),
            result.get(f"fp_{pos}_count", 0),
            len(subset),
        )

    # Cross-reference with L3 EDA finding
    view_eda = eda_summary.get("view_position", {})
    eda_gap = view_eda.get("gap_pp")
    risk_flag = view_eda.get("risk_flag", False)

    result["eda_ap_pa_gap_pp"] = eda_gap
    result["eda_ap_pa_risk_flag"] = risk_flag

    if risk_flag and eda_gap:
        actual_recall_gap = abs(result.get("recall_AP", 0) - result.get("recall_PA", 0)) * 100

        logger.info(
            "EDA AP/PA risk flag was raised (gap=%.1fpp). Actual recall gap in test set: %.1fpp",
            eda_gap,
            actual_recall_gap,
        )

    return result


# ─── Full Failure Analysis Pipeline ───────────────────────────────────────────


def run_failure_analysis(
    config_path: str,
    test_df: pd.DataFrame,
    run_id: str,
) -> dict:
    """
    Orchestrate the complete failure analysis pipeline.

    Steps:
      1. Load model, threshold, EDA summary
      2. Extract FP/FN cases with triage tier AND model confidence
      3. Analyse confidence patterns (using max(P, 1-P) — not P alone)
      4. Analyse demographic breakdown with bootstrap CI
      5. Analyse view position errors (recall + precision)
      6. Save fp_cases.parquet and fn_cases.parquet
      7. Log all stats to MLflow (same run as L6/L7)
      8. Write failure_report.md
    """
    config = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = float(Path(THRESHOLD_PATH).read_text().strip())

    logger.info("Starting failure analysis. threshold=%.4f", threshold)

    # ── Load EDA summary ──────────────────────────────────────────────────────
    eda_summary = {}
    if Path(EDA_SUMMARY_PATH).exists():
        eda_summary = json.loads(Path(EDA_SUMMARY_PATH).read_text())
    else:
        logger.warning("%s not found. AP/PA cross-reference will be limited.", EDA_SUMMARY_PATH)

    # ── Build test DataLoader ─────────────────────────────────────────────────
    _, _, test_loader = create_dataloaders(test_df, test_df, test_df, config)

    # ── Load model ────────────────────────────────────────────────────────────
    model, _ = load_model_and_config(config_path, device)

    # ── Extract failure cases ─────────────────────────────────────────────────
    fp_df, fn_df, all_df = extract_failure_cases(model, test_loader, test_df, threshold, device)

    # ── Analyse patterns ──────────────────────────────────────────────────────
    conf_stats = analyse_confidence_patterns(fp_df, fn_df, all_df)
    demog_stats = analyse_demographic_breakdown(all_df)
    view_stats = analyse_view_position_errors(fp_df, fn_df, all_df, eda_summary)

    # ── Save failure case artifacts ───────────────────────────────────────────
    Path("artifacts").mkdir(exist_ok=True)
    fp_df.to_parquet(FP_PARQUET_PATH, index=False)
    fn_df.to_parquet(FN_PARQUET_PATH, index=False)

    logger.info("Saved %s and %s", FP_PARQUET_PATH, FN_PARQUET_PATH)

    all_stats = {
        "fp_count": len(fp_df),
        "fn_count": len(fn_df),
        "tp_count": int((all_df["error_type"] == "TP").sum()),
        "tn_count": int((all_df["error_type"] == "TN").sum()),
        "threshold": threshold,
        **conf_stats,
        **{k: v for k, v in demog_stats.items() if isinstance(v, (int, float)) and v is not None},
        **{
            k: v
            for k, v in view_stats.items()
            if isinstance(v, (int, float, bool)) and v is not None
        },
    }

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in all_stats.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and not (isinstance(v, float) and np.isnan(v))
            }
        )

    # ── Write failure report ──────────────────────────────────────────────────
    _write_failure_report(
        fp_df=fp_df,
        fn_df=fn_df,
        all_df=all_df,
        conf_stats=conf_stats,
        demog_stats=demog_stats,
        view_stats=view_stats,
        eda_summary=eda_summary,
        threshold=threshold,
        config=config,
    )

    logger.info(
        "Failure analysis complete. FP=%d  FN=%d  Report: %s",
        len(fp_df),
        len(fn_df),
        FAILURE_REPORT_PATH,
    )

    return all_stats


# ─── Report Writer ─────────────────────────────────────────────────────────────


def _write_failure_report(
    fp_df,
    fn_df,
    all_df,
    conf_stats,
    demog_stats,
    view_stats,
    eda_summary,
    threshold,
    config,
) -> None:
    """Write the complete failure analysis report."""

    n_susp = int((all_df["binary_label"] == 1).sum())
    n_norm = int((all_df["binary_label"] == 0).sum())
    fp_rate = len(fp_df) / max(1, n_norm) * 100
    fn_rate = len(fn_df) / max(1, n_susp) * 100

    # ── Tier + confidence breakdown strings ───────────────────────────────────
    def conf_block(prefix, n_total, label):
        if n_total == 0:
            return f"No {label} cases."

        high = conf_stats.get(f"{prefix}_high_conf_count", 0)
        mod = conf_stats.get(f"{prefix}_mod_conf_count", 0)
        low = conf_stats.get(f"{prefix}_low_conf_count", 0)
        mc = conf_stats.get(f"{prefix}_mean_confidence", float("nan"))
        mp = conf_stats.get(f"{prefix}_mean_probability", float("nan"))

        return (
            f"High confidence (≥0.80): {high} ({high / n_total * 100:.1f}%) | "
            f"Moderate (0.65–0.79): {mod} ({mod / n_total * 100:.1f}%) | "
            f"Low (<0.65): {low} ({low / n_total * 100:.1f}%)\n"
            f"  Mean model_confidence: {mc:.4f} | Mean P(Suspicious): {mp:.4f}"
        )

    # ── Demographic section ───────────────────────────────────────────────────
    demog_lines = []
    for gender in ["M", "F"]:
        rk = f"recall_gender_{gender}"
        ck = f"count_gender_{gender}"
        ci = f"recall_gender_{gender}_ci"
        if rk in demog_stats and demog_stats[rk] is not None:
            demog_lines.append(
                f"  Gender {gender}: recall={demog_stats[rk]:.4f} "
                f"95% CI={demog_stats.get(ci, 'n/a')} "
                f"(n={demog_stats.get(ck, '?')})"
            )

    for age in ["under_40", "40_to_60", "over_60"]:
        rk = f"recall_age_{age}"
        ck = f"count_age_{age}"
        ci = f"recall_age_{age}_ci"
        if rk in demog_stats and demog_stats[rk] is not None:
            demog_lines.append(
                f"  Age {age}: recall={demog_stats[rk]:.4f} "
                f"95% CI={demog_stats.get(ci, 'n/a')} "
                f"(n={demog_stats.get(ck, '?')})"
            )

    # ── View position section ─────────────────────────────────────────────────
    view_lines = []
    for pos in ["AP", "PA"]:
        if f"recall_{pos}" in view_stats:
            view_lines.append(
                f"  {pos}: recall={view_stats[f'recall_{pos}']:.4f} | "
                f"precision={view_stats.get(f'precision_{pos}', float('nan')):.4f} | "
                f"FN={view_stats.get(f'fn_{pos}_count', '?')} | "
                f"FP={view_stats.get(f'fp_{pos}_count', '?')} | "
                f"n={view_stats.get(f'n_{pos}', '?')}"
            )

    eda_gap = view_stats.get("eda_ap_pa_gap_pp")
    risk_flag = view_stats.get("eda_ap_pa_risk_flag", False)
    eda_line = (
        f"L3 EDA AP/PA risk flag: {'⚠️ RAISED' if risk_flag else '✅ Not raised'} "
        f"(EDA gap={eda_gap:.1f}pp)"
        if eda_gap is not None
        else "L3 EDA summary not available."
    )

    # ── FN sample for clinical advisor (includes Finding Labels) ─────────────
    fn_sample = fn_df.nsmallest(10, "model_confidence")  # most confident wrong FNs first
    fn_lines = []

    for _, row in fn_sample.iterrows():
        finding_label = str(row.get("Finding Labels", "Unknown"))[:40]
        fn_lines.append(
            f"  - {Path(str(row['image_path'])).name} | "
            f"P(Susp)={row['probability']:.4f} | "
            f"model_conf={row['model_confidence']:.4f} ({row['conf_level']}) | "
            f"Triage={row['triage_tier']} | "
            f"NLP label: **{finding_label}** | "
            f"Age={row.get('Patient Age', '?')} | "
            f"Gender={row.get('Patient Gender', '?')} | "
            f"View={row.get('View Position', '?')}"
        )

    fn_block = "\n".join(fn_lines) if fn_lines else "  No false negatives in test set."

    _ = f"""# Failure Analysis Report — P4 Radiology AI

## 60 Seconds Academy — AI & ML

**Decision threshold:** {threshold:.4f} (tuned on calibrated validation — Decision 14)

**Total test images:** {len(all_df):,} ({n_susp:,} Suspicious, {n_norm:,} Normal)

---

## Conceptual Clarification: Triage Tier vs Model Confidence

These are two separate concepts. Do not conflate them.

**Triage Tier (routing decision — based on P(Suspicious)):**

Determines which clinical queue the image enters.

- Tier1 (P ≥ 0.80): auto-priority Suspicious queue
- Tier2 (0.50 ≤ P < 0.80): standard Suspicious queue
- Tier3 (threshold ≤ P < 0.50): soft-flag — human review required
- Normal (P < threshold): predicted Normal

**Model Confidence (uncertainty — based on max(P, 1-P)):**

Measures how certain the model is, in either direction.

- P=0.95 → confidence=0.95 (highly confident: Suspicious)
- P=0.02 → confidence=0.98 (highly confident: Normal — this is NOT "low confidence")
- P=0.52 → confidence=0.52 (genuinely uncertain — near 50/50 boundary)

**Important note on threshold:**

With decision threshold={threshold:.4f}, some images predicted Suspicious will have low

model confidence (e.g., P=0.42 → confidence=0.58). This is expected and correct —

the system flags them because the cost of missing a Suspicious finding (fn_weight=5)

outweighs the precision cost. Low-confidence Suspicious predictions should receive

mandatory human-in-the-loop review before clinical action.

---

## Summary of Errors

| Error Type | Count | Rate |

|---|---|---|

| False Positives (FP) | {len(fp_df):,} | {fp_rate:.1f}% of Normal images |

| False Negatives (FN) | {len(fn_df):,} | {fn_rate:.1f}% of Suspicious images |

| True Positives (TP) | {conf_stats.get("tp_count", "?"):,} | — |

| True Negatives (TN) | {conf_stats.get("tn_count", "?"):,} | — |

FNs are the primary clinical concern (fn_weight=5.0 — a missed finding costs 5x more

than an unnecessary review).

---

## Error Confidence Distribution

*Uses model_confidence = max(P(Suspicious), 1-P(Suspicious)), NOT triage tier.*

*This correctly identifies whether the model was certain about a wrong answer.*

### False Positives ({len(fp_df)} Normal images incorrectly flagged as Suspicious)

{conf_block("fp", len(fp_df), "FP")}

**Interpretation:**

- HIGH-confidence FPs: model was very certain it was Suspicious — systematic over-prediction.
  Most likely cause: spurious correlation (AP imaging characteristics, specific Normal
  anatomy patterns). See view position analysis below.
- LOW-confidence FPs: expected at a low threshold. Not a primary concern.

### False Negatives ({len(fn_df)} Suspicious images missed, predicted Normal)

{conf_block("fn", len(fn_df), "FN")}

**Interpretation:**

- HIGH-confidence FNs (model_confidence ≥ 0.80): MODEL WAS VERY CERTAIN IT WAS NORMAL
  — but was wrong. These are the most dangerous errors. The model actively rejected
  pathological evidence. Root cause: images look Normal by every learned feature.
  NOT a threshold issue. May indicate a data gap for this image type.
- LOW-confidence FNs (confidence < 0.65): near the decision boundary. Some recoverable
  by lowering threshold — but at a precision cost. Quantify with threshold sweep from L7.

---

## Root Cause Hypotheses

| # | Observation | Hypothesis | Type | Evidence Needed |

|---|-------------|------------|------|-----------------|

| 1 | [populate: e.g., FN high-conf count=X] | Model confidently predicted Normal for Suspicious images — data gap for this image type | Data gap | Clinical advisor review of sampled high-conf FNs |

| 2 | [populate: e.g., FP high-conf AP count=X] | Model over-predicts Suspicious for AP images — spurious correlation with imaging position | Spurious correlation | Confirmed if AP FP rate >> PA FP rate (see below) |

| 3 | [populate from demographic analysis] | [Hypothesis for largest recall disparity] | [Data/spurious/noise] | [Bootstrap CI — if CI excludes zero, likely real] |

**Hypothesis types:**

- *Data gap*: underrepresented image type in training → model has no learned signal
- *Spurious correlation*: model learned a non-causal feature (position, brightness)
- *Label noise*: NLP extraction error — the label may be wrong, not the model
- *Noise*: subgroup gap within bootstrap CI — may not be real

---

## Demographic Breakdown

{chr(10).join(demog_lines) if demog_lines else "  Demographic columns not available."}

**Note on confounding variables:**

Subgroup recall differences are not necessarily independent. If elderly patients

predominantly receive AP-view X-rays (bedridden/immobile), an age-based recall gap

may actually be a view-position effect rather than an age effect. Do not interpret

subgroup gaps in isolation — check for overlap with view position patterns below

before drawing conclusions.

**Interpretation guideline:**

If the bootstrap 95% CI for a recall gap excludes zero, the gap is likely robust.

If the CI spans zero, the gap may be sampling noise and should not be reported as a finding.

---

## View Position Error Analysis

{chr(10).join(view_lines) if view_lines else "  View Position column not available."}

{eda_line}

**Interpretation:**

- If AP recall << PA recall: model has more difficulty detecting pathology in AP images.
  Consistent with the L3 EDA risk flag — imaging position correlated with Suspicious labels.
- If AP precision << PA precision: model over-flags AP images (spurious correlation confirmed).
  High-confidence AP FPs are the strongest evidence: model is systematically certain about
  Normal AP images being Suspicious.
- [populate: actual AP vs PA recall gap = ?pp | AP vs PA precision gap = ?pp]

---

## L3 EDA Finding Cross-Reference

| L3 Risk Flag | Confirmed in L8? | Evidence |

|---|---|---|

| AP/PA Suspicious rate gap = {f"{eda_gap:.1f}pp" if eda_gap else "N/A"} | [Yes/No/Partial — populate] | [AP recall=X, PA recall=Y, gap=Zpp] |

| Long-tail patient dominance | [Yes/No/Not checked] | [Any patient appearing multiple times in FP/FN?] |

| Pixel intensity risk flag | [Not checked in this version — check in L9 Grad-CAM] | — |

| Gender Suspicious rate difference | [Yes/No — populate from demographic section] | [Recall M=X 95% CI=[a,b], F=Y 95% CI=[c,d]] |

---

## Sampled False Negative Cases for Clinical Advisor

The following {min(len(fn_df), 10)} FN cases are sorted by **lowest model_confidence** —

the images where the model was most uncertain. The **NLP label** is the disease string

extracted from the original radiology report. Use it to assess whether the finding is

genuinely visible in the image (model miss) or whether the NLP extraction may be

incorrect (label noise).

{fn_block}

**To access full FP/FN case lists (for L9 Grad-CAM visualisation):**

```python
import pandas as pd
fn_df = pd.read_parquet("artifacts/fn_cases.parquet")
fp_df = pd.read_parquet("artifacts/fp_cases.parquet")
"""
