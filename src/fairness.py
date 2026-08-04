"""
Fairness evaluation pipeline for P4 Radiology AI.

PRIMARY METRIC: Equal Opportunity (recall parity)

SECONDARY METRIC: Calibration fairness (ECE and Brier per subgroup)

IMPORTANT FRAMING:

Equal Opportunity is the correct primary metric for THIS use case, not a

universally superior metric. The choice is justified by the harm model:

fn_weight=5 >> fp_weight=1, so differential miss rates are the primary harm.

Other fairness notions (equalized odds, calibration fairness, predictive

parity) are also monitored. See decisions.md Decision 16.

CRITICAL STATISTICAL FIX:

Bootstrap CI is computed for the GAP directly, not just per-subgroup recalls.

Overlapping per-subgroup CIs do NOT directly test whether the gap is

statistically significant. This is a common statistical error in fairness

analysis.

CALIBRATION FAIRNESS:

Equal recall with unequal calibration means one subgroup's confidence scores

are less trustworthy — clinicians get misleading probability information.

Subgroup ECE and Brier score are computed alongside recall parity.

CAUSAL NOTE:

Age, gender, and view position are causally entangled. An age recall gap

may be mediated by view position (elderly → AP-view → harder images).

Causal structure is documented but cannot be resolved from observational data.

See fairness_report.md Section 5.

Connections:

  Consumes: all_predictions_df with predicted_label, binary_label, metadata

  Produces: reports/fairness_report.md

  Produces: reports/model_card.md

  Feeds: MLflow run (same run_id as L6-L9)

"""

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

FAIRNESS_REPORT_PATH = "reports/fairness_report.md"
MODEL_CARD_PATH = "reports/model_card.md"
EDA_SUMMARY_PATH = "reports/eda/eda_summary.json"

# Fairness threshold: max acceptable recall gap (policy-defined, not scientific law)
FAIRNESS_THRESHOLD = 0.05  # 5pp — Decision 16

# Minimum Suspicious cases for a reliable recall estimate
MIN_N_SUSPICIOUS = 50


# ─── Subgroup Recall ──────────────────────────────────────────────────────────


def compute_subgroup_recall(
    all_df: pd.DataFrame,
    subgroup_col: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute recall and 95% bootstrap CI for each subgroup value.

    Bootstrap CI is computed per-subgroup (for reporting).
    For gap significance, use compute_gap_ci() which bootstraps the gap directly.

    Subgroups with N_suspicious < MIN_N_SUSPICIOUS (50) are flagged as
    low-reliability — their recall estimate is reported but marked as
    statistically unreliable.

    Returns:
        dict keyed by subgroup value, each with:
          recall, ci_lower, ci_upper, n_total, n_suspicious,
          low_reliability (bool: True if n_suspicious < MIN_N_SUSPICIOUS)
    """
    if subgroup_col not in all_df.columns:
        logger.warning("Column '%s' not in DataFrame — skipping.", subgroup_col)
        return {}

    rng = np.random.default_rng(seed=seed)
    results = {}

    for group_val in sorted(all_df[subgroup_col].dropna().unique()):
        group_df = all_df[all_df[subgroup_col] == group_val]
        suspicious = group_df[group_df["binary_label"] == 1]
        n_total = len(group_df)
        n_susp = len(suspicious)
        low_rel = n_susp < MIN_N_SUSPICIOUS

        if n_susp == 0:
            logger.warning(
                "Subgroup %s=%s: 0 Suspicious cases — recall undefined.",
                subgroup_col,
                group_val,
            )
            results[str(group_val)] = {
                "recall": None,
                "ci_lower": None,
                "ci_upper": None,
                "n_total": n_total,
                "n_suspicious": 0,
                "low_reliability": True,
            }
            continue

        recall = float((suspicious["predicted_label"] == 1).sum() / n_susp)

        # Bootstrap CI (within subgroup — for reporting only)
        boots = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_susp, n_susp, replace=True)
            sample = suspicious.iloc[idx]
            boots.append(float((sample["predicted_label"] == 1).sum() / n_susp))

        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))

        results[str(group_val)] = {
            "recall": round(recall, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n_total": n_total,
            "n_suspicious": n_susp,
            "low_reliability": low_rel,
        }

        rel_note = " ⚠️  LOW RELIABILITY (n_suspicious < 50)" if low_rel else ""
        logger.info(
            "Equal Opportunity — %s=%s: recall=%.4f 95%%CI=[%.4f,%.4f] (n_susp=%d, n_total=%d)%s",
            subgroup_col,
            group_val,
            recall,
            ci_lo,
            ci_hi,
            n_susp,
            n_total,
            rel_note,
        )

    return results


# ─── Gap CI — Bootstrap the Gap Directly ─────────────────────────────────────


def compute_gap_bootstrap_ci(
    all_df: pd.DataFrame,
    subgroup_col: str,
    group_a: str,
    group_b: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap the recall gap distribution directly between two subgroups.

    WHY NOT USE PER-SUBGROUP CIs:
    Overlapping per-subgroup 95% CIs do NOT imply the gap is non-significant.
    Two groups can both have CI=[0.80, 0.90] while the gap CI is [0.01, 0.09]
    (gap is real). The correct test is to bootstrap the gap itself.

    For each bootstrap resample:
      1. Resample Suspicious cases within group A (stratified within subgroup)
      2. Resample Suspicious cases within group B (separately)
      3. Compute recall for each
      4. Record gap = recall_A - recall_B

    The 2.5th and 97.5th percentiles of the gap distribution = 95% CI for the gap.
    If CI excludes zero: gap is statistically established at α=0.05.

    NOTE ON MULTIPLE COMPARISONS:
    If testing N pairwise comparisons, Bonferroni correction divides α by N.
    For 5 comparisons across dimensions, adjusted α=0.01 → CI must exclude
    zero at the 99% level. This function uses α=0.05 (exploratory framing).
    For confirmatory fairness audits, apply Bonferroni correction.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed=seed)
    df_a = all_df[(all_df[subgroup_col] == group_a) & (all_df["binary_label"] == 1)]
    df_b = all_df[(all_df[subgroup_col] == group_b) & (all_df["binary_label"] == 1)]

    if len(df_a) == 0 or len(df_b) == 0:
        return float("nan"), float("nan"), float("nan")

    recall_a = float((df_a["predicted_label"] == 1).sum() / len(df_a))
    recall_b = float((df_b["predicted_label"] == 1).sum() / len(df_b))
    point = recall_a - recall_b

    gap_boots = []
    for _ in range(n_bootstrap):
        s_a = df_a.iloc[rng.choice(len(df_a), len(df_a), replace=True)]
        s_b = df_b.iloc[rng.choice(len(df_b), len(df_b), replace=True)]

        r_a = float((s_a["predicted_label"] == 1).sum() / len(df_a))
        r_b = float((s_b["predicted_label"] == 1).sum() / len(df_b))

        gap_boots.append(r_a - r_b)

    ci_lo = float(np.percentile(gap_boots, 2.5))
    ci_hi = float(np.percentile(gap_boots, 97.5))

    return round(point, 4), round(ci_lo, 4), round(ci_hi, 4)


# ─── Recall Gap Matrix ────────────────────────────────────────────────────────


def compute_recall_gap_matrix(
    subgroup_recalls: dict,
    all_df: pd.DataFrame,
    subgroup_col: str,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute pairwise recall gaps with direct gap bootstrap CIs.

    Uses compute_gap_bootstrap_ci() — not per-subgroup CI overlap.

    Gap significance: CI excludes zero → gap is statistically established.
    A gap > FAIRNESS_THRESHOLD (0.05) is flagged as a fairness concern.
    """
    valid_groups = {k: v for k, v in subgroup_recalls.items() if v["recall"] is not None}
    groups = list(valid_groups.keys())
    gaps = {}

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            gap_pt, gap_lo, gap_hi = compute_gap_bootstrap_ci(
                all_df, subgroup_col, a, b, n_bootstrap=n_bootstrap
            )

            abs_gap = abs(gap_pt)
            gap_sig = (gap_lo > 0) or (gap_hi < 0)  # CI excludes zero
            concern = abs_gap > FAIRNESS_THRESHOLD

            key = f"{a}_vs_{b}"
            gaps[key] = {
                "group_a": a,
                "group_b": b,
                "recall_a": valid_groups[a]["recall"],
                "recall_b": valid_groups[b]["recall"],
                "gap": abs_gap,
                "gap_signed": gap_pt,
                "gap_ci_lower": gap_lo,
                "gap_ci_upper": gap_hi,
                "gap_significant": gap_sig,
                "concern": concern,
            }

            level = "⚠️  CONCERN" if concern else "✅ ok"
            sig = "(significant)" if gap_sig else "(not significant)"
            logger.info(
                "Gap %s vs %s: %.4f 95%%CI=[%.4f,%.4f] %s %s",
                a,
                b,
                abs_gap,
                gap_lo,
                gap_hi,
                level,
                sig,
            )

    return gaps


# ─── Calibration Fairness ─────────────────────────────────────────────────────


def compute_subgroup_calibration(
    all_df: pd.DataFrame,
    subgroup_col: str,
    n_bins: int = 10,
) -> dict:
    """
    Compute ECE and Brier score per subgroup (calibration fairness).

    CALIBRATION FAIRNESS:
    A model with equal recall across groups may still be systematically
    over- or under-confident for specific subgroups. If the model's
    probabilities are poorly calibrated for one group, clinicians
    reading that group's predictions receive misleading confidence information.

    ECE = Σ_b (|bin_b|/N) × |mean_confidence(bin_b) - accuracy(bin_b)|
    Brier = mean((p_predicted - y_true)²)

    If ECE for one subgroup is significantly higher than another, the
    model's probabilities are less trustworthy for the higher-ECE group.

    Args:
        all_df:       predictions DataFrame with probability, binary_label,
                      predicted_label, and subgroup_col
        subgroup_col: demographic column name
        n_bins:       number of calibration bins (default 10)

    Returns:
        dict keyed by subgroup value with ece and brier_score
    """
    if subgroup_col not in all_df.columns or "probability" not in all_df.columns:
        logger.warning(
            "Column '%s' or 'probability' not in DataFrame — skipping calibration.",
            subgroup_col,
        )
        return {}

    results = {}

    for group_val in sorted(all_df[subgroup_col].dropna().unique()):
        group_df = all_df[all_df[subgroup_col] == group_val]
        if len(group_df) == 0:
            continue

        probs = group_df["probability"].values
        labels = group_df["binary_label"].values

        # ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(labels)

        for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue

            bin_conf = probs[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

        # Brier score
        brier = float(np.mean((probs - labels) ** 2))

        results[str(group_val)] = {
            "ece": round(float(ece), 4),
            "brier_score": round(brier, 4),
            "n_total": len(group_df),
        }

        logger.info(
            "Calibration fairness — %s=%s: ECE=%.4f, Brier=%.4f (n=%d)",
            subgroup_col,
            group_val,
            ece,
            brier,
            len(group_df),
        )

    return results


# ─── Full Fairness Evaluation ─────────────────────────────────────────────────


def run_fairness_evaluation(
    config_path: str,
    all_predictions_df: pd.DataFrame,
    run_id: str,
) -> dict:
    """
    Orchestrate Equal Opportunity + calibration fairness across all dimensions.

    Dimensions evaluated:
      1. Patient Gender (M vs F)
      2. Patient Age groups (< 40, 40-60, ≥ 60)
      3. View Position (AP vs PA)

    For each dimension:
      a. Recall per subgroup with per-subgroup CI (for reporting)
      b. Pairwise recall gap with DIRECT gap bootstrap CI (for significance)
      c. Calibration fairness (ECE + Brier per subgroup)

    MULTIPLE COMPARISONS NOTE:
    This analysis is framed as EXPLORATORY, not confirmatory.
    For confirmatory fairness audit (regulatory submission), apply
    Bonferroni correction: adjusted α = 0.05 / n_comparisons.
    Document this distinction in the fairness report.

    CAUSAL CONFOUNDING:
    Age and View Position are causally entangled. An age recall gap
    may be mediated by view position. Causal structure is documented
    but cannot be resolved from observational data alone.
    """
    config = yaml.safe_load(open(config_path))
    df = all_predictions_df.copy()

    # Create age group column
    if "Patient Age" in df.columns:
        df["age_group"] = pd.cut(
            df["Patient Age"],
            bins=[0, 40, 60, 200],
            labels=["under_40", "40_to_60", "over_60"],
            right=False,
        ).astype(str)

    results = {}

    for dim_name, col_name in [
        ("gender", "Patient Gender"),
        ("age", "age_group"),
        ("view_position", "View Position"),
    ]:
        recalls = compute_subgroup_recall(df, col_name)
        gaps = compute_recall_gap_matrix(recalls, df, col_name)
        calibration = compute_subgroup_calibration(df, col_name)

        any_concern = any(v["concern"] for v in gaps.values())

        results[dim_name] = {
            "recalls": recalls,
            "gaps": gaps,
            "calibration": calibration,
            "any_concern": any_concern,
        }

    any_concern = any(results[d]["any_concern"] for d in ["gender", "age", "view_position"])
    results["any_fairness_concern"] = any_concern
    results["fairness_threshold"] = FAIRNESS_THRESHOLD

    logger.info(
        "Fairness: gender_concern=%s age_concern=%s view_concern=%s overall=%s",
        results["gender"]["any_concern"],
        results["age"]["any_concern"],
        results["view_position"]["any_concern"],
        any_concern,
    )

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow_metrics = {}

    for dim_name in ["gender", "age", "view_position"]:
        dim = results[dim_name]

        for gv, stats in dim["recalls"].items():
            if stats["recall"] is not None:
                sg = gv.replace(" ", "_")
                mlflow_metrics[f"fairness_{dim_name}_{sg}_recall"] = stats["recall"]
                mlflow_metrics[f"fairness_{dim_name}_{sg}_ci_lo"] = stats["ci_lower"]
                mlflow_metrics[f"fairness_{dim_name}_{sg}_ci_hi"] = stats["ci_upper"]

        for pk, gd in dim["gaps"].items():
            mlflow_metrics[f"fairness_{dim_name}_{pk}_gap"] = gd["gap"]
            mlflow_metrics[f"fairness_{dim_name}_{pk}_concern"] = float(gd["concern"])
            mlflow_metrics[f"fairness_{dim_name}_{pk}_sig"] = float(gd["gap_significant"])

        for gv, cal in dim["calibration"].items():
            sg = gv.replace(" ", "_")
            mlflow_metrics[f"fairness_{dim_name}_{sg}_ece"] = cal["ece"]
            mlflow_metrics[f"fairness_{dim_name}_{sg}_brier"] = cal["brier_score"]

    mlflow_metrics["fairness_any_concern"] = float(any_concern)
    mlflow_metrics["equal_opportunity_pass"] = float(not any_concern)
    mlflow_metrics["fairness_threshold"] = FAIRNESS_THRESHOLD

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(mlflow_metrics)

    _write_fairness_report(results, config)

    return results


# ─── Fairness Report Writer ───────────────────────────────────────────────────


def _write_fairness_report(results: dict, config: dict) -> None:
    """Write reports/fairness_report.md."""

    threshold = results["fairness_threshold"]
    any_c = results["any_fairness_concern"]
    status = "⚠️  FAIRNESS CONCERNS DETECTED" if any_c else "✅ ALL FAIRNESS CHECKS PASSED"

    def fmt_recalls(recalls, calibration, label):
        if not recalls:
            return f"  {label}: no data\n"

        lines = [f"\n**{label} — Equal Opportunity (Recall Parity):**\n"]
        lines.append("| Subgroup | Recall | Per-subgroup 95% CI | N Susp | Reliability |")
        lines.append("|---|---|---|---|---|")

        for g, s in sorted(recalls.items()):
            if s["recall"] is None:
                lines.append(f"| {g} | N/A | N/A | 0 | ❌ No data |")
            else:
                rel = "⚠️  Low (n<50)" if s["low_reliability"] else "✅ OK"
                lines.append(
                    f"| {g} | {s['recall']:.4f} | [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}] "
                    f"| {s['n_suspicious']} | {rel} |"
                )

        if calibration:
            lines.append(f"\n**{label} — Calibration Fairness (ECE and Brier):**\n")
            lines.append("| Subgroup | ECE | Brier Score | N |")
            lines.append("|---|---|---|---|")

            for g, c in sorted(calibration.items()):
                lines.append(
                    f"| {g} | {c['ece']:.4f} | {c['brier_score']:.4f} | {c['n_total']:,} |"
                )

        return "\n".join(lines)

    def fmt_gaps(gaps, label):
        if not gaps:
            return f"  {label}: no pairwise gaps\n"

        lines = [f"\n**{label} — Recall Gap (direct bootstrap CI):**\n"]
        lines.append("| Pair | Gap | 95% CI (gap) | Significant? | Concern? |")
        lines.append("|---|---|---|---|---|")

        for _, g in sorted(gaps.items()):
            sig = "✅ Yes" if g["gap_significant"] else "No"
            con = "⚠️  YES" if g["concern"] else "✅ No"
            lines.append(
                f"| {g['group_a']} vs {g['group_b']} | {g['gap']:.4f} | "
                f"[{g['gap_ci_lower']:.4f}, {g['gap_ci_upper']:.4f}] | {sig} | {con} |"
            )

        lines.append(
            "\n*Gap CI computed by bootstrapping the gap distribution directly "
            "(not from per-subgroup CI overlap — which is statistically incorrect).*"
        )

        return "\n".join(lines)

    eda_note = ""
    if Path(EDA_SUMMARY_PATH).exists():
        eda = json.loads(Path(EDA_SUMMARY_PATH).read_text())
        ap_gap = eda.get("view_position", {}).get("gap_pp")
        if ap_gap:
            eda_note = f"\n*L3 EDA AP/PA Suspicious rate gap was {ap_gap:.1f}pp.*\n"

    report = f"""# Fairness Evaluation Report — P4 Radiology AI

## 60 Seconds Academy — AI & ML

**Primary metric:** Equal Opportunity (recall parity)

**Secondary metric:** Calibration fairness (ECE + Brier per subgroup)

**Fairness threshold:** recall gap ≤ {threshold:.2f} ({threshold * 100:.0f}pp) — policy-defined, not scientific law

**Overall status:** {status}

---

## Fairness Metric Framing

Equal Opportunity is the primary metric for this use case, not the universally

correct metric. The choice is justified by the clinical harm model (fn_weight=5):

differential miss rates are the primary clinical harm, so recall parity is the

primary fairness requirement.

**Multiple fairness notions monitored:**

- Equal Opportunity (recall parity): primary gate

- Calibration fairness (ECE/Brier per subgroup): secondary check

- Demographic parity and equalized odds: not primary for this use case

  (see decisions.md Decision 16 for full rationale)

**Multiple comparisons note:**

This analysis is EXPLORATORY. Testing fairness across multiple dimensions

increases Type I error (false fairness alarms). For a confirmatory fairness

audit (regulatory submission), apply Bonferroni correction:

  adjusted α = 0.05 / comparisons

A gap CI must exclude zero at the 99% level for confirmed significance.

---

## Dimension 1: Gender Fairness

{fmt_recalls(results["gender"]["recalls"], results["gender"]["calibration"], "Gender")}

{fmt_gaps(results["gender"]["gaps"], "Gender")}

**Status:** {"⚠️  Fairness concern" if results["gender"]["any_concern"] else "✅ No concern (gap within threshold)"}

---

## Dimension 2: Age Group Fairness

Age groups: under_40 (< 40), 40_to_60 (40–60), over_60 (≥ 60)

{fmt_recalls(results["age"]["recalls"], results["age"]["calibration"], "Age Group")}

{fmt_gaps(results["age"]["gaps"], "Age Group")}

**Status:** {"⚠️  Fairness concern" if results["age"]["any_concern"] else "✅ No concern (gap within threshold)"}

---

## Dimension 3: View Position Fairness (AP vs PA)

{eda_note}

{fmt_recalls(results["view_position"]["recalls"], results["view_position"]["calibration"], "View Position")}

{fmt_gaps(results["view_position"]["gaps"], "View Position")}

**Status:** {"⚠️  Fairness concern" if results["view_position"]["any_concern"] else "✅ No concern (gap within threshold)"}

---

## Causal Confounding

Age, gender, and view position are causally entangled — they are not independent

fairness dimensions:

- Elderly patients → more likely bedridden → more AP-view X-rays → different image quality

- Elderly patients → higher disease prevalence → images may be more complex

An observed age recall gap may be mediated by view position. If so, improving

AP-view recall would fix both the age and view position gaps simultaneously. If

the age gap is direct (model fails on elderly anatomy independently), view-position

stratification will not resolve it.

**This causal structure cannot be resolved from observational data alone.**

It is documented here and in the model card as a known limitation.

---

## Label Bias Consideration

NIH labels are NLP-extracted from radiology reports. NLP extraction quality may

vary by subgroup — reports written for different patient demographics may use

different terminology, verbosity, or clinical conventions. An apparent model

fairness gap may partially reflect label bias (different label quality across

subgroups) rather than purely model bias.

Evidence from L8 clinical advisor review: [populate — did the advisor find

plausible labels for the lower-recall subgroup's FN cases? or suspect labels?]

---

## Fairness-Aware Threshold Option

{"**⚠️ A fairness concern was detected. Consider the fairness-aware threshold option below.**" if any_c else "No action required at current threshold."}

If recall gap > {threshold:.2f}, a lower threshold for the disadvantaged subgroup

restores recall parity at the cost of higher FP rate for that subgroup.

| | Uniform threshold (current) | Subgroup threshold (option) |

|---|---|---|

| Clinical equity | Gap remains | Recall parity restored |

| FP rate | Uniform | Higher for disadvantaged group |

| Regulatory | No demographic decisions | May need explicit justification |

| Status | Default | Requires sign-off (Decision 17) |

---

## Summary

| Dimension | Max Recall Gap | Gap Significant? | ECE Range | Status |

|---|---|---|---|---|

| Gender | {max((v["gap"] for v in results["gender"]["gaps"].values()), default=0):.4f} | [populate] | [populate] | {"⚠️  CONCERN" if results["gender"]["any_concern"] else "✅ Pass"} |

| Age Group | {max((v["gap"] for v in results["age"]["gaps"].values()), default=0):.4f} | [populate] | [populate] | {"⚠️  CONCERN" if results["age"]["any_concern"] else "✅ Pass"} |

| View Position | {max((v["gap"] for v in results["view_position"]["gaps"].values()), default=0):.4f} | [populate] | [populate] | {"⚠️  CONCERN" if results["view_position"]["any_concern"] else "✅ Pass"} |

**Equal Opportunity gate:** {"❌ NOT MET" if any_c else "✅ PASSED"}

"""

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(FAIRNESS_REPORT_PATH).write_text(report)
    logger.info("Fairness report written to %s", FAIRNESS_REPORT_PATH)


# ─── Model Card ───────────────────────────────────────────────────────────────


def write_model_card(
    config_path: str,
    eval_results: dict,
    fairness_results: dict,
    failure_stats: dict,
    gradcam_summary_path: str = "reports/gradcam/summary.md",
) -> None:
    """Write reports/model_card.md (Mitchell et al. 2019 structure)."""

    config = yaml.safe_load(open(config_path))
    eval_cfg = config["evaluation"]

    threshold = eval_results.get("threshold", "N/A")
    recall = eval_results.get("recall", "N/A")
    precision = eval_results.get("precision", "N/A")
    auc_roc = eval_results.get("auc_roc", "N/A")
    auc_pr = eval_results.get("auc_pr", "N/A")
    brier = eval_results.get("brier", "N/A")
    brier_b = eval_results.get("brier_naive", "N/A")
    r_ci = eval_results.get("recall_ci", ("N/A", "N/A"))
    cost_r = eval_results.get("cost_stats", {}).get("cost_reduction_pct", "N/A")
    mcnemar_p = eval_results.get("mcnemar", {}).get("p_value", "N/A")
    all_pass = eval_results.get("quality_gates", {}).get("all_pass", False)

    fair_any = fairness_results.get("any_fairness_concern", False)
    fp_count = failure_stats.get("fp_count", "N/A")
    fn_count = failure_stats.get("fn_count", "N/A")
    fn_high = failure_stats.get("fn_high_conf_count", "N/A")
    fp_high = failure_stats.get("fp_high_conf_count", "N/A")

    gradcam_note = "[See reports/gradcam/summary.md]"
    if Path(gradcam_summary_path).exists():
        gradcam_note = (
            "Grad-CAM heatmaps generated for FN, FP, TP, and TN priority cases "
            "(sorted by model confidence). Clinical observations in reports/gradcam/summary.md."
        )

    # Format threshold safely
    if isinstance(threshold, float):
        threshold_str = f"{threshold:.4f}"
    else:
        threshold_str = str(threshold)

    # Format recall safely
    if isinstance(recall, float):
        recall_str = f"{recall:.4f}"
    else:
        recall_str = str(recall)

    # Format precision safely
    if isinstance(precision, float):
        precision_str = f"{precision:.4f}"
    else:
        precision_str = str(precision)

    # Format auc_roc safely
    if isinstance(auc_roc, float):
        auc_roc_str = f"{auc_roc:.4f}"
    else:
        auc_roc_str = str(auc_roc)

    # Format auc_pr safely
    if isinstance(auc_pr, float):
        auc_pr_str = f"{auc_pr:.4f}"
    else:
        auc_pr_str = str(auc_pr)

    # Format brier safely
    if isinstance(brier, float):
        brier_str = f"{brier:.4f}"
    else:
        brier_str = str(brier)

    # Format brier_b safely
    if isinstance(brier_b, float):
        brier_b_str = f"{brier_b:.4f}"
    else:
        brier_b_str = str(brier_b)

    # Format CI values safely
    if isinstance(r_ci[0], float):
        ci_lo_str = f"{r_ci[0]:.4f}"
    else:
        ci_lo_str = str(r_ci[0])

    if isinstance(r_ci[1], float):
        ci_hi_str = f"{r_ci[1]:.4f}"
    else:
        ci_hi_str = str(r_ci[1])

    # Format cost reduction safely
    if isinstance(cost_r, float):
        cost_r_str = f"{cost_r:.1f}"
    else:
        cost_r_str = str(cost_r)

    # Format mcnemar p safely
    if isinstance(mcnemar_p, float):
        mcnemar_str = f"{mcnemar_p:.4f}"
    else:
        mcnemar_str = str(mcnemar_p)

    card = f"""# Model Card — P4 Radiology AI Image Pipeline

## 60 Seconds Academy — AI & ML

*Following Mitchell et al. (2019). Integrates findings from L7 (metrics), L8 (failure analysis), L9 (explainability), L10 (fairness).*

---

## 1. Model Details

| Property | Value |

|---|---|

| Architecture | EfficientNet-B0, ImageNet pretrained, two-phase fine-tuning |

| Task | Binary: Normal vs Suspicious (frontal chest X-ray) |

| Decision threshold | {threshold_str} (calibrated validation, Decision 14) |

| Embedding dim | 1,280 (penultimate layer, P5 monitoring) |

| Parameters | ~5.3M total |

| Training | Phase 1: frozen backbone; Phase 2: full fine-tune at phase1_lr/10 |

| Explainability | Grad-CAM (backbone.features[-1]), target layer verified |

---

## 2. Intended Use

**Primary use:** Automated first-pass screening of frontal chest X-rays.

Flags Suspicious images for radiologist review. Screening aid — reduces workload.

**OUT OF SCOPE — must NOT be used for:**

- Definitive diagnosis of any specific disease

- Multi-disease classification (binary only)

- Paediatric-only populations (underrepresented in training)

- Replacement of radiologist review

- Deployment without prospective validation at deployment site

- Populations significantly different from NIH Clinical Center demographics

- Clinical decisions without qualified oversight

---

## 3. Factors

**Factors known to affect performance:**

| Factor | Direction | Evidence Source |

|---|---|---|

| View Position (AP vs PA) | [populate: AP/PA recall gap] | L10 fairness, L8 failure analysis |

| Patient Age | [populate: max age group gap] | L10 fairness |

| Patient Gender | [populate: M vs F gap] | L10 fairness |

| Calibration by subgroup | [populate: ECE range across groups] | L10 calibration fairness |

| Image quality | Degraded performance expected | L8 failure analysis |

| Label noise | Performance ceiling bounded | data/data_card.md |

**Causal entanglement note:** Age, view position, and severity are causally entangled.

An age-based recall gap may be mediated by view position (elderly → AP views).

See reports/fairness_report.md Section "Causal Confounding."

**Known spurious correlations:** [Populate from L9 Grad-CAM observations]

---

## 4. Metrics

**Test set performance (threshold={threshold_str}):**

| Metric | Value | 95% CI | Gate | Status |

|---|---|---|---|---|

| **Recall** | {recall_str} | [{ci_lo_str}, {ci_hi_str}] | ≥ {eval_cfg["recall_threshold"]:.2f} | {"✅ PASS" if all_pass else "[populate]"} |

| Precision | {precision_str} | — | ≥ {eval_cfg["precision_threshold"]:.2f} | [populate] |

| AUC-ROC | {auc_roc_str} | — | ≥ {eval_cfg["auc_threshold"]:.2f} | [populate] |

| AUC-PR | {auc_pr_str} | — | — | — |

| Brier Score | {brier_str} | — | < {brier_b_str} | [populate] |

| Equal Opportunity | [populate from L10] | — | max gap ≤ 0.05 | {"⚠️ CONCERN" if fair_any else "✅ PASS"} |

Cost reduction vs naive baseline: {cost_r_str}% (fn_weight=5, fp_weight=1)

McNemar's p vs naive: {mcnemar_str}

---

## 5. Evaluation Data

| Property | Value |

|---|---|

| Source | NIH ChestX-ray14 (test split — never seen during training or threshold tuning) |

| Split | Patient-level (no patient in both train and test) |

| Test images | ~16,800 (15% of dataset) |

| Class balance | ~54% Normal, ~46% Suspicious |

| Threshold tuning | On CALIBRATED VALIDATION split (Decision 14) |

---

## 6. Training Data

| Property | Value |

|---|---|

| Source | NIH ChestX-ray14 (112,120 frontal-view X-rays) |

| Labels | NLP-extracted from radiology reports — known error rate |

| Training images | ~78,400 (70% patient-level) |

| Augmentation | Horizontal flip (clinical review pending Decision 8), rotation ±15°, colour jitter |

| Label noise | Present — see data/data_card.md for details |

| Single-site bias | NIH Clinical Center, Washington DC only |

---

## 7. Quantitative Analyses

**Equal Opportunity (recall parity) — key results:**

| Dimension | Max Gap | Gap CI excludes 0? | Calibration (ECE range) | Status |

|---|---|---|---|---|

| Gender | {max((v["gap"] for v in fairness_results.get("gender", {}).get("gaps", {}).values()), default=0):.4f} | [populate] | [populate from L10] | {"⚠️" if fairness_results.get("gender", {}).get("any_concern") else "✅"} |

| Age group | {max((v["gap"] for v in fairness_results.get("age", {}).get("gaps", {}).values()), default=0):.4f} | [populate] | [populate] | {"⚠️" if fairness_results.get("age", {}).get("any_concern") else "✅"} |

| View Position | {max((v["gap"] for v in fairness_results.get("view_position", {}).get("gaps", {}).values()), default=0):.4f} | [populate] | [populate] | {"⚠️" if fairness_results.get("view_position", {}).get("any_concern") else "✅"} |

*Gap significance tested by bootstrapping the gap distribution directly.*

*Full subgroup recall tables and calibration in reports/fairness_report.md.*

**Error analysis (from L8):**

- False Negatives: {fn_count} | High-confidence FNs: {fn_high}

- False Positives: {fp_count} | High-confidence FPs: {fp_high}

**Explainability:** {gradcam_note}

---

## 8. Ethical Considerations

1. **Label noise:** NLP-extracted labels have a known error rate. Model performance ceiling is bounded by label quality.

2. **Label bias by subgroup:** NLP extraction quality may vary by demographic — reports for different subgroups may use different terminology. An apparent fairness gap may partially reflect label bias, not model bias.

3. **Single-site bias:** All training data from NIH Clinical Center. Generalisation to other sites not validated.

4. **Racial/ethnic representation gap:** No racial/ethnic labels in NIH dataset. Bias along these dimensions cannot be measured or mitigated without demographic labels. Historical evidence shows medical imaging AI can exhibit racial performance gaps.

5. **Causal entanglement:** Observed demographic fairness gaps have entangled causal pathways. Mitigation requires understanding which causal path drives the gap.

6. **Grad-CAM as auditing tool, not explanation:** Spatial heatmaps show localisation, not reasoning. A visually plausible heatmap is necessary but not sufficient evidence of correct clinical understanding.

7. **Fairness-aware threshold not implemented:** If fairness gaps are confirmed, subgroup-specific thresholds are documented as an option (Decision 17). Not deployed pending regulatory review.

8. **Horizontal flip augmentation:** Simulates situs inversus (Decision 8). Pending clinical advisor sign-off.

---

## 9. Caveats and Recommendations

**Mandatory deployment constraints:**

| Constraint | Rationale |

|---|---|

| Tier 3 predictions require human review | Model confidence < 0.65 |

| AP-view images: additional scrutiny if view gap confirmed | AP/PA correlation risk (L8/L10) |

| No paediatric-only use | Underrepresented in training |

| Prospective validation required | Single-site training |

| Recalibration at new site | Prevalence may differ |

| Monitor embedding drift (P5) | Distribution shift expected |

| Monthly subgroup recall review | Fairness can drift |

**Monitoring plan (P5):**

- Penultimate embedding distribution: KS test + PSI monthly

- PSI > 0.2 → retraining evaluation triggered

- Subgroup recall re-evaluated after any model update

- Calibration re-checked after population shift

**Reproducibility:**

- Git hash: [MLflow run params]

- DVC data hash: [MLflow run params]

- Split manifest hash: [MLflow run params]

- Config hash: [MLflow run params]

- MLflow run_id: [L6 training]

"""

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(MODEL_CARD_PATH).write_text(card)
    logger.info("Model card written to %s", MODEL_CARD_PATH)
