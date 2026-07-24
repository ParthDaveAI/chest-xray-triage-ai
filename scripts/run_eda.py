"""
EDA Script — P4 Radiology AI Image Pipeline

60 Seconds Academy — AI & ML

CRITICAL PROTOCOL: ALL ANALYSIS RUNS ON TRAINING SPLIT ONLY.

Rationale: Analysing the full dataset before splitting means the test
set distribution influences training design decisions (augmentation
choices, AP/PA mitigation, subgroup selection). This is data snooping —
a form of leakage. In strict ML protocol, the test set is hidden until
final evaluation in L7.

Execution:
  1. prepare_dataset() creates train/val/test splits
  2. All EDA functions receive train_df only
  3. Test set remains untouched

Run with:
  uv run python scripts/run_eda.py

Outputs (all to reports/eda/):
  class_distribution.png
  demographic_distributions.png
  view_position_correlation.png
  patient_image_count.png
  pixel_intensity_distribution.png
  image_quality_samples.png
  eda_summary.json       ← structured output consumed by L8 and L10
"""

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

# Must be set before any other matplotlib import
# Agg = non-interactive backend — required for CI and headless servers
matplotlib.use("Agg")

# Global seed for all sampling operations in this script
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eda")

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG_PATH = "config/training_config.yaml"
REPORTS_DIR = Path("reports/eda")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds for automated warnings and hard failures
AP_PA_GAP_WARN_THRESHOLD    = 10.0   # pp — log warning and raise risk flag
AP_PA_GAP_FAIL_THRESHOLD    = 25.0   # pp — raise ValueError, pipeline must not proceed
INTENSITY_DIFF_THRESHOLD    = 20.0   # mean pixel intensity units
DOMINANT_PATIENT_THRESHOLD  = 0.005  # fraction of total training images (0.5%)


# ── Section 1: Load and Prepare (Training Split Only) ─────────────────────────

def validate_and_load(
    config: dict,
) -> tuple[pd.DataFrame, str]:
    """
    Validate schema, run prepare_dataset(), return TRAINING split only.

    WHY TRAINING SPLIT ONLY:
    Analysing the full pre-split dataset allows the test distribution to
    influence design decisions (augmentation, stratification strategy,
    subgroup selection). This is data snooping. EDA must operate only on
    the training distribution — the same distribution the model will train on.

    Returns:
        (train_df, split_hash)
        train_df: training split DataFrame with image_path and binary_label
        split_hash: SHA256 of split manifest — links EDA to a specific split
    """
    from src.data_prep import prepare_dataset, validate_dataset_schema

    # Load full CSV for schema validation only
    full_df = pd.read_csv(config["data"]["labels_path"])
    validate_dataset_schema(full_df)
    logger.info("Schema validation passed on full CSV.")

    # Create splits — EDA uses train_df only
    train_df, val_df, test_df, split_hash = prepare_dataset(
        labels_csv_path=config["data"]["labels_path"],
        images_dir=config["data"]["dataset_path"],
        config=config,
    )
    logger.info(
        "Using TRAINING SPLIT ONLY for EDA: %d images, %d patients",
        len(train_df), train_df["Patient ID"].nunique(),
    )
    logger.info("Split hash: %s", split_hash)

    return train_df, split_hash


# ── Section 2: Class Distribution ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> dict:
    """
    Plot Normal vs Suspicious distribution in the training split.

    Important: this reflects the TRAINING distribution, not the full dataset.
    Any class imbalance seen here is what the model will be trained on
    and what class weights (L6) will be calibrated to.
    """
    counts = df["binary_label"].value_counts().sort_index()
    total  = len(df)
    susp_pct   = counts.get(1, 0) / total * 100
    normal_pct = counts.get(0, 0) / total * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(
        ["Normal (0)", "Suspicious (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=colors, edgecolor="black", linewidth=0.8,
    )

    for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({count / total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )

    ax.set_title(
        "Class Distribution — Training Split Only\n(NIH ChestX-ray14 Binary)",
        fontsize=12, pad=12,
    )
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=120)
    plt.close()

    logger.info(
        "Training split class distribution — Normal: %d (%.1f%%), Suspicious: %d (%.1f%%)",
        counts.get(0, 0), normal_pct, counts.get(1, 0), susp_pct,
    )

    return {"normal_count": int(counts.get(0, 0)), "suspicious_count": int(counts.get(1, 0)),
            "normal_pct": round(normal_pct, 2), "suspicious_pct": round(susp_pct, 2)}


# ── Section 3: Demographic Distributions ──────────────────────────────────────

def plot_demographic_distributions(df: pd.DataFrame) -> dict:
    """
    Plot age and gender distributions in the training split by class.

    These demographics are NOT model features — they are excluded due to
    inference-time unavailability.

    Purpose of this analysis:
      1. Document training distribution demographics for the Data Card
      2. Identify any demographic skew that will affect fairness evaluation (L10)
      3. Select appropriate subgroups for recall gap analysis in L10

    NOTE ON CAUSAL VS CORRELATIONAL:
    Even though age correlates with disease prevalence (older patients have
    higher rates of suspicious findings), age is not used as a model feature
    because the causal pathway runs through visible pathology in the image,
    not through patient demographics. Using age as a feature trains the model
    on a correlation rather than the causal signal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Demographic Distributions — Training Split", fontsize=13)

    # Age distribution by class
    label_color_map = {"Normal": "#4CAF50", "Suspicious": "#F44336"}
    df_plot = df.copy()
    df_plot["label_name"] = df_plot["binary_label"].map({0: "Normal", 1: "Suspicious"})

    for label_name, group in df_plot.groupby("label_name"):
        axes[0].hist(
            group["Patient Age"], bins=30, alpha=0.6, label=label_name,
            color=label_color_map.get(label_name, "gray"),
            edgecolor="black", linewidth=0.3,
        )

    axes[0].set_title("Age Distribution by Class")
    axes[0].set_xlabel("Patient Age (years)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # Gender by class
    gender_label = (
        df_plot.groupby(["Patient Sex", "label_name"])
        .size()
        .unstack(fill_value=0)
    )

    gender_label.plot(
        kind="bar", ax=axes[1],
        color={"Normal": "#4CAF50", "Suspicious": "#F44336"},
        edgecolor="black", linewidth=0.5,
    )
    axes[1].set_title("Gender Distribution by Class")
    axes[1].set_xlabel("Patient Sex")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Class")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "demographic_distributions.png", dpi=120)
    plt.close()

    gender_stats = {}
    for gender in ["M", "F"]:
        subset = df[df["Patient Sex"] == gender]
        if len(subset) > 0:
            susp_pct = subset["binary_label"].mean() * 100
            gender_stats[gender] = {
                "count": len(subset),
                "suspicious_pct": round(susp_pct, 2),
            }
            logger.info("Gender %s: %d images, Suspicious=%.1f%%",
                        gender, len(subset), susp_pct)

    return {"gender_stats": gender_stats}


# ── Section 4: AP vs PA View Position Correlation ─────────────────────────────

def check_view_position_correlation(df: pd.DataFrame) -> dict:
    """
    Measure the AP vs PA Suspicious rate gap in the TRAINING split.

    CLINICAL CONTEXT:
    AP (Anterior-Posterior): portable X-ray for bedridden/critical patients.
    Systematically sicker population than ambulatory PA patients.

    If AP images have a much higher Suspicious rate, the model may learn
    imaging position as a discriminative signal rather than pathology.

    PATIENT-LEVEL DOMINANT VIEW ASSIGNMENT (for stratification mitigation):
    Because splits are patient-level (L2), stratification must also be
    patient-level. A patient may have both AP and PA images. Resolution:
    assign each patient to a view position group based on their dominant
    view type — if >50% of a patient's images are AP, classify as AP;
    otherwise classify as PA. This allows patient-level stratification
    without breaking the patient-level split guarantee.

    THRESHOLDS:
      Warning: gap > 10pp — log warning, set risk_flag=True
      Hard fail: gap > 25pp — raise ValueError, pipeline must not proceed
                 An extreme gap of 25pp indicates the spurious correlation
                 is severe enough to make training unreliable.

    Returns:
        dict with ap_susp_pct, pa_susp_pct, gap_pp, risk_flag,
        dominant_view_per_patient (DataFrame for mitigation use)
    """
    if "View Position" not in df.columns:
        logger.warning("No 'View Position' column — AP/PA check skipped.")
        return {}

    # Compute Suspicious rate per view position in training split
    view_stats = (
        df.groupby("View Position")["binary_label"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "suspicious", "count": "total"})
    )
    view_stats["suspicious_pct"] = view_stats["suspicious"] / view_stats["total"] * 100

    logger.info("Training split view position stats:\n%s", view_stats.to_string())

    ap_pct = float(view_stats.loc["AP", "suspicious_pct"]) if "AP" in view_stats.index else 0.0
    pa_pct = float(view_stats.loc["PA", "suspicious_pct"]) if "PA" in view_stats.index else 0.0
    gap    = abs(ap_pct - pa_pct)

    # Compute dominant view per patient for mitigation reference
    patient_view_counts = df.groupby(["Patient ID", "View Position"]).size().unstack(fill_value=0)
    patient_view_counts["dominant_view"] = patient_view_counts.apply(
        lambda row: "AP" if row.get("AP", 0) > row.get("PA", 0) else "PA", axis=1
    )
    dominant_view_df = patient_view_counts[["dominant_view"]].reset_index()
    ap_patient_pct = (dominant_view_df["dominant_view"] == "AP").mean() * 100

    logger.info(
        "Dominant view assignment: %.1f%% of training patients classified as AP-dominant",
        ap_patient_pct,
    )

    if gap > AP_PA_GAP_FAIL_THRESHOLD:
        raise ValueError(
            f"AP/PA SPURIOUS CORRELATION CRITICAL: gap={gap:.1f}pp "
            f"exceeds hard failure threshold ({AP_PA_GAP_FAIL_THRESHOLD}pp).\n"
            f"AP Suspicious rate: {ap_pct:.1f}%, PA: {pa_pct:.1f}%.\n"
            f"Training on this data without mitigation is unreliable.\n"
            f"Apply patient-level dominant-view stratified split and re-run."
        )
    elif gap > AP_PA_GAP_WARN_THRESHOLD:
        logger.warning(
            "AP/PA SPURIOUS CORRELATION RISK: gap=%.1f pp (> %.1f pp threshold).\n"
            "  AP Suspicious: %.1f%%, PA Suspicious: %.1f%%\n"
            "  Model may learn imaging position as discriminative signal.\n"
            "  Proposed mitigation: stratify patient split by dominant view position.\n"
            "  Assign patients to AP/PA group: >50%% of images in that position.\n"
            "  Document in decisions.md Decision 5.",
            gap, AP_PA_GAP_WARN_THRESHOLD, ap_pct, pa_pct,
        )
    else:
        logger.info(
            "AP/PA gap=%.1f pp — within acceptable range (< %.1f pp). No mitigation required.",
            gap, AP_PA_GAP_WARN_THRESHOLD,
        )

    # Plot
    overall_susp_pct = df["binary_label"].mean() * 100
    positions  = view_stats.index.tolist()
    susp_pcts  = view_stats["suspicious_pct"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = []
    for p in susp_pcts:
        if abs(p - overall_susp_pct) > AP_PA_GAP_WARN_THRESHOLD:
            bar_colors.append("#F44336")
        else:
            bar_colors.append("#2196F3")

    bars = ax.bar(positions, susp_pcts, color=bar_colors, edgecolor="black", linewidth=0.8)

    for bar, pct in zip(bars, susp_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.4,
                f"{pct:.1f}%", ha="center", fontsize=11)

    ax.axhline(y=overall_susp_pct, color="black", linestyle="--",
               linewidth=1.2, label=f"Overall: {overall_susp_pct:.1f}%")
    ax.set_title("Suspicious Rate by View Position\n(Training Split Only)", fontsize=11)
    ax.set_xlabel("View Position")
    ax.set_ylabel("Suspicious Rate (%)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    if gap > AP_PA_GAP_WARN_THRESHOLD:
        ax.text(0.98, 0.97,
                f"⚠️  Gap = {gap:.1f}pp\n(> {AP_PA_GAP_WARN_THRESHOLD}pp threshold)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#B71C1C",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", alpha=0.8))

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "view_position_correlation.png", dpi=120)
    plt.close()

    return {
        "ap_susp_pct": round(ap_pct, 2),
        "pa_susp_pct": round(pa_pct, 2),
        "gap_pp": round(gap, 2),
        "risk_flag": gap > AP_PA_GAP_WARN_THRESHOLD,
        "ap_patient_pct": round(ap_patient_pct, 2),
    }


# ── Section 5: Patient Long-Tail Distribution ──────────────────────────────────

def plot_patient_image_count_distribution(df: pd.DataFrame) -> dict:
    """
    Analyse images-per-patient distribution in the training split.

    WHY THIS MATTERS:
    A patient with 50 images in training contributes 50x more gradient
    updates than a typical patient with 1-2 images. The model may learn
    to recognise that specific patient's anatomy rather than generalised
    pathological patterns.

    Patient-level splitting (L2) ensures this patient does not appear in
    test. But the dominance effect within training remains.

    MITIGATION OPTIONS (see decisions.md Decision 6):
      A — Do nothing (if no patient exceeds 0.5% threshold)
      B — Cap images per patient (simple, direct)
      C — Patient-level weighted sampling (complex but preserves full data)

    THRESHOLD: 0.5% of total training images
    """
    images_per_patient = df.groupby("Patient ID").size().sort_values(ascending=False)
    total         = len(df)
    top_count     = int(images_per_patient.iloc[0])
    top_pct       = top_count / total * 100
    threshold_pct = DOMINANT_PATIENT_THRESHOLD * 100

    logger.info(
        "Patient image count (training split):\n"
        "  Patients:           %d\n"
        "  Total images:       %d\n"
        "  Mean per patient:   %.1f\n"
        "  Median per patient: %.1f\n"
        "  Max per patient:    %d (%.3f%% of training)",
        df["Patient ID"].nunique(), total,
        images_per_patient.mean(), images_per_patient.median(),
        top_count, top_pct,
    )

    if top_pct > threshold_pct:
        logger.warning(
            "LONG-TAIL DOMINANCE: Top patient has %d images (%.3f%% > %.1f%% threshold).\n"
            "Document in data/feature_registry.md. Consider Decision 6 mitigation.",
            top_count, top_pct, threshold_pct,
        )
    else:
        logger.info(
            "Long-tail check passed: top patient %.3f%% < %.1f%% threshold.",
            top_pct, threshold_pct,
        )

    top_10 = images_per_patient.head(10)
    logger.info("Top 10 patients by training image count:\n%s", top_10.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Patient Image Count Distribution — Training Split", fontsize=13)

    axes[0].hist(images_per_patient.values, bins=50, color="#2196F3",
                 edgecolor="black", linewidth=0.3)
    axes[0].set_title("All Training Patients (full range)")
    axes[0].set_xlabel("Images per Patient")
    axes[0].set_ylabel("Number of Patients")
    axes[0].spines[["top", "right"]].set_visible(False)

    median_count = int(images_per_patient.median())
    high_vol = images_per_patient[images_per_patient > median_count]

    axes[1].hist(high_vol.values, bins=40, color="#FF9800",
                 edgecolor="black", linewidth=0.3)
    axes[1].set_title(f"High-Volume Patients (> {median_count} images)")
    axes[1].set_xlabel("Images per Patient")
    axes[1].set_ylabel("Number of Patients")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "patient_image_count.png", dpi=120)
    plt.close()

    return {
        "total_patients": df["Patient ID"].nunique(),
        "total_images": int(total),
        "top_patient_image_count": top_count,
        "top_patient_pct": round(top_pct, 3),
        "threshold_exceeded": bool(top_pct > threshold_pct),
    }


# ── Section 6: Image Quality Check ────────────────────────────────────────────

def check_image_quality(
    df: pd.DataFrame,
    images_dir: str,
    n_per_class: int = 5,
) -> None:
    """
    Sample and display images from each class for visual quality inspection.

    Images are loaded as thumbnails (224×224) to avoid I/O bottlenecks.
    At original resolution, loading hundreds of images sequentially is
    heavily I/O bound. Thumbnail loading is sufficient for visual quality
    checking — we only need to confirm images are valid frontal X-rays.

    Context manager (with Image.open() as img) ensures file handles are
    released immediately, preventing OS-level "Too many open files" errors
    if n_per_class is increased later.

    This is a MANUAL check. The output image must be opened and reviewed
    before proceeding to L4. The script cannot automate clinical validity
    assessment — that requires human judgment.
    """
    fig, axes = plt.subplots(2, n_per_class, figsize=(4 * n_per_class, 8))
    fig.suptitle(
        "Image Quality Check — Training Split Samples\n"
        "Row 1: Normal (0)    Row 2: Suspicious (1)\n"
        "MANUAL REVIEW REQUIRED: confirm these are valid frontal chest X-rays",
        fontsize=10,
    )

    for row_idx, label in enumerate([0, 1]):
        label_subset = df[df["binary_label"] == label]
        samples = label_subset.sample(
            min(n_per_class, len(label_subset)), random_state=42
        )

        for col_idx, (_, row) in enumerate(samples.iterrows()):
            img_path = Path(images_dir) / row["Image Index"]
            ax = axes[row_idx, col_idx]

            if img_path.exists():
                try:
                    # Context manager ensures file handle is released
                    # thumbnail() for I/O efficiency — sufficient for visual QC
                    with Image.open(img_path) as img:
                        img.thumbnail((224, 224))
                        img_array = np.array(img.convert("L"))

                    ax.imshow(img_array, cmap="gray", vmin=0, vmax=255)
                    finding_text = str(row.get("Finding Labels", ""))[:20]
                    ax.set_title(
                        f"{'Normal' if label == 0 else 'Suspicious'}\n{finding_text}",
                        fontsize=7, pad=2,
                    )
                except Exception as e:
                    ax.text(0.5, 0.5, f"Load error:\n{e}", ha="center", va="center",
                            transform=ax.transAxes, color="red", fontsize=7)
            else:
                ax.text(0.5, 0.5, "FILE NOT\nFOUND", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=8)

            ax.axis("off")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "image_quality_samples.png", dpi=100)
    plt.close()

    logger.info(
        "Image quality samples saved to %s/image_quality_samples.png\n"
        "ACTION REQUIRED: Open this file and confirm:\n"
        "  - Images are frontal chest X-rays (not corrupted, not blank)\n"
        "  - Normal images show clear lung fields\n"
        "  - Suspicious images may show subtle abnormalities",
        REPORTS_DIR,
    )


# ── Section 7: Pixel Intensity Distribution ────────────────────────────────────

def plot_pixel_intensity_distribution(
    df: pd.DataFrame,
    images_dir: str,
    n_sample: int = 300,
) -> dict:
    """
    Measure mean AND std dev pixel intensity by class in the training split.

    WHY BOTH MEAN AND STD:
    Mean intensity alone is insufficient. Two distributions can have the
    same mean but different variance — one class may have consistently
    moderate intensity while the other has high-contrast images. The
    model could learn contrast patterns as discriminative features.

    RISK:
    Mean difference > 20 units: model may learn brightness as feature
    Std dev ratio > 1.5: model may learn contrast as feature

    Both risks are partially mitigated by colour jitter augmentation
    (brightness=0.2, contrast=0.2 in training_config.yaml).

    I/O EFFICIENCY:
    Images loaded at thumbnail size (224×224) — sufficient for intensity
    statistics. Full-resolution loading of 600 medical images is heavily
    I/O bound and adds several minutes to script runtime unnecessarily.
    """
    normal_means, suspicious_means = [], []
    normal_stds,  suspicious_stds  = [], []

    sampled = df.sample(min(n_sample * 2, len(df)), random_state=42)

    for _, row in sampled.iterrows():
        img_path = Path(images_dir) / row["Image Index"]
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                img.thumbnail((224, 224))  # I/O efficiency
                arr = np.array(img.convert("L"), dtype=np.float32)

            if row["binary_label"] == 0:
                normal_means.append(arr.mean())
                normal_stds.append(arr.std())
            else:
                suspicious_means.append(arr.mean())
                suspicious_stds.append(arr.std())
        except Exception:
            continue

        if len(normal_means) >= n_sample and len(suspicious_means) >= n_sample:
            break

    n_mean  = np.mean(normal_means)
    s_mean  = np.mean(suspicious_means)
    n_std   = np.mean(normal_stds)
    s_std   = np.mean(suspicious_stds)

    mean_diff = abs(n_mean - s_mean)
    std_ratio = max(n_std, s_std) / (min(n_std, s_std) + 1e-8)

    logger.info(
        "Pixel intensity (training split):\n"
        "  Normal    — mean: %.1f, std: %.1f\n"
        "  Suspicious — mean: %.1f, std: %.1f\n"
        "  Mean difference: %.1f (threshold: %.1f)\n"
        "  Std ratio:       %.2f (threshold: 1.5)",
        n_mean, n_std, s_mean, s_std, mean_diff, INTENSITY_DIFF_THRESHOLD, std_ratio,
    )

    if mean_diff > INTENSITY_DIFF_THRESHOLD:
        logger.warning(
            "PIXEL INTENSITY MEAN RISK: difference %.1f > %.1f threshold.\n"
            "Colour jitter augmentation (configured) partially mitigates this.",
            mean_diff, INTENSITY_DIFF_THRESHOLD,
        )

    if std_ratio > 1.5:
        logger.warning(
            "PIXEL INTENSITY CONTRAST RISK: std ratio %.2f > 1.5 threshold.\n"
            "Colour jitter contrast augmentation partially mitigates this.",
            std_ratio,
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Pixel Intensity Statistics — Training Split\n"
        "(Images loaded at 224×224 thumbnail for I/O efficiency)",
        fontsize=11,
    )

    # Mean intensity histograms
    axes[0].hist(normal_means, bins=40, alpha=0.65, label=f"Normal (mean={n_mean:.1f})",
                 color="#4CAF50", edgecolor="black", linewidth=0.3)
    axes[0].hist(suspicious_means, bins=40, alpha=0.65,
                 label=f"Suspicious (mean={s_mean:.1f})",
                 color="#F44336", edgecolor="black", linewidth=0.3)
    axes[0].axvline(n_mean, color="#1B5E20", linestyle="--", linewidth=1.5)
    axes[0].axvline(s_mean, color="#B71C1C", linestyle="--", linewidth=1.5)
    axes[0].set_title("Mean Pixel Intensity per Image")
    axes[0].set_xlabel("Mean Intensity (0=black, 255=white)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Std dev histograms
    axes[1].hist(normal_stds, bins=40, alpha=0.65, label=f"Normal (mean std={n_std:.1f})",
                 color="#4CAF50", edgecolor="black", linewidth=0.3)
    axes[1].hist(suspicious_stds, bins=40, alpha=0.65,
                 label=f"Suspicious (mean std={s_std:.1f})",
                 color="#F44336", edgecolor="black", linewidth=0.3)
    axes[1].set_title("Pixel Intensity Std Dev per Image (Contrast)")
    axes[1].set_xlabel("Std Dev of Pixel Values")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pixel_intensity_distribution.png", dpi=120)
    plt.close()

    return {
        "normal_mean_intensity": round(n_mean, 2),
        "suspicious_mean_intensity": round(s_mean, 2),
        "mean_intensity_diff": round(mean_diff, 2),
        "normal_mean_std": round(n_std, 2),
        "suspicious_mean_std": round(s_std, 2),
        "std_ratio": round(std_ratio, 3),
        "mean_risk_flag": bool(mean_diff > INTENSITY_DIFF_THRESHOLD),
        "contrast_risk_flag": bool(std_ratio > 1.5),
    }


# ── Section 8: Leakage Documentation ─────────────────────────────────────────

def document_leakage_risks() -> None:
    """
    Log the complete target leakage analysis.

    INFERENCE CONTRACT: POST /predict/image receives ONLY a raw PNG file.
    No metadata, no demographics, no CSV columns are available at serving time.

    CAUSAL VS CORRELATIONAL:
    Even features that are theoretically harmless correlates (e.g., age
    correlates with disease prevalence) are excluded because:
      1. They are not available at inference time (inference contract violation)
      2. They train the model on correlation rather than causation
      3. Any population shift that changes the demographic correlation
         would silently degrade performance

    The causal feature is the pathological pattern visible in the image.
    That is the only defensible model input.
    """
    logger.info(
        "\n%s\n"
        "TARGET LEAKAGE ANALYSIS — TRAINING SPLIT EDA\n"
        "%s\n\n"
        "Inference contract: only raw PNG image available at POST /predict/image\n\n"
        "EXCLUDED (available in CSV, absent at inference time):\n"
        "  Patient Age      — inference contract violation\n"
        "                     causal: age correlates with disease, not causes it\n"
        "  Patient Sex   — inference contract violation\n"
        "                     reserved for fairness evaluation (L10) only\n"
        "  View Position    — inference contract violation\n"
        "                     AP/PA spurious correlation risk documented above\n"
        "  Follow-up Number — inference contract violation\n"
        "  Finding Labels   — direct target leakage (IS the label)\n\n"
        "CONCLUSION: Feature set = raw image only. No target leakage.\n"
        "%s",
        "=" * 60, "=" * 60, "=" * 60,
    )


# ── Section 9: Save EDA Summary ───────────────────────────────────────────────

def save_eda_summary(
    split_hash: str,
    class_stats: dict,
    demographic_stats: dict,
    view_stats: dict,
    long_tail_stats: dict,
    intensity_stats: dict,
) -> None:
    """
    Save structured EDA findings to eda_summary.json.

    WHY A JSON SUMMARY:
    EDA plots are for human review. eda_summary.json is for downstream
    programmatic consumption:
      - L8 failure_analysis.py reads view_stats and intensity_stats for
        context when interpreting FP/FN patterns
      - L10 fairness.py reads demographic_stats to confirm subgroup
        selection for recall gap analysis
      - P5 drift monitoring reads this as the training distribution baseline

    The split_hash links the EDA findings to a specific data split,
    ensuring the summary is not accidentally used with a different split.
    """
    summary = {
        "eda_version": "1.0",
        "eda_scope": "training_split_only",
        "split_hash": split_hash,
        "class_distribution": class_stats,
        "demographic_stats": demographic_stats,
        "view_position": view_stats,
        "long_tail": long_tail_stats,
        "pixel_intensity": intensity_stats,
    }

    summary_path = REPORTS_DIR / "eda_summary.json"
    with open(summary_path, "w", newline="\n") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info("EDA summary saved to %s", summary_path)
    logger.info(
        "This file is consumed by:\n"
        "  - L8 failure_analysis.py (interpretation context)\n"
        "  - L10 fairness.py (demographic subgroup confirmation)\n"
        "  - P5 drift monitoring (training distribution baseline)"
    )


# ── Main Runner ────────────────────────────────────────────────────────────────

def run_full_eda() -> dict:
    """
    Run the complete EDA pipeline on the TRAINING SPLIT ONLY.

    Returns the summary dict and saves eda_summary.json.
    """
    config = yaml.safe_load(open(CONFIG_PATH))
    images_dir = config["data"]["dataset_path"]

    logger.info(
        "Starting EDA pipeline.\n"
        "Protocol: training split only — test set remains hidden until L7."
    )

    # Step 1: Load training split
    train_df, split_hash = validate_and_load(config)

    # Step 2–7: Run all analysis on train_df
    class_stats       = plot_class_distribution(train_df)
    demographic_stats = plot_demographic_distributions(train_df)
    view_stats        = check_view_position_correlation(train_df)
    long_tail_stats   = plot_patient_image_count_distribution(train_df)
    check_image_quality(train_df, images_dir)
    intensity_stats   = plot_pixel_intensity_distribution(train_df, images_dir)
    document_leakage_risks()

    # Step 8: Save structured summary for downstream use
    save_eda_summary(
        split_hash, class_stats, demographic_stats,
        view_stats, long_tail_stats, intensity_stats,
    )

    logger.info(
        "\nEDA COMPLETE — Training split only.\n"
        "All outputs in %s\n\n"
        "MANUAL REVIEW REQUIRED:\n"
        "  Open reports/eda/image_quality_samples.png\n"
        "  Confirm images are valid frontal chest X-rays\n\n"
        "NEXT STEPS:\n"
        "  1. Note AP/PA gap from view_position_correlation.png\n"
        "  2. Populate data/feature_registry.md with real values\n"
        "  3. Update decisions.md Decision 5 (AP/PA) and Decision 6 (long-tail)",
        REPORTS_DIR,
    )

    return {
        "split_hash": split_hash,
        "class_stats": class_stats,
        "view_stats": view_stats,
        "long_tail_stats": long_tail_stats,
        "intensity_stats": intensity_stats,
    }


if __name__ == "__main__":
    run_full_eda()