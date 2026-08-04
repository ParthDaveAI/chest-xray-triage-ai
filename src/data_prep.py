"""
Data preparation pipeline for P4 Radiology AI.

Responsibilities:
  1. Validate dataset schema and label distribution before any processing
  2. Load NIH ChestX-ray14 CSV and create binary labels
  3. Build image_path column pointing to actual PNG files
  4. Enforce missing-file failure threshold
  5. Patient-level train/val/test split (prevents data leakage)
  6. Verify zero patient overlap across splits
  7. Log and warn on class distribution skew across splits
  8. Save and hash the split manifest for reproducibility

Critical design decisions:
  - np.sort() on patient IDs before splitting: guarantees determinism
    regardless of CSV row order. Without sorting, random_state=42 produces
    different splits if the CSV is re-ordered upstream.
  - newline='\n' when writing manifest JSON: prevents cross-OS hash
    instability (Windows \r\n vs Linux/macOS \n).
  - Patient-level split: no patient's images appear in more than one split.
    Image-level split causes anatomy memorisation leakage.
  - See decisions.md for full rationale on all design choices.

Data contract to L4 (dataset.py):
  Each returned DataFrame must contain at minimum:
    - image_path (str): absolute path to PNG file
    - binary_label (int): 0 (Normal) or 1 (Suspicious)
  Additional columns (Patient ID, Age, Gender, View Position) are preserved
  for fairness evaluation in L10 but are never accessed by dataset.py.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

NORMAL_LABEL = "No Finding"

# Required columns in Data_Entry_2017.csv
REQUIRED_COLUMNS = [
    "Image Index",
    "Finding Labels",
    "Patient ID",
    "Patient Age",
    "Patient Sex",
    "View Position",
]


# ── Data Validation ────────────────────────────────────────────────────────────


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """
    Validate that the loaded CSV matches expected schema and distribution.

    Run BEFORE any label mapping or splitting. Catches:
      - Missing required columns (renamed/corrupted CSV)
      - Null values in critical columns
      - Unexpected label values after binary mapping
      - Class distribution outside expected range (30–70% Suspicious)

    Raises:
        ValueError: with a specific message if any check fails.
                    Pipeline must not proceed past this point on failure.
    """
    # Check 1: Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset schema validation failed. Missing columns: {missing_cols}\n"
            f"Expected columns: {REQUIRED_COLUMNS}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Check that Data_Entry_2017.csv was downloaded correctly."
        )

    # Check 2: No nulls in critical columns
    null_counts = df[["Image Index", "Finding Labels", "Patient ID"]].isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Null values found in critical columns:\n{null_counts[null_counts > 0]}\n"
            f"Dataset may be corrupted or incompletely downloaded."
        )

    # Check 3: Class distribution sanity after binary mapping
    binary_labels = df["Finding Labels"].apply(lambda x: 0 if str(x).strip() == NORMAL_LABEL else 1)
    suspicious_pct = binary_labels.mean() * 100
    if not (30.0 <= suspicious_pct <= 70.0):
        raise ValueError(
            f"Class distribution outside expected range.\n"
            f"Suspicious: {suspicious_pct:.1f}% (expected 30–70%)\n"
            f"This suggests a label mapping error, corrupted CSV, or incorrect dataset version."
        )

    logger.info(
        "Schema validation passed. Suspicious: %.1f%%, Normal: %.1f%%",
        suspicious_pct,
        100 - suspicious_pct,
    )


# ── Binary Label Creation ──────────────────────────────────────────────────────


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NIH 14-class labels to binary Normal/Suspicious.

    NIH Finding Labels format:
      Single:    "No Finding"  or  "Cardiomegaly"
      Multiple:  "Cardiomegaly|Effusion"

    Mapping rule:
      "No Finding" → 0 (Normal)
      Any other value → 1 (Suspicious)

    Args:
        df: DataFrame with 'Finding Labels' column

    Returns:
        Copy of df with added 'binary_label' column (int: 0 or 1)
    """
    df = df.copy()
    df["binary_label"] = df["Finding Labels"].apply(
        lambda x: 0 if str(x).strip() == NORMAL_LABEL else 1
    )

    normal_count = (df["binary_label"] == 0).sum()
    suspicious_count = (df["binary_label"] == 1).sum()
    total = len(df)

    logger.info(
        "Binary labels created. Normal: %d (%.1f%%), Suspicious: %d (%.1f%%)",
        normal_count,
        normal_count / total * 100,
        suspicious_count,
        suspicious_count / total * 100,
    )

    return df


# ── Patient-Level Split ────────────────────────────────────────────────────────


def create_patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    distribution_warning_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at patient level to prevent data leakage.

    WHY PATIENT-LEVEL, NOT IMAGE-LEVEL:
    The same patient may have multiple X-ray images (follow-up visits).
    An image-level split places images from the same patient into both
    train and test. The model then memorises patient anatomy rather than
    learning to detect pathology. Patient-level split ensures test
    performance reflects generalisation to unseen patients — the actual
    clinical deployment scenario.

    SORTING IS MANDATORY (non-determinism trap):
    df["Patient ID"].unique() returns elements in CSV row order.
    If the CSV row order ever changes, even with random_state=42, the
    split changes. np.sort() guarantees the same ordering regardless of
    upstream CSV changes, making the split truly deterministic.

    TEMPORAL LEAKAGE — KNOWN LIMITATION:
    The NIH dataset includes Follow-up Numbers. A fully rigorous split
    would be time-based (train on earlier scans, test on later). This
    project uses patient-level splitting instead. Temporal leakage from
    longitudinal patient data is a known, documented limitation. In
    production deployment, time-based splitting would be mandatory.

    STRATIFICATION AND DISTRIBUTION CHECK:
    Patient-level splitting does not guarantee class balance across splits
    (unlike stratified image-level splitting). After splitting, the
    Suspicious percentage per split is logged. A warning is raised if any
    split deviates by more than distribution_warning_threshold percentage
    points from the overall distribution. The split is not rejected —
    patient-level integrity takes priority over class balance — but the
    deviation is documented.

    Args:
        df: DataFrame with 'Patient ID' and 'binary_label' columns
        train_ratio: fraction of unique patients for training (default 0.70)
        val_ratio: fraction of unique patients for validation (default 0.15)
        seed: random seed — deterministic only when combined with np.sort()
        distribution_warning_threshold: warn if split Suspicious% deviates
            more than this many percentage points from overall (default 5.0)

    Returns:
        (train_df, val_df, test_df) DataFrames. No patient ID appears in
        more than one split. Verified by _verify_no_patient_overlap().
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    overall_suspicious_pct = df["binary_label"].mean() * 100

    # MANDATORY: sort for determinism — see docstring
    unique_patients = np.sort(df["Patient ID"].unique())

    # Split 1: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
    )

    # Split 2: val vs test from remaining pool
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test_size,
        random_state=seed,
    )

    train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["Patient ID"].isin(test_patients)].reset_index(drop=True)

    # Verify leakage prevention
    _verify_no_patient_overlap(train_patients, val_patients, test_patients)

    # Log split statistics
    for split_name, split_df, patient_arr in [
        ("Train", train_df, train_patients),
        ("Validation", val_df, val_patients),
        ("Test", test_df, test_patients),
    ]:
        suspicious_pct = split_df["binary_label"].mean() * 100
        deviation = abs(suspicious_pct - overall_suspicious_pct)
        deviation_flag = (
            " ⚠️ DISTRIBUTION DEVIATION" if deviation > distribution_warning_threshold else ""
        )
        logger.info(
            "%s: %d patients, %d images, Suspicious=%.1f%% (overall=%.1f%%, deviation=%.1f%%)%s",
            split_name,
            len(patient_arr),
            len(split_df),
            suspicious_pct,
            overall_suspicious_pct,
            deviation,
            deviation_flag,
        )
        if deviation > distribution_warning_threshold:
            logger.warning(
                "%s split Suspicious%% (%.1f%%) deviates %.1f pp from overall (%.1f%%). "
                "This is expected with patient-level splitting but should be documented "
                "in data/data_card.md.",
                split_name,
                suspicious_pct,
                deviation,
                overall_suspicious_pct,
            )

    return train_df, val_df, test_df


def _verify_no_patient_overlap(
    train_patients: np.ndarray,
    val_patients: np.ndarray,
    test_patients: np.ndarray,
) -> None:
    """
    Assert zero patient ID overlap across all three splits.

    This is the data leakage guarantee. If any overlap is detected,
    the pipeline must stop — proceeding would produce metrics that
    do not reflect generalisation to unseen patients.

    Raises:
        ValueError: with overlap counts per pair if any overlap found.
    """
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    tv = len(train_set & val_set)
    tt = len(train_set & test_set)
    vt = len(val_set & test_set)

    if tv > 0 or tt > 0 or vt > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED — patient overlap found:\n"
            f"  Train ∩ Val:  {tv} patients\n"
            f"  Train ∩ Test: {tt} patients\n"
            f"  Val ∩ Test:   {vt} patients\n"
            f"This indicates a bug in the splitting logic. "
            f"Do not proceed until overlap is zero."
        )

    logger.info("Leakage check passed: zero patient overlap across all splits.")


# ── Split Manifest Hash ────────────────────────────────────────────────────────


def save_and_hash_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: str = "data/split_manifest.json",
) -> str:
    """
    Save patient ID assignments and compute a cross-OS-stable SHA256 hash.

    PURPOSE — REPRODUCIBILITY GUARANTEE:
    The manifest records which patients were assigned to each split.
    SHA256 hash of this file = fingerprint of experimental conditions.
    Same hash = same patients = same splits = reproducible experiment.
    This hash is logged to MLflow in L6 alongside git commit hash,
    DVC data hash, and config hash to complete the four-part
    reproducibility chain.

    CROSS-OS STABILITY:
    Using newline='\n' when writing the JSON file ensures identical
    bytes on Windows (default \r\n) and Linux/macOS (\n).
    Without this, the same split produces different SHA256 hashes
    on different operating systems, silently breaking the contract.

    Args:
        train_df, val_df, test_df: split DataFrames with 'Patient ID' column
        save_path: path to write the manifest JSON

    Returns:
        SHA256 hex digest string (64 characters)
    """
    manifest = {
        "split_config": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "sort_before_split": True,  # documents that np.sort() is applied
        },
        "train_patient_ids": sorted(train_df["Patient ID"].unique().tolist()),
        "val_patient_ids": sorted(val_df["Patient ID"].unique().tolist()),
        "test_patient_ids": sorted(test_df["Patient ID"].unique().tolist()),
        "train_image_count": len(train_df),
        "val_image_count": len(val_df),
        "test_image_count": len(test_df),
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # newline='\n' is MANDATORY for cross-OS hash stability
    # Windows default (\r\n) vs Linux/macOS (\n) produces different hashes
    with open(save_path, "w", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    file_hash = hashlib.sha256(Path(save_path).read_bytes()).hexdigest()

    logger.info("Split manifest written: %s\nSHA256: %s", save_path, file_hash)

    return file_hash


# ── Full Preparation Pipeline ──────────────────────────────────────────────────


def prepare_dataset(
    labels_csv_path: str,
    images_dir: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    End-to-end data preparation pipeline.

    Steps:
      1. Load Data_Entry_2017.csv
      2. Validate schema and class distribution
      3. Create binary labels
      4. Build image_path column
      5. Enforce missing-file failure threshold
      6. Optional: subsample for development mode
      7. Patient-level split
      8. Save and hash split manifest

    Args:
        labels_csv_path: path to Data_Entry_2017.csv
        images_dir: directory containing all PNG image files
        config: loaded training_config.yaml dict

    Returns:
        (train_df, val_df, test_df, split_hash)
        Data contract to dataset.py (L4):
          Each DataFrame contains at minimum:
            image_path (str): absolute path to PNG
            binary_label (int): 0 or 1
          Plus preserved metadata: Patient ID, Patient Age,
          Patient Sex, View Position (for L10 fairness evaluation)

    Raises:
        ValueError: if schema validation fails, missing files exceed
                    threshold, or patient overlap is detected.
    """
    logger.info("Starting data preparation pipeline...")

    data_cfg = config["data"]

    # Step 1: Load
    df = pd.read_csv(labels_csv_path)
    logger.info("Loaded CSV: %d rows, %d columns", len(df), len(df.columns))

    # Step 2: Validate schema and distribution BEFORE any processing
    validate_dataset_schema(df)

    # Step 3: Binary labels
    df = create_binary_labels(df)

    # Step 4: Build image paths
    df["image_path"] = df["Image Index"].apply(lambda x: str(Path(images_dir) / x))

    # Step 5: Enforce missing-file failure threshold
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    missing_count = (~exists_mask).sum()
    missing_pct = missing_count / len(df)
    threshold = data_cfg.get("missing_file_threshold", 0.05)

    if missing_pct > threshold:
        raise ValueError(
            f"Missing file threshold exceeded: {missing_count} files missing "
            f"({missing_pct * 100:.1f}% > threshold {threshold * 100:.1f}%).\n"
            f"Verify all image archives are fully extracted.\n"
            f"Expected image count: 112,120"
        )

    if missing_count > 0:
        logger.warning(
            "%d files missing (%.2f%% — within threshold). Removing from dataset.",
            missing_count,
            missing_pct * 100,
        )

    df = df[exists_mask].reset_index(drop=True)
    logger.info("After file check: %d images available", len(df))

    # Step 6: Optional development subset
    if data_cfg.get("use_subset", False):
        subset_size = data_cfg.get("subset_size", 10000)
        # Stratified sampling to preserve class balance in subset
        df = (
            df.groupby("binary_label", group_keys=False)
            .apply(
                lambda x: x.sample(
                    min(len(x), subset_size // 2), random_state=data_cfg["random_seed"]
                )
            )
            .reset_index(drop=True)
        )
        logger.warning(
            "SUBSET MODE: using %d images. "
            "Quality gates will NOT pass. For pipeline verification only.",
            len(df),
        )

    # Step 7: Patient-level split
    train_df, val_df, test_df = create_patient_level_split(
        df,
        train_ratio=data_cfg["train_split"],
        val_ratio=data_cfg["val_split"],
        seed=data_cfg["random_seed"],
    )

    # Step 8: Save and hash split manifest
    split_hash = save_and_hash_split_manifest(train_df, val_df, test_df)

    logger.info("Data preparation pipeline complete. Split hash: %s", split_hash)

    return train_df, val_df, test_df, split_hash
