"""
Unit tests for src/failure_analysis.py — L8 contracts.

Tests (5):

  U20: P=0.02 → model_confidence=0.98, conf_level=High (NOT Low)

  U21: P=0.52 → model_confidence=0.52, conf_level=Low

  U22: Triage tier boundaries at P=0.80 (Tier1) and P=0.50 (Tier2)

  U23: FP+FN+TP+TN == total test set size

  U24: FP rows binary_label=0; FN rows binary_label=1

"""

import numpy as np
import pytest
import torch
from pytest import approx

pytestmark = pytest.mark.unit


def test_u20_confidence_high_for_low_prob():
    """U20: P=0.02 → model_confidence=0.98 (High) — NOT Low Confidence.

    P=0.02 means the model is 98% certain it is Normal.

    The PREVIOUS BUG used P(Suspicious) alone as confidence, which labeled
    P=0.02 as 'Low Confidence' — completely backwards.

    The fix: confidence = max(P, 1-P).
    """
    from src.failure_analysis import assign_triage_and_confidence

    result = assign_triage_and_confidence(np.array([0.02]), threshold=0.35)

    assert float(result["model_confidence"].iloc[0]) == approx(0.98, abs=1e-4), (
        "P=0.02 must give model_confidence=0.98 (98% confident Normal)"
    )
    assert result["conf_level"].iloc[0] == "High", (
        "P=0.02 must be conf_level='High' — model is highly certain, just about Normal"
    )


def test_u21_confidence_low_for_ambiguous():
    """U21: P=0.52 → model_confidence=0.52 (Low) — genuinely uncertain."""
    from src.failure_analysis import assign_triage_and_confidence

    result = assign_triage_and_confidence(np.array([0.52]), threshold=0.35)

    assert float(result["model_confidence"].iloc[0]) == approx(0.52, abs=1e-4)
    assert result["conf_level"].iloc[0] == "Low"


def test_u22_triage_tier_boundaries():
    """U22: Tier boundaries — P=0.80 inclusive Tier1, P=0.50 inclusive Tier2."""
    from src.failure_analysis import assign_triage_and_confidence

    threshold = 0.35

    cases = [
        (0.95, "Tier1"),
        (0.80, "Tier1"),  # boundary inclusive
        (0.79, "Tier2"),
        (0.50, "Tier2"),  # boundary inclusive
        (0.49, "Tier3"),  # above threshold, below Tier2
        (0.36, "Tier3"),
        (0.34, "Normal"),  # below threshold
        (0.02, "Normal"),
    ]

    for prob, expected in cases:
        result = assign_triage_and_confidence(np.array([prob]), threshold)
        actual = result["triage_tier"].iloc[0]
        assert actual == expected, f"P={prob}: expected '{expected}', got '{actual}'"


def test_u23_failure_counts_sum_to_total(synthetic_dataframe, test_config, synthetic_model):
    """U23: FP+FN+TP+TN == total test set size."""
    from src.dataset import create_dataloaders
    from src.failure_analysis import extract_failure_cases

    _, _, loader = create_dataloaders(
        synthetic_dataframe,
        synthetic_dataframe,
        synthetic_dataframe,
        test_config,
        num_workers=0,
    )

    fp, fn, all_df = extract_failure_cases(
        synthetic_model, loader, synthetic_dataframe, 0.5, torch.device("cpu")
    )

    total = len(all_df)
    counts = sum(
        [
            len(fp),
            len(fn),
            int((all_df["error_type"] == "TP").sum()),
            int((all_df["error_type"] == "TN").sum()),
        ]
    )

    assert counts == total


def test_u24_failure_case_labels(synthetic_dataframe, test_config, synthetic_model):
    """U24: FP rows have binary_label=0; FN rows have binary_label=1."""
    from src.dataset import create_dataloaders
    from src.failure_analysis import extract_failure_cases

    _, _, loader = create_dataloaders(
        synthetic_dataframe,
        synthetic_dataframe,
        synthetic_dataframe,
        test_config,
        num_workers=0,
    )

    fp, fn, _ = extract_failure_cases(
        synthetic_model, loader, synthetic_dataframe, 0.5, torch.device("cpu")
    )

    if len(fp) > 0:
        assert (fp["binary_label"] == 0).all()

    if len(fn) > 0:
        assert (fn["binary_label"] == 1).all()
