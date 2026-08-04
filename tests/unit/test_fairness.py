"""
Unit tests for src/fairness.py — L10 contracts.

Tests (4):

  U25: Zero Suspicious subgroup → recall=None, no crash

  U26: Gap CI point estimate matches direct recall difference

  U27: FAIRNESS_THRESHOLD == 0.05

  U28: Calibration ECE in [0, 1] per subgroup

"""

import numpy as np
import pandas as pd
import pytest
from pytest import approx

pytestmark = pytest.mark.unit


@pytest.fixture
def fairness_df():
    np.random.seed(42)
    n = 200

    return pd.DataFrame(
        {
            "binary_label": np.array([1] * 100 + [0] * 100),
            "predicted_label": np.array([1] * 80 + [0] * 20 + [0] * 60 + [1] * 40),
            "probability": np.random.beta(2, 3, n),
            "Patient Gender": ["M"] * 100 + ["F"] * 100,
            "Patient Age": np.random.randint(20, 80, n),
        }
    )


def test_u25_zero_suspicious_subgroup(fairness_df):
    """U25: Subgroup with 0 Suspicious → recall=None, no crash."""
    from src.fairness import compute_subgroup_recall

    df_no = fairness_df.copy()
    df_no.loc[df_no["Patient Gender"] == "M", "binary_label"] = 0

    recalls = compute_subgroup_recall(df_no, "Patient Gender", n_bootstrap=20)

    assert recalls["M"]["recall"] is None
    assert recalls["M"]["low_reliability"] is True


def test_u26_gap_ci_point_estimate(fairness_df):
    """U26: compute_gap_bootstrap_ci point estimate matches direct recall difference.

    This tests that we bootstrap the GAP directly — not compute per-subgroup
    CIs and infer gap significance from overlap (which is statistically wrong).
    """
    from src.fairness import compute_gap_bootstrap_ci, compute_subgroup_recall

    recalls = compute_subgroup_recall(fairness_df, "Patient Gender", n_bootstrap=50)

    pt, lo, hi = compute_gap_bootstrap_ci(fairness_df, "Patient Gender", "M", "F", n_bootstrap=50)

    # Skip if we don't have both groups with Suspicious cases
    if recalls["M"]["recall"] is not None and recalls["F"]["recall"] is not None:
        expected = recalls["M"]["recall"] - recalls["F"]["recall"]
        assert pt == approx(expected, abs=0.01)
        assert lo <= hi
    else:
        # Both groups need Suspicious cases for valid test
        pytest.skip("Not enough Suspicious cases in both gender groups")


def test_u27_fairness_threshold_locked():
    """U27: FAIRNESS_THRESHOLD locked at 0.05 (Decision 16)."""
    from src.fairness import FAIRNESS_THRESHOLD

    assert FAIRNESS_THRESHOLD == approx(0.05), "FAIRNESS_THRESHOLD must be 0.05 per Decision 16"


def test_u28_calibration_ece_range(fairness_df):
    """U28: Subgroup ECE in [0, 1] for all groups."""
    from src.fairness import compute_subgroup_calibration

    cal = compute_subgroup_calibration(fairness_df, "Patient Gender")

    for group, stats in cal.items():
        assert 0.0 <= stats["ece"] <= 1.0
        assert 0.0 <= stats["brier_score"] <= 1.0
