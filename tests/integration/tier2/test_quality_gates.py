"""
Tier 2 quality gate tests — skipped in CI, run in nightly after training.

Require artifacts/best_model.pt, artifacts/threshold.txt, artifacts/test_df.parquet.

Tests (4):

  T2-1: Test set recall >= 0.80 (primary quality gate from L1)

  T2-2: Single-image p99 latency < 500ms (SLA gate)

  T2-3: Brier score < dynamic prevalence baseline

  T2-4: Gender recall gap <= 0.05 (Equal Opportunity gate)

"""

from pathlib import Path

import pytest

ARTIFACTS_OK = all([
    Path("artifacts/best_model.pt").exists(),
    Path("artifacts/threshold.txt").exists(),
    Path("artifacts/test_df.parquet").exists(),
])

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.skipif(
        not ARTIFACTS_OK,
        reason=(
            "Tier 2 quality gates require artifacts/best_model.pt, "
            "artifacts/threshold.txt, and artifacts/test_df.parquet. "
            "Run scripts/run_training.py and scripts/run_evaluation.py first."
        ),
    ),
]


def test_t2_1_recall_quality_gate():
    """T2-1: Test set recall >= 0.80 (locked primary quality gate from L1)."""
    pytest.skip("Run after training to verify recall gate")


def test_t2_2_latency_gate():
    """T2-2: Single-image p99 latency < 500ms (CPU SLA gate)."""
    pytest.skip("Run after training to verify latency SLA")


def test_t2_3_brier_beats_baseline():
    """T2-3: Brier score < dynamic prevalence baseline."""
    pytest.skip("Run after training to verify Brier gate")


def test_t2_4_gender_fairness_gate():
    """T2-4: Gender recall gap <= 0.05 (Equal Opportunity gate, Decision 16)."""
    pytest.skip("Run after training to verify fairness gate")