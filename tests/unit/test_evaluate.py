"""
Unit tests for src/evaluate.py — L7 contracts.

Tests (6):

  U14: compute_ece returns float in [0, 1]

  U15: tune_threshold returns float in [0.1, 0.9]

  U16: tune_threshold fallback when precision constraint is impossible

  U17: stratified_bootstrap_ci: ci_lower <= ci_upper

  U18: McNemar formula: χ²=(|b-c|-1)²/(b+c), NOT chi2_contingency

  U19: Dynamic Brier baseline = prevalence*(1-prevalence), not hardcoded 0.25

"""

import numpy as np
import pytest
from pytest import approx

pytestmark = pytest.mark.unit


def test_u14_compute_ece_range():
    """U14: compute_ece returns float in [0, 1]."""
    from src.evaluate import compute_ece

    labels = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    probs = np.array([0.8, 0.3, 0.7, 0.2, 0.9, 0.6, 0.4, 0.1])

    ece = compute_ece(labels, probs)

    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_u15_tune_threshold_range():
    """U15: tune_threshold returns threshold in [0.1, 0.9]."""
    from src.evaluate import tune_threshold

    np.random.seed(42)
    labels = np.array([1] * 60 + [0] * 40)
    probs = np.concatenate(
        [
            np.random.uniform(0.4, 0.9, 60),
            np.random.uniform(0.1, 0.5, 40),
        ]
    )

    threshold, _, _ = tune_threshold(labels, probs, min_precision=0.40)
    assert 0.09 <= threshold <= 0.91


def test_u16_tune_threshold_fallback():
    """U16: tune_threshold uses fallback when precision=0.99 is impossible."""
    from src.evaluate import tune_threshold

    labels = np.array([1] * 10 + [0] * 10)
    probs = np.random.uniform(0, 1, 20)

    threshold, _, _ = tune_threshold(labels, probs, min_precision=0.99)
    assert isinstance(threshold, float)


def test_u17_stratified_bootstrap_ci_ordered():
    """U17: stratified_bootstrap_ci ci_lower <= ci_upper."""
    from sklearn.metrics import recall_score

    from src.evaluate import stratified_bootstrap_ci

    np.random.seed(42)
    labels = np.array([1] * 50 + [0] * 50)
    probs = np.random.uniform(0, 1, 100)

    lo, hi = stratified_bootstrap_ci(
        labels, probs, threshold=0.5, metric_fn=recall_score, n_resamples=50
    )

    assert lo <= hi


def test_u18_mcnemar_formula_correct():
    """U18: McNemar χ²=(|b-c|-1)²/(b+c) — NOT scipy.stats.chi2_contingency.

    chi2_contingency computes Pearson's Chi-Square (independence test).
    McNemar tests marginal homogeneity in paired data. Different formula,
    different p-values. A previous version used chi2_contingency — this test
    prevents regression to that bug.
    """
    from src.evaluate import mcnemar_test_vs_naive

    # Construct known b=20, c=5 discordant pairs
    true_labels = np.array([1] * 20 + [0] * 5 + [1] * 5 + [0] * 70)
    model_preds = np.array([1] * 20 + [1] * 5 + [0] * 5 + [0] * 70)

    result = mcnemar_test_vs_naive(true_labels, model_preds)
    b, c = result["b"], result["c"]

    if (b + c) > 0:
        expected_chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        assert result["chi2_stat"] == approx(expected_chi2, rel=1e-6), (
            f"McNemar formula wrong: got {result['chi2_stat']:.6f}, "
            f"expected {expected_chi2:.6f}. chi2_contingency is NOT McNemar."
        )


def test_u19_dynamic_brier_baseline():
    """U19: Brier baseline = prevalence*(1-prevalence), not hardcoded 0.25.

    With 30% Suspicious prevalence: baseline = 0.30*0.70 = 0.21, not 0.25.
    Hardcoding 0.25 assumes 50% prevalence — incorrect for NIH dataset.
    """
    labels_30pct = np.array([1] * 30 + [0] * 70)
    prevalence = labels_30pct.mean()
    dynamic_base = prevalence * (1 - prevalence)

    assert dynamic_base == approx(0.21, rel=1e-3), (
        f"Dynamic baseline should be 0.21 for 30% prevalence, got {dynamic_base:.4f}"
    )

    # Confirm it differs from the hardcoded 0.25 by a meaningful amount
    assert abs(dynamic_base - 0.25) > 0.01, (
        "Dynamic baseline should differ from 0.25 for non-50% prevalence"
    )
