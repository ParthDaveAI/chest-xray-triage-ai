"""
Unit tests for data preparation — L2 contracts.

Tests (3):

  U1: No-patient-overlap check raises ValueError when patient IDs overlap

  U2: No-patient-overlap check passes when splits are disjoint

  U3: Patient ID sorting is deterministic (np.sort, not arbitrary order)

"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_u1_patient_overlap_raises():
    """U1: _verify_no_patient_overlap raises ValueError when splits share patient IDs.

    This is the L2 leakage prevention circuit breaker. If a patient appears
    in both train and test, evaluation metrics are optimistically biased.
    The circuit breaker must fire — this test proves it does.
    """
    from src.data_prep import _verify_no_patient_overlap

    train_ids = np.array([1, 2, 3, 4, 5])
    val_ids = np.array([6, 7, 8, 9, 10])
    test_ids = np.array([4, 5, 11, 12, 13])  # 4 and 5 overlap with train

    with pytest.raises(ValueError, match="overlap"):
        _verify_no_patient_overlap(train_ids, val_ids, test_ids)


def test_u2_patient_overlap_passes_disjoint():
    """U2: _verify_no_patient_overlap passes when splits are disjoint."""
    from src.data_prep import _verify_no_patient_overlap

    train_ids = np.array([1, 2, 3])
    val_ids = np.array([4, 5, 6])
    test_ids = np.array([7, 8, 9])

    # Should not raise
    _verify_no_patient_overlap(train_ids, val_ids, test_ids)


def test_u3_patient_id_sort_deterministic():
    """U3: np.sort(patient_ids) produces deterministic order regardless of input order.

    The L2 contract requires np.sort() — not Python's set() or arbitrary order —
    to guarantee reproducible splits across OS and Python versions.
    """
    ids_a = np.array([5, 3, 1, 4, 2])
    ids_b = np.array([2, 4, 1, 3, 5])

    assert np.array_equal(np.sort(ids_a), np.sort(ids_b))
    assert np.sort(ids_a).tolist() == [1, 2, 3, 4, 5]
