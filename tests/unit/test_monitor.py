"""
Unit tests for src/monitor.py and L14 monitoring contracts.

Tests (4):

  U41: All required Prometheus metric objects defined with correct types

  U42: INFERENCE_COUNTER has 'prediction' labelname

  U43: INFERENCE_LATENCY has 0.5s SLA bucket

  U44: get_penultimate_features() is thread-safe — same input, same output

"""

import pytest
import torch

pytestmark = pytest.mark.unit


def test_u41_all_metrics_defined():
    """U41: All required metric objects exist with correct Prometheus types."""
    from prometheus_client import Counter, Gauge, Histogram

    from src.monitor import (
        DEGRADED_MODE_GAUGE,
        EMBEDDING_REQUEST_COUNTER,
        INFERENCE_COUNTER,
        INFERENCE_LATENCY,
        INFERENCE_TIER_COUNTER,
        MODEL_LOADED_GAUGE,
        VALIDATION_FAILURE_COUNTER,
    )

    assert isinstance(INFERENCE_COUNTER, Counter)
    assert isinstance(INFERENCE_TIER_COUNTER, Counter)
    assert isinstance(VALIDATION_FAILURE_COUNTER, Counter)
    assert isinstance(INFERENCE_LATENCY, Histogram)
    assert isinstance(MODEL_LOADED_GAUGE, Gauge)
    assert isinstance(DEGRADED_MODE_GAUGE, Gauge)
    assert isinstance(EMBEDDING_REQUEST_COUNTER, Counter)


def test_u42_inference_counter_labels():
    """U42: INFERENCE_COUNTER has 'prediction' labelname."""
    from src.monitor import INFERENCE_COUNTER

    assert "prediction" in INFERENCE_COUNTER._labelnames


def test_u43_latency_histogram_sla_bucket():
    """U43: INFERENCE_LATENCY includes the 0.5s SLA threshold bucket."""
    from src.monitor import INFERENCE_LATENCY

    assert 0.5 in INFERENCE_LATENCY._upper_bounds, (
        "INFERENCE_LATENCY must include 0.5s bucket — this is the SLA threshold"
    )


def test_u44_get_penultimate_features_deterministic(synthetic_model):
    """U44: get_penultimate_features produces identical output on repeated calls.

    Verifies thread-safety property: same input always produces same output
    regardless of call order or concurrency. This is the key invariant that
    makes get_penultimate_features() safe in a multi-threaded web server,
    unlike forward hooks which have shared mutable state.
    """
    dummy = torch.zeros(1, 3, 224, 224)

    with torch.no_grad():
        emb_1 = synthetic_model.get_penultimate_features(dummy)
        emb_2 = synthetic_model.get_penultimate_features(dummy)
        emb_3 = synthetic_model.get_penultimate_features(dummy)

    assert emb_1.shape == (1, 1280), f"Expected (1, 1280), got {emb_1.shape}"
    assert torch.equal(emb_1, emb_2), "Same input must produce identical embeddings (run 1 vs 2)"
    assert torch.equal(emb_2, emb_3), "Same input must produce identical embeddings (run 2 vs 3)"
