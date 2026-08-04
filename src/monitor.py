"""
Prometheus metric definitions for P4 Radiology AI serving.

Kept separate from serve.py to:
  - Prevent circular imports
  - Allow independent testing of metric definitions
  - Centralise all metric names in one place

LABEL CARDINALITY:
  All labels have bounded, low cardinality (2–6 values).
  Never add patient ID, file name, or request ID as a label.
  High-cardinality labels cause Prometheus OOM.

THREAD SAFETY:
  All prometheus_client types are thread-safe.
  predict_image() and get_embedding() run in FastAPI's threadpool.

SINGLE-WORKER MODE:
  Standard prometheus_client (single-process) is correct for --workers 1.
  Multi-worker deployments require PROMETHEUS_MULTIPROC_DIR.
"""

from prometheus_client import Counter, Gauge, Histogram

INFERENCE_COUNTER = Counter(
    name          = "p4_inference_total",
    documentation = "Total inference requests by prediction class",
    labelnames    = ["prediction"],     # "Normal" | "Suspicious"
)

INFERENCE_TIER_COUNTER = Counter(
    name          = "p4_inference_tier_total",
    documentation = "Total inference requests by triage tier",
    labelnames    = ["tier"],           # "Tier1" | "Tier2" | "Tier3" | "Normal"
)

VALIDATION_FAILURE_COUNTER = Counter(
    name          = "p4_validation_failures_total",
    documentation = "Total image validation failures by failure type",
    labelnames    = ["failure_type"],
    # Values: "invalid_format" | "too_small" | "blank" | "too_large"
    #         | "corrupted" | "size_limit"
)

INFERENCE_LATENCY = Histogram(
    name          = "p4_inference_latency_seconds",
    documentation = "End-to-end inference latency in seconds",
    # Buckets sized for ML serving SLA:
    #   0.05s = fast path target
    #   0.10s = typical EfficientNet-B0 CPU inference
    #   0.20s = with full validate_image overhead
    #   0.50s = SLA threshold (p99 must be below this)
    #   1.0s  = degraded performance (trigger alert)
    #   2.0s  = severely degraded (page on-call)
    buckets       = [0.05, 0.10, 0.20, 0.50, 1.0, 2.0, float("inf")],
)

EMBEDDING_LATENCY = Histogram(
    name          = "p4_embedding_latency_seconds",
    documentation = "Embedding extraction latency in seconds",
    buckets       = [0.05, 0.10, 0.20, 0.50, 1.0, float("inf")],
)

MODEL_LOADED_GAUGE = Gauge(
    name          = "p4_model_loaded",
    documentation = "1 if model loaded and ready, 0 if not loaded",
)

DEGRADED_MODE_GAUGE = Gauge(
    name          = "p4_degraded_mode",
    documentation = "1 if API is in degraded mode, 0 if healthy",
)

EMBEDDING_REQUEST_COUNTER = Counter(
    name          = "p4_embedding_requests_total",
    documentation = "Total embedding extraction requests (internal monitoring only)",
)