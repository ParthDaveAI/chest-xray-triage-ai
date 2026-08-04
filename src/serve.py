"""
FastAPI serving application for P4 Radiology AI.

CRITICAL ARCHITECTURAL DECISION — def vs async def:

The prediction endpoint uses synchronous `def`, NOT `async def`.

PyTorch CPU inference is CPU-bound (blocking). If placed in an `async def`

route, it blocks the asyncio event loop: health checks fail, concurrent

requests queue, Kubernetes probes time out.

`def` routes are automatically delegated to FastAPI's threadpool — the event

loop remains free during inference. This is the correct pattern for all

synchronous CPU-bound ML workloads.

PREPROCESSING CONSISTENCY:

serve.py imports get_inference_transform from src.dataset — same function as

ChestXRayDataset(mode="val") and evaluate.py. Training/serving skew is

structurally prevented.

PRODUCTION BUNDLE:

model + threshold + config_hash = coupled artifact. All three must match.

Config hash comparison detects deployment drift at startup.

model_hash in response provides prediction provenance.

VALIDATE_IMAGE:

Must run before every inference. The model produces logits for any tensor.

7-step validation including PIL decompression bomb protection.

CPU-ONLY:

~80ms inference satisfies 500ms p99 SLA. See decisions.md Decision 18.
"""

import hashlib
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import Response as FastAPIResponse
from PIL import Image, ImageStat
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, ConfigDict, Field

from src.dataset import get_inference_transform
from src.model import ChestXRayClassifier
from src.monitor import (
    INFERENCE_COUNTER,
    INFERENCE_TIER_COUNTER,
    VALIDATION_FAILURE_COUNTER,
    INFERENCE_LATENCY,
    EMBEDDING_LATENCY,
    MODEL_LOADED_GAUGE,
    DEGRADED_MODE_GAUGE,
    EMBEDDING_REQUEST_COUNTER,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CONFIG_PATH    = "config/training_config.yaml"
MODEL_PATH     = "artifacts/best_model.pt"
THRESHOLD_PATH = "artifacts/threshold.txt"

MIN_IMAGE_SIZE_PX   = 32        # minimum dimension
MAX_IMAGE_DIM_PX    = 8000      # maximum dimension — caps memory spike from giant images
MAX_IMAGE_PIXELS    = 50_000_000  # PIL decompression bomb cap (50 megapixels)
MIN_IMAGE_VARIANCE  = 10.0      # below this, image is blank
MAX_BYTES           = 10 * 1024 * 1024  # 10MB file size limit

# Three-tier thresholds (design canvas, L8)
TIER1_MIN = 0.80
TIER2_MIN = 0.50
CONF_HIGH = 0.80
CONF_MOD  = 0.65

# Accepted MIME types (magic-byte validation in validate_image)
ACCEPTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

# Internal service header for /embeddings endpoint
INTERNAL_SERVICE_HEADER = "X-Internal-Service"
INTERNAL_SERVICE_VALUE  = "p4-monitoring"

# PIL decompression bomb protection — set before any Image.open() calls
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# ── Pydantic v2 Schemas — Strict Mode ─────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Structured triage decision. All fields are strictly typed.
    ConfigDict(strict=True) prevents silent type coercion (e.g., str→float).
    model_hash provides prediction provenance for audit trail.
    """
    model_config = ConfigDict(strict=True)

    prediction:            str   = Field(..., description="Normal or Suspicious")
    probability:           float = Field(..., ge=0.0, le=1.0)
    triage_tier:           str   = Field(..., description="Tier1/Tier2/Tier3/Normal")
    model_confidence:      float = Field(..., ge=0.5, le=1.0)
    conf_level:            str   = Field(..., description="High/Moderate/Low")
    threshold_used:        float = Field(...)
    requires_human_review: bool  = Field(..., description="True for Tier3 or low confidence")
    model_hash:            str   = Field(..., description="MD5 of best_model.pt — prediction provenance")
    inference_ms:          float = Field(...)
    request_id:            str   = Field(default_factory=lambda: str(uuid.uuid4())[:8])


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    status:         str
    model_loaded:   bool
    degraded:       bool
    config_drift:   bool  = Field(False, description="True if serving config differs from training config")
    threshold:      Optional[float] = None
    model_hash:     Optional[str]   = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    detail:     str
    error_code: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])


class EmbeddingResponse(BaseModel):
    """
    Embedding endpoint response with Pydantic-enforced dimensionality.

    Field(min_length=1280, max_length=1280) guarantees exactly 1280 values —
    not just documentation, but enforced at serialisation time.
    """
    model_config = ConfigDict(strict=True)

    embedding:     list[float] = Field(
        ...,
        min_length   = 1280,
        max_length   = 1280,
        description  = "1280-dim EfficientNet-B0 avgpool embedding",
    )
    embedding_dim: int   = Field(1280)
    model_hash:    str   = Field(...)
    inference_ms:  float = Field(...)
    request_id:    str   = Field(default_factory=lambda: str(uuid.uuid4())[:8])


# ── Serving Components Singleton ───────────────────────────────────────────────

class ServingComponents(NamedTuple):
    model:        ChestXRayClassifier
    transform:    object                # transforms.Compose
    threshold:    float
    config:       dict
    model_hash:   str                   # MD5 of best_model.pt bytes
    config_drift: bool                  # True if current config ≠ training config


_components:       Optional[ServingComponents] = None
_serving_degraded: bool = False
_degraded_reason:  str  = ""


def get_serving_components() -> Optional[ServingComponents]:
    """
    Lazy singleton: load and cache model, transform, threshold on first call.

    Config hash comparison:
    Computes MD5 of current training_config.yaml and compares against
    artifacts/model_metadata.json (written by run_training.py) to detect
    deployment drift — serving environment config changed since training.

    Model hash:
    Computes MD5 of best_model.pt bytes for prediction provenance logging.
    Included in every PredictionResponse so predictions can be traced to
    a specific checkpoint.
    """
    global _components, _serving_degraded, _degraded_reason

    if _components is not None:
        return _components

    if _serving_degraded:
        return None

    try:
        config = yaml.safe_load(open(CONFIG_PATH))

        # Load model (CPU only — Decision 18)
        model = ChestXRayClassifier(config)
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.eval()

        # Import preprocessing from dataset.py — same as evaluate.py (skew prevention)
        transform = get_inference_transform(config)

        # Load locked threshold
        threshold = float(Path(THRESHOLD_PATH).read_text().strip())

        # Compute model hash for prediction provenance
        model_hash = hashlib.md5(Path(MODEL_PATH).read_bytes()).hexdigest()[:12]

        # Config hash comparison — detect deployment drift
        current_config_hash = hashlib.md5(Path(CONFIG_PATH).read_bytes()).hexdigest()
        config_drift        = False
        metadata_path       = Path("artifacts/model_metadata.json")

        if metadata_path.exists():
            import json
            metadata          = json.loads(metadata_path.read_text())
            training_hash     = metadata.get("config_hash", "unknown")
            config_drift      = (current_config_hash != training_hash)

            if config_drift:
                logger.warning(
                    "CONFIG DRIFT DETECTED: serving config hash (%s) differs from "
                    "training config hash (%s). Threshold was tuned for the training config. "
                    "Predictions may be miscalibrated.",
                    current_config_hash[:8], training_hash[:8],
                )
        else:
            logger.info(
                "artifacts/model_metadata.json not found — skipping config drift check. "
                "Run scripts/run_training.py to generate metadata."
            )

        logger.info(
            "Production bundle loaded:\n"
            "  Model:       %s (hash: %s)\n"
            "  Threshold:   %.4f\n"
            "  Config hash: %s\n"
            "  Config drift: %s\n"
            "  Device:      CPU (Decision 18)",
            MODEL_PATH, model_hash, threshold,
            current_config_hash[:8], config_drift,
        )

        _components = ServingComponents(
            model=model, transform=transform, threshold=threshold,
            config=config, model_hash=model_hash, config_drift=config_drift,
        )

        return _components

    except Exception as e:
        _serving_degraded = True
        _degraded_reason  = str(e)
        logger.error(
            "SERVING DEGRADED — production bundle load failed:\n%s",
            e, exc_info=True,
        )
        return None


# ── validate_image ─────────────────────────────────────────────────────────────

def validate_image(image_bytes: bytes) -> Image.Image:
    """
    7-step image validation before inference.

    The model produces logits for ANY input tensor — including blank images
    and corrupted files. This function is the quality gate.

    Step ordering is intentional — cheapest checks first:
      1. PIL decompression bomb cap (set at module level via Image.MAX_IMAGE_PIXELS)
      2. MIME validation — magic bytes whitelist (PNG: 0x89504E47, JPEG: 0xFFD8FF)
      3. PIL.open + verify() — catches corrupted/truncated files
      4. Maximum spatial dimensions — prevents memory spike from giant images
      5. Minimum spatial dimensions — rejects thumbnails and corrupted stubs
      6. Not blank — pixel variance check
      7. RGB conversion — handles grayscale, RGBA, palette-mode X-rays

    PHI SAFETY: No image metadata is logged.

    Raises:
        HTTPException 422: any validation failure with descriptive detail
    """
    # Step 1: PIL decompression bomb protection is set at module level (Image.MAX_IMAGE_PIXELS)
    # PIL raises DecompressionBombError automatically for oversized images

    # Step 2: MIME magic-byte validation
    # Checks actual file content, not just Content-Type header
    if len(image_bytes) < 4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File too small to be a valid image.",
        )

    magic = image_bytes[:4]
    is_png  = magic[:4] == b"\x89PNG"
    is_jpeg = magic[:2] == b"\xff\xd8"

    if not (is_png or is_jpeg):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File format not accepted. Only PNG and JPEG are supported. "
                   "File magic bytes do not match PNG or JPEG.",
        )

    # Step 3: PIL open + verify
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes))  # re-open — verify() closes
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image file is corrupted or cannot be read by the imaging library.",
        )

    # Step 4: Maximum spatial dimensions
    w, h = image.size
    if w > MAX_IMAGE_DIM_PX or h > MAX_IMAGE_DIM_PX:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image dimensions ({w}×{h}px) exceed maximum "
                   f"({MAX_IMAGE_DIM_PX}×{MAX_IMAGE_DIM_PX}px). "
                   f"Chest X-rays should not exceed this size.",
        )

    # Step 5: Minimum spatial dimensions
    if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image too small ({w}×{h}px). "
                   f"Minimum: {MIN_IMAGE_SIZE_PX}×{MIN_IMAGE_SIZE_PX}px.",
        )

    # Step 6: Not blank — compute variance on greyscale
    image_rgb = image.convert("RGB")
    stat      = ImageStat.Stat(image_rgb.convert("L"))
    variance  = stat.var[0]

    if variance < MIN_IMAGE_VARIANCE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image appears blank (variance={variance:.2f} < {MIN_IMAGE_VARIANCE}). "
                   f"No diagnostic information present.",
        )

    # Step 7: RGB conversion — already done above
    return image_rgb


# ── FastAPI Application ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: configure threads, load components, warmup inference.
    Shutdown: log graceful shutdown.

    L14 ADDITIONS:
      - Prometheus gauge initialisation (model_loaded, degraded_mode)
      - Pre-initialise all label combinations to 0 for Grafana dashboards
    """
    # CPU thread count: 1 minimises latency for single-image inference
    # Thread spawn overhead + lock contention exceeds parallelism benefit for batch=1
    torch.set_num_threads(1)
    logger.info("PyTorch CPU threads set to 1 (single-image inference optimisation).")

    logger.info("API starting — loading production bundle...")
    components = get_serving_components()

    if components is not None:
        # Warmup inference: triggers PyTorch kernel compilation and memory allocation
        # First real inference would pay 200–500ms cold-start cost without this
        logger.info("Running warmup inference...")
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            _ = components.model(dummy)
        logger.info("Warmup complete. API ready.")
        MODEL_LOADED_GAUGE.set(1)
        DEGRADED_MODE_GAUGE.set(0)
    else:
        logger.error("Startup failed — API is in degraded mode.")
        MODEL_LOADED_GAUGE.set(0)
        DEGRADED_MODE_GAUGE.set(1)

    # Pre-initialise all known label combinations to 0.
    # Without this, Prometheus time series don't exist until first increment.
    # Grafana dashboards show "No Data" on rate() queries until first event.
    for pred in ["Normal", "Suspicious"]:
        INFERENCE_COUNTER.labels(prediction=pred).inc(0)

    for tier in ["Tier1", "Tier2", "Tier3", "Normal"]:
        INFERENCE_TIER_COUNTER.labels(tier=tier).inc(0)

    for ft in ["invalid_format", "too_small", "blank", "too_large", "corrupted", "size_limit"]:
        VALIDATION_FAILURE_COUNTER.labels(failure_type=ft).inc(0)

    logger.info("Prometheus labels pre-initialised.")

    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="P4 Radiology AI — Chest X-Ray Triage API",
    description=(
        "Binary Normal/Suspicious classification for frontal chest X-rays. "
        "Tier3 and low-confidence predictions require mandatory human review. "
        "CPU-only inference. 500ms p99 SLA target."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Liveness check. Returns HTTP 200 always.
    Degraded status visible in response body.
    """
    comps = _components

    return HealthResponse(
        status       = "degraded" if _serving_degraded else "healthy",
        model_loaded = comps is not None,
        degraded     = _serving_degraded,
        config_drift = comps.config_drift if comps else False,
        threshold    = comps.threshold if comps else None,
        model_hash   = comps.model_hash if comps else None,
    )


@app.post("/predict/image", response_model=PredictionResponse)
def predict_image(
    file: UploadFile = File(
        ...,
        description="Frontal chest X-ray PNG or JPEG. Max 10MB.",
    ),
):
    """
    Classify a frontal chest X-ray as Normal or Suspicious.

    SYNCHRONOUS def — NOT async def:
    PyTorch CPU inference is CPU-bound (blocking). An async def endpoint
    would block the asyncio event loop, making health probes unresponsive
    and serialising concurrent requests. `def` routes run in FastAPI's
    threadpool, keeping the event loop free.

    Raises:
      503: serving components not loaded (degraded mode)
      422: image validation failed
      413: file too large
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()   # monotonic — safe for Prometheus histograms

    # ── 1. File size check BEFORE reading — OOM prevention ──────────────────────
    # file.size from Content-Length header. Check first, read after.
    # A 2GB upload would exhaust container RAM if read before checking.
    # Note: Content-Length can be absent or spoofed; post-read check is fallback.

    if file.size and file.size > MAX_BYTES:
        VALIDATION_FAILURE_COUNTER.labels(failure_type="size_limit").inc()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file.size // (1024*1024)}MB) exceeds 10MB limit.",
        )

    # Synchronous read — correct in def (threadpool) route, not in async route
    image_bytes = file.file.read()

    # Post-read size guard (handles absent/spoofed Content-Length)
    if len(image_bytes) > MAX_BYTES:
        VALIDATION_FAILURE_COUNTER.labels(failure_type="size_limit").inc()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(image_bytes) // (1024*1024)}MB) exceeds 10MB limit.",
        )

    # ── 2. validate_image BEFORE degraded check ──────────────────────────────────
    # Malformed inputs always return 422 regardless of server state.
    # Clients must distinguish "my upload is invalid" from "server is broken".
    try:
        image_pil = validate_image(image_bytes)
    except HTTPException as e:
        detail = str(e.detail).lower()
        if "format" in detail or "magic" in detail:
            ft = "invalid_format"
        elif "small" in detail or "minimum" in detail:
            ft = "too_small"
        elif "blank" in detail or "variance" in detail:
            ft = "blank"
        elif "large" in detail or "dimension" in detail or "exceed" in detail:
            ft = "too_large"
        else:
            ft = "corrupted"
        VALIDATION_FAILURE_COUNTER.labels(failure_type=ft).inc()
        raise

    # ── 3. Degraded mode check AFTER validation ───────────────────────────────────
    comps = get_serving_components()
    if comps is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model serving unavailable. Reason: {_degraded_reason}",
        )

    model, transform, threshold, config, model_hash, _ = comps

    # ── Inference ──────────────────────────────────────────────────────────────────
    # transform = get_inference_transform() from dataset.py
    # Same function as evaluate.py — training/serving skew structurally prevented
    input_tensor = transform(image_pil).unsqueeze(0)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits    = model(input_tensor)                # (1, 2) raw logits
        prob_susp = torch.softmax(logits, dim=1)[0, 1].item()

    # ── Threshold + tier assignment ───────────────────────────────────────────
    predicted_label = int(prob_susp >= threshold)
    prediction_str  = "Suspicious" if predicted_label == 1 else "Normal"

    # Triage tier: based on P(Suspicious) — routing decision
    # Model confidence: based on max(P, 1-P) — uncertainty measure
    # These are independent concepts (L8)

    if prob_susp >= TIER1_MIN:
        triage_tier = "Tier1"
    elif prob_susp >= TIER2_MIN:
        triage_tier = "Tier2"
    elif prob_susp >= threshold:
        triage_tier = "Tier3"
    else:
        triage_tier = "Normal"

    model_confidence = max(prob_susp, 1.0 - prob_susp)

    if model_confidence >= CONF_HIGH:
        conf_level = "High"
    elif model_confidence >= CONF_MOD:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    # requires_human_review: operational expression of model card Tier3 constraint
    requires_review = (conf_level == "Low") or (triage_tier == "Tier3")

    inference_ms = (time.perf_counter() - t_start) * 1000

    # ── Prometheus observations ────────────────────────────────────────────────────
    INFERENCE_COUNTER.labels(prediction=prediction_str).inc()
    INFERENCE_TIER_COUNTER.labels(tier=triage_tier).inc()
    INFERENCE_LATENCY.observe(inference_ms / 1000.0)  # histogram in seconds

    # PHI-safe log: no image content, no filename
    logger.info(
        "request_id=%s pred=%s prob=%.4f tier=%s conf=%.4f(%s) "
        "review=%s threshold=%.4f model_hash=%s ms=%.1f",
        request_id, prediction_str, prob_susp, triage_tier,
        model_confidence, conf_level, requires_review,
        threshold, model_hash, inference_ms,
    )

    return PredictionResponse(
        prediction            = prediction_str,
        probability           = round(prob_susp, 4),
        triage_tier           = triage_tier,
        model_confidence      = round(model_confidence, 4),
        conf_level            = conf_level,
        threshold_used        = round(threshold, 4),
        requires_human_review = requires_review,
        model_hash            = model_hash,
        inference_ms          = round(inference_ms, 1),
        request_id            = request_id,
    )


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics in exposition format (text/plain).

    Scrape interval: 15–30s is typical for ML serving.

    NOT authenticated — assume network perimeter security (VPC/firewall).
    Never expose this endpoint on a public load balancer without firewall rules.

    PromQL examples:
      p99 latency: histogram_quantile(0.99, rate(p4_inference_latency_seconds_bucket[5m]))
      prediction rate: rate(p4_inference_total[5m])
      validation failures: rate(p4_validation_failures_total[5m])
    """
    return FastAPIResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/embeddings", response_model=EmbeddingResponse)
def get_embedding(
    file: UploadFile = File(
        ...,
        description="Frontal chest X-ray PNG or JPEG. Max 10MB.",
    ),
    x_internal_service: str = Header(
        default=None,
        alias="X-Internal-Service",
    ),
):
    """
    Extract 1280-dim penultimate-layer embedding for P5 drift monitoring.

    INTERNAL ENDPOINT — NOT for clinical users.

    SECURITY NOTE:
    The X-Internal-Service header check is defence-in-depth, NOT real security.
    HTTP headers are trivially forgeable. Real security requires:
      - mTLS: service mesh (Istio/Linkerd) mutual certificate authentication
      - VPC isolation: endpoint not routable from public internet
      - Service mesh identity: Kubernetes RBAC on service accounts

    In this deployment, the load balancer MUST strip X-Internal-Service from
    all external requests so only internal services can supply it.

    Do NOT deploy this endpoint on a public load balancer without verifying
    header-stripping configuration.

    THREAD SAFETY:
    Uses model.get_penultimate_features() — a stateless method that runs
    backbone.features → backbone.avgpool through standard forward operations.
    No forward hooks are used. Hooks on a shared model singleton in a
    multi-threaded server cause cross-request embedding contamination.

    SAMPLING:
    Do not call this endpoint for every inference request.
    Recommended: 5–10% random sample or reservoir sampling.
    Collecting all embeddings produces unsustainable storage and compute costs.

    JSON FORMAT NOTE:
    Returns 1280 floats as a JSON list. Production would use binary protocols
    (protobuf, Arrow, gRPC) for efficiency at scale.

    PHI SAFETY:
    Embedding vectors encode image content. Never log them.
    Only aggregate statistics (count, norm) are logged.
    """
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()

    # ── Security boundary check ───────────────────────────────────────────────
    if x_internal_service != INTERNAL_SERVICE_VALUE:
        logger.warning(
            "request_id=%s /embeddings: unauthorized — incorrect or missing "
            "X-Internal-Service header", request_id,
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "This endpoint requires X-Internal-Service: p4-monitoring. "
                "Internal monitoring use only."
            ),
        )

    # ── Size + validation ─────────────────────────────────────────────────────
    if file.size and file.size > MAX_BYTES:
        raise HTTPException(413, detail="File exceeds 10MB limit.")

    image_bytes = file.file.read()

    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(413, detail="File exceeds 10MB limit.")

    image_pil = validate_image(image_bytes)

    # ── Degraded check ────────────────────────────────────────────────────────
    comps = get_serving_components()
    if comps is None:
        raise HTTPException(503, detail=f"Model unavailable: {_degraded_reason}")

    model, transform, _, _, model_hash, _ = comps

    # ── Thread-safe embedding extraction ──────────────────────────────────────
    # Uses model.get_penultimate_features() — NO hooks.
    # Forward hooks on a shared singleton cause cross-thread contamination:
    #   Thread A and Thread B both register hooks. When either runs a forward
    #   pass, ALL hooks fire — embeddings from one request land in another.
    # get_penultimate_features() is a pure functional operation:
    #   same input → same output, regardless of concurrent calls.

    input_tensor = transform(image_pil).unsqueeze(0)   # (1, 3, 224, 224)

    with torch.no_grad():
        embedding_tensor = model.get_penultimate_features(input_tensor)
        embedding_arr = (
            embedding_tensor.detach().squeeze().cpu().numpy().astype(np.float32)
        )

    # ── Validate finiteness ───────────────────────────────────────────────────
    if not np.isfinite(embedding_arr).all():
        nan_n = int(np.isnan(embedding_arr).sum())
        inf_n = int(np.isinf(embedding_arr).sum())
        logger.error(
            "request_id=%s Non-finite embedding: nan=%d inf=%d",
            request_id, nan_n, inf_n,
        )
        raise HTTPException(500, detail="Embedding contains non-finite values.")

    inference_ms = (time.perf_counter() - t_start) * 1000

    # ── PHI-safe logging (aggregate only, never the vector itself) ────────────
    EMBEDDING_REQUEST_COUNTER.inc()
    EMBEDDING_LATENCY.observe(inference_ms / 1000.0)

    logger.info(
        "request_id=%s embedding dim=%d norm=%.4f model_hash=%s ms=%.1f",
        request_id, len(embedding_arr),
        float(np.linalg.norm(embedding_arr)),
        model_hash, inference_ms,
    )

    return EmbeddingResponse(
        embedding     = embedding_arr.tolist(),
        embedding_dim = len(embedding_arr),
        model_hash    = model_hash,
        inference_ms  = round(inference_ms, 1),
        request_id    = request_id,
    )