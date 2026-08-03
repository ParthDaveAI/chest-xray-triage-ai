"""
Tier 1 integration tests — always run in CI, no model artifacts required.

The API operates in degraded mode (model not loaded). Input validation
(validate_image) runs BEFORE the degraded check (L11 errata fix), so
malformed inputs correctly return 422 even in degraded mode.

Tests (4):

  T1-1: GET /health returns 200 with required fields

  T1-2: Malformed image (tiny) returns 422 even in degraded mode

  T1-3: Blank image returns 422 even in degraded mode

  T1-4: predict_image is def (not async) — event loop safety

"""

import inspect

import io

import numpy as np

import pytest

from fastapi.testclient import TestClient

from PIL import Image

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")

def client():
    """
    FastAPI TestClient using context manager to trigger lifespan events.

    CRITICAL: TestClient(app) WITHOUT `with` bypasses the lifespan
    asynccontextmanager — startup warmup, model loading, and thread config
    do not execute. Always use `with TestClient(app) as c: yield c` to
    match production uvicorn behaviour.
    """
    from src.serve import app

    with TestClient(app) as c:
        yield c


def _make_png(w=256, h=256, uniform=False):
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    if not uniform:
        arr[50:100, 50:100]   = 220
        arr[150:200, 150:200] = 30
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_t1_1_health_endpoint(client):
    """T1-1: GET /health returns 200 with all required fields."""
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    for field in ["status", "model_loaded", "degraded", "config_drift"]:
        assert field in body, f"Missing field: {field}"


def test_t1_2_tiny_image_422_in_degraded_mode(client):
    """T1-2: Tiny image (10×10) returns 422 even when model not loaded.

    Requires L11 errata fix: validate_image must run BEFORE degraded check.
    This confirms the correct client-visible contract: malformed input is
    always a 4xx error, never a 5xx infrastructure error.
    """
    tiny = _make_png(10, 10)
    resp = client.post("/predict/image",
                       files={"file": ("tiny.png", tiny, "image/png")})

    assert resp.status_code == 422, \
        f"Expected 422 for tiny image, got {resp.status_code}. " \
        f"Check that validate_image() runs before degraded mode check in serve.py."


def test_t1_3_blank_image_422_in_degraded_mode(client):
    """T1-3: Blank image returns 422 even when model not loaded."""
    blank = _make_png(256, 256, uniform=True)
    resp  = client.post("/predict/image",
                        files={"file": ("blank.png", blank, "image/png")})

    assert resp.status_code == 422


def test_t1_4_predict_endpoint_is_sync():
    """T1-4: predict_image is def (not async def) — event loop safety.

    async def with CPU-bound PyTorch inference blocks the asyncio event loop.
    All concurrent requests and health probes queue behind the inference.
    def routes run in FastAPI's threadpool, keeping the event loop free.
    """
    from src.serve import predict_image

    assert not inspect.iscoroutinefunction(predict_image), \
        "predict_image MUST be synchronous def, not async def. " \
        "CPU-bound PyTorch inference in async def blocks the event loop, " \
        "making health probes unresponsive under load."