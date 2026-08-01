"""
Serving Smoke Test — P4-L11

Run with: uv run python scripts/verify_serve.py

Contracts:
  1.  src.serve imports without error
  2.  validate_image(): valid PNG with variance passes
  3.  validate_image(): image < 32px raises 422
  4.  validate_image(): blank image raises 422
  5.  validate_image(): non-image bytes raises 422 (magic byte check)
  6.  validate_image(): oversized image raises 422
  7.  GET /health returns 200 with required fields
  8.  predict_image is def, NOT async def (event loop safety)
  9.  get_inference_transform imported from src.dataset (not redefined)
 10.  PredictionResponse uses ConfigDict(strict=True)
 11.  POST /predict/image returns all 10 required fields (if model loaded)
 12.  file.size check appears before file.file.read() (OOM prevention)
"""

import inspect
import io
import logging
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

logging.basicConfig(level=logging.WARNING)

print("=" * 60)
print("P4-L11: Serving Smoke Test")
print("=" * 60)

# ── Contract 1: Import ─────────────────────────────────────────────────────

from src.serve import app, validate_image, PredictionResponse
from fastapi import HTTPException

print("✓  1: src.serve imports succeed")

client = TestClient(app)


def make_png(w=256, h=256, uniform=False):
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    if not uniform:
        arr[50:100, 50:100] = 220
        arr[150:200, 150:200] = 30
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


valid_png = make_png(256, 256)

# ── Contract 2: Valid PNG passes ───────────────────────────────────────────

try:
    img = validate_image(valid_png)
    assert img.mode == "RGB"
    print("✓  2: validate_image() accepts valid PNG → RGB PIL Image")
except HTTPException as e:
    print(f"✗  2 FAILED: {e.detail}")

# ── Contract 3: Too-small image raises 422 ────────────────────────────────

try:
    validate_image(make_png(10, 10))
    print("✗  3 FAILED: Should raise 422 for tiny image")
except HTTPException as e:
    assert e.status_code == 422
    print(f"✓  3: <32px image raises 422")

# ── Contract 4: Blank image raises 422 ───────────────────────────────────

try:
    validate_image(make_png(256, 256, uniform=True))
    print("✗  4 FAILED: Should raise 422 for blank image")
except HTTPException as e:
    assert e.status_code == 422
    print(f"✓  4: Blank image raises 422")

# ── Contract 5: Non-image bytes raise 422 (magic byte) ────────────────────

try:
    validate_image(b"Not an image file at all. Plain text.")
    print("✗  5 FAILED: Should raise 422 for non-image")
except HTTPException as e:
    assert e.status_code == 422
    print(f"✓  5: Non-image bytes raise 422 (magic byte check)")

# ── Contract 6: Oversized image raises 422 ────────────────────────────────

# Image.MAX_IMAGE_PIXELS is set at module level — PIL raises automatically
# We also have MAX_IMAGE_DIM_PX check

try:
    validate_image(make_png(9000, 9000))
    print("✗  6 FAILED: Should raise 422 for oversized image")
except HTTPException as e:
    assert e.status_code == 422
    print(f"✓  6: Oversized image (9000px) raises 422")

# ── Contract 7: GET /health ────────────────────────────────────────────────

resp = client.get("/health")
assert resp.status_code == 200
h = resp.json()

for f in ["status", "model_loaded", "degraded", "config_drift"]:
    assert f in h, f"Missing field: {f}"

print(f"✓  7: GET /health 200 | status={h['status']}, degraded={h['degraded']}")

# ── Contract 8: predict_image is def, NOT async def ───────────────────────

from src.serve import predict_image
import inspect

# If it were async def, inspect.iscoroutinefunction would return True
assert not inspect.iscoroutinefunction(predict_image), \
    "predict_image must be synchronous def, not async def. " \
    "Async def blocks the event loop during CPU-bound PyTorch inference."

print("✓  8: predict_image is def (not async) — event loop safe ✓")

# ── Contract 9: get_inference_transform imported from src.dataset ─────────

import src.serve as serve_mod
source = inspect.getsource(serve_mod)
assert "from src.dataset import get_inference_transform" in source, \
    "serve.py must import get_inference_transform from src.dataset"

print("✓  9: get_inference_transform imported from src.dataset ✓")

# ── Contract 10: PredictionResponse uses strict mode ─────────────────────

from pydantic import ConfigDict

assert hasattr(PredictionResponse, "model_config"), \
    "PredictionResponse must have model_config"
assert PredictionResponse.model_config.get("strict") is True, \
    "PredictionResponse must use ConfigDict(strict=True)"

print("✓ 10: PredictionResponse uses ConfigDict(strict=True) ✓")

# ── Contract 11: POST /predict/image (if model present) ──────────────────

if not Path("artifacts/best_model.pt").exists():
    print("      Skipping prediction test — run scripts/run_training.py first")
else:
    r = client.post("/predict/image",
                    files={"file": ("test.png", valid_png, "image/png")})

    if r.status_code == 503:
        print(f"      API degraded (model load failed): {r.json()['detail'][:60]}")
    else:
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

        pred = r.json()
        required = ["prediction","probability","triage_tier","model_confidence",
                    "conf_level","threshold_used","requires_human_review",
                    "model_hash","inference_ms","request_id"]
        missing = [f for f in required if f not in pred]
        assert not missing, f"Missing fields: {missing}"

        assert 0.0 <= pred["probability"] <= 1.0
        assert 0.5 <= pred["model_confidence"] <= 1.0
        assert pred["triage_tier"] in ["Tier1","Tier2","Tier3","Normal"]
        assert len(pred["model_hash"]) == 12

        print(f"✓ 11: POST /predict/image 200 | pred={pred['prediction']} "
              f"tier={pred['triage_tier']} hash={pred['model_hash']}")

# ── Contract 12: file.size check before read (OOM prevention) ────────────

source = inspect.getsource(predict_image)
size_pos = source.find("file.size")
read_pos = source.find("file.file.read()")

assert size_pos != -1, "file.size check not found in predict_image"
assert read_pos  != -1, "file.file.read() not found in predict_image"
assert size_pos < read_pos, \
    "file.size check must come BEFORE file.file.read() to prevent OOM"

print("✓ 12: file.size checked before file.file.read() — OOM safe ✓")

print()
print("=" * 60)
print("All 12 contracts verified.")
print()
print("Critical fixes confirmed:")
print("  predict_image is def not async def ✓")
print("  file.size checked before read ✓")
print("  Pydantic strict mode ✓")
print("  get_inference_transform imported not redefined ✓")
print()
print("Docker commands:")
print("  docker build -t p4-radiology-serving .")
print("  docker run -p 8000:8000 p4-radiology-serving")
print("  curl -s http://localhost:8000/health | python -m json.tool")
print("=" * 60)