# Multi-stage CPU-only Docker build — P4 Radiology AI serving
# Decision 18: CPU-only (EfficientNet-B0 ~80ms CPU inference within 500ms SLA)
#
# CRITICAL: --extra-index-url https://download.pytorch.org/whl/cpu
# Without this, Linux pip/uv installs CUDA-bundled torch (~2.5GB wheel) even
# when no GPU is present, producing a ~5GB container despite CPU-only intent.
# This single line is the difference between a 1.5GB and 5GB image.

# ── Stage 1: Builder ──────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r pyproject.toml


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

# Security: non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/uvicorn    /usr/local/bin/uvicorn

# Copy application
COPY src/     ./src/
COPY config/  ./config/

# Copy production artifacts
# Production: mount via secrets or object storage; baked in for portfolio
COPY artifacts/best_model.pt   ./artifacts/best_model.pt
COPY artifacts/threshold.txt   ./artifacts/threshold.txt

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Single worker per container — scale horizontally
CMD ["uvicorn", "src.serve:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "5"]