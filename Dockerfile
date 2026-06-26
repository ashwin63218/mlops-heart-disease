# ---------------------------------------------------------------------------
# Stage 1 — Builder
# Install dependencies into a virtual env so they can be copied cleanly.
# This avoids polluting the final image with build tools.
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies (needed for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# Lean final image — no compilers, no build cache.
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and trained model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Ensure the data/processed dir exists (preprocessor is embedded in pipeline,
# but some code paths check for the directory)
RUN mkdir -p data/processed data/raw

# Switch to non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker will mark container unhealthy if /health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Environment defaults (override at runtime with -e flags)
ENV MODEL_BACKEND=sklearn \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]