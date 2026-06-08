# Multi-stage Dockerfile for the inference server.
#
# Build:
#   docker build -t logistics-qa-lora:latest .
#
# Run (CPU / mock backend - for CI and local dev):
#   docker run -p 8000:8000 -e USE_MOCK=true logistics-qa-lora:latest
#
# Run (GPU + real adapter):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/artifacts:/app/artifacts \
#     -e ADAPTER_PATH=/app/artifacts/checkpoints/final \
#     logistics-qa-lora:latest

# ---- Build stage ------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install build deps for any packages with native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---- Runtime stage ----------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root user - best practice; the container has no business running as root.
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/bash app

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --chown=app:app server ./server

USER app

EXPOSE 8000

# Single worker - each uvicorn worker would load its own copy of the model into
# VRAM. For a 7B model on a single GPU, workers=1 is the right call.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
