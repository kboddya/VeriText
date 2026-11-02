# syntax=docker/dockerfile:1
# check=skip=SecretsUsedInArgOrEnv
# Multi-platform API container (linux/arm64 and linux/amd64)

# ==============================================================================
# Stage 1: Dependency Builder
# ==============================================================================
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch first (defaults to lightweight CPU wheels; override for CUDA)
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
RUN pip install --upgrade pip setuptools wheel && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.1.0"; \
    else \
        pip install "torch>=2.1.0"; \
    fi

# Install project dependencies from pyproject.toml and strip non-runtime artifacts
COPY pyproject.toml README.md ./
RUN mkdir -p backend/api && \
    touch backend/__init__.py backend/api/__init__.py && \
    pip install --no-build-isolation . && \
    pip uninstall -y pip wheel && \
    rm -rf /opt/venv/lib/python3.11/site-packages/backend \
           /opt/venv/lib/python3.11/site-packages/torch/test \
           /opt/venv/lib/python3.11/site-packages/torch/include \
           /opt/venv/lib/python3.11/site-packages/torch/bin/test_* \
           /opt/venv/lib/python3.11/site-packages/torch/bin/*Test \
           /opt/venv/lib/python3.11/site-packages/torch/bin/protoc* \
           backend build *.egg-info && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# Stage 2: Slim Production Runtime
# ==============================================================================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    HOME=/home/veritext \
    HF_HOME=/home/veritext/.cache/huggingface \
    VERITEXT_CACHE_DIR=/home/veritext/.cache/veritext \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Install minimal OpenMP runtime library required by PyTorch CPU & create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 veritext \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/veritext --shell /usr/sbin/nologin veritext \
    && mkdir -p /app /home/veritext/.cache/huggingface /home/veritext/.cache/veritext \
    && chown -R veritext:veritext /app /home/veritext

COPY --from=builder /opt/venv /opt/venv

USER veritext
WORKDIR /app

# Preload model weights into image at build time (cached independently of backend code)
ARG PRELOAD_MODELS=true
ARG HF_TOKEN=""
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
    echo "Preloading VeriText model weights into Docker image..." && \
    HF_TOKEN="${HF_TOKEN}" python3 -c "import json, time; \
from pathlib import Path; \
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
models = ['toiletsandpaper/russian-ai-text-detector-bert', 'mdrakibali/deberta-ai-detector-v3']; \
[ (AutoTokenizer.from_pretrained(m), AutoModelForSequenceClassification.from_pretrained(m)) for m in models ]; \
sync_dir = Path('/home/veritext/.cache/veritext'); \
sync_dir.mkdir(parents=True, exist_ok=True); \
now = time.time(); \
(sync_dir / 'model_sync.json').write_text(json.dumps({'last_checks': {m: now for m in models}}, indent=2)); \
print('Model weights and sync state successfully preloaded!')" && \
    rm -rf /home/veritext/.cache/huggingface/xet \
           /home/veritext/.cache/huggingface/hub/.locks \
           /tmp/*; \
    fi

# Copy application source last for sub-second rebuilds on code changes
COPY --chown=veritext:veritext pyproject.toml README.md /app/
COPY --chown=veritext:veritext backend/ /app/backend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
