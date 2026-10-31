# Multi-platform API container (linux/arm64 and linux/amd64)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cached Python dependencies layer
RUN pip install --no-cache-dir --upgrade pip \
    "brotli>=1.2.0" \
    "brotlicffi>=1.2.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.28.0" \
    "python-multipart>=0.0.9" \
    "pydantic>=2.0.0" \
    "pypdf>=4.0.0" \
    "python-docx>=1.1.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "safetensors>=0.4.0" \
    "huggingface_hub>=0.20.0" \
    "torch>=2.1.0" \
    "transformers>=4.38.0"

# Preload model weights into image at build time (enabled by default)
ARG PRELOAD_MODELS=true
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
    echo "Preloading VeriText model weights into Docker image..." && \
    python3 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('toiletsandpaper/russian-ai-text-detector-bert'); \
AutoModelForSequenceClassification.from_pretrained('toiletsandpaper/russian-ai-text-detector-bert'); \
AutoTokenizer.from_pretrained('mdrakibali/deberta-ai-detector-v3'); \
AutoModelForSequenceClassification.from_pretrained('mdrakibali/deberta-ai-detector-v3'); \
print('Model weights successfully preloaded!')"; \
    fi

# Copy backend source code and install
COPY pyproject.toml README.md ./
COPY backend/ ./backend/
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
