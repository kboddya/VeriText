# Multi-platform standalone CLI container (linux/arm64 and linux/amd64)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip "brotli>=1.2.0" "brotlicffi>=1.2.0"

COPY pyproject.toml README.md ./
COPY backend/ ./backend/
RUN pip install --no-cache-dir .

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

WORKDIR /data

ENTRYPOINT ["veritext"]
CMD ["--help"]
