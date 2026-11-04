from __future__ import annotations

import os
from typing import Optional


def detect_device() -> str:
    """Detect the best available compute device (CUDA / MPS / CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE: str = detect_device()

MODELS_CATALOG: dict[str, dict[str, str]] = {
    "ru": {
        "id": "toiletsandpaper/russian-ai-text-detector-bert",
        "name": "RuBERT-AI (GPT-4o, DeepSeek, GigaChat, YandexGPT)",
        "lang": "ru",
    },
    "en": {
        "id": "mdrakibali/deberta-ai-detector-v3",
        "name": "DeBERTa-v3 (ChatGPT, Claude, Gemini, Llama)",
        "lang": "en",
    },
}

FALLBACK_EN_MODEL_ID: str = "openai-community/roberta-base-openai-detector"
MAX_INPUT_CHARS: int = 98304
MAX_TOKEN_LENGTH: int = 512
WINDOW_TOKEN_SIZE: int = 480
WINDOW_TOKEN_STRIDE: int = 240
MAX_WINDOWS: int = 10

# Humanizer Paraphrasers Catalog
PARAPHRASERS_CATALOG: dict[str, dict[str, str]] = {
    "ru": {
        "id": "cointegrated/rut5-base-paraphraser",
        "name": "RuT5-Base Paraphraser",
        "lang": "ru",
    },
    "en": {
        "id": "humarin/chatgpt_paraphraser_on_T5_base",
        "name": "ChatGPT Paraphraser on T5-base",
        "lang": "en",
    },
}

# Optional Remote/Local LLM configuration
def detect_local_ollama() -> tuple[Optional[str], Optional[str]]:
    """Detect if local Ollama daemon is running on localhost:11434 with models available."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "VeriText"})
        with urllib.request.urlopen(req, timeout=0.8) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                if models:
                    model_name = models[0].get("name", "gemma2:9b")
                    return "http://127.0.0.1:11434/v1", model_name
    except Exception:
        pass
    return None, None


_OLLAMA_URL, _OLLAMA_MODEL = detect_local_ollama()
LLM_BASE_URL: Optional[str] = os.environ.get("VERITEXT_LLM_BASE_URL", _OLLAMA_URL)
LLM_MODEL: str = os.environ.get("VERITEXT_LLM_MODEL", _OLLAMA_MODEL or "gpt-4o-mini")
LLM_API_KEY: str = os.environ.get("VERITEXT_LLM_API_KEY", "")
LLM_TIMEOUT: float = float(os.environ.get("VERITEXT_LLM_TIMEOUT", "60.0"))

# Humanizer decoding strength presets
HUMANIZER_STRENGTH_PRESETS: dict[str, dict[str, float | int]] = {
    "light": {
        "num_candidates": 2,
        "temperature": 0.65,
        "top_p": 0.90,
        "repetition_penalty": 1.15,
    },
    "balanced": {
        "num_candidates": 3,
        "temperature": 0.85,
        "top_p": 0.92,
        "repetition_penalty": 1.20,
    },
    "aggressive": {
        "num_candidates": 4,
        "temperature": 1.00,
        "top_p": 0.95,
        "repetition_penalty": 1.25,
    },
}

# Length parity bounds
LENGTH_PARITY_MIN_RATIO: float = 0.65
LENGTH_PARITY_MAX_RATIO: float = 1.40

