from __future__ import annotations


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
