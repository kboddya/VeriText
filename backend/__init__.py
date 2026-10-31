from __future__ import annotations

from typing import TYPE_CHECKING
from .config import DEVICE, MODELS_CATALOG, detect_device
from .markers import detect_stylistic_markers
from .parsers import extract_text_from_file
from .preprocessing import detect_dominant_language, normalize_text_for_detector

if TYPE_CHECKING:
    from .engine import ModernAIDetector

__all__ = [
    "__version__",
    "DEVICE",
    "MODELS_CATALOG",
    "ModernAIDetector",
    "detect_device",
    "detect_dominant_language",
    "detect_stylistic_markers",
    "extract_text_from_file",
    "normalize_text_for_detector",
]

__version__ = "0.3.0"


def __getattr__(name: str):
    if name == "ModernAIDetector":
        from .engine import ModernAIDetector

        return ModernAIDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
