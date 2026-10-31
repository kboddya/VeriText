from __future__ import annotations

import re

STYLISTIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Russian generative clichés (GPT-4o, YandexGPT, GigaChat, DeepSeek)
    (
        re.compile(
            r'\b(важно отметить|следует подчеркнуть|нельзя не упомянуть|стоит обратить внимание|необходимо выделить)\b',
            re.IGNORECASE,
        ),
        "Formal opening cliché (typical for GPT / YandexGPT)",
    ),
    (
        re.compile(
            r'\b(в современном мире|в заключение стоит сказать|подводя итоги|резюмируя вышесказанное|исходя из вышеизложенного)\b',
            re.IGNORECASE,
        ),
        "Formulaic conclusion structuring",
    ),
    (
        re.compile(
            r'\b(является неотъемлемой частью|играет ключевую роль|неотъемлемый элемент|фундаментальную роль)\b',
            re.IGNORECASE,
        ),
        "Semantic LLM cliché",
    ),
    (
        re.compile(
            r'\b(широкий спектр|неуклонно растет|проливает свет на|глубокое понимание)\b',
            re.IGNORECASE,
        ),
        "High-frequency synthetic phrasing pattern",
    ),
    (
        re.compile(
            r'\b(как искусственный интеллект|как языковая модель)\b',
            re.IGNORECASE,
        ),
        "Explicit LLM self-reference disclaimer",
    ),
    # English generative clichés (ChatGPT, Claude, Gemini)
    (
        re.compile(
            r'\b(it is important to note|it is worth noting|it should be noted|it is crucial to understand)\b',
            re.IGNORECASE,
        ),
        "Standard ChatGPT preamble cliché",
    ),
    (
        re.compile(
            r'\b(a testament to|testament of|beacon of|plays a crucial role|plays a pivotal role)\b',
            re.IGNORECASE,
        ),
        "Classic Claude / GPT rhetoric cliché",
    ),
    (
        re.compile(
            r'\b(delve into|delving into|tapestry of|rich tapestry|ever-evolving landscape)\b',
            re.IGNORECASE,
        ),
        "High-frequency RLHF alignment vocabulary marker",
    ),
    (
        re.compile(
            r'\b(in conclusion|in summary|to wrap up|furthermore|moreover)\b',
            re.IGNORECASE,
        ),
        "Formulaic connective transition marker",
    ),
]


def detect_stylistic_markers(text: str) -> list[dict[str, str | int]]:
    """Detect characteristic AI-generated markers, clichés, and stylistic patterns."""
    if not text:
        return []

    detected: list[dict[str, str | int]] = []
    for pattern, desc in STYLISTIC_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            detected.append({
                "phrase": matches[0],
                "description": desc,
                "count": len(matches),
            })
    return detected
