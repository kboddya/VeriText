from __future__ import annotations

import re
from typing import Optional

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


RU_SCRUB_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # Preambles and formulaic leading clauses
    (
        re.compile(
            r'\b(важно отметить|следует подчеркнуть|нельзя не упомянуть|стоит обратить внимание|необходимо выделить|необходимо отметить|следует отметить|важно помнить|стоит отметить)\s*(,\s*что|\s+что)?\b,?\s*',
            re.IGNORECASE,
        ),
        "",
        "Formal opening cliché preamble",
    ),
    (
        re.compile(
            r'\b(таким образом,\s*можно сделать вывод,\s*что|таким образом,\s*можно сделать вывод|таким образом,\s*следует отметить,\s*что|таким образом,\s*следует подчеркнуть)\b,?\s*',
            re.IGNORECASE,
        ),
        "",
        "Formulaic concluding deduction",
    ),
    (
        re.compile(
            r'\b(подводя итоги,\s*следует подчеркнуть|подводя итоги,\s*следует сказать|подводя итоги,\s*можно отметить|подводя итоги)\b,?\s*',
            re.IGNORECASE,
        ),
        "",
        "Formulaic conclusion structuring",
    ),
    (
        re.compile(
            r'\b(резюмируя вышесказанное|исходя из вышеизложенного|в заключение стоит сказать,\s*что|в заключение стоит сказать)\b,?\s*',
            re.IGNORECASE,
        ),
        "",
        "Formulaic summary boilerplate",
    ),
    (
        re.compile(
            r'\b(в современном мире|в современном обществе|в эпоху цифровизации)\b,?\s*',
            re.IGNORECASE,
        ),
        "",
        "Generic temporal opening cliché",
    ),
    (
        re.compile(
            r'\b(как искусственный интеллект|как языковая модель)\b,?\s*(я\s+)?',
            re.IGNORECASE,
        ),
        "",
        "Explicit LLM self-reference disclaimer",
    ),
    (
        re.compile(
            r'\b(не секрет,\s*что|как известно,\s*что|очевидно,\s*что)\b,?\s*',
            re.IGNORECASE,
        ),
        "",
        "Presumptive rhetorical boilerplate",
    ),
    # Mid-sentence semantic clichés
    (
        re.compile(r'\bявляется неотъемлемой частью\b', re.IGNORECASE),
        "входит в состав",
        "Synthetic semantic cliché",
    ),
    (
        re.compile(r'\bиграет ключевую роль\b', re.IGNORECASE),
        "имеет большое значение",
        "Overused importance cliché",
    ),
    (
        re.compile(r'\bнеотъемлемый элемент\b', re.IGNORECASE),
        "важная часть",
        "Synthetic phrasing pattern",
    ),
    (
        re.compile(r'\bфундаментальную роль\b', re.IGNORECASE),
        "важную роль",
        "Inflated rhetoric cliché",
    ),
    (
        re.compile(r'\bширокий спектр\b', re.IGNORECASE),
        "множество",
        "High-frequency synthetic phrasing",
    ),
    (
        re.compile(r'\bнеуклонно растет\b', re.IGNORECASE),
        "растет",
        "Synthetic intensification",
    ),
    (
        re.compile(r'\bпроливает свет на\b', re.IGNORECASE),
        "объясняет",
        "Formulaic idiom",
    ),
    (
        re.compile(r'\bглубокое понимание\b', re.IGNORECASE),
        "понимание",
        "Superficial emphasis cliché",
    ),
    (
        re.compile(r'\bна регулярной основе\b', re.IGNORECASE),
        "регулярно",
        "Bureaucratic phrasing",
    ),
    (
        re.compile(r'\bв целях повышения\b', re.IGNORECASE),
        "для повышения",
        "Bureaucratic phrasing",
    ),
    (
        re.compile(r'\bоказывает существенное влияние на\b', re.IGNORECASE),
        "влияет на",
        "Wordy verb phrase cliché",
    ),
]


EN_SCRUB_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # Preambles and formulaic leading clauses
    (
        re.compile(
            r"\b(in today's fast-paced digital world|in today's fast-paced world|in today's digital world|in today's digital age|in the modern era)\b,?\s*",
            re.IGNORECASE,
        ),
        "",
        "Generic temporal opening boilerplate",
    ),
    (
        re.compile(
            r"\b(it is important to note|it is important to remember|it is worth noting|it should be noted|it is crucial to understand|it is essential to recognize)\s*(that)?\b,?\s*",
            re.IGNORECASE,
        ),
        "",
        "Standard ChatGPT preamble cliché",
    ),
    (
        re.compile(
            r"\b(it goes without saying that|needless to say)\b,?\s*",
            re.IGNORECASE,
        ),
        "",
        "Formulaic filler preamble",
    ),
    (
        re.compile(
            r"\b(in conclusion|in summary|to sum up|to wrap up|all things considered)\b,?\s*",
            re.IGNORECASE,
        ),
        "",
        "Formulaic conclusion marker",
    ),
    (
        re.compile(
            r"\b(as an ai language model|as an ai|as a large language model)\b,?\s*",
            re.IGNORECASE,
        ),
        "",
        "Explicit LLM self-reference disclaimer",
    ),
    (
        re.compile(
            r"\b(furthermore|moreover)\b,?\s*",
            re.IGNORECASE,
        ),
        "",
        "Formulaic connective transition marker",
    ),
    # Mid-sentence semantic clichés
    (
        re.compile(r"\ba testament to\b", re.IGNORECASE),
        "proof of",
        "Classic Claude / GPT rhetoric cliché",
    ),
    (
        re.compile(r"\btestament of\b", re.IGNORECASE),
        "evidence of",
        "Classic Claude / GPT rhetoric cliché",
    ),
    (
        re.compile(r"\bbeacon of\b", re.IGNORECASE),
        "symbol of",
        "Classic Claude / GPT rhetoric cliché",
    ),
    (
        re.compile(r"\bplays a crucial role\b", re.IGNORECASE),
        "is essential",
        "Inflated importance cliché",
    ),
    (
        re.compile(r"\bplays a pivotal role\b", re.IGNORECASE),
        "is central",
        "Inflated importance cliché",
    ),
    (
        re.compile(r"\bdelve into\b", re.IGNORECASE),
        "examine",
        "High-frequency RLHF alignment vocabulary marker",
    ),
    (
        re.compile(r"\bdelving into\b", re.IGNORECASE),
        "examining",
        "High-frequency RLHF alignment vocabulary marker",
    ),
    (
        re.compile(r"\bcomplex tapestry\b", re.IGNORECASE),
        "complexity",
        "Classic LLM tapestry metaphor",
    ),
    (
        re.compile(r"\brich tapestry\b", re.IGNORECASE),
        "diversity",
        "Classic LLM tapestry metaphor",
    ),
    (
        re.compile(r"\btapestry of\b", re.IGNORECASE),
        "web of",
        "Classic LLM tapestry metaphor",
    ),
    (
        re.compile(r"\bever-evolving landscape\b", re.IGNORECASE),
        "evolving field",
        "High-frequency RLHF alignment vocabulary marker",
    ),
    (
        re.compile(r"\bcornerstone of\b", re.IGNORECASE),
        "foundation for",
        "Structural metaphor cliché",
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


def scrub_cliches(text: str, lang: str = "auto") -> tuple[str, list[dict[str, str]]]:
    """
    Stage 1 De-cliché Scrubber:
    Removes and replaces formulaic AI boilerplate, leading preambles, and clichés.
    Restores sentence initial capitalization and cleans punctuation anomalies.
    Returns:
        (scrubbed_text, removed_markers)
    """
    if not text or not text.strip():
        return text, []

    if lang == "ru":
        rules = RU_SCRUB_RULES
    elif lang == "en":
        rules = EN_SCRUB_RULES
    else:
        rules = RU_SCRUB_RULES + EN_SCRUB_RULES

    removed: list[dict[str, str]] = []
    cleaned = text

    for pattern, repl, desc in rules:
        matches = [m.group(0) for m in pattern.finditer(cleaned)]
        if matches:
            for m in matches:
                phrase = m.strip(" \t\n,")
                if phrase:
                    removed.append({
                        "phrase": phrase,
                        "description": desc,
                        "replacement": repl,
                    })
            cleaned = pattern.sub(repl, cleaned)

    # Post-processing normalization
    # 1. Clean spacing before punctuation
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+([,.:;!?])', r'\1', cleaned)
    # 2. Clean duplicate commas or comma-period anomalies
    cleaned = re.sub(r',\s*,+', ', ', cleaned)
    cleaned = re.sub(r',\s*([.!?])', r'\1', cleaned)
    cleaned = re.sub(r'(?<=[.!?])\s*,\s*', ' ', cleaned)
    # 3. Clean leading comma or whitespace per line
    cleaned = re.sub(r'^\s*,\s*', '', cleaned)
    cleaned = re.sub(r'(?<=\n)\s*,\s*', '', cleaned)
    # 4. Sentence initial capitalization restoration
    cleaned = re.sub(
        r'(^|[.!?…]\s+|\n+)(["«\'\s]*)([a-zа-яё])',
        lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
        cleaned,
    )
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()

    return cleaned, removed
