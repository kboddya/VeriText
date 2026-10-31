from __future__ import annotations

import re
from .config import MAX_INPUT_CHARS

# Canonical normalization rules for the Russian model (LLMTrace / aiw-ru)
RU_NORMALIZE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]'), ''),
    (re.compile(r'[\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\t\r\f\v]'), ' '),
    (re.compile(r'^[ ]*(?:```|~~~).*$', re.MULTILINE), ''),
    (re.compile(r'^[ ]*(?:[-*_=][ ]*){3,}$', re.MULTILINE), ''),
    (re.compile(r'^[ ]*\|?(?:[ ]*:?-+:?[ ]*\|)+[ ]*(?::?-+:?)?[ ]*$', re.MULTILINE), ''),
    (re.compile(r'^[ ]*#{1,6}[ ]+', re.MULTILINE), ''),
    (re.compile(r'^[ ]*(?:>[ ]?)+', re.MULTILINE), ''),
    (re.compile(r'^[ ]*(?:[-*+\u2022\u00b7\u25aa\u25cf\u25e6\u2023\u2013\u2014]|\d{1,3}[.)])[ ]+', re.MULTILINE), ''),
    (re.compile(r'!\[([^\]\n]*)\]\([^)\n]*\)'), r'\1'),
    (re.compile(r'\[([^\]\n]+)\]\([^)\n]*\)'), r'\1'),
    (re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>\n]*)?/?>'), ' '),
    (re.compile(r'\*+|~~|`+'), ''),
    (re.compile(r'(?<!\w)_+|_\+(?!\w)'), ''),
    (re.compile(r'\|'), ' '),
    (re.compile(r'[\u00ab\u00bb\u201e\u201c\u201d\u201f\u2033]'), '"'),
    (re.compile(r'[\u2018\u2019\u201a\u201b\u2032]'), "'"),
    (re.compile(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]'), '-'),
    (re.compile(r'\u2026'), '...'),
    (re.compile(r'ё'), 'е'),
    (re.compile(r'Ё'), 'Е'),
    (re.compile(r'\s+'), ' '),
]

# Normalization rules for the English model
EN_NORMALIZE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]'), ''),
    (re.compile(r'[\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000\t\r\f\v]'), ' '),
    (re.compile(r'^[ ]*(?:```|~~~).*$', re.MULTILINE), ''),
    (re.compile(r'^[ ]*(?:[-*_=][ ]*){3,}$', re.MULTILINE), ''),
    (re.compile(r'^[ ]*#{1,6}[ ]+', re.MULTILINE), ''),
    (re.compile(r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>\n]*)?/?>'), ' '),
    (re.compile(r'[\u201c\u201d\u201f\u2033]'), '"'),
    (re.compile(r'[\u2018\u2019\u201a\u201b\u2032]'), "'"),
    (re.compile(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]'), '-'),
    (re.compile(r'\u2026'), '...'),
    (re.compile(r'\s+'), ' '),
]

_CYRILLIC_RE = re.compile(r'[а-яёА-ЯЁ]')
_LATIN_RE = re.compile(r'[a-zA-Z]')


def normalize_text_for_detector(text: str, lang: str = "ru") -> str:
    """
    Canonical text preprocessing prior to tokenizer input.
    Eliminates false positives caused by typography (ё, «», —, ...) and Markdown artifacts.
    """
    if not text:
        return ""

    res = text[:MAX_INPUT_CHARS]
    rules = RU_NORMALIZE_RULES if lang == "ru" else EN_NORMALIZE_RULES
    for pattern, repl in rules:
        res = pattern.sub(repl, res)
    return res.strip()


def detect_dominant_language(text: str) -> str:
    """
    Detect the dominant language of the text ('ru' or 'en') based on character ratio.
    """
    if not text or not text.strip():
        return "ru"
    cyrillic_chars = len(_CYRILLIC_RE.findall(text))
    latin_chars = len(_LATIN_RE.findall(text))
    total_alpha = cyrillic_chars + latin_chars
    if total_alpha == 0:
        return "ru"

    if cyrillic_chars >= latin_chars or (cyrillic_chars / total_alpha >= 0.30):
        return "ru"
    return "en"


def get_confidence_status(ai_percentage: float) -> dict:
    """
    Categorize AI generation probability into confidence grades and risk levels.
    Provides clear semantic status instead of a blunt binary verdict.
    """
    pct = max(0.0, min(100.0, float(ai_percentage)))
    if pct < 20.0:
        return {
            "code": "verified_human",
            "label": "Verified Human",
            "grade": "Grade A (Human)",
            "description": "High-confidence human authorship with natural lexical variety and burstiness.",
            "color": "emerald",
            "risk_level": "Minimal",
            "badge_class": "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
        }
    elif pct < 45.0:
        return {
            "code": "likely_human",
            "label": "Likely Human / Formal Style",
            "grade": "Grade B (Likely Human)",
            "description": "Predominantly human characteristics with structured or formal terminology.",
            "color": "teal",
            "risk_level": "Low",
            "badge_class": "bg-teal-500/10 text-teal-300 border-teal-500/30",
        }
    elif pct <= 60.0:
        return {
            "code": "uncertain_borderline",
            "label": "Uncertain / Borderline Content",
            "grade": "Grade C (Borderline)",
            "description": "Score near 50% threshold. Typical for legal/historical charters, formulaic encyclopedic prose, or mixed hybrid human-AI text.",
            "color": "amber",
            "risk_level": "Moderate / Indeterminate",
            "badge_class": "bg-amber-500/10 text-amber-300 border-amber-500/30",
        }
    elif pct <= 85.0:
        return {
            "code": "likely_ai",
            "label": "Likely AI-Generated",
            "grade": "Grade D (Likely AI)",
            "description": "Substantial synthetic phrasing, uniform rhythm, and low perplexity typical of LLMs.",
            "color": "orange",
            "risk_level": "High",
            "badge_class": "bg-orange-500/10 text-orange-400 border-orange-500/30",
        }
    else:
        return {
            "code": "high_confidence_ai",
            "label": "High-Confidence AI Generation",
            "grade": "Grade F (Confirmed AI)",
            "description": "Unmistakable synthetic signatures (ChatGPT, Claude, DeepSeek, or Llama).",
            "color": "rose",
            "risk_level": "Critical",
            "badge_class": "bg-rose-500/10 text-rose-400 border-rose-500/30",
        }

