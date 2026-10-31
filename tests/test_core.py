from __future__ import annotations

import unittest
from backend.preprocessing import (
    detect_dominant_language,
    get_confidence_status,
    normalize_text_for_detector,
)
from backend.markers import detect_stylistic_markers
from backend.parsers import extract_text_from_file


class TestCoreDetector(unittest.TestCase):
    """Unit test suite for text preprocessing, language detection, markers, and parsers."""

    def test_normalize_russian_typography_and_markdown(self) -> None:
        raw = "## Заголовок\n\nЧёрт возьми — «счастье»… **Важно**!"
        normalized = normalize_text_for_detector(raw, lang="ru")
        self.assertNotIn("ё", normalized)
        self.assertNotIn("«", normalized)
        self.assertNotIn("—", normalized)
        self.assertNotIn("…", normalized)
        self.assertNotIn("##", normalized)
        self.assertNotIn("**", normalized)
        self.assertEqual('Заголовок Черт возьми - "счастье"... Важно!', normalized)

    def test_detect_dominant_language(self) -> None:
        self.assertEqual(
            detect_dominant_language("Привет, это русский текст для проверки."),
            "ru",
        )
        self.assertEqual(
            detect_dominant_language(
                "This paper presents a novel approach by Ivanov to optimize sparse networks."
            ),
            "en",
        )

    def test_detect_stylistic_markers(self) -> None:
        text = "В современном мире важно отметить, что ИИ играет ключевую роль."
        markers = detect_stylistic_markers(text)
        self.assertEqual(len(markers), 3)

    def test_extract_text_from_markdown(self) -> None:
        md_bytes = (
            "# Document Title\n\n"
            "First sufficiently long paragraph of text to verify extraction.\n\n"
            "Second sufficiently long paragraph of text to verify separation."
        ).encode("utf-8")
        full_text, paragraphs = extract_text_from_file("sample.md", md_bytes)
        self.assertIn("First sufficiently long paragraph", full_text)
        self.assertGreaterEqual(len(paragraphs), 2)

    def test_extract_unsupported_extension(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported file format"):
            extract_text_from_file("archive.zip", b"123")

    def test_get_confidence_status(self) -> None:
        self.assertEqual(get_confidence_status(10.0)["code"], "verified_human")
        self.assertEqual(get_confidence_status(35.0)["code"], "likely_human")
        self.assertEqual(get_confidence_status(48.0)["code"], "uncertain_borderline")
        self.assertEqual(get_confidence_status(75.0)["code"], "likely_ai")
        self.assertEqual(get_confidence_status(95.0)["code"], "high_confidence_ai")


if __name__ == "__main__":
    unittest.main()
