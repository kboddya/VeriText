from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.reports import generate_humanize_markdown_report, generate_markdown_report
from backend.cli import score_text


class TestMarkdownReportGenerator(unittest.TestCase):
    """Unit test suite for English Markdown report generation in both Web and CLI modes."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_generate_markdown_report_structure(self) -> None:
        sample_data = {
            "filename": "sample_document.docx",
            "label": "ai",
            "score": 3.42,
            "human_ai_scale": 88.5,
            "monolithic_scale": 88.5,
            "paragraph_scale": 85.0,
            "mode": "ru",
            "model_used": "RuBERT-AI",
            "device": "cpu",
            "markers": [
                {
                    "phrase": "важно отметить",
                    "description": "Formal opening cliché",
                    "count": 2,
                }
            ],
            "full_text": "Paragraph one.\n\nParagraph two.",
            "summary": {
                "word_count": 4,
                "char_count": 29,
                "total_paragraphs": 2,
                "analyzed_paragraphs": 2,
                "ai_paragraphs": 2,
                "human_paragraphs": 0,
                "ai_paragraph_pct": 100.0,
            },
            "paragraphs": [
                {
                    "index": 1,
                    "text": "Paragraph one.",
                    "label": "ai",
                    "human_ai_scale": 91.2,
                    "word_count": 2,
                },
                {
                    "index": 2,
                    "text": "Paragraph two.",
                    "label": "ai",
                    "human_ai_scale": 85.8,
                    "word_count": 2,
                },
            ],
        }

        md = generate_markdown_report(sample_data, created_at="2026-11-01 12:00:00 UTC")
        self.assertIn("# 🛡️ VeriText — AI Text & Document Authenticity Report", md)
        self.assertIn("## 📋 Inspection Metadata", md)
        self.assertIn("## 🎯 Executive Verdict & Confidence Status", md)
        self.assertIn("## 📊 Document Statistics & Metrics", md)
        self.assertIn("sample_document.docx", md)
        self.assertIn("RuBERT-AI", md)
        self.assertIn("88.50%", md)
        self.assertIn("85.00%", md)
        self.assertIn("важно отметить", md)
        self.assertIn("Paragraph #1 — 🔴 AI-Generated", md)
        self.assertIn("Paragraph #2 — 🔴 AI-Generated", md)
        self.assertIn("```text\nParagraph one.\n\nParagraph two.\n```", md)

    def test_api_report_markdown_endpoint_single(self) -> None:
        payload = {
            "data": {
                "filename": "test.txt",
                "label": "human",
                "human_ai_scale": 5.0,
                "full_text": "Hello world!",
            },
            "report_type": "single",
        }
        resp = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/markdown", resp.headers["content-type"])
        self.assertIn('attachment; filename="report_test.md"', resp.headers["content-disposition"])
        self.assertIn("# 🛡️ VeriText — AI Text & Document Authenticity Report", resp.text)
        self.assertIn("Grade A (Human)", resp.text)

    def test_cli_report_file_export(self) -> None:
        class DummyDetector:
            device_str = "cpu"

            def detect_language(self, text: str) -> str:
                return "ru"

            def score_single(self, text: str, mode: str = "auto") -> dict:
                return {
                    "label": "human",
                    "score": -2.5,
                    "human_ai_scale": 15.0,
                    "model_used": "DummyModel",
                }

            def score_batch(self, texts: list[str], mode: str = "auto", batch_size: int = 16) -> list[dict]:
                return [
                    {
                        "label": "human",
                        "score": -2.5,
                        "human_ai_scale": 15.0,
                        "ai_probability": 0.15,
                    }
                    for _ in texts
                ]

        with TemporaryDirectory() as tmp_dir:
            out_md_path = Path(tmp_dir) / "test_report.md"
            sample_text = "Sample text for testing CLI report export."

            code = score_text(
                detector=DummyDetector(),  # type: ignore
                text=sample_text,
                mode="auto",
                as_json=False,
                as_markdown=False,
                report_path=out_md_path,
            )
            self.assertEqual(code, 0)
            self.assertTrue(out_md_path.exists())
            content = out_md_path.read_text(encoding="utf-8")
            self.assertIn("# 🛡️ VeriText — AI Text & Document Authenticity Report", content)
            self.assertIn("CLI Text Input", content)
            self.assertIn("15.00%", content)

    def test_generate_humanize_markdown_report_structure(self) -> None:
        sample_data = {
            "filename": "humanized_doc.docx",
            "original_text": "Before paragraph.",
            "humanized_text": "After paragraph.",
            "mode": "ru",
            "effective_language": "ru",
            "strength": "balanced",
            "before_scale": 88.5,
            "after_scale": 12.0,
            "delta_scale": 76.5,
            "before_confidence": {"grade": "Grade D", "label": "Likely AI"},
            "after_confidence": {"grade": "Grade A", "label": "Verified Human"},
            "summary": {
                "word_count_before": 2,
                "word_count_after": 2,
                "char_count_before": 17,
                "char_count_after": 16,
                "total_paragraphs": 1,
                "modified_paragraphs_count": 1,
                "removed_cliches_count": 1,
            },
            "removed_markers": [
                {
                    "phrase": "важно отметить",
                    "description": "Formal opening cliché",
                    "replacement": "",
                    "count": 1,
                }
            ],
            "paragraphs": [
                {
                    "index": 1,
                    "original_text": "Before paragraph.",
                    "humanized_text": "After paragraph.",
                    "before_scale": 88.5,
                    "after_scale": 12.0,
                    "delta_scale": 76.5,
                    "stage_applied": "full",
                }
            ],
        }
        md = generate_humanize_markdown_report(sample_data, created_at="2026-11-03 12:00:00 UTC")
        self.assertIn("# ✨ VeriText — AI Text Humanization & Transformation Report", md)
        self.assertIn("## 📋 Transformation Metadata", md)
        self.assertIn("## 🎯 Executive Verdict & Confidence Evolution", md)
        self.assertIn("## 🔍 Scrubbed Generative Clichés & Stylistic Markers", md)
        self.assertIn("## 📑 Paragraph-by-Paragraph Transformation Breakdown", md)
        self.assertIn("humanized_doc.docx", md)
        self.assertIn("88.50%", md)
        self.assertIn("12.00%", md)
        self.assertIn("-76.50%", md)
        self.assertIn("важно отметить", md)
        self.assertIn("```text\nAfter paragraph.\n```", md)

    def test_api_report_markdown_endpoint_humanize(self) -> None:
        payload = {
            "data": {
                "filename": "article.md",
                "before_scale": 80.0,
                "after_scale": 20.0,
                "delta_scale": 60.0,
                "paragraphs": [],
                "summary": {},
            },
            "report_type": "humanize",
            "filename": "article.md",
        }
        resp = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/markdown", resp.headers["content-type"])
        self.assertIn('attachment; filename="report_humanized_article.md"', resp.headers["content-disposition"])
        self.assertIn("# ✨ VeriText — AI Text Humanization & Transformation Report", resp.text)

    def test_generate_humanize_markdown_report_missing_or_none_scales(self) -> None:
        """Ensure defensive handling of None, missing, or invalid scale values in paragraphs and root."""
        sample_data = {
            "original_text": "First sentence.",
            "humanized_text": "Second sentence.",
            "before_scale": None,
            "after_scale": None,
            "paragraphs": [
                {
                    "index": 1,
                    "original_text": "First sentence.",
                    "humanized_text": "Second sentence.",
                    "before_scale": None,
                    "after_scale": None,
                    "delta_scale": None,
                },
                {
                    "index": 2,
                    "original_text": "Another sentence.",
                    "humanized_text": "Another sentence.",
                    # before_scale and after_scale intentionally missing
                },
            ],
        }
        # Must generate cleanly without raising TypeError or ValueError
        md = generate_humanize_markdown_report(sample_data)
        self.assertIn("# ✨ VeriText — AI Text Humanization & Transformation Report", md)
        self.assertIn("0.00%", md)


if __name__ == "__main__":
    unittest.main()
