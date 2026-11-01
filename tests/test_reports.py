from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.reports import generate_batch_markdown_report, generate_markdown_report
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

    def test_generate_batch_markdown_report(self) -> None:
        batch_data = {
            "filename": "dataset_eval.csv",
            "summary": {
                "total_rows": 50,
                "ai_count": 20,
                "human_count": 30,
                "ai_percentage": 40.0,
                "text_column": "content",
                "mode": "auto",
                "device": "cpu",
            },
            "preview_rows": [
                {"id": 1, "content": "Text one", "ai_label": "human", "ai_probability_pct": 12.0},
                {"id": 2, "content": "Text two", "ai_label": "ai", "ai_probability_pct": 95.0},
            ],
        }

        md = generate_batch_markdown_report(batch_data, created_at="2026-11-01 12:00:00 UTC")
        self.assertIn("# 📊 VeriText — Batch Dataset Analysis Report", md)
        self.assertIn("## 📋 Dataset Metadata", md)
        self.assertIn("## 📈 Summary Classification Results", md)
        self.assertIn("dataset_eval.csv", md)
        self.assertIn("40.00%", md)
        self.assertIn("60.00%", md)
        self.assertIn("Text one", md)
        self.assertIn("Text two", md)

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

    def test_api_report_markdown_endpoint_batch(self) -> None:
        payload = {
            "data": {
                "filename": "table.csv",
                "summary": {
                    "total_rows": 10,
                    "ai_count": 2,
                    "human_count": 8,
                    "ai_percentage": 20.0,
                },
                "preview_rows": [{"col": "val"}],
            },
            "report_type": "batch",
        }
        resp = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/markdown", resp.headers["content-type"])
        self.assertIn('attachment; filename="report_table.md"', resp.headers["content-disposition"])
        self.assertIn("# 📊 VeriText — Batch Dataset Analysis Report", resp.text)

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


if __name__ == "__main__":
    unittest.main()
