"""
Challenger Test Suite for Milestone 3 (R3 - Backend API Endpoints & Reporting).
Adversarial and empirical stress-testing of FastAPI routes, schemas, cancellation, and reporting.
"""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.cancellation import OperationCancelled, cancellation_manager
from tests.test_e2e_humanizer import (
    ControlledCriticDetector,
    MockClassificationModel,
    MockSeq2SeqModel,
    MockTokenizer,
)


@pytest.fixture(autouse=True)
def mock_transformers_for_challenger() -> Any:
    """Mock heavy transformers and external LLM to run lightweight, isolated deterministic tests."""
    def mock_llm_candidates_fn(self: Any, text: str, num_candidates: int, temperature: float) -> list[str]:
        return [
            "Transformed natural candidate version with authentic cadence.",
            "Alternative refined phrasing emphasizing clear semantic expression.",
        ][:num_candidates]

    patches = [
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer()),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=MockClassificationModel(),
        ),
        patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=MockSeq2SeqModel()),
        patch("backend.humanizer.ModernTextHumanizer._generate_llm_candidates", mock_llm_candidates_fn),
    ]
    import backend.cache as cache_mod

    if hasattr(cache_mod, "AutoTokenizer"):
        patches.append(
            patch.object(
                cache_mod,
                "AutoTokenizer",
                MagicMock(from_pretrained=MagicMock(return_value=MockTokenizer())),
            )
        )
    if hasattr(cache_mod, "AutoModelForSequenceClassification"):
        patches.append(
            patch.object(
                cache_mod,
                "AutoModelForSequenceClassification",
                MagicMock(from_pretrained=MagicMock(return_value=MockClassificationModel())),
            )
        )
    if hasattr(cache_mod, "AutoModelForSeq2SeqLM"):
        patches.append(
            patch.object(
                cache_mod,
                "AutoModelForSeq2SeqLM",
                MagicMock(from_pretrained=MagicMock(return_value=MockSeq2SeqModel())),
            )
        )

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


class TestChallengerHumanizeEndpoint(unittest.TestCase):
    """Empirical challenge tests for POST /api/humanize."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_humanize_valid_russian_text(self) -> None:
        """Verify successful humanization of standard Russian text with all schema keys."""
        payload = {
            "text": "В современном мире важно отметить, что искусственный интеллект играет ключевую роль.",
            "mode": "ru",
            "strength": "balanced",
            "request_id": "challenger-ru-1",
        }
        res = self.client.post("/api/humanize", json=payload)
        self.assertEqual(res.status_code, 200, f"Failed: {res.text}")
        data = res.json()

        # Validate schema contract
        expected_keys = [
            "original_text",
            "humanized_text",
            "mode",
            "effective_language",
            "strength",
            "request_id",
            "before_scale",
            "after_scale",
            "delta_scale",
            "before_paragraph_scale",
            "after_paragraph_scale",
            "before_confidence",
            "after_confidence",
            "summary",
            "removed_markers",
            "paragraphs",
        ]
        for key in expected_keys:
            self.assertIn(key, data, f"Missing key in response: {key}")

        self.assertEqual(data["effective_language"], "ru")
        self.assertEqual(data["mode"], "ru")
        self.assertEqual(data["strength"], "balanced")
        self.assertEqual(data["request_id"], "challenger-ru-1")
        self.assertEqual(
            data["delta_scale"],
            round(data["before_scale"] - data["after_scale"], 2),
        )
        self.assertIsInstance(data["paragraphs"], list)
        self.assertGreaterEqual(len(data["paragraphs"]), 1)

    def test_humanize_valid_english_text(self) -> None:
        """Verify successful humanization of standard English text."""
        payload = {
            "text": "It is important to remember that rigorous software testing ensures production reliability.",
            "mode": "en",
            "strength": "aggressive",
            "request_id": "challenger-en-1",
        }
        res = self.client.post("/api/humanize", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["effective_language"], "en")
        self.assertEqual(data["strength"], "aggressive")
        self.assertIsInstance(data["summary"], dict)
        self.assertIn("word_count_before", data["summary"])
        self.assertIn("word_count_after", data["summary"])

    def test_humanize_empty_string_returns_400(self) -> None:
        """Verify POST /api/humanize with empty string returns HTTP 400."""
        res = self.client.post("/api/humanize", json={"text": "", "mode": "auto"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Text cannot be empty", res.json().get("detail", ""))

    def test_humanize_whitespace_only_returns_400(self) -> None:
        """Verify POST /api/humanize with various whitespace characters returns HTTP 400."""
        for ws in ["   ", "\t\t", "\n\n\r\n", "  \t \n \r  "]:
            res = self.client.post("/api/humanize", json={"text": ws, "mode": "auto"})
            self.assertEqual(res.status_code, 400, f"Expected 400 for whitespace '{ws!r}', got {res.status_code}")
            self.assertIn("Text cannot be empty", res.json().get("detail", ""))

    def test_humanize_missing_text_field_returns_422_or_400(self) -> None:
        """Verify POST /api/humanize with missing 'text' field is rejected."""
        # Pydantic schemas might default text to "" which then triggers the route's 400 check
        res = self.client.post("/api/humanize", json={"mode": "auto"})
        self.assertIn(res.status_code, (400, 422))

    def test_humanize_invalid_text_types(self) -> None:
        """Verify non-string inputs for text are rejected by Pydantic validation (422)."""
        for bad_val in [12345, True, ["some", "list"], {"key": "val"}]:
            res = self.client.post("/api/humanize", json={"text": bad_val})
            self.assertEqual(res.status_code, 422, f"Expected 422 for {bad_val!r}, got {res.status_code}")

    def test_humanize_x_request_id_header(self) -> None:
        """Verify X-Request-ID header is propagated when request_id is omitted from body."""
        payload = {"text": "Simple test paragraph for header tracking.", "mode": "en"}
        headers = {"X-Request-ID": "header-trace-999"}
        res = self.client.post("/api/humanize", json=payload, headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("request_id"), "header-trace-999")

    def test_humanize_single_character_boundary(self) -> None:
        """Boundary test: 1-character text should process gracefully without splitting crashes."""
        res = self.client.post("/api/humanize", json={"text": "A", "mode": "auto"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["summary"]["total_paragraphs"], 1)

    def test_humanize_punctuation_and_symbols_boundary(self) -> None:
        """Boundary test: Text with only punctuation and numbers should not crash."""
        res = self.client.post("/api/humanize", json={"text": "12345! ... ??? ---", "mode": "auto"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["summary"]["total_paragraphs"], 1)

    def test_humanize_multiparagraph_complex_formatting(self) -> None:
        """Boundary test: Text with emojis, code formatting, bullet points, and multiple paragraphs."""
        complex_text = (
            "✨ Paragraph 1: First concept with emojis and *markdown* formatting.\n\n"
            "Paragraph 2: Second concept with `inline code` and numbers (42, 100%).\n\n"
            "Paragraph 3: Third paragraph concluding the narrative."
        )
        res = self.client.post("/api/humanize", json={"text": complex_text, "mode": "auto"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["summary"]["total_paragraphs"], 3)
        self.assertEqual(len(data["paragraphs"]), 3)


class TestChallengerCancellationEndpoints(unittest.TestCase):
    """Empirical challenge tests for POST /api/cancel and /api/abort."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_cancel_by_path_success(self) -> None:
        """Verify POST /api/cancel/{request_id} successfully signals cancellation."""
        token = "test-token-cancel-path"
        cancellation_manager.register(token)
        try:
            res = self.client.post(f"/api/cancel/{token}")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["request_id"], token)
            self.assertTrue(data["aborted"])
            self.assertEqual(data["message"], "Cancellation signal sent")
        finally:
            cancellation_manager.unregister(token)

    def test_abort_by_path_success(self) -> None:
        """Verify POST /api/abort/{request_id} successfully signals cancellation."""
        token = "test-token-abort-path"
        cancellation_manager.register(token)
        try:
            res = self.client.post(f"/api/abort/{token}")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["request_id"], token)
            self.assertTrue(data["aborted"])
            self.assertEqual(data["message"], "Cancellation signal sent")
        finally:
            cancellation_manager.unregister(token)

    def test_cancel_by_body_success(self) -> None:
        """Verify POST /api/cancel with JSON body successfully signals cancellation."""
        token = "test-token-cancel-body"
        cancellation_manager.register(token)
        try:
            res = self.client.post("/api/cancel", json={"request_id": token})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["request_id"], token)
            self.assertTrue(data["aborted"])
            self.assertEqual(data["message"], "Cancellation signal sent")
        finally:
            cancellation_manager.unregister(token)

    def test_abort_by_body_success(self) -> None:
        """Verify POST /api/abort with JSON body successfully signals cancellation."""
        token = "test-token-abort-body"
        cancellation_manager.register(token)
        try:
            res = self.client.post("/api/abort", json={"request_id": token})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["request_id"], token)
            self.assertTrue(data["aborted"])
            self.assertEqual(data["message"], "Cancellation signal sent")
        finally:
            cancellation_manager.unregister(token)

    def test_cancel_nonexistent_request_id(self) -> None:
        """Verify cancellation of an unknown request_id returns aborted=False gracefully."""
        res = self.client.post("/api/cancel/nonexistent-token-xyz")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["aborted"])
        self.assertEqual(data["message"], "Request not found or already completed")

    def test_cancel_repeated_calls_idempotent(self) -> None:
        """Verify cancelling the same token twice: first returns True, second returns True (token remains cancelled)."""
        token = "test-token-idempotent"
        cancellation_manager.register(token)
        try:
            res1 = self.client.post(f"/api/cancel/{token}")
            self.assertEqual(res1.status_code, 200)
            self.assertTrue(res1.json()["aborted"])

            # Second call while still registered
            res2 = self.client.post(f"/api/cancel/{token}")
            self.assertEqual(res2.status_code, 200)
            self.assertTrue(res2.json()["aborted"])
        finally:
            cancellation_manager.unregister(token)

    def test_in_flight_humanize_cancellation_returns_499(self) -> None:
        """Empirically test that cancelling an in-flight humanize request raises HTTP 499."""
        from backend.api import routes

        token = "in-flight-cancel-token"

        # Hook into humanizer to trigger cancellation mid-execution
        orig_method = routes.humanizer.humanize_text

        def canceling_humanize_text(*args: Any, **kwargs: Any) -> Any:
            cancellation_manager.cancel(token)
            # Call original, which checks cancel_event.is_set
            return orig_method(*args, **kwargs)

        with patch.object(routes.humanizer, "humanize_text", side_effect=canceling_humanize_text):
            res = self.client.post(
                "/api/humanize",
                json={
                    "text": "This is a paragraph that will be cancelled in flight.",
                    "request_id": token,
                },
            )
            self.assertEqual(res.status_code, 499)
            self.assertIn("Humanization cancelled by user", res.json().get("detail", ""))


class TestChallengerReportMarkdownEndpoint(unittest.TestCase):
    """Empirical challenge tests for POST /api/report/markdown with report_type='humanize'."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_report_humanize_complete_data_structure(self) -> None:
        """Verify complete report generation includes all 6 core sections and proper markdown table."""
        payload = {
            "data": {
                "filename": "quarterly_analysis.md",
                "mode": "en",
                "effective_language": "en",
                "strength": "balanced",
                "original_text": "Original text paragraph.",
                "humanized_text": "Humanized transformed paragraph.",
                "before_scale": 88.0,
                "after_scale": 12.0,
                "delta_scale": 76.0,
                "before_confidence": {"grade": "Grade D", "label": "Likely AI"},
                "after_confidence": {"grade": "Grade A", "label": "Verified Human"},
                "summary": {
                    "word_count_before": 25,
                    "word_count_after": 24,
                    "char_count_before": 150,
                    "char_count_after": 145,
                    "total_paragraphs": 1,
                    "modified_paragraphs_count": 1,
                    "removed_cliches_count": 2,
                    "original_ai_paragraphs": 1,
                    "humanized_ai_paragraphs": 0,
                },
                "removed_markers": [
                    {"phrase": "In conclusion", "category": "boilerplate", "replacement": ""},
                    {"phrase": "delve into", "category": "cliche", "replacement": "explore"},
                ],
                "paragraphs": [
                    {
                        "index": 1,
                        "original_text": "In conclusion, we must delve into the metrics.",
                        "humanized_text": "We explore the metrics.",
                        "before_scale": 88.0,
                        "after_scale": 12.0,
                        "delta_scale": 76.0,
                        "before_label": "Likely AI",
                        "after_label": "Verified Human",
                        "stage_applied": "full",
                        "changed": True,
                    }
                ],
            },
            "report_type": "humanize",
            "filename": "quarterly_analysis.md",
        }
        res = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "text/markdown; charset=utf-8")
        self.assertIn("report_humanized_quarterly_analysis.md", res.headers.get("content-disposition", ""))

        md = res.text
        # Check required sections
        self.assertIn("# ✨ VeriText — AI Text Humanization & Transformation Report", md)
        self.assertIn("## 📋 Transformation Metadata", md)
        self.assertIn("## 🎯 Executive Verdict & Confidence Evolution", md)
        self.assertIn("## 🔍 Scrubbed Generative Clichés & Stylistic Markers", md)
        self.assertIn("## 📑 Paragraph-by-Paragraph Transformation Breakdown", md)
        self.assertIn("## 📜 Full Transformed Text", md)
        self.assertIn("<summary>Click to expand full humanized text</summary>", md)
        self.assertIn("*Report automatically generated by VeriText AI Text Humanizer.*", md)

    def test_report_humanize_minimal_sparse_data(self) -> None:
        """Verify report generation does not crash on sparse/empty dictionaries."""
        payload = {
            "data": {},
            "report_type": "humanize",
        }
        res = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(res.status_code, 200, f"Sparse data failed: {res.text}")
        md = res.text
        self.assertIn("# ✨ VeriText — AI Text Humanization & Transformation Report", md)
        self.assertIn("## 🎯 Executive Verdict", md)

    def test_report_humanize_path_traversal_sanitization(self) -> None:
        """Verify behavior when filename contains directory traversal characters.
        Finding: Routes currently preserve traversal tokens in Content-Disposition.
        """
        payload = {
            "data": {"before_scale": 50.0, "after_scale": 20.0},
            "report_type": "humanize",
            "filename": "../../../etc/passwd",
        }
        res = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(res.status_code, 200)
        disposition = res.headers.get("content-disposition", "")
        # Observed: 'attachment; filename="report_humanized_../../../etc/passwd.md"'
        self.assertIn("report_humanized_", disposition)


class TestChallengerHumanizeDocumentEndpoint(unittest.TestCase):
    """Empirical challenge tests for POST /api/humanize-document."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_humanize_document_empty_file_rejected(self) -> None:
        """Verify 0-byte file upload returns HTTP 400."""
        res = self.client.post(
            "/api/humanize-document",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"mode": "auto"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Document contains no readable text", res.json().get("detail", ""))

    def test_humanize_document_valid_txt(self) -> None:
        """Verify uploading a plain .txt file succeeds and returns HumanizeDocumentResponse."""
        content = b"This is a valid document text file ready for humanization."
        res = self.client.post(
            "/api/humanize-document",
            files={"file": ("notes.txt", content, "text/plain")},
            data={"mode": "en", "strength": "balanced"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("filename"), "notes.txt")
        self.assertEqual(data.get("file_type"), "txt")
        self.assertIn("before_scale", data)
        self.assertIn("after_scale", data)

    def test_humanize_document_unsupported_extensions_rejected(self) -> None:
        """Verify unsupported extensions (.csv, .zip, .exe, .tar) are rejected with HTTP 400."""
        unsupported = [
            ("data.csv", b"a,b,c\n1,2,3", "text/csv"),
            ("archive.zip", b"PK\x03\x04...", "application/zip"),
            ("program.exe", b"MZ...", "application/octet-stream"),
        ]
        for fname, content, mime in unsupported:
            res = self.client.post(
                "/api/humanize-document",
                files={"file": (fname, content, mime)},
                data={"mode": "auto"},
            )
            self.assertEqual(res.status_code, 400, f"Expected 400 for {fname}, got {res.status_code}")
            self.assertIn("Unsupported file format", res.json().get("detail", ""))
