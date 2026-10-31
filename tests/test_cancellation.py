from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.cancellation import (
    CancellationManager,
    OperationCancelled,
    cancellation_manager,
)
from backend.engine import ModernAIDetector


class TestCancellation(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_cancellation_manager_lifecycle(self) -> None:
        mgr = CancellationManager()
        self.assertEqual(mgr.active_count(), 0)

        ev = mgr.register("req-1")
        self.assertFalse(ev.is_set())
        self.assertFalse(mgr.is_cancelled("req-1"))
        self.assertEqual(mgr.active_count(), 1)

        success = mgr.cancel("req-1")
        self.assertTrue(success)
        self.assertTrue(ev.is_set())
        self.assertTrue(mgr.is_cancelled("req-1"))

        # Cancel unknown request
        self.assertFalse(mgr.cancel("unknown-req"))

        mgr.unregister("req-1")
        self.assertEqual(mgr.active_count(), 0)
        self.assertFalse(mgr.is_cancelled("req-1"))

    def test_engine_score_single_cancelled(self) -> None:
        detector = ModernAIDetector()
        with self.assertRaises(OperationCancelled):
            detector.score_single(
                "Тестовый текст для быстрой проверки аборта операции.",
                is_cancelled=lambda: True,
            )

    def test_engine_score_batch_cancelled(self) -> None:
        detector = ModernAIDetector()
        with self.assertRaises(OperationCancelled):
            detector.score_batch(
                ["Параграф 1", "Параграф 2"],
                is_cancelled=lambda: True,
            )

    def test_api_abort_endpoints(self) -> None:
        # Register a token
        cancellation_manager.register("active-test-req")
        try:
            # Abort via path parameter
            res = self.client.post("/api/abort/active-test-req")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["aborted"])
            self.assertEqual(data["request_id"], "active-test-req")

            # Try aborting an already non-existent token
            res2 = self.client.post("/api/abort/non-existent-req")
            self.assertEqual(res2.status_code, 200)
            self.assertFalse(res2.json()["aborted"])

            # Abort via JSON body
            cancellation_manager.register("active-body-req")
            res3 = self.client.post("/api/abort", json={"request_id": "active-body-req"})
            self.assertEqual(res3.status_code, 200)
            self.assertTrue(res3.json()["aborted"])
        finally:
            cancellation_manager.unregister("active-test-req")
            cancellation_manager.unregister("active-body-req")

    def test_health_check_reports_active_requests(self) -> None:
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("active_requests", data)
        self.assertIsInstance(data["active_requests"], int)
        self.assertNotIn("Apple Silicon", data.get("device_name", ""))

    def test_api_score_returns_499_when_cancelled(self) -> None:
        from unittest.mock import patch

        def fake_score_single(text, mode="auto", is_cancelled=None):
            raise OperationCancelled("Analysis aborted by user.")

        with patch("backend.api.routes.engine.score_single", side_effect=fake_score_single):
            res = self.client.post(
                "/api/score",
                json={"text": "Некоторый текст для теста", "mode": "ru"},
                headers={"X-Request-ID": "test-abort-score"},
            )
            self.assertEqual(res.status_code, 499)
            self.assertIn("cancelled", res.json()["detail"].lower())

    def test_api_score_document_returns_499_when_cancelled(self) -> None:
        from unittest.mock import patch

        def fake_score_single(text, mode="auto", is_cancelled=None):
            raise OperationCancelled("Analysis aborted by user.")

        with patch("backend.api.routes.engine.score_single", side_effect=fake_score_single):
            file_content = b"# Title\n\nSome paragraph for testing document analysis cancellation."
            res = self.client.post(
                "/api/score-document",
                files={"file": ("test.md", file_content, "text/markdown")},
                data={"mode": "auto"},
                headers={"X-Request-ID": "test-abort-doc"},
            )
            self.assertEqual(res.status_code, 499)
            self.assertIn("cancelled", res.json()["detail"].lower())

    def test_api_score_file_returns_499_when_cancelled(self) -> None:
        from unittest.mock import patch

        def fake_score_batch(texts, mode="auto", batch_size=16, is_cancelled=None):
            raise OperationCancelled("Analysis aborted by user.")

        with patch("backend.api.routes.engine.score_batch", side_effect=fake_score_batch):
            csv_content = b"text\nRow one text\nRow two text\n"
            res = self.client.post(
                "/api/score-file",
                files={"file": ("dataset.csv", csv_content, "text/csv")},
                data={"mode": "auto", "batch_size": 32},
                headers={"X-Request-ID": "test-abort-batch"},
            )
            self.assertEqual(res.status_code, 499)
            self.assertIn("cancelled", res.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
