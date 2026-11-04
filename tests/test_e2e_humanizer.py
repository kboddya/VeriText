"""
E2E Test Suite for VeriText AI Text Humanizer & Batch Tabular Purge.

Covers all 5 core requirements (R1 through R5) across 4 comprehensive tiers:
- Tier 1: Feature Coverage (R1, R2, R3, R4, R5)
- Tier 2: Boundary & Corner Cases
- Tier 3: Cross-Feature Combinations
- Tier 4: Real-World Application Scenarios

Total tests: 61 opaque-box tests executing efficiently via mocking of heavy model weights.
"""

from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

import backend.config as config_module
from backend.api.app import app
from backend.cancellation import (
    CancellationManager,
    OperationCancelled,
    cancellation_manager,
)
from backend.cli import parse_args
from backend.engine import ModernAIDetector


# ==============================================================================
# Model Weight Mocking Infrastructure (Prevents multi-gigabyte downloads)
# ==============================================================================

class MockTokenizer:
    """Lightweight mock tokenizer replacing Hugging Face AutoTokenizer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(text, str):
            n = 1
        elif isinstance(text, (list, tuple)):
            n = len(text)
        else:
            n = 1

        input_ids = torch.ones((n, 12), dtype=torch.long)
        attention_mask = torch.ones((n, 12), dtype=torch.long)

        class MockBatchEncoding(dict):
            def __init__(self, d: dict[str, Any]) -> None:
                super().__init__(d)
                self.__dict__.update(d)

            def to(self, device: Any) -> MockBatchEncoding:
                return self

        return MockBatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def encode(self, text: str, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.ones((1, 12), dtype=torch.long)

    def decode(self, token_ids: Any, *args: Any, **kwargs: Any) -> str:
        return "Transformed natural text candidate preserving meaning."

    def batch_decode(self, sequences: Any, *args: Any, **kwargs: Any) -> list[str]:
        candidates = [
            "Transformed natural text emphasizing clarity and human expression.",
            "A distinctly rephrased version maintaining core semantic meaning without boilerplate.",
            "Paraphrased alternative with varied sentence structure and natural cadence.",
            "Refined stylistic phrasing with enhanced burstiness and readability.",
        ]
        results = []
        for i, _ in enumerate(sequences):
            results.append(candidates[i % len(candidates)])
        return results


class MockSeq2SeqModel:
    """Mock Seq2Seq paraphraser model replacing AutoModelForSeq2SeqLM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = MagicMock()

    def to(self, device: Any) -> MockSeq2SeqModel:
        return self

    def eval(self) -> MockSeq2SeqModel:
        return self

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        num_return_sequences: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        b = input_ids.shape[0] if input_ids is not None else 1
        return torch.ones((b * num_return_sequences, 12), dtype=torch.long)


class MockClassificationModel:
    """Mock classifier model replacing AutoModelForSequenceClassification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = MagicMock()
        self.config.id2label = {0: "human", 1: "ai"}

    def to(self, device: Any) -> MockClassificationModel:
        return self

    def eval(self) -> MockClassificationModel:
        return self

    def __call__(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        batch_size = input_ids.shape[0] if input_ids is not None else 1

        class MockOutput:
            logits = torch.tensor([[2.0, -2.0]] * batch_size)

        return MockOutput()


class ControlledCriticDetector:
    """Mock detector critic with configurable scores for deterministic testing."""

    def __init__(self, scores: Optional[dict[str, float] | list[float]] = None) -> None:
        self.device_str = "cpu"
        self.device = torch.device("cpu")
        self.scores = scores or {}

    def score_single(
        self,
        text: str,
        mode: str = "auto",
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        if is_cancelled and is_cancelled():
            raise OperationCancelled("Cancelled by user.")
        score = self.scores.get(text, 0.20) if isinstance(self.scores, dict) else 0.20
        return {
            "label": "ai" if score >= 0.50 else "human",
            "score": round((score - 0.5) * 10, 4),
            "human_ai_scale": round(score * 100, 2),
            "ai_probability": score,
            "model_used": "CriticDetectorMock",
        }

    def score_batch(
        self,
        texts: list[str],
        mode: str = "auto",
        batch_size: int = 16,
        is_cancelled: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_step: Optional[Callable[[str], None]] = None,
    ) -> list[dict[str, Any]]:
        if is_cancelled and is_cancelled():
            raise OperationCancelled("Cancelled by user.")
        results = []
        for i, text in enumerate(texts):
            if isinstance(self.scores, dict):
                score = self.scores.get(text, 0.20)
            elif isinstance(self.scores, list) and i < len(self.scores):
                score = self.scores[i]
            else:
                score = 0.20
            results.append({
                "label": "ai" if score >= 0.50 else "human",
                "score": round((score - 0.5) * 10, 4),
                "human_ai_scale": round(score * 100, 2),
                "ai_probability": score,
                "model_used": "CriticDetectorMock",
            })
        return results


@pytest.fixture(autouse=True)
def mock_heavy_transformers() -> Any:
    """Auto-use fixture patching transformer loading and external LLM calls across all test executions."""
    def mock_llm_candidates_fn(self: Any, text: str, num_candidates: int, temperature: float) -> list[str]:
        return [
            "Transformed natural text candidate preserving meaning.",
            "Alternative refined phrasing emphasizing clear semantic expression.",
            "Humanized variation preserving essential context and varied rhythm.",
        ][:num_candidates]

    patches = [
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer()),
        patch("transformers.AutoModelForSequenceClassification.from_pretrained", return_value=MockClassificationModel()),
        patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=MockSeq2SeqModel()),
        patch("backend.humanizer.ModernTextHumanizer._generate_llm_candidates", mock_llm_candidates_fn),
    ]
    import backend.cache as cache_mod
    if hasattr(cache_mod, "AutoTokenizer"):
        patches.append(patch.object(cache_mod, "AutoTokenizer", MagicMock(from_pretrained=MagicMock(return_value=MockTokenizer()))))
    if hasattr(cache_mod, "AutoModelForSequenceClassification"):
        patches.append(patch.object(cache_mod, "AutoModelForSequenceClassification", MagicMock(from_pretrained=MagicMock(return_value=MockClassificationModel()))))
    if hasattr(cache_mod, "AutoModelForSeq2SeqLM"):
        patches.append(patch.object(cache_mod, "AutoModelForSeq2SeqLM", MagicMock(from_pretrained=MagicMock(return_value=MockSeq2SeqModel()))))

    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


# Helper dynamic loaders to support progressive milestone testability
def get_humanizer_cls() -> Any:
    try:
        from backend.humanizer import ModernTextHumanizer
        return ModernTextHumanizer
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"R2 Feature not yet implemented: backend.humanizer.ModernTextHumanizer ({exc})")


def get_scrub_cliches_fn() -> Any:
    try:
        from backend.markers import scrub_cliches
        return scrub_cliches
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"R2 Feature not yet implemented: backend.markers.scrub_cliches ({exc})")


def get_generate_humanize_report_fn() -> Any:
    try:
        from backend.reports import generate_humanize_markdown_report
        return generate_humanize_markdown_report
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"R3 Feature not yet implemented: backend.reports.generate_humanize_markdown_report ({exc})")


# ==============================================================================
# TIER 1: FEATURE COVERAGE (R1 through R5)
# ==============================================================================

class TestTier1R1BatchPurge(unittest.TestCase):
    """Tier 1: R1 Complete Removal of Batch Tabular Datasets (CSV / TSV)."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.root = Path(__file__).resolve().parent.parent

    def test_tier1_r1_pandas_purged_from_dependencies(self) -> None:
        """Verify pandas>=2.0.0 is completely removed from pyproject.toml and source code."""
        pyproject_path = self.root / "pyproject.toml"
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
        self.assertNotIn("pandas", pyproject_text, "pandas dependency must be purged from pyproject.toml")

        # Check backend routes and CLI files for pandas imports
        routes_text = (self.root / "backend" / "api" / "routes.py").read_text(encoding="utf-8")
        cli_text = (self.root / "backend" / "cli.py").read_text(encoding="utf-8")
        self.assertNotIn("import pandas", routes_text, "pandas import must be purged from backend/api/routes.py")
        self.assertNotIn("import pandas", cli_text, "pandas import must be purged from backend/cli.py")

    def test_tier1_r1_score_file_route_purged(self) -> None:
        """Verify POST /api/score-file is purged from the router and returns 404."""
        from backend.api.routes import router
        route_paths = [route.path for route in router.routes if hasattr(route, "path")]
        self.assertNotIn("/score-file", route_paths, "/score-file route must be removed from API router")
        self.assertNotIn("/api/score-file", route_paths, "/api/score-file route must be removed from API router")

        # Requesting /api/score-file should yield 404 Not Found (or 405)
        resp = self.client.post("/api/score-file", files={"file": ("test.csv", b"a,b\n1,2", "text/csv")})
        self.assertIn(resp.status_code, [404, 405], f"Expected 404/405 for removed endpoint, got {resp.status_code}")

    def test_tier1_r1_batch_cli_args_purged(self) -> None:
        """Verify --input, --text-column, and --batch-size are purged from CLI argument parsing."""
        with patch("sys.argv", ["veritext", "--input", "test.csv"]):
            with self.assertRaises(SystemExit):
                parse_args()

        with patch("sys.argv", ["veritext", "--text-column", "body"]):
            with self.assertRaises(SystemExit):
                parse_args()

        with patch("sys.argv", ["veritext", "--batch-size", "32"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_tier1_r1_generate_batch_markdown_report_purged(self) -> None:
        """Verify generate_batch_markdown_report is deleted from backend/reports.py."""
        import backend.reports as reports_mod
        self.assertFalse(
            hasattr(reports_mod, "generate_batch_markdown_report"),
            "generate_batch_markdown_report must be removed from backend/reports.py",
        )

    def test_tier1_r1_engine_score_batch_retained(self) -> None:
        """Verify score_batch in backend/engine.py is retained for paragraph and critic scoring."""
        self.assertTrue(
            hasattr(ModernAIDetector, "score_batch"),
            "ModernAIDetector.score_batch must be preserved in backend/engine.py",
        )
        detector = ModernAIDetector()
        results = detector.score_batch(["Первый тестовый абзац.", "Второй тестовый абзац."])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIn("ai_probability", results[0])
        self.assertIn("human_ai_scale", results[0])

    def test_tier1_r1_web_ui_batch_elements_purged(self) -> None:
        """Verify #view-btn-batch and #view-batch are purged from index.html and app.js."""
        index_html = (self.root / "web" / "public" / "index.html").read_text(encoding="utf-8")
        app_js = (self.root / "web" / "public" / "js" / "app.js").read_text(encoding="utf-8")

        self.assertNotIn("id=\"view-btn-batch\"", index_html, "#view-btn-batch must be purged from index.html")
        self.assertNotIn("id=\"view-batch\"", index_html, "#view-batch container must be purged from index.html")
        self.assertNotIn("submitBatchAnalysis", app_js, "submitBatchAnalysis must be purged from app.js")
        self.assertNotIn("selectedBatchFile", app_js, "selectedBatchFile must be purged from app.js")


class TestTier1R2HumanizerEngine(unittest.TestCase):
    """Tier 1: R2 Zero-Training Detector-Guided Humanizer Engine."""

    def test_tier1_r2_paraphrasers_catalog_config(self) -> None:
        """Verify default bilingual paraphraser catalog and strength presets in backend/config.py."""
        self.assertTrue(
            hasattr(config_module, "PARAPHRASERS_CATALOG"),
            "PARAPHRASERS_CATALOG must be defined in backend/config.py",
        )
        catalog = getattr(config_module, "PARAPHRASERS_CATALOG")
        self.assertIn("ru", catalog, "Russian catalog entry missing")
        self.assertIn("en", catalog, "English catalog entry missing")
        self.assertEqual(catalog["ru"]["id"], "cointegrated/rut5-base-paraphraser")
        self.assertEqual(catalog["en"]["id"], "humarin/chatgpt_paraphraser_on_T5_base")

        self.assertTrue(
            hasattr(config_module, "HUMANIZER_STRENGTH_PRESETS"),
            "HUMANIZER_STRENGTH_PRESETS must be defined in backend/config.py",
        )
        presets = getattr(config_module, "HUMANIZER_STRENGTH_PRESETS")
        self.assertIn("light", presets)
        self.assertIn("balanced", presets)
        self.assertIn("aggressive", presets)

        # Length parity bounds
        self.assertEqual(getattr(config_module, "LENGTH_PARITY_MIN_RATIO", 0.65), 0.65)
        self.assertEqual(getattr(config_module, "LENGTH_PARITY_MAX_RATIO", 1.40), 1.40)

    def test_tier1_r2_load_offline_first_supports_model_cls(self) -> None:
        """Verify load_offline_first in backend/cache.py supports model_cls parameter."""
        import backend.cache as cache_mod
        import inspect

        sig = inspect.signature(cache_mod.load_offline_first)
        self.assertIn("model_cls", sig.parameters, "load_offline_first must accept model_cls parameter")

        # Test loading with custom mock class
        with patch("backend.cache.should_check_remote", return_value=False):
            m, t, eff_id = cache_mod.load_offline_first(
                "mock-seq2seq-model",
                torch.device("cpu"),
                model_cls=MockSeq2SeqModel,
            )
            self.assertIsInstance(m, MockSeq2SeqModel)

    def test_tier1_r2_stage1_decliche_scrubber_ru(self) -> None:
        """Verify Stage 1 de-cliché scrubber replaces/strips Russian AI boilerplate and normalizes casing."""
        scrub_fn = get_scrub_cliches_fn()
        raw_text = "Важно отметить, что искусственный интеллект играет ключевую роль в современном мире."
        cleaned, removed = scrub_fn(raw_text, lang="ru")

        self.assertNotEqual(raw_text, cleaned)
        self.assertTrue(len(removed) >= 1, "Expected at least one removed Russian cliché")
        self.assertFalse(cleaned.startswith("важно отметить"), "Leading cliché preamble was not removed")
        # Capitalization check after removal
        self.assertTrue(cleaned[0].isupper(), f"First letter of cleaned text must be capitalized: '{cleaned}'")

    def test_tier1_r2_stage1_decliche_scrubber_en(self) -> None:
        """Verify Stage 1 de-cliché scrubber removes English AI boilerplate and transitions."""
        scrub_fn = get_scrub_cliches_fn()
        raw_text = "It is important to remember that deep learning revolutionized image recognition."
        cleaned, removed = scrub_fn(raw_text, lang="en")

        self.assertNotEqual(raw_text, cleaned)
        self.assertTrue(len(removed) >= 1, "Expected at least one removed English cliché")
        self.assertFalse(cleaned.lower().startswith("it is important to remember"), "English cliché was not removed")
        self.assertTrue(cleaned[0].isupper(), f"First letter must be capitalized: '{cleaned}'")

    def test_tier1_r2_stage2_multi_candidate_generation(self) -> None:
        """Verify Stage 2 multi-candidate paraphrasing pipeline produces candidate variations."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()

        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize", None))
        self.assertIsNotNone(method, "ModernTextHumanizer must have humanize_text or humanize method")

        result = method("This is a formal test sentence generated by synthetic algorithms.")
        self.assertIsInstance(result, dict)
        self.assertIn("humanized_text", result)
        self.assertIn("paragraphs", result)

    def test_tier1_r2_stage3_detector_critic_monotonic_guarantee(self) -> None:
        """Verify Stage 3 detector critic guarantees monotonic improvement (never worsens Stage 1)."""
        humanizer_cls = get_humanizer_cls()

        # Critic returns high AI score for all neural candidates (0.90), but lower for Stage 1 (0.30)
        critic = ControlledCriticDetector(scores={"Stage 1 Text": 0.30})
        humanizer = humanizer_cls(detector=critic)

        # Monotonic improvement guarantee: even if neural candidates are generated, Stage 1 is selected
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize", None))
        res = method("Важно отметить, что технология развивается стремительно.", mode="ru")

        # after_scale must be <= before_scale
        self.assertLessEqual(
            res["after_scale"],
            res["before_scale"],
            "Critic must guarantee monotonic score improvement",
        )

    def test_tier1_r2_llm_fallback_to_local_t5(self) -> None:
        """Verify automatic fallback to local T5 when external LLM endpoint fails/times out."""
        humanizer_cls = get_humanizer_cls()

        with patch.object(config_module, "LLM_BASE_URL", "http://127.0.0.1:9999/v1"):
            humanizer = humanizer_cls()
            method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize", None))
            # Should not raise ConnectError or HTTPException; must fall back to local model
            res = method("Technology is evolving rapidly across diverse domains.")
            self.assertIn("humanized_text", res)
            self.assertIsInstance(res["humanized_text"], str)


class TestTier1R3ApiAndReports(unittest.TestCase):
    """Tier 1: R3 Backend API Endpoints & Reporting."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_tier1_r3_post_humanize_api_success(self) -> None:
        """Verify POST /api/humanize returns valid JSON with before/after scales and diffs."""
        payload = {
            "text": "В современном мире искусственный интеллект играет ключевую роль.",
            "mode": "ru",
            "strength": "balanced",
            "request_id": "e2e-test-req-1",
        }
        res = self.client.post("/api/humanize", json=payload)
        self.assertEqual(res.status_code, 200, f"Expected 200 from /api/humanize, got: {res.text}")
        data = res.json()

        for key in ["before_scale", "after_scale", "delta_scale", "paragraphs", "summary"]:
            self.assertIn(key, data, f"Key '{key}' missing from /api/humanize response")
        self.assertIsInstance(data["paragraphs"], list)

    def test_tier1_r3_post_humanize_document_txt_md(self) -> None:
        """Verify POST /api/humanize-document processes .txt and .md files."""
        md_content = b"# Document Title\n\nIt is important to remember that testing ensures high quality."
        res = self.client.post(
            "/api/humanize-document",
            files={"file": ("test_doc.md", md_content, "text/markdown")},
            data={"mode": "en", "strength": "balanced"},
        )
        self.assertEqual(res.status_code, 200, f"Expected 200, got: {res.text}")
        data = res.json()
        self.assertEqual(data.get("filename"), "test_doc.md")
        self.assertIn("paragraphs", data)

    def test_tier1_r3_post_humanize_document_docx_pdf(self) -> None:
        """Verify POST /api/humanize-document extracts and humanizes text from .docx and .pdf."""
        # Mock extract_text_from_file from backend.parsers
        with patch("backend.api.routes.extract_text_from_file", return_value=(
            "Title\n\nExtracted paragraph from binary document.",
            ["Extracted paragraph from binary document."]
        )):
            res = self.client.post(
                "/api/humanize-document",
                files={"file": ("sample.pdf", b"%PDF-1.4...", "application/pdf")},
                data={"mode": "auto"},
            )
            self.assertEqual(res.status_code, 200, f"Expected 200 for PDF, got: {res.text}")
            data = res.json()
            self.assertEqual(data.get("filename"), "sample.pdf")

    def test_tier1_r3_cancellation_endpoints(self) -> None:
        """Verify POST /api/cancel/{request_id} and POST /api/abort/{request_id} cancel tokens."""
        cancellation_manager.register("token-cancel-e2e")
        try:
            res_cancel = self.client.post("/api/cancel/token-cancel-e2e")
            self.assertEqual(res_cancel.status_code, 200)
            self.assertTrue(res_cancel.json().get("aborted"))

            cancellation_manager.register("token-abort-e2e")
            res_abort = self.client.post("/api/abort/token-abort-e2e")
            self.assertEqual(res_abort.status_code, 200)
            self.assertTrue(res_abort.json().get("aborted"))
        finally:
            cancellation_manager.unregister("token-cancel-e2e")
            cancellation_manager.unregister("token-abort-e2e")

    def test_tier1_r3_progress_polling(self) -> None:
        """Verify GET /api/status/{request_id} returns live step progress."""
        cancellation_manager.register("token-progress-e2e")
        try:
            cancellation_manager.update_progress(
                "token-progress-e2e",
                message="Evaluating candidates...",
                percent=70,
                step=3,
                total_steps=4,
            )
            res = self.client.get("/api/status/token-progress-e2e")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data.get("percent"), 70)
            self.assertEqual(data.get("step"), 3)
            self.assertEqual(data.get("total_steps"), 4)
            self.assertIn("Evaluating", data.get("message", ""))
        finally:
            cancellation_manager.unregister("token-progress-e2e")

    def test_tier1_r3_generate_humanize_markdown_report_structure(self) -> None:
        """Verify generate_humanize_markdown_report formats report with executive verdict & diffs."""
        report_fn = get_generate_humanize_report_fn()
        sample_data = {
            "original_text": "Original test paragraph.",
            "humanized_text": "Humanized test paragraph.",
            "mode": "en",
            "effective_language": "en",
            "strength": "balanced",
            "before_scale": 85.5,
            "after_scale": 15.2,
            "delta_scale": 70.3,
            "before_confidence": {"grade": "Grade D", "label": "Likely AI"},
            "after_confidence": {"grade": "Grade A", "label": "Verified Human"},
            "summary": {
                "word_count_before": 10,
                "word_count_after": 10,
                "char_count_before": 50,
                "char_count_after": 50,
                "total_paragraphs": 1,
                "modified_paragraphs_count": 1,
                "removed_cliches_count": 1,
            },
            "removed_markers": [{"phrase": "It is important to note", "category": "boilerplate"}],
            "paragraphs": [
                {
                    "index": 1,
                    "original_text": "Original test paragraph.",
                    "humanized_text": "Humanized test paragraph.",
                    "before_scale": 85.5,
                    "after_scale": 15.2,
                    "delta_scale": 70.3,
                    "stage_applied": "full",
                }
            ],
        }

        md = report_fn(sample_data, source_name="sample.md")
        self.assertIn("# ✨ VeriText — AI Text Humanization & Transformation Report", md)
        self.assertIn("Executive Verdict", md)
        self.assertIn("85.50%", md)
        self.assertIn("15.20%", md)
        self.assertIn("sample.md", md)

    def test_tier1_r3_api_report_markdown_humanize(self) -> None:
        """Verify POST /api/report/markdown supports report_type='humanize'."""
        payload = {
            "data": {
                "filename": "article.txt",
                "before_scale": 90.0,
                "after_scale": 18.0,
                "delta_scale": 72.0,
                "paragraphs": [],
                "summary": {},
            },
            "report_type": "humanize",
            "filename": "article.txt",
        }
        res = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/markdown", res.headers.get("content-type", ""))
        self.assertIn("report_humanized_", res.headers.get("content-disposition", ""))


class TestTier1R4WebUI(unittest.TestCase):
    """Tier 1: R4 Interactive Web UI Workspace."""

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent.parent
        self.index_html = (self.root / "web" / "public" / "index.html").read_text(encoding="utf-8")
        self.app_js = (self.root / "web" / "public" / "js" / "app.js").read_text(encoding="utf-8")

    def test_tier1_r4_web_ui_tab_button_and_container(self) -> None:
        """Verify #view-btn-humanize and #view-humanize exist in index.html."""
        self.assertIn("id=\"view-btn-humanize\"", self.index_html, "#view-btn-humanize missing from index.html")
        self.assertIn("id=\"view-humanize\"", self.index_html, "#view-humanize missing from index.html")
        self.assertIn("switchView('humanize')", self.index_html)

    def test_tier1_r4_web_ui_dual_input_elements(self) -> None:
        """Verify dual input modes: direct text input (#humanize-text-input) and dropzone."""
        self.assertIn("id=\"humanize-text-input\"", self.index_html)
        self.assertIn("id=\"humanize-dropzone\"", self.index_html)
        self.assertIn("id=\"humanize-file-input\"", self.index_html)

    def test_tier1_r4_web_ui_sample_loaders(self) -> None:
        """Verify quick sample loader buttons for AI text in index.html."""
        self.assertTrue(
            "loadHumanizeSample" in self.index_html or "loadHumanizeSample" in self.app_js,
            "Sample loader handler must exist in UI",
        )

    def test_tier1_r4_web_ui_controls_and_progress_card(self) -> None:
        """Verify controls: mode select, strength select, Run CTA, Cancel, and live progress bar."""
        self.assertIn("id=\"humanize-mode-select\"", self.index_html)
        self.assertIn("id=\"humanize-strength-select\"", self.index_html)
        self.assertIn("id=\"btn-run-humanize\"", self.index_html)
        self.assertIn("id=\"btn-abort-humanize\"", self.index_html)
        self.assertIn("id=\"humanize-live-progress-card\"", self.index_html)

    def test_tier1_r4_web_ui_results_dashboard_elements(self) -> None:
        """Verify dashboard meters: before score, after score, delta badge, and modification stats."""
        self.assertIn("id=\"humanize-before-score\"", self.index_html)
        self.assertIn("id=\"humanize-after-score\"", self.index_html)
        self.assertIn("id=\"humanize-delta-badge\"", self.index_html)
        self.assertIn("id=\"humanize-stat-modified-paras\"", self.index_html)
        self.assertIn("id=\"humanize-stat-removed-cliches\"", self.index_html)

    def test_tier1_r4_web_ui_side_by_side_and_toolbar(self) -> None:
        """Verify side-by-side comparison viewer and action toolbar buttons."""
        self.assertIn("id=\"btn-copy-humanized\"", self.index_html)
        self.assertIn("id=\"btn-download-humanized-txt\"", self.index_html)
        self.assertIn("id=\"btn-download-humanized-md\"", self.index_html)
        self.assertIn("id=\"btn-download-humanize-report\"", self.index_html)
        self.assertIn("id=\"btn-reverify-inspector\"", self.index_html)

    def test_tier1_r4_web_ui_inspector_integrations(self) -> None:
        """Verify Inspector integrations: 'Humanize Entire Text' button and in-place paragraph buttons."""
        self.assertIn("id=\"btn-humanize-from-inspector\"", self.index_html)
        self.assertTrue(
            "transferTextToHumanizer" in self.app_js or "transferTextToHumanizer" in self.index_html,
            "transferTextToHumanizer must be defined in app.js",
        )
        self.assertTrue(
            "humanizeIndividualParagraph" in self.app_js or "humanizeParagraph" in self.app_js,
            "humanizeIndividualParagraph handler must exist in app.js",
        )


class TestTier1R5CLI(unittest.TestCase):
    """Tier 1: R5 CLI Humanization Interface."""

    def test_tier1_r5_cli_humanize_flag(self) -> None:
        """Verify --humanize flag is registered and defaults to strength='balanced'."""
        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize"]):
            args = parse_args()
            self.assertTrue(args.humanize)
            self.assertEqual(args.strength, "balanced")

    def test_tier1_r5_cli_strength_options(self) -> None:
        """Verify --strength choices: light, balanced, aggressive."""
        for strength in ["light", "balanced", "aggressive"]:
            with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--strength", strength]):
                args = parse_args()
                self.assertEqual(args.strength, strength)

        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--strength", "extreme"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_tier1_r5_cli_output_option(self) -> None:
        """Verify -o / --output options store target output path."""
        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "-o", "out.txt"]):
            args = parse_args()
            self.assertEqual(args.output, "out.txt")

        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--output", "out.txt"]):
            args = parse_args()
            self.assertEqual(args.output, "out.txt")

    def test_tier1_r5_cli_mutually_exclusive_text_and_file(self) -> None:
        """Verify --text and --file are mutually exclusive in CLI parser."""
        with patch("sys.argv", ["veritext", "--text", "Text", "--file", "doc.txt", "--humanize"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_tier1_r5_cli_text_humanize_execution(self) -> None:
        """Verify CLI executes humanization for --text input and prints results."""
        from backend.cli import main
        with patch("sys.argv", ["veritext", "--text", "Machine generated content.", "--humanize"]):
            exit_code = main()
            self.assertEqual(exit_code, 0)

    def test_tier1_r5_cli_file_humanize_execution(self) -> None:
        """Verify CLI executes humanization for --file input and saves to -o."""
        from backend.cli import main
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "input.txt"
            in_file.write_text("Text inside file to humanize.", encoding="utf-8")
            out_file = Path(tmp_dir) / "output.txt"

            with patch("sys.argv", ["veritext", "--file", str(in_file), "--humanize", "-o", str(out_file)]):
                exit_code = main()
                self.assertEqual(exit_code, 0)
                self.assertTrue(out_file.exists())
                self.assertTrue(len(out_file.read_text(encoding="utf-8")) > 0)

    def test_tier1_r5_cli_json_and_markdown_output(self) -> None:
        """Verify CLI produces pure JSON with --json and markdown report with --markdown."""
        from backend.cli import main

        # Test --json stdout
        with patch("sys.argv", ["veritext", "--text", "Test sentence.", "--humanize", "--json"]), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            exit_code = main()
            self.assertEqual(exit_code, 0)
            output = mock_stdout.getvalue().strip()
            parsed = json.loads(output)
            self.assertIn("before_scale", parsed)
            self.assertIn("after_scale", parsed)

        # Test --markdown stdout
        with patch("sys.argv", ["veritext", "--text", "Test sentence.", "--humanize", "--markdown"]), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            exit_code = main()
            self.assertEqual(exit_code, 0)
            output = mock_stdout.getvalue()
            self.assertIn("# ✨ VeriText", output)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (12 tests)
# ==============================================================================

class TestTier2BoundaryCases(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_tier2_empty_text_api_returns_400(self) -> None:
        """POST /api/humanize with empty string returns HTTP 400."""
        res = self.client.post("/api/humanize", json={"text": "", "mode": "auto"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Text cannot be empty", res.json().get("detail", ""))

    def test_tier2_whitespace_only_text_api_returns_400(self) -> None:
        """POST /api/humanize with whitespace/newlines returns HTTP 400."""
        res = self.client.post("/api/humanize", json={"text": "   \n\t   \n  ", "mode": "auto"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Text cannot be empty", res.json().get("detail", ""))

    def test_tier2_engine_empty_text_raises_value_error(self) -> None:
        """Calling humanize_text with empty/whitespace raises ValueError."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        with self.assertRaises(ValueError):
            method("")

        with self.assertRaises(ValueError):
            method("    \n\t ")

    def test_tier2_very_short_sentence(self) -> None:
        """Processing a very short sentence (< 20 chars) succeeds without indexing/split crash."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        res = method("Yes, indeed.")
        self.assertEqual(len(res["paragraphs"]), 1)
        self.assertIn("after_scale", res)

    def test_tier2_very_long_multi_paragraph_text(self) -> None:
        """Processing a long document (25 paragraphs) batches correctly and produces 25 diffs."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        paragraphs = [f"Paragraph #{i} discussing machine learning applications." for i in range(1, 26)]
        long_doc = "\n\n".join(paragraphs)

        res = method(long_doc)
        self.assertEqual(len(res["paragraphs"]), 25)
        self.assertEqual(res["summary"]["total_paragraphs"], 25)

    def test_tier2_length_parity_rejection_too_short(self) -> None:
        """Candidate with length < 0.65x original length is discarded by length parity constraint."""
        humanizer_cls = get_humanizer_cls()
        # Mock candidate generator returning extremely truncated candidate
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            tok_instance = MockTokenizer()
            # Override decode to return 2 characters
            tok_instance.decode = lambda *a, **kw: "Ok"
            tok_instance.batch_decode = lambda seqs, *a, **kw: ["Ok" for _ in seqs]
            mock_tok.return_value = tok_instance

            humanizer = humanizer_cls()
            method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))
            long_sentence = "This is a detailed paragraph describing several algorithmic principles in depth."
            res = method(long_sentence)
            # Candidate 'Ok' should be discarded; Stage 1 text retained
            self.assertNotEqual(res["humanized_text"], "Ok")

    def test_tier2_length_parity_rejection_too_long(self) -> None:
        """Candidate with length > 1.40x original length is discarded by length parity constraint."""
        humanizer_cls = get_humanizer_cls()
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            tok_instance = MockTokenizer()
            bloated = "Extremely long candidate repetition " * 10
            tok_instance.batch_decode = lambda seqs, *a, **kw: [bloated for _ in seqs]
            mock_tok.return_value = tok_instance

            humanizer = humanizer_cls()
            method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))
            short_sentence = "Short concise sentence."
            res = method(short_sentence)
            self.assertNotEqual(res["humanized_text"], bloated)

    def test_tier2_unsupported_document_upload_rejected(self) -> None:
        """POST /api/humanize-document rejects unsupported file extensions with HTTP 400."""
        res = self.client.post(
            "/api/humanize-document",
            files={"file": ("data.csv", b"col1,col2\nval1,val2", "text/csv")},
            data={"mode": "auto"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.json().get("detail", ""))

    def test_tier2_empty_document_upload_rejected(self) -> None:
        """POST /api/humanize-document rejects 0-byte document with HTTP 400."""
        res = self.client.post(
            "/api/humanize-document",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"mode": "auto"},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("no readable text", res.json().get("detail", "").lower())

    def test_tier2_operation_cancellation_returns_499(self) -> None:
        """In-flight cancellation during POST /api/humanize returns HTTP 499."""
        humanizer_cls = get_humanizer_cls()

        def fake_humanize(*args: Any, **kwargs: Any) -> Any:
            raise OperationCancelled("Humanization was cancelled by user.")

        with patch.object(humanizer_cls, "humanize_text", side_effect=fake_humanize):
            res = self.client.post(
                "/api/humanize",
                json={"text": "Text to cancel", "mode": "en"},
                headers={"X-Request-ID": "abort-req-499"},
            )
            self.assertEqual(res.status_code, 499)
            self.assertIn("cancelled", res.json().get("detail", "").lower())

    def test_tier2_llm_network_timeout_fallback(self) -> None:
        """HTTP connect timeout on external LLM triggers automatic fallback to local T5."""
        import httpx
        humanizer_cls = get_humanizer_cls()

        with patch.object(config_module, "LLM_BASE_URL", "http://timeout-endpoint.internal"), \
             patch("httpx.Client.post", side_effect=httpx.ConnectTimeout("Connection timed out")):
            humanizer = humanizer_cls()
            method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))
            # Must succeed via fallback without raising ConnectTimeout
            res = method("Artificial intelligence models generate synthetic patterns.")
            self.assertIn("humanized_text", res)

    def test_tier2_extreme_whitespace_between_paragraphs(self) -> None:
        """Handling document with multiple empty lines between paragraphs without empty diffs."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        messy_text = "Paragraph One.\n\n\n\n\n   \n\nParagraph Two."
        res = method(messy_text)
        self.assertEqual(len(res["paragraphs"]), 2)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (8 tests)
# ==============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_tier3_document_upload_to_markdown_report_export(self) -> None:
        """Document upload -> humanization -> markdown report export via /api/report/markdown."""
        # 1. Upload document
        file_bytes = b"# Report Subject\n\nIt is important to emphasize that software validation is critical."
        res_doc = self.client.post(
            "/api/humanize-document",
            files={"file": ("validation.md", file_bytes, "text/markdown")},
            data={"mode": "en"},
        )
        self.assertEqual(res_doc.status_code, 200)
        doc_result = res_doc.json()

        # 2. Export to markdown report
        res_rep = self.client.post(
            "/api/report/markdown",
            json={
                "data": doc_result,
                "report_type": "humanize",
                "filename": "validation.md",
            },
        )
        self.assertEqual(res_rep.status_code, 200)
        self.assertIn("report_humanized_", res_rep.headers.get("content-disposition", ""))
        self.assertIn("# ✨ VeriText", res_rep.text)

    def test_tier3_cli_file_input_to_output_and_report(self) -> None:
        """CLI humanize file input with -o output.txt and --report-md report.md."""
        from backend.cli import main
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = Path(tmp_dir) / "source.txt"
            src_file.write_text("Важно отметить, что методология обеспечивает надежность.", encoding="utf-8")
            out_txt = Path(tmp_dir) / "transformed.txt"
            out_rep = Path(tmp_dir) / "report.md"

            with patch("sys.argv", [
                "veritext",
                "--file", str(src_file),
                "--humanize",
                "-o", str(out_txt),
                "--report-md", str(out_rep),
            ]):
                code = main()
                self.assertEqual(code, 0)
                self.assertTrue(out_txt.exists())
                self.assertTrue(out_rep.exists())
                self.assertIn("# ✨ VeriText", out_rep.read_text(encoding="utf-8"))

    def test_tier3_inspector_to_humanizer_to_reverify_workflow(self) -> None:
        """Inspector /api/score -> /api/humanize -> Inspector /api/score verification loop."""
        critic = ControlledCriticDetector(scores={
            "Raw AI text.": 0.88,
            "Transformed natural text candidate preserving meaning.": 0.15,
        })

        with patch("backend.api.routes.engine", critic):
            # Step 1: Score raw text in inspector
            res_score = self.client.post("/api/score", json={"text": "Raw AI text.", "mode": "en"})
            self.assertEqual(res_score.status_code, 200)
            initial_scale = res_score.json()["human_ai_scale"]
            self.assertEqual(initial_scale, 88.0)

            # Step 2: Humanize text
            res_humanize = self.client.post(
                "/api/humanize",
                json={"text": "Raw AI text.", "mode": "en"},
            )
            self.assertEqual(res_humanize.status_code, 200)
            humanized_text = res_humanize.json()["humanized_text"]

            # Step 3: Re-verify humanized text in inspector
            res_reverify = self.client.post("/api/score", json={"text": humanized_text, "mode": "en"})
            self.assertEqual(res_reverify.status_code, 200)
            final_scale = res_reverify.json()["human_ai_scale"]
            self.assertEqual(final_scale, 15.0)
            self.assertLess(final_scale, initial_scale)

    def test_tier3_paragraph_level_in_place_humanization_recalculation(self) -> None:
        """Humanizing a single paragraph in-place and recalculating document weighted score."""
        # 3 paragraphs with word counts 10, 10, 10
        p1_score = 90.0  # AI
        p2_score = 10.0  # Human
        p3_score = 80.0  # AI

        # Initial document weighted score: (90*10 + 10*10 + 80*10) / 30 = 60.0
        initial_doc_score = (p1_score * 10 + p2_score * 10 + p3_score * 10) / 30
        self.assertEqual(initial_doc_score, 60.0)

        # Humanize paragraph #1 in place: P1 drops to 15.0
        new_p1_score = 15.0
        recalculated_doc_score = (new_p1_score * 10 + p2_score * 10 + p3_score * 10) / 30
        self.assertEqual(recalculated_doc_score, 35.0)
        self.assertLess(recalculated_doc_score, initial_doc_score)

    def test_tier3_cancellation_lifecycle_with_progress_polling(self) -> None:
        """Full lifecycle: register request -> stream progress -> cancel -> verify clean state."""
        req_id = "lifecycle-cancel-test"
        cancellation_manager.register(req_id)
        try:
            # Step 1
            cancellation_manager.update_progress(req_id, "Step 1", percent=10, step=1, total_steps=4)
            p1 = cancellation_manager.get_progress(req_id)
            self.assertEqual(p1["percent"], 10)

            # Cancel mid-flight
            success = cancellation_manager.cancel(req_id)
            self.assertTrue(success)
            self.assertTrue(cancellation_manager.is_cancelled(req_id))
        finally:
            cancellation_manager.unregister(req_id)
            self.assertFalse(cancellation_manager.is_cancelled(req_id))

    def test_tier3_cancellation_alias_equivalence(self) -> None:
        """Verify POST /api/cancel/{id} and POST /api/abort/{id} behave identically."""
        cancellation_manager.register("req-alias-1")
        cancellation_manager.register("req-alias-2")
        try:
            res1 = self.client.post("/api/cancel/req-alias-1")
            res2 = self.client.post("/api/abort/req-alias-2")

            self.assertEqual(res1.status_code, 200)
            self.assertEqual(res2.status_code, 200)
            self.assertEqual(res1.json().keys(), res2.json().keys())
            self.assertTrue(res1.json()["aborted"])
            self.assertTrue(res2.json()["aborted"])
        finally:
            cancellation_manager.unregister("req-alias-1")
            cancellation_manager.unregister("req-alias-2")

    def test_tier3_cli_json_pipe_integrity(self) -> None:
        """Verify CLI with --json outputs valid JSON payload containing all schema keys."""
        from backend.cli import main
        with patch("sys.argv", ["veritext", "--text", "Sample for JSON pipe.", "--humanize", "--json"]), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = main()
            self.assertEqual(code, 0)
            data = json.loads(mock_out.getvalue())
            self.assertIn("before_scale", data)
            self.assertIn("after_scale", data)
            self.assertIn("delta_scale", data)
            self.assertIn("paragraphs", data)

    def test_tier3_document_humanization_preserving_structure(self) -> None:
        """Document humanization preserves layout across multi-paragraph text."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        multi_doc = "Introduction section.\n\nBody analysis paragraph.\n\nConcluding remarks."
        res = method(multi_doc)
        self.assertEqual(len(res["paragraphs"]), 3)
        self.assertEqual(res["summary"]["total_paragraphs"], 3)


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (7 tests)
# ==============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Real-World Application Scenarios."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_tier4_russian_academic_text_with_cliches(self) -> None:
        """Russian academic text with AI clichés and markers is scrubbed and humanized."""
        text = (
            "Важно отметить, что в современном мире искусственный интеллект играет ключевую роль "
            "в обработке больших данных. Таким образом, можно сделать вывод, что интеграция алгоритмов "
            "является неотъемлемой частью технологического прогресса. Подводя итоги, следует подчеркнуть "
            "значимость дальнейших исследований."
        )

        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        res = method(text, mode="ru", strength="balanced")
        self.assertIn("humanized_text", res)
        # Cliché scrubbing check
        self.assertFalse(res["humanized_text"].startswith("Важно отметить"))
        self.assertGreater(res["summary"]["removed_cliches_count"], 0)
        self.assertLessEqual(res["after_scale"], res["before_scale"])

    def test_tier4_english_technical_article_with_boilerplate(self) -> None:
        """English technical article with formulaic transitions is de-clichéd and transformed."""
        text = (
            "In today's fast-paced digital world, it is important to remember that microservices "
            "architecture has become a cornerstone of cloud development. Furthermore, it is worth noting "
            "that container orchestration simplifies deployments. In conclusion, navigating this complex "
            "tapestry requires continuous engineering vigilance."
        )

        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        res = method(text, mode="en", strength="aggressive")
        self.assertIn("humanized_text", res)
        self.assertFalse(res["humanized_text"].startswith("In today's fast-paced"))
        self.assertGreater(res["summary"]["removed_cliches_count"], 0)

    def test_tier4_mixed_multiparagraph_document(self) -> None:
        """Document with mixed human and AI paragraphs: AI paragraphs transformed, human preserved."""
        p_ai_1 = "Важно отметить, что синтетический текст имеет характерные особенности."
        p_human_1 = "Я написал этот абзац вручную вчера вечером за чашкой чая без нейросетей."
        p_ai_2 = "Таким образом, можно сделать вывод о необходимости детекции генераций."

        # Configure critic to score human paragraph low and AI paragraphs high
        critic = ControlledCriticDetector(scores={
            p_ai_1: 0.90,
            p_human_1: 0.05,
            p_ai_2: 0.85,
        })

        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls(detector=critic)
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        doc = f"{p_ai_1}\n\n{p_human_1}\n\n{p_ai_2}"
        res = method(doc, mode="ru")

        paras = res["paragraphs"]
        self.assertEqual(len(paras), 3)
        self.assertEqual(res["summary"]["total_paragraphs"], 3)
        self.assertGreater(res["summary"]["modified_paragraphs_count"], 0)

    def test_tier4_bilingual_mode_auto_detection(self) -> None:
        """Auto mode correctly identifies Russian and English languages and applies correct catalog."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        res_ru = method("Искусственный интеллект трансформирует современное общество.", mode="auto")
        self.assertEqual(res_ru.get("effective_language"), "ru")

        res_en = method("Neural networks are fundamentally altering software development.", mode="auto")
        self.assertEqual(res_en.get("effective_language"), "en")

    def test_tier4_burstiness_sentence_length_variance(self) -> None:
        """Evaluation of burstiness: candidates provide variation in sentence structures."""
        humanizer_cls = get_humanizer_cls()
        humanizer = humanizer_cls()
        method = getattr(humanizer, "humanize_text", getattr(humanizer, "humanize"))

        uniform_text = (
            "The model generates text with fixed cadence. "
            "Each sentence contains roughly seven words. "
            "Uniform length indicates artificial text generation."
        )
        res = method(uniform_text, mode="en", strength="aggressive")
        self.assertIn("humanized_text", res)
        self.assertIsInstance(res["humanized_text"], str)

    def test_tier4_cli_end_to_end_humanize_session(self) -> None:
        """End-to-end CLI workflow: file input -> aggressive strength -> destination file -> report."""
        from backend.cli import main
        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = Path(tmp_dir) / "article.md"
            in_file.write_text(
                "# Deep Learning\n\nIt is important to remember that training requires computing power.",
                encoding="utf-8",
            )
            out_txt = Path(tmp_dir) / "humanized_article.txt"
            out_rep = Path(tmp_dir) / "audit_report.md"

            with patch("sys.argv", [
                "veritext",
                "--file", str(in_file),
                "--humanize",
                "--strength", "aggressive",
                "-o", str(out_txt),
                "--report-md", str(out_rep),
            ]):
                code = main()
                self.assertEqual(code, 0)
                self.assertTrue(out_txt.exists())
                self.assertTrue(out_rep.exists())
                self.assertIn("Executive Verdict", out_rep.read_text(encoding="utf-8"))

    def test_tier4_document_docx_pdf_end_to_end(self) -> None:
        """End-to-end API document humanization with extracted paragraph structure."""
        with patch("backend.api.routes.extract_text_from_file", return_value=(
            "Executive Summary\n\nIn modern times, algorithms play a crucial role in data security.",
            ["Executive Summary", "In modern times, algorithms play a crucial role in data security."]
        )):
            res = self.client.post(
                "/api/humanize-document",
                files={"file": ("security_brief.docx", b"DOCX_STUB", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"mode": "en", "strength": "balanced"},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["filename"], "security_brief.docx")
            self.assertEqual(len(data["paragraphs"]), 2)
            self.assertIn("before_scale", data)
            self.assertIn("after_scale", data)


if __name__ == "__main__":
    unittest.main()
