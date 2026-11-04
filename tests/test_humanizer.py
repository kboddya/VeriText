"""Unit tests for ModernTextHumanizer and de-cliché scrubbing engine."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Optional

import torch

from backend.cancellation import OperationCancelled
import backend.config as config
from backend.humanizer import ModernTextHumanizer, split_sentences, split_paragraphs
from backend.markers import scrub_cliches, detect_stylistic_markers


class MockTokenizer:
    """Fast in-memory mock tokenizer."""

    def __call__(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        n = len(text) if isinstance(text, (list, tuple)) else 1
        input_ids = torch.ones((n, 8), dtype=torch.long)
        attention_mask = torch.ones((n, 8), dtype=torch.long)

        class MockBatchEncoding(dict):
            def __init__(self, d: dict[str, Any]) -> None:
                super().__init__(d)
                self.__dict__.update(d)

            def to(self, device: Any) -> MockBatchEncoding:
                return self

        return MockBatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def decode(self, token_ids: Any, *args: Any, **kwargs: Any) -> str:
        return "Humanized candidate phrase preserving semantics."

    def batch_decode(self, sequences: Any, *args: Any, **kwargs: Any) -> list[str]:
        candidates = [
            "Natural phrasing emphasizing authentic expression and clear ideas.",
            "Refined alternative text with varied rhythm and vocabulary.",
            "Distinctly rewritten candidate eliminating formulaic boilerplate.",
            "Authentic human-like variation with natural stylistic cadence.",
        ]
        return [candidates[i % len(candidates)] for i, _ in enumerate(sequences)]


class MockSeq2SeqModel:
    """Fast in-memory mock Seq2Seq model."""

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
        return torch.ones((b * num_return_sequences, 8), dtype=torch.long)


class MockCriticDetector:
    """Deterministic mock detector critic."""

    def __init__(self, scores: Optional[dict[str, float]] = None, default_score: float = 0.25) -> None:
        self.scores = scores or {}
        self.default_score = default_score
        self.device_str = "cpu"
        self.device = torch.device("cpu")

    def score_single(
        self,
        text: str,
        mode: str = "auto",
        is_cancelled: Optional[Any] = None,
    ) -> dict[str, Any]:
        if is_cancelled and is_cancelled():
            raise OperationCancelled("Cancelled by user.")
        score = self.scores.get(text, self.default_score)
        return {
            "label": "ai" if score >= 0.50 else "human",
            "score": round((score - 0.5) * 10, 4),
            "human_ai_scale": round(score * 100, 2),
            "ai_probability": score,
            "model_used": "MockCriticDetector",
        }

    def score_batch(
        self,
        texts: list[str],
        mode: str = "auto",
        batch_size: int = 16,
        is_cancelled: Optional[Any] = None,
        on_progress: Optional[Any] = None,
        on_step: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        if is_cancelled and is_cancelled():
            raise OperationCancelled("Cancelled by user.")
        results = []
        for text in texts:
            results.append(self.score_single(text, mode=mode, is_cancelled=is_cancelled))
        return results


class TestR2ModernTextHumanizer(unittest.TestCase):
    """Unit test suite for R2 ModernTextHumanizer and Stage 1-3 pipeline."""

    def setUp(self) -> None:
        self.tok_patcher = patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer())
        self.model_patcher = patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=MockSeq2SeqModel())
        self.cache_sync_patcher = patch("backend.cache.should_check_remote", return_value=False)

        def default_llm_cands(text: str, num_candidates: int, temperature: float) -> list[str]:
            return [
                "Natural humanized phrasing with varied syntax.",
                "Alternative authentic expression preserving core ideas.",
            ][:num_candidates]

        self.llm_patcher = patch.object(ModernTextHumanizer, "_generate_llm_candidates", side_effect=default_llm_cands)

        self.mock_tok = self.tok_patcher.start()
        self.mock_model = self.model_patcher.start()
        self.mock_cache_sync = self.cache_sync_patcher.start()
        self.mock_llm = self.llm_patcher.start()

    def tearDown(self) -> None:
        self.llm_patcher.stop()
        self.cache_sync_patcher.stop()
        self.model_patcher.stop()
        self.tok_patcher.stop()

    def test_stage1_decliche_scrubbing_ru(self) -> None:
        """Stage 1 strips Russian preamble boilerplate and normalizes casing."""
        raw = "Важно отметить, что современные технологии развиваются ускоренными темпами."
        cleaned, removed = scrub_cliches(raw, lang="ru")

        self.assertNotEqual(raw, cleaned)
        self.assertFalse(cleaned.lower().startswith("важно отметить"))
        self.assertTrue(cleaned[0].isupper(), f"Sentence must begin with capital letter: '{cleaned}'")
        self.assertGreater(len(removed), 0)
        self.assertEqual(removed[0]["phrase"], "Важно отметить, что")

    def test_stage1_decliche_scrubbing_en(self) -> None:
        """Stage 1 strips English boilerplate preamble and restores sentence casing."""
        raw = "It is important to remember that deep neural networks require substantial computing power."
        cleaned, removed = scrub_cliches(raw, lang="en")

        self.assertNotEqual(raw, cleaned)
        self.assertFalse(cleaned.lower().startswith("it is important to remember"))
        self.assertTrue(cleaned[0].isupper(), f"Sentence must begin with capital letter: '{cleaned}'")
        self.assertGreater(len(removed), 0)

    def test_stage1_mid_sentence_replacements(self) -> None:
        """Stage 1 replaces overused synthetic clichés with natural synonyms."""
        raw_ru = "Алгоритм играет ключевую роль и является неотъемлемой частью системы."
        cleaned_ru, removed_ru = scrub_cliches(raw_ru, lang="ru")
        self.assertNotIn("играет ключевую роль", cleaned_ru)
        self.assertNotIn("является неотъемлемой частью", cleaned_ru)
        self.assertEqual(len(removed_ru), 2)

        raw_en = "This framework is a testament to the complex tapestry of modern software."
        cleaned_en, removed_en = scrub_cliches(raw_en, lang="en")
        self.assertNotIn("a testament to", cleaned_en)
        self.assertNotIn("complex tapestry", cleaned_en)
        self.assertEqual(len(removed_en), 2)

    def test_sentence_splitter_abbreviations(self) -> None:
        """Sentence splitter protects known abbreviations without splitting mid-sentence."""
        text_en = "Dr. Smith and Mr. Jones arrived at 8 a.m. e.g. for the conference. They presented findings."
        sentences_en = split_sentences(text_en)
        self.assertEqual(len(sentences_en), 2)
        self.assertIn("Dr. Smith and Mr. Jones", sentences_en[0])
        self.assertIn("They presented findings.", sentences_en[1])

        text_ru = "В 2024 г. начался эксперимент т.е. первый этап. Результаты подтвердились."
        sentences_ru = split_sentences(text_ru)
        self.assertEqual(len(sentences_ru), 2)
        self.assertIn("2024 г.", sentences_ru[0])
        self.assertIn("т.е.", sentences_ru[0])

    def test_stage2_candidate_generation_and_burstiness(self) -> None:
        """Stage 2 generates candidate paraphrases and returns required response fields."""
        critic = MockCriticDetector(default_score=0.20)
        humanizer = ModernTextHumanizer(detector=critic)

        text = "Synthetic text generation has seen unprecedented acceleration in recent years."
        res = humanizer.humanize_text(text, mode="en", strength="aggressive")

        self.assertIn("humanized_text", res)
        self.assertIn("before_scale", res)
        self.assertIn("after_scale", res)
        self.assertIn("delta_scale", res)
        self.assertIn("summary", res)
        self.assertIn("paragraphs", res)
        self.assertEqual(len(res["paragraphs"]), 1)

    def test_stage2_length_parity_filtering(self) -> None:
        """Stage 2 discards candidates that violate length parity bounds [0.65, 1.40]."""
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_fn:
            tok_inst = MockTokenizer()
            # Return candidate with 3 characters (violates 0.65x for a 70 char sentence)
            tok_inst.batch_decode = lambda seqs, *a, **kw: ["Too short" for _ in seqs]
            mock_tok_fn.return_value = tok_inst

            critic = MockCriticDetector(default_score=0.10)
            humanizer = ModernTextHumanizer(detector=critic)
            input_text = "This is a moderately long sentence intended to test length parity rejection bounds."
            res = humanizer.humanize_text(input_text, mode="en")

            # "Too short" should be rejected; Stage 1 text retained
            self.assertNotEqual(res["humanized_text"], "Too short")

    def test_stage3_monotonic_improvement_guarantee(self) -> None:
        """Stage 3 detector critic mathematically guarantees non-regressive AI score."""
        orig_text = "Важно отметить, что алгоритмы работают стабильно."
        stage1_text, _ = scrub_cliches(orig_text, lang="ru")

        # Critic scores: original=0.85, Stage 1=0.40, all neural candidates=0.95
        critic = MockCriticDetector(
            scores={
                orig_text: 0.85,
                stage1_text: 0.40,
            },
            default_score=0.95,
        )
        humanizer = ModernTextHumanizer(detector=critic)
        res = humanizer.humanize_text(orig_text, mode="ru")

        self.assertLessEqual(res["after_scale"], res["before_scale"])
        self.assertLessEqual(res["after_scale"], 40.0)

    def test_llm_fallback_on_network_error(self) -> None:
        """External LLM failure transparently falls back to local T5 models."""
        import httpx

        with patch.object(config, "LLM_BASE_URL", "http://unreachable-llm.domain:8080"), \
             patch("httpx.Client.post", side_effect=httpx.ConnectError("Failed to connect")):
            critic = MockCriticDetector(default_score=0.15)
            humanizer = ModernTextHumanizer(detector=critic)

            # Execution must succeed via fallback without raising ConnectError
            res = humanizer.humanize_text("Machine learning models transform information processing.")
            self.assertIn("humanized_text", res)
            self.assertIsInstance(res["humanized_text"], str)

    def test_operation_cancellation(self) -> None:
        """OperationCancelled is raised immediately when is_cancelled() returns True."""
        critic = MockCriticDetector()
        humanizer = ModernTextHumanizer(detector=critic)

        with self.assertRaises(OperationCancelled):
            humanizer.humanize_text(
                "Test paragraph for cancellation.",
                is_cancelled=lambda: True,
            )

    def test_empty_or_whitespace_input_raises_value_error(self) -> None:
        """Calling humanize_text with empty or whitespace-only string raises ValueError."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector())

        with self.assertRaises(ValueError):
            humanizer.humanize_text("")

        with self.assertRaises(ValueError):
            humanizer.humanize_text("   \n\t   ")

    def test_multi_paragraph_structure_preservation(self) -> None:
        """Multi-paragraph document preserves paragraph count and layout."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector())
        doc = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."

        res = humanizer.humanize_text(doc)
        self.assertEqual(len(res["paragraphs"]), 3)
        self.assertEqual(res["summary"]["total_paragraphs"], 3)
        self.assertEqual(res["paragraphs"][0]["index"], 1)
        self.assertEqual(res["paragraphs"][1]["index"], 2)
        self.assertEqual(res["paragraphs"][2]["index"], 3)

    def test_bilingual_mode_auto_detection(self) -> None:
        """Auto mode correctly identifies Russian and English text."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector())

        res_ru = humanizer.humanize_text("Экспериментальные данные демонстрируют устойчивость.", mode="auto")
        self.assertEqual(res_ru["effective_language"], "ru")

        res_en = humanizer.humanize_text("Experimental benchmarks demonstrate robust reliability.", mode="auto")
        self.assertEqual(res_en["effective_language"], "en")

    def test_markdown_code_block_strictly_preserved(self) -> None:
        """Markdown code blocks (fenced with ```) are 100% preserved byte-for-byte without humanization."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector(default_score=0.85))
        code_block = (
            "```python\n"
            "# Важно отметить, что данный комментарий в коде не должен изменяться\n"
            "def calculate_metrics(data: list) -> dict:\n"
            "    # В современном мире это играет ключевую роль\n"
            "\n"
            "    total = sum(data)\n"
            "    return {'total': total, 'count': len(data)}\n"
            "```"
        )
        md_text = f"Важно отметить, что вводная часть статьи требует внимания.\n\n{code_block}\n\nВ заключение стоит подчеркнуть итоги исследования."

        res = humanizer.humanize_text(md_text, mode="ru")
        self.assertEqual(len(res["paragraphs"]), 3)
        code_p = res["paragraphs"][1]

        self.assertTrue(code_p.get("is_code"), "Code block must be marked as is_code=True")
        self.assertEqual(code_p["stage_applied"], "code")
        self.assertEqual(code_p["humanized_text"], code_block, "Code block must be strictly identical byte-for-byte")
        self.assertIn(code_block, res["humanized_text"], "Full humanized text must contain exact original code block")

    def test_inline_code_in_prose_preserved(self) -> None:
        """Inline code (`...`) inside regular paragraphs is protected from alteration and translation."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector(default_score=0.20))
        text = "Важно отметить, что функция `ModernTextHumanizer.humanize_text()` играет ключевую роль в пайплайне."

        res = humanizer.humanize_text(text, mode="ru")
        self.assertIn("`ModernTextHumanizer.humanize_text()`", res["humanized_text"])

    def test_markdown_table_preserved(self) -> None:
        """Markdown tables are recognized as structured content and preserved intact."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector())
        table = (
            "| Feature | Status | Description |\n"
            "| :--- | :---: | :--- |\n"
            "| Detector | Active | Evaluates continuous AI probability |\n"
            "| Humanizer | Active | Refines and diversifies cadence |"
        )
        text = f"Table overview:\n\n{table}\n\nConcluding summary."
        res = humanizer.humanize_text(text, mode="en")
        self.assertEqual(len(res["paragraphs"]), 3)
        self.assertTrue(res["paragraphs"][1].get("is_code"))
        self.assertEqual(res["paragraphs"][1]["humanized_text"], table)

    def test_external_llm_active_bypasses_local_t5_completely(self) -> None:
        """When an external LLM is connected and responding, local T5 model is NEVER loaded or called."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector(default_score=0.85))

        # Mock external LLM probe to report active Ollama
        mock_llm_info = {
            "configured": True,
            "active": True,
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "gemma2:9b",
            "provider": "Ollama (gemma2:9b)",
        }

        # Mock LLM candidate generator
        mock_llm_candidates = [
            "Естественный перефразированный вариант текста с живым синтаксисом.",
            "Более прямое и понятное изложение основной идеи статьи.",
        ]

        with patch.object(humanizer, "get_llm_info", return_value=mock_llm_info), \
             patch.object(humanizer, "_generate_llm_candidates", return_value=mock_llm_candidates) as mock_gen_llm, \
             patch.object(humanizer, "_generate_local_t5_candidates") as mock_gen_t5, \
             patch.object(humanizer, "_load_paraphraser") as mock_load_t5:

            text = "Важно отметить, что данный текст должен быть очеловечен без загрузки локальных весов в оперативную память."
            res = humanizer.humanize_text(text, mode="ru")

            # Verify external LLM was called
            self.assertTrue(mock_gen_llm.called)

            # Strict requirement: local T5 must NEVER be loaded into memory or called
            mock_gen_t5.assert_not_called()
            mock_load_t5.assert_not_called()
            self.assertEqual(len(humanizer._paraphrasers), 0)

            # Metadata verification
            self.assertTrue(res["external_llm_active"])
            self.assertFalse(res["local_model_loaded"])
            self.assertEqual(res["generator_model"], "Ollama (gemma2:9b)")
            self.assertEqual(res["summary"]["generator_provider"], "Ollama (gemma2:9b)")
            self.assertTrue(res["summary"]["external_llm_active"])
            self.assertFalse(res["summary"]["local_model_loaded"])

    def test_external_llm_active_empty_candidates_still_does_not_call_local_t5(self) -> None:
        """Even if external LLM returns no candidates, local T5 is strictly not loaded or downloaded when LLM is active."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector(default_score=0.80))

        mock_llm_info = {
            "configured": True,
            "active": True,
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "gemma2:9b",
            "provider": "Ollama (gemma2:9b)",
        }

        with patch.object(humanizer, "get_llm_info", return_value=mock_llm_info), \
             patch.object(humanizer, "_generate_llm_candidates", return_value=[]), \
             patch.object(humanizer, "_generate_local_t5_candidates") as mock_gen_t5, \
             patch.object(humanizer, "_load_paraphraser") as mock_load_t5:

            text = "Проверяем строгое отсутствие fallback на скачивание или подгрузку локальной модели."
            res = humanizer.humanize_text(text, mode="ru")

            mock_gen_t5.assert_not_called()
            mock_load_t5.assert_not_called()
            self.assertFalse(res["local_model_loaded"])

    def test_paragraph_progress_reporting(self) -> None:
        """Verify on_progress callback receives live paragraph updates (current and total)."""
        humanizer = ModernTextHumanizer(detector=MockCriticDetector(default_score=0.80))

        progress_calls: list[tuple[int, int, str, Optional[int], Optional[int]]] = []

        def mock_progress(pct: int, step: int = 1, msg: str = "", cur_p: Optional[int] = None, tot_p: Optional[int] = None) -> None:
            progress_calls.append((pct, step, msg, cur_p, tot_p))

        text = (
            "Первый концептуальный абзац для проверки статуса очеловечивания.\n\n"
            "Второй исследовательский абзац с важными выводами.\n\n"
            "Третий заключительный абзац документа."
        )

        with patch.object(humanizer, "get_llm_info", return_value={"configured": True, "active": True, "provider": "Mock LLM", "model": "mock"}), \
             patch.object(humanizer, "_generate_llm_candidates", return_value=["Улучшенный абзац с живой речью."]):
            res = humanizer.humanize_text(text, mode="ru", on_progress=mock_progress)

        self.assertEqual(len(res["paragraphs"]), 3)
        self.assertGreater(len(progress_calls), 3)

        # Check that paragraph numbers were reported
        cur_paras = [c[3] for c in progress_calls if c[3] is not None]
        tot_paras = [c[4] for c in progress_calls if c[4] is not None]

        self.assertIn(1, cur_paras)
        self.assertIn(2, cur_paras)
        self.assertIn(3, cur_paras)
        self.assertTrue(all(tp == 3 for tp in tot_paras))


if __name__ == "__main__":
    unittest.main()
