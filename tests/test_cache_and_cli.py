from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

import backend.cache as cache_module
from backend.cache import (
    CHECK_INTERVAL_SECONDS,
    get_last_check,
    record_check,
    should_check_remote,
    load_offline_first,
)
from backend.cli import parse_args


class TestCacheAndCLI(unittest.TestCase):
    """Test suite for offline-first caching, 1-week sync policy, and CLI flags."""

    def test_record_and_get_last_check(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            test_sync_file = Path(tmp_dir) / "test_sync.json"
            with patch.object(cache_module, "SYNC_FILE", test_sync_file), patch.object(
                cache_module, "CACHE_DIR", Path(tmp_dir)
            ):
                # Initially empty
                self.assertIsNone(get_last_check("model-test-id"))
                self.assertTrue(should_check_remote("model-test-id"))

                # Record now
                now = time.time()
                record_check("model-test-id", timestamp=now)
                self.assertEqual(get_last_check("model-test-id"), now)
                self.assertFalse(should_check_remote("model-test-id"))

                # Past 8 days ago -> should_check_remote is True
                eight_days_ago = now - (CHECK_INTERVAL_SECONDS + 100)
                record_check("model-test-id", timestamp=eight_days_ago)
                self.assertTrue(should_check_remote("model-test-id"))

    def test_load_offline_first_uses_local_files_only_first(self) -> None:
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        steps_recorded = []

        def on_step(msg: str) -> None:
            steps_recorded.append(msg)

        with patch("backend.cache.AutoTokenizer.from_pretrained") as mock_tok, patch(
            "backend.cache.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_mdl, patch("backend.cache.should_check_remote", return_value=False):
            mock_tok.return_value = mock_tokenizer
            mock_mdl.return_value = mock_model

            device = MagicMock()
            m, t, eff_id = load_offline_first("fake/test-model", device, on_step=on_step)

            # Verified local_files_only was set to True
            mock_tok.assert_called_once_with("fake/test-model", local_files_only=True)
            mock_mdl.assert_called_once_with("fake/test-model", local_files_only=True)
            mock_model.to.assert_called_once_with(device)
            mock_model.eval.assert_called_once()
            self.assertEqual(eff_id, "fake/test-model")
            self.assertTrue(any("Подгрузка весов" in s for s in steps_recorded))

    def test_load_offline_first_falls_back_to_remote_when_local_absent(self) -> None:
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        steps_recorded = []

        def mock_tok_side_effect(model_id, local_files_only=False):
            if local_files_only:
                raise OSError("Not found in local cache")
            return mock_tokenizer

        with patch("backend.cache.AutoTokenizer.from_pretrained", side_effect=mock_tok_side_effect) as mock_tok, patch(
            "backend.cache.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model
        ) as mock_mdl, patch("backend.cache.record_check") as mock_rec:
            device = MagicMock()
            m, t, eff_id = load_offline_first(
                "fake/remote-model", device, on_step=lambda s: steps_recorded.append(s)
            )

            # Local attempt failed, so second attempt had local_files_only=False
            self.assertEqual(mock_tok.call_count, 2)
            self.assertEqual(mock_tok.call_args_list[0].kwargs.get("local_files_only"), True)
            self.assertEqual(mock_tok.call_args_list[1].kwargs.get("local_files_only"), False)
            mock_rec.assert_called_once_with("fake/remote-model")
            self.assertTrue(any("Загрузка модели из Hugging Face" in s for s in steps_recorded))

    def test_cli_parse_log_flag(self) -> None:
        with patch("sys.argv", ["veritext", "--text", "Test string", "--log"]):
            args = parse_args()
            self.assertTrue(args.log)
            self.assertEqual(args.text, "Test string")

        with patch("sys.argv", ["veritext", "--text", "Test string"]):
            args = parse_args()
            self.assertFalse(args.log)


if __name__ == "__main__":
    unittest.main()
