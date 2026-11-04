from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from backend.cli import humanize_cli, main, parse_args


class TestCliHumanize(unittest.TestCase):
    """Unit and functional test suite for VeriText CLI Humanization Interface (R5)."""

    def test_parse_args_humanize_defaults(self) -> None:
        """Verify --humanize sets humanize=True and defaults strength='balanced' and output=None."""
        with patch("sys.argv", ["veritext", "--text", "Sample text", "--humanize"]):
            args = parse_args()
            self.assertTrue(args.humanize)
            self.assertEqual(args.strength, "balanced")
            self.assertIsNone(args.output)
            self.assertEqual(args.text, "Sample text")

    def test_parse_args_strength_choices(self) -> None:
        """Verify valid strength choices are parsed and invalid ones raise SystemExit."""
        for strength in ["light", "balanced", "aggressive"]:
            with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--strength", strength]):
                args = parse_args()
                self.assertEqual(args.strength, strength)

        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--strength", "invalid_preset"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_parse_args_output_flags(self) -> None:
        """Verify both -o and --output options store the destination file path."""
        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "-o", "dest.txt"]):
            args = parse_args()
            self.assertEqual(args.output, "dest.txt")

        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--output", "/tmp/dest.txt"]):
            args = parse_args()
            self.assertEqual(args.output, "/tmp/dest.txt")

    def test_parse_args_report_and_markdown_aliases(self) -> None:
        """Verify --report / --report-md and --md / --markdown flag aliases."""
        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--report", "rep.md", "--md"]):
            args = parse_args()
            self.assertEqual(args.report_md, Path("rep.md"))
            self.assertTrue(args.markdown)

        with patch("sys.argv", ["veritext", "--text", "Sample", "--humanize", "--report-md", "rep.md", "--markdown"]):
            args = parse_args()
            self.assertEqual(args.report_md, Path("rep.md"))
            self.assertTrue(args.markdown)

    def test_parse_args_mutually_exclusive_text_and_file(self) -> None:
        """Verify supplying both --text and --file is rejected."""
        with patch("sys.argv", ["veritext", "--text", "Hello", "--file", "doc.txt", "--humanize"]):
            with self.assertRaises(SystemExit):
                parse_args()

    @patch("backend.cli.ModernTextHumanizer")
    def test_humanize_cli_text_execution(self, mock_humanizer_cls: MagicMock) -> None:
        """Verify humanize_cli processes text and renders transformation summary to stdout."""
        mock_instance = MagicMock()
        mock_humanizer_cls.return_value = mock_instance
        mock_instance.humanize_text.return_value = {
            "original_text": "Machine text.",
            "humanized_text": "Human phrasing.",
            "mode": "auto",
            "effective_language": "en",
            "strength": "balanced",
            "before_scale": 85.0,
            "after_scale": 15.0,
            "delta_scale": 70.0,
            "paragraphs": [
                {
                    "index": 1,
                    "original_text": "Machine text.",
                    "humanized_text": "Human phrasing.",
                    "before_scale": 85.0,
                    "after_scale": 15.0,
                    "delta_scale": 70.0,
                    "changed": True,
                }
            ],
            "removed_markers": [{"phrase": "it is important to remember", "count": 1}],
        }

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = humanize_cli(text="Machine text.", mode="en", strength="balanced", show_progress=False)
            self.assertEqual(code, 0)
            output = mock_out.getvalue()
            self.assertIn("✨ VeriText AI Text Humanizer — Transformation Summary", output)
            self.assertIn("Continuous AI Probability", output)
            self.assertIn("Confidence Grade", output)
            self.assertIn("Paragraph-Weighted AI", output)
            self.assertIn("Stylistic Markers", output)
            self.assertIn("Changed Paragraphs", output)
            self.assertIn("Total Words / Chars", output)
            self.assertIn("Humanized Text Output:", output)
            self.assertIn("Human phrasing.", output)

    @patch("backend.cli.ModernTextHumanizer")
    def test_humanize_cli_file_execution(self, mock_humanizer_cls: MagicMock) -> None:
        """Verify humanize_cli ingests text from a file and runs transformation."""
        mock_instance = MagicMock()
        mock_humanizer_cls.return_value = mock_instance
        mock_instance.humanize_text.return_value = {
            "original_text": "File machine content.",
            "humanized_text": "File human content.",
            "mode": "auto",
            "effective_language": "en",
            "strength": "balanced",
            "before_scale": 90.0,
            "after_scale": 20.0,
            "delta_scale": 70.0,
            "paragraphs": [],
            "removed_markers": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_file = Path(tmp_dir) / "input.txt"
            sample_file.write_text("File machine content.", encoding="utf-8")

            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                code = humanize_cli(file_path=sample_file, show_progress=False)
                self.assertEqual(code, 0)
                output = mock_out.getvalue()
                self.assertIn("File human content.", output)

    @patch("backend.cli.ModernTextHumanizer")
    def test_humanize_cli_json_output(self, mock_humanizer_cls: MagicMock) -> None:
        """Verify as_json=True prints strictly valid JSON with required schema fields."""
        mock_instance = MagicMock()
        mock_humanizer_cls.return_value = mock_instance
        mock_instance.humanize_text.return_value = {
            "original_text": "AI sample.",
            "humanized_text": "Transformed sample.",
            "mode": "auto",
            "effective_language": "en",
            "strength": "balanced",
            "before_scale": 88.0,
            "after_scale": 12.0,
            "delta_scale": 76.0,
            "paragraphs": [],
            "removed_markers": [],
        }

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = humanize_cli(text="AI sample.", as_json=True, show_progress=False)
            self.assertEqual(code, 0)
            data = json.loads(mock_out.getvalue().strip())
            self.assertIn("before_scale", data)
            self.assertIn("after_scale", data)
            self.assertIn("delta_scale", data)
            self.assertIn("paragraphs", data)
            self.assertIn("summary", data)
            self.assertEqual(data["humanized_text"], "Transformed sample.")

    @patch("backend.cli.ModernTextHumanizer")
    def test_humanize_cli_markdown_output(self, mock_humanizer_cls: MagicMock) -> None:
        """Verify as_markdown=True prints Markdown report to stdout."""
        mock_instance = MagicMock()
        mock_humanizer_cls.return_value = mock_instance
        mock_instance.humanize_text.return_value = {
            "original_text": "AI text.",
            "humanized_text": "Human text.",
            "mode": "en",
            "effective_language": "en",
            "strength": "balanced",
            "before_scale": 80.0,
            "after_scale": 20.0,
            "delta_scale": 60.0,
            "paragraphs": [],
            "removed_markers": [],
        }

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            code = humanize_cli(text="AI text.", as_markdown=True, show_progress=False)
            self.assertEqual(code, 0)
            output = mock_out.getvalue()
            self.assertIn("# ✨ VeriText", output)
            self.assertIn("Executive Verdict", output)

    @patch("backend.cli.ModernTextHumanizer")
    def test_humanize_cli_output_file_saving(self, mock_humanizer_cls: MagicMock) -> None:
        """Verify humanize_cli writes transformed text to output_path and creates parent dirs."""
        mock_instance = MagicMock()
        mock_humanizer_cls.return_value = mock_instance
        mock_instance.humanize_text.return_value = {
            "original_text": "Original text.",
            "humanized_text": "Saved humanized text.",
            "mode": "auto",
            "effective_language": "en",
            "strength": "balanced",
            "before_scale": 75.0,
            "after_scale": 25.0,
            "delta_scale": 50.0,
            "paragraphs": [],
            "removed_markers": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_output = Path(tmp_dir) / "nested" / "folder" / "result.txt"
            code = humanize_cli(text="Original text.", output_path=nested_output, show_progress=False)
            self.assertEqual(code, 0)
            self.assertTrue(nested_output.exists())
            self.assertEqual(nested_output.read_text(encoding="utf-8"), "Saved humanized text.")

    @patch("backend.cli.ModernTextHumanizer")
    def test_humanize_cli_report_export(self, mock_humanizer_cls: MagicMock) -> None:
        """Verify humanize_cli generates and exports Markdown report to report_path."""
        mock_instance = MagicMock()
        mock_humanizer_cls.return_value = mock_instance
        mock_instance.humanize_text.return_value = {
            "original_text": "Document content.",
            "humanized_text": "Refined content.",
            "mode": "ru",
            "effective_language": "ru",
            "strength": "aggressive",
            "before_scale": 95.0,
            "after_scale": 10.0,
            "delta_scale": 85.0,
            "paragraphs": [],
            "removed_markers": [],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_file = Path(tmp_dir) / "humanize_audit.md"
            code = humanize_cli(text="Document content.", report_path=report_file, show_progress=False)
            self.assertEqual(code, 0)
            self.assertTrue(report_file.exists())
            report_text = report_file.read_text(encoding="utf-8")
            self.assertIn("# ✨ VeriText", report_text)
            self.assertIn("Executive Verdict", report_text)

    def test_humanize_cli_missing_input_returns_error(self) -> None:
        """Verify humanize_cli returns 1 when neither text nor file_path is provided."""
        code = humanize_cli(text=None, file_path=None, show_progress=False)
        self.assertEqual(code, 1)

    @patch("backend.cli.humanize_cli")
    def test_main_routes_to_humanize_cli(self, mock_humanize_cli: MagicMock) -> None:
        """Verify main() routes execution to humanize_cli when args.humanize is True."""
        mock_humanize_cli.return_value = 0
        with patch("sys.argv", [
            "veritext",
            "--text", "Test dispatch",
            "--humanize",
            "--strength", "aggressive",
            "-o", "out.txt",
            "--json",
        ]):
            code = main()
            self.assertEqual(code, 0)
            mock_humanize_cli.assert_called_once()
            _, kwargs = mock_humanize_cli.call_args
            self.assertEqual(kwargs["text"], "Test dispatch")
            self.assertEqual(kwargs["strength"], "aggressive")
            self.assertEqual(kwargs["output_path"], "out.txt")
            self.assertTrue(kwargs["as_json"])


if __name__ == "__main__":
    unittest.main()
