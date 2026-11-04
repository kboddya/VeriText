from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Silence third-party noise (PyTorch redirects, Transformers warnings/progress bars, HF)
# UNLESS --log is explicitly requested.
VERBOSE_LOG = "--log" in sys.argv
if not VERBOSE_LOG:
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("veritext").setLevel(logging.WARNING)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("veritext").setLevel(logging.DEBUG)

import transformers

if not VERBOSE_LOG:
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()

from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .engine import ModernAIDetector
from .humanizer import ModernTextHumanizer
from .markers import detect_stylistic_markers
from .parsers import extract_text_from_file
from .preprocessing import get_confidence_status
from .reports import generate_humanize_markdown_report, generate_markdown_report

logger = logging.getLogger("veritext.cli")


def create_progress(enabled: bool) -> Progress:
    console = Console(stderr=True)
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not enabled,
    )


def parse_args() -> argparse.Namespace:
    # Allow running 'veritext web' or 'veritext ui' without dashes
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"web", "ui"}:
        sys.argv[1] = "--web"

    parser = argparse.ArgumentParser(description="VeriText — AI Text & Document Detector CLI")
    parser.add_argument(
        "--mode",
        choices=["auto", "ru", "en"],
        default="auto",
        help="Language model selection: auto, ru (RuBERT) or en (DeBERTa-v3)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Inference compute device",
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument(
        "--markdown",
        "--md",
        action="store_true",
        help="Output verification report in Markdown (.md) format to stdout",
    )
    parser.add_argument(
        "--report-md",
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save verification report in Markdown (.md) format to specified file path",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable verbose service logs (PyTorch, Transformers, Hugging Face, debug messages)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for web server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web server (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open web browser on launch",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", default=None, help="Input raw text string to analyze")
    group.add_argument("--file", default=None, help="Document file path (.pdf, .docx, .md, .txt)")
    group.add_argument(
        "--web",
        "--ui",
        action="store_true",
        help="Launch interactive VeriText Web UI in browser",
    )

    parser.add_argument(
        "--humanize",
        action="store_true",
        help="Run detector-guided AI text humanization",
    )
    parser.add_argument(
        "--strength",
        choices=["light", "balanced", "aggressive"],
        default="balanced",
        help="Humanization intensity preset",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output destination file path",
    )
    return parser.parse_args()


def score_text(
    detector: ModernAIDetector,
    text: str,
    mode: str,
    as_json: bool,
    as_markdown: bool = False,
    report_path: Path | None = None,
    show_progress: bool = True,
) -> int:
    with create_progress(enabled=show_progress) as progress:
        task = progress.add_task("[1/4] Detecting text language...", total=4)

        text_lang = detector.detect_language(text)
        effective_mode = text_lang if mode == "auto" else mode
        logger.info("Language detected: %s (mode=%s)", effective_mode, mode)

        def on_step(sub_msg: str) -> None:
            progress.update(task, description=f"[2/4] {sub_msg}")

        progress.update(task, advance=1, description=f"[2/4] Loading model weights ({effective_mode})...")
        if hasattr(detector, "ensure_model_loaded"):
            effective_key, model_name = detector.ensure_model_loaded(
                effective_mode,
                sample_text=text,
                on_step=on_step,
            )
        else:
            effective_key = effective_mode
            model_name = getattr(detector, "model_name", "AI Detector")

        progress.update(task, advance=1, description=f"[3/4] Analyzing text & stylistic markers ({model_name})...")
        res = detector.score_single(text[:30000], mode=effective_mode)
        markers = detect_stylistic_markers(text)

        # Granular paragraph breakdown
        raw_paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        if len(raw_paras) <= 1:
            lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 35]
            raw_paras = lines if lines else [text]

        paras_to_score = raw_paras[:150]
        scored_paragraphs = []

        if len(paras_to_score) > 1:
            batch_results = detector.score_batch(paras_to_score, mode=effective_mode, batch_size=16)
            ai_para_count = sum(1 for r in batch_results if r["label"] == "ai")
            human_para_count = len(batch_results) - ai_para_count

            weights = [max(1, len(p.split())) for p in paras_to_score]
            total_weight = sum(weights)
            weighted_ai_sum = sum(
                r["ai_probability"] * w for r, w in zip(batch_results, weights)
            )
            paragraph_scale = (
                round((weighted_ai_sum / total_weight) * 100, 2)
                if total_weight > 0
                else res["human_ai_scale"]
            )

            for idx, (p_text, r) in enumerate(zip(paras_to_score, batch_results)):
                scored_paragraphs.append({
                    "index": idx + 1,
                    "text": p_text,
                    "label": r["label"],
                    "score": r["score"],
                    "human_ai_scale": r["human_ai_scale"],
                    "word_count": len(p_text.split()),
                })
            ai_paragraph_pct = round((ai_para_count / len(paras_to_score) * 100), 2)
        else:
            ai_para_count = 1 if res["label"] == "ai" else 0
            human_para_count = 0 if res["label"] == "ai" else 1
            paragraph_scale = res["human_ai_scale"]
            ai_paragraph_pct = 100.0 if res["label"] == "ai" else 0.0
            scored_paragraphs = [{
                "index": 1,
                "text": text,
                "label": res["label"],
                "score": res["score"],
                "human_ai_scale": res["human_ai_scale"],
                "word_count": len(text.split()),
            }]

        progress.update(task, advance=1, description="[4/4] Finalizing verification verdict...")

    words = len(text.split())
    chars = len(text)
    monolithic_scale = res["human_ai_scale"]
    confidence_status = get_confidence_status(monolithic_scale)

    payload = {
        "label": res["label"],
        "ai_probability_pct": monolithic_scale,
        "score": res["score"],
        "human_ai_scale": monolithic_scale,
        "monolithic_scale": monolithic_scale,
        "paragraph_scale": paragraph_scale,
        "overall_scale": monolithic_scale,
        "confidence_status": confidence_status,
        "mode": mode,
        "model_used": res["model_used"],
        "device": detector.device_str,
        "markers": markers,
        "full_text": text,
        "summary": {
            "overall_label": res["label"],
            "overall_scale": monolithic_scale,
            "monolithic_scale": monolithic_scale,
            "paragraph_scale": paragraph_scale,
            "confidence_status": confidence_status,
            "word_count": words,
            "char_count": chars,
            "total_paragraphs": len(raw_paras),
            "analyzed_paragraphs": len(paras_to_score),
            "ai_paragraphs": ai_para_count,
            "human_paragraphs": human_para_count,
            "ai_paragraph_pct": ai_paragraph_pct,
        },
        "paragraphs": scored_paragraphs,
    }

    if report_path:
        md_content = generate_markdown_report(payload, source_name="CLI Text Input")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md_content, encoding="utf-8")
        if not as_markdown and not as_json:
            print(f"Markdown report saved to: {report_path}", file=sys.stderr)

    if as_markdown:
        print(generate_markdown_report(payload, source_name="CLI Text Input"))
    elif as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        label_str = "AI-Generated (AI)" if res["label"] == "ai" else "Human-Written (Human)"
        print("=" * 60)
        print(f"Verdict:         {label_str} [{confidence_status['grade']}]")
        print(f"AI Probability:  {res['human_ai_scale']}%")
        print(f"Paragraph Scale: {paragraph_scale}%")
        print(f"Risk Level:      {confidence_status['risk_level']}")
        print(f"Model Used:      {res['model_used']}")
        print(f"Words / Chars:   {words} / {chars}")
        print("=" * 60)
        if markers:
            print("Detected Generative Markers:")
            for m in markers:
                print(f" - «{m['phrase']}»: {m['description']} (x{m['count']})")
        if report_path:
            print(f"Report file:     {report_path}")
    return 0


def score_document(
    detector: ModernAIDetector,
    file_path: Path,
    mode: str,
    as_json: bool,
    as_markdown: bool = False,
    report_path: Path | None = None,
    show_progress: bool = True,
) -> int:
    with create_progress(enabled=show_progress) as progress:
        task = progress.add_task(f"[1/5] Extracting text from {file_path.name}...", total=5)
        content = file_path.read_bytes()
        full_text, paragraphs = extract_text_from_file(file_path.name, content)

        progress.update(task, advance=1, description="[2/5] Detecting document language...")
        doc_lang = detector.detect_language(full_text)
        effective_mode = doc_lang if mode == "auto" else mode
        logger.info("Document language detected: %s", effective_mode)

        def on_step(sub_msg: str) -> None:
            progress.update(task, description=f"[3/5] {sub_msg}")

        progress.update(task, advance=1, description=f"[3/5] Loading model weights ({effective_mode})...")
        if hasattr(detector, "ensure_model_loaded"):
            effective_key, model_name = detector.ensure_model_loaded(
                effective_mode,
                sample_text=full_text,
                on_step=on_step,
            )
        else:
            effective_key = effective_mode
            model_name = getattr(detector, "model_name", "AI Detector")

        progress.update(task, advance=1, description=f"[4/5] Analyzing {len(paragraphs)} paragraphs ({model_name})...")
        overall = detector.score_single(full_text[:30000], mode=effective_mode)
        markers = detect_stylistic_markers(full_text)

        paras_to_score = paragraphs[:150]
        paras_results = detector.score_batch(paras_to_score, mode=effective_mode)

        ai_paras = sum(1 for r in paras_results if r["label"] == "ai")
        human_paras = len(paras_results) - ai_paras

        scored_paragraphs = []
        if paras_results:
            weights = [max(1, len(p.split())) for p in paras_to_score]
            total_weight = sum(weights)
            weighted_ai_sum = sum(
                res["ai_probability"] * w for res, w in zip(paras_results, weights)
            )
            ai_pct = (
                round((weighted_ai_sum / total_weight) * 100, 2)
                if total_weight > 0
                else overall["human_ai_scale"]
            )

            for idx, (p_text, r) in enumerate(zip(paras_to_score, paras_results)):
                scored_paragraphs.append({
                    "index": idx + 1,
                    "text": p_text,
                    "label": r["label"],
                    "score": r["score"],
                    "human_ai_scale": r["human_ai_scale"],
                    "word_count": len(p_text.split()),
                })
            ai_paragraph_pct = round((ai_paras / len(paras_results) * 100), 2)
        else:
            ai_pct = overall["human_ai_scale"]
            ai_paragraph_pct = 100.0 if overall["label"] == "ai" else 0.0

        progress.update(task, advance=1, description="[5/5] Generating verification report...")

    words = len(full_text.split())
    chars = len(full_text)
    monolithic_scale = overall["human_ai_scale"]
    confidence_status = get_confidence_status(monolithic_scale)

    payload = {
        "filename": file_path.name,
        "overall_label": "ai" if ai_pct >= 50.0 else "human",
        "label": overall["label"],
        "overall_ai_pct": ai_pct,
        "human_ai_scale": monolithic_scale,
        "monolithic_scale": monolithic_scale,
        "paragraph_scale": ai_pct,
        "confidence_status": confidence_status,
        "model_used": overall["model_used"],
        "device": detector.device_str,
        "mode": mode,
        "markers": markers,
        "full_text": full_text,
        "summary": {
            "overall_label": overall["label"],
            "overall_scale": monolithic_scale,
            "monolithic_scale": monolithic_scale,
            "paragraph_scale": ai_pct,
            "confidence_status": confidence_status,
            "word_count": words,
            "char_count": chars,
            "total_paragraphs": len(paragraphs),
            "analyzed_paragraphs": len(paras_results),
            "ai_paragraphs": ai_paras,
            "human_paragraphs": human_paras,
            "ai_paragraph_pct": ai_paragraph_pct,
        },
        "paragraphs": scored_paragraphs,
    }

    if report_path:
        md_content = generate_markdown_report(payload, source_name=file_path.name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md_content, encoding="utf-8")
        if not as_markdown and not as_json:
            print(f"Markdown report saved to: {report_path}", file=sys.stderr)

    if as_markdown:
        print(generate_markdown_report(payload, source_name=file_path.name))
    elif as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        label_str = "AI-Generated (AI)" if payload["overall_label"] == "ai" else "Human-Written (Human)"
        print("=" * 60)
        print(f"Document:        {file_path.name}")
        print(f"Verdict:         {label_str} [{confidence_status['grade']}]")
        print(f"AI Content Share:{ai_pct}%")
        print(f"AI Paragraphs:   {ai_paras} of {len(paras_results)}")
        print(f"Human Paragraphs:{human_paras} of {len(paras_results)}")
        print(f"Risk Level:      {confidence_status['risk_level']}")
        print(f"Model Used:      {overall['model_used']}")
        print("=" * 60)
        if markers:
            print("Detected Generative Markers:")
            for m in markers:
                print(f" - «{m['phrase']}»: {m['description']} (x{m['count']})")
        if report_path:
            print(f"Report file:     {report_path}")
    return 0


def humanize_cli(
    text: str | None = None,
    file_path: Path | str | None = None,
    mode: str = "auto",
    strength: str = "balanced",
    device: str | None = None,
    output_path: Path | str | None = None,
    as_json: bool = False,
    as_markdown: bool = False,
    report_path: Path | None = None,
    show_progress: bool = True,
) -> int:
    """Execute detector-guided AI text humanization from CLI."""
    if text is not None:
        raw_text = text
        source_name = "CLI Text Input"
    elif file_path is not None:
        p = Path(file_path)
        content = p.read_bytes()
        raw_text, _ = extract_text_from_file(p.name, content)
        source_name = p.name
    else:
        logger.error("No input text or file provided for humanization.")
        return 1

    humanizer = ModernTextHumanizer(device=device)

    with create_progress(enabled=show_progress) as progress:
        task = progress.add_task("[1/4] Scrubbing AI clichés & formulaic transitions...", total=100)

        def on_progress(pct: Any, *p_args: Any) -> None:
            effective_pct = int(pct) if isinstance(pct, (int, float)) else 25
            if len(p_args) >= 2:
                step_idx = p_args[0]
                msg = p_args[1]
                progress.update(task, completed=effective_pct, description=f"[{step_idx}/4] {msg}")
            elif len(p_args) == 1:
                step_idx = pct
                progress.update(task, completed=min(100, int(step_idx * 25)))
            else:
                progress.update(task, completed=effective_pct)

        raw_res = humanizer.humanize_text(
            raw_text,
            mode=mode,
            strength=strength,
            on_progress=on_progress,
        )
        progress.update(task, completed=4, description="[4/4] Finalizing verification & delta metrics...")

    before_scale = float(raw_res.get("before_scale", 0.0))
    after_scale = float(raw_res.get("after_scale", 0.0))
    delta_scale = float(raw_res.get("delta_scale", round(before_scale - after_scale, 2)))
    before_para_scale = float(raw_res.get("before_paragraph_scale", before_scale))
    after_para_scale = float(raw_res.get("after_paragraph_scale", after_scale))

    before_conf = raw_res.get("before_confidence")
    if not isinstance(before_conf, dict) or "grade" not in before_conf:
        before_conf = get_confidence_status(before_scale)

    after_conf = raw_res.get("after_confidence")
    if not isinstance(after_conf, dict) or "grade" not in after_conf:
        after_conf = get_confidence_status(after_scale)

    orig_full = raw_res.get("original_text", raw_text)
    hum_full = raw_res.get("humanized_text", raw_text)

    paragraphs: list[dict[str, Any]] = []
    for idx, p in enumerate(raw_res.get("paragraphs") or [], 1):
        b_sc = float(p.get("before_scale", 0.0))
        a_sc = float(p.get("after_scale", 0.0))
        d_sc = float(p.get("delta_scale", round(b_sc - a_sc, 2)))
        orig_p = p.get("original_text", "")
        hum_p = p.get("humanized_text", "")
        is_code = bool(p.get("is_code", False))
        changed = p.get("changed", orig_p != hum_p)
        stage_applied = p.get("stage_applied", "code" if is_code else ("full" if changed else "unchanged"))

        b_c = p.get("before_confidence")
        if not isinstance(b_c, dict) or "grade" not in b_c:
            b_c = get_confidence_status(b_sc)

        a_c = p.get("after_confidence")
        if not isinstance(a_c, dict) or "grade" not in a_c:
            a_c = get_confidence_status(a_sc)

        paragraphs.append({
            "index": p.get("index", idx),
            "original_text": orig_p,
            "humanized_text": hum_p,
            "before_scale": b_sc,
            "after_scale": a_sc,
            "delta_scale": d_sc,
            "before_label": p.get("before_label", "ai" if b_sc >= 50.0 else "human"),
            "after_label": p.get("after_label", "ai" if a_sc >= 50.0 else "human"),
            "before_confidence": b_c,
            "after_confidence": a_c,
            "changed": changed,
            "stage_applied": stage_applied,
            "is_code": is_code,
        })

    raw_summary = raw_res.get("summary") or {}
    words_before = raw_summary.get("word_count_before", len(orig_full.split()) if orig_full else 0)
    words_after = raw_summary.get("word_count_after", len(hum_full.split()) if hum_full else 0)
    chars_before = raw_summary.get("char_count_before", len(orig_full))
    chars_after = raw_summary.get("char_count_after", len(hum_full))
    total_paras = raw_summary.get("total_paragraphs", len(paragraphs) or 1)
    modified_paras = raw_summary.get("modified_paragraphs_count", sum(1 for p in paragraphs if p["changed"]))
    removed_cliches = raw_summary.get("removed_cliches_count", len(raw_res.get("removed_markers", [])))

    summary = {
        "word_count_before": words_before,
        "word_count_after": words_after,
        "char_count_before": chars_before,
        "char_count_after": chars_after,
        "total_paragraphs": total_paras,
        "modified_paragraphs_count": modified_paras,
        "removed_cliches_count": removed_cliches,
        "original_ai_paragraphs": raw_summary.get("original_ai_paragraphs", sum(1 for p in paragraphs if p["before_scale"] >= 50.0)),
        "humanized_ai_paragraphs": raw_summary.get("humanized_ai_paragraphs", sum(1 for p in paragraphs if p["after_scale"] >= 50.0)),
    }

    result: dict[str, Any] = {
        "original_text": orig_full,
        "humanized_text": hum_full,
        "mode": raw_res.get("mode", mode),
        "effective_language": raw_res.get("effective_language", "en"),
        "strength": raw_res.get("strength", strength),
        "request_id": raw_res.get("request_id", "cli-humanize"),
        "before_scale": before_scale,
        "after_scale": after_scale,
        "delta_scale": delta_scale,
        "before_paragraph_scale": before_para_scale,
        "after_paragraph_scale": after_para_scale,
        "before_confidence": before_conf,
        "after_confidence": after_conf,
        "summary": summary,
        "removed_markers": raw_res.get("removed_markers", []),
        "paragraphs": paragraphs,
    }
    if file_path is not None:
        result["filename"] = Path(file_path).name

    out_p: Path | None = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(hum_full, encoding="utf-8")

    rep_p: Path | None = None
    if report_path:
        rep_p = Path(report_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        report_md = generate_humanize_markdown_report(
            result,
            source_name=source_name,
            filename=source_name,
        )
        rep_p.write_text(report_md, encoding="utf-8")

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif as_markdown:
        print(generate_humanize_markdown_report(result, source_name=source_name, filename=source_name))
    else:
        console = Console(file=sys.stdout)
        console.print("=" * 70)
        console.print("✨ VeriText AI Text Humanizer — Transformation Summary", style="bold cyan")
        console.print("=" * 70)

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="bold", no_wrap=True)
        table.add_column("Original (Before)", justify="center")
        table.add_column("Humanized (After)", justify="center")
        table.add_column("Net Change (Delta)", justify="center")

        # Row 1: Continuous AI Probability
        delta_prob_str = f"[bold green]-{delta_scale:.2f}%[/bold green]" if delta_scale >= 0 else f"[bold red]+{-delta_scale:.2f}%[/bold red]"
        table.add_row(
            "Continuous AI Probability",
            f"{before_scale:.2f}%",
            f"{after_scale:.2f}%",
            delta_prob_str,
        )

        # Row 2: Confidence Grade
        b_grade = before_conf.get("grade", "N/A")
        a_grade = after_conf.get("grade", "N/A")
        grade_delta = "[bold green]Improved[/bold green]" if b_grade != a_grade else "[dim]Unchanged[/dim]"
        table.add_row(
            "Confidence Grade",
            str(b_grade),
            str(a_grade),
            grade_delta,
        )

        # Row 3: Paragraph-Weighted AI
        para_delta = round(before_para_scale - after_para_scale, 2)
        para_delta_str = f"[bold green]-{para_delta:.2f}%[/bold green]" if para_delta >= 0 else f"[bold red]+{-para_delta:.2f}%[/bold red]"
        table.add_row(
            "Paragraph-Weighted AI",
            f"{before_para_scale:.2f}%",
            f"{after_para_scale:.2f}%",
            para_delta_str,
        )

        # Row 4: Stylistic Markers
        markers_delta_str = f"[bold green]-{removed_cliches} scrubbed[/bold green]" if removed_cliches > 0 else "[dim]0 scrubbed[/dim]"
        table.add_row(
            "Stylistic Markers",
            f"{removed_cliches} detected",
            "0 detected",
            markers_delta_str,
        )

        # Row 5: Changed Paragraphs
        mod_pct = (modified_paras / max(1, total_paras)) * 100
        table.add_row(
            "Changed Paragraphs",
            f"0 / {total_paras}",
            f"{modified_paras} / {total_paras}",
            f"{mod_pct:.1f}% modified",
        )

        # Row 6: Total Words / Chars
        word_delta = words_after - words_before
        table.add_row(
            "Total Words / Chars",
            f"{words_before} / {chars_before}",
            f"{words_after} / {chars_after}",
            f"{word_delta:+d} words",
        )

        console.print(table)
        console.print("-" * 70)
        console.print("[bold]Humanized Text Output:[/bold]")
        console.print("-" * 70)
        print(hum_full)
        console.print("=" * 70)

        if out_p:
            print(f"Humanized text saved to: {out_p}")
        if rep_p:
            print(f"Markdown report saved to: {rep_p}")

    return 0


def run_web(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    log_level: str = "warning",
) -> int:
    import threading
    import webbrowser
    import uvicorn
    from rich.console import Console

    console = Console(stderr=True)
    url = f"http://{host}:{port}"

    console.print()
    console.print(f"[bold green]✓[/bold green] [bold]VeriText Web UI starting at:[/bold] [cyan underline]{url}[/cyan underline]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    console.print()

    if open_browser:
        def _open():
            import time
            time.sleep(1.0)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    try:
        uvicorn.run(
            "backend.api.app:app",
            host=host,
            port=port,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]VeriText Web UI server stopped.[/yellow]")
    return 0


def run_web_entrypoint() -> int:
    if "--web" not in sys.argv:
        sys.argv.insert(1, "--web")
    return main()


def main() -> int:
    args = parse_args()
    if args.web:
        log_level = "info" if args.log else "warning"
        return run_web(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            log_level=log_level,
        )

    show_progress = not args.log and sys.stderr.isatty() and not (args.json or args.markdown)

    device = None if args.device == "auto" else args.device

    if args.humanize:
        return humanize_cli(
            text=args.text,
            file_path=Path(args.file) if args.file is not None else None,
            mode=args.mode,
            strength=args.strength,
            device=device,
            output_path=args.output,
            as_json=args.json,
            as_markdown=args.markdown,
            report_path=args.report_md,
            show_progress=show_progress,
        )

    detector = ModernAIDetector(device=device)

    if args.text is not None:
        return score_text(
            detector,
            args.text,
            args.mode,
            args.json,
            as_markdown=args.markdown,
            report_path=args.report_md,
            show_progress=show_progress,
        )
    elif args.file is not None:
        return score_document(
            detector,
            Path(args.file),
            args.mode,
            args.json,
            as_markdown=args.markdown,
            report_path=args.report_md,
            show_progress=show_progress,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
