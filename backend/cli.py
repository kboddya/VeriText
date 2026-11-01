from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from .engine import ModernAIDetector
from .markers import detect_stylistic_markers
from .parsers import extract_text_from_file
from .preprocessing import get_confidence_status
from .reports import generate_batch_markdown_report, generate_markdown_report


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", default=None, help="Input raw text string to analyze")
    group.add_argument("--file", default=None, help="Document file path (.pdf, .docx, .md, .txt)")
    group.add_argument("--input", default=None, help="Tabular dataset file path (.csv, .tsv)")

    parser.add_argument("--text-column", default=None, help="Target text column name for --input")
    parser.add_argument("--output", default=None, help="Output destination path for scored .csv")
    return parser.parse_args()


def score_text(
    detector: ModernAIDetector,
    text: str,
    mode: str,
    as_json: bool,
    as_markdown: bool = False,
    report_path: Path | None = None,
) -> int:
    text_lang = detector.detect_language(text)
    effective_mode = text_lang if mode == "auto" else mode

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
) -> int:
    content = file_path.read_bytes()
    full_text, paragraphs = extract_text_from_file(file_path.name, content)

    doc_lang = detector.detect_language(full_text)
    effective_mode = doc_lang if mode == "auto" else mode

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


def score_dataset(
    detector: ModernAIDetector,
    input_path: Path,
    output_path: Path | None,
    text_column: str | None,
    mode: str,
    batch_size: int,
    report_path: Path | None = None,
    as_markdown: bool = False,
) -> int:
    sep = "\t" if input_path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(input_path, sep=sep, low_memory=False)

    col = text_column
    if not col:
        for c in ("text", "content", "generation", "prompt", "body", "текст", "сообщение"):
            if c in frame.columns:
                col = c
                break
        if not col:
            col = str(frame.columns[0])

    texts = frame[col].fillna("").astype(str).tolist()
    preds = detector.score_batch(texts, mode=mode, batch_size=batch_size)

    frame["ai_label"] = [p["label"] for p in preds]
    frame["ai_probability_pct"] = [p["human_ai_scale"] for p in preds]

    total_rows = len(preds)
    ai_count = sum(1 for p in preds if p["label"] == "ai")
    human_count = total_rows - ai_count
    ai_pct = round((ai_count / total_rows * 100), 2) if total_rows > 0 else 0.0

    batch_payload = {
        "filename": input_path.name,
        "summary": {
            "total_rows": total_rows,
            "ai_count": ai_count,
            "human_count": human_count,
            "ai_percentage": ai_pct,
            "text_column": col,
            "mode": mode,
            "device": detector.device_str,
        },
        "preview_rows": frame.head(25).to_dict(orient="records"),
    }

    if report_path:
        md_content = generate_batch_markdown_report(batch_payload, filename=input_path.name)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md_content, encoding="utf-8")
        if not as_markdown:
            print(f"Markdown report saved to: {report_path}", file=sys.stderr)

    if as_markdown:
        print(generate_batch_markdown_report(batch_payload, filename=input_path.name))
    else:
        if output_path is None:
            frame.to_csv(sys.stdout, sep=sep, index=False)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, sep=sep, index=False)
            print(f"Scored dataset saved to: {output_path}", file=sys.stderr)
    return 0


def main() -> int:
    args = parse_args()
    device = None if args.device == "auto" else args.device
    detector = ModernAIDetector(device=device)

    if args.text is not None:
        return score_text(
            detector,
            args.text,
            args.mode,
            args.json,
            as_markdown=args.markdown,
            report_path=args.report_md,
        )
    elif args.file is not None:
        return score_document(
            detector,
            Path(args.file),
            args.mode,
            args.json,
            as_markdown=args.markdown,
            report_path=args.report_md,
        )
    elif args.input is not None:
        return score_dataset(
            detector,
            Path(args.input),
            Path(args.output) if args.output else None,
            args.text_column,
            args.mode,
            args.batch_size,
            report_path=args.report_md,
            as_markdown=args.markdown,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
