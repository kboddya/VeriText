from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

from .engine import ModernAIDetector
from .markers import detect_stylistic_markers
from .parsers import extract_text_from_file


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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", default=None, help="Input raw text string to analyze")
    group.add_argument("--file", default=None, help="Document file path (.pdf, .docx, .md, .txt)")
    group.add_argument("--input", default=None, help="Tabular dataset file path (.csv, .tsv)")

    parser.add_argument("--text-column", default=None, help="Target text column name for --input")
    parser.add_argument("--output", default=None, help="Output destination path for scored .csv")
    return parser.parse_args()


def score_text(detector: ModernAIDetector, text: str, mode: str, as_json: bool) -> int:
    res = detector.score_single(text, mode=mode)
    markers = detect_stylistic_markers(text)

    payload = {
        "label": res["label"],
        "ai_probability_pct": res["human_ai_scale"],
        "score": res["score"],
        "model_used": res["model_used"],
        "device": detector.device_str,
        "markers": markers,
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        label_str = "AI-Generated (AI)" if res["label"] == "ai" else "Human-Written (Human)"
        print("=" * 50)
        print(f"Verdict:         {label_str}")
        print(f"AI Probability:  {res['human_ai_scale']}%")
        print(f"Logit (Score):   {res['score']}")
        print(f"Model Used:      {res['model_used']}")
        print("=" * 50)
        if markers:
            print("Detected Generative Markers:")
            for m in markers:
                print(f" - «{m['phrase']}»: {m['description']}")
    return 0


def score_document(detector: ModernAIDetector, file_path: Path, mode: str, as_json: bool) -> int:
    content = file_path.read_bytes()
    full_text, paragraphs = extract_text_from_file(file_path.name, content)

    doc_lang = detector.detect_language(full_text)
    effective_mode = doc_lang if mode == "auto" else mode

    overall = detector.score_single(full_text[:30000], mode=effective_mode)
    paras_results = detector.score_batch(paragraphs[:100], mode=effective_mode)

    ai_paras = sum(1 for r in paras_results if r["label"] == "ai")
    human_paras = len(paras_results) - ai_paras

    if paras_results:
        weights = [max(1, len(p.split())) for p in paragraphs[:100]]
        total_weight = sum(weights)
        weighted_ai_sum = sum(
            res["ai_probability"] * w for res, w in zip(paras_results, weights)
        )
        ai_pct = (
            round((weighted_ai_sum / total_weight) * 100, 2)
            if total_weight > 0
            else overall["human_ai_scale"]
        )
    else:
        ai_pct = overall["human_ai_scale"]

    payload = {
        "filename": file_path.name,
        "overall_label": "ai" if ai_pct >= 50.0 else "human",
        "overall_ai_pct": ai_pct,
        "model_used": overall["model_used"],
        "total_paragraphs": len(paragraphs),
        "analyzed_paragraphs": len(paras_results),
        "ai_paragraphs": ai_paras,
        "human_paragraphs": human_paras,
    }

    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        label_str = "AI-Generated (AI)" if payload["overall_label"] == "ai" else "Human-Written (Human)"
        print("=" * 60)
        print(f"Document:        {file_path.name}")
        print(f"Verdict:         {label_str}")
        print(f"AI Content Share:{ai_pct}%")
        print(f"AI Paragraphs:   {ai_paras} of {len(paras_results)}")
        print(f"Human Paragraphs:{human_paras} of {len(paras_results)}")
        print(f"Model Used:      {overall['model_used']}")
        print("=" * 60)
    return 0


def score_dataset(
    detector: ModernAIDetector,
    input_path: Path,
    output_path: Path | None,
    text_column: str | None,
    mode: str,
    batch_size: int,
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
        return score_text(detector, args.text, args.mode, args.json)
    elif args.file is not None:
        return score_document(detector, Path(args.file), args.mode, args.json)
    elif args.input is not None:
        return score_dataset(
            detector,
            Path(args.input),
            Path(args.output) if args.output else None,
            args.text_column,
            args.mode,
            args.batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
