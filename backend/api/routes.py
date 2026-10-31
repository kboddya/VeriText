from __future__ import annotations

import io
from typing import Optional
import pandas as pd
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import DEVICE, MODELS_CATALOG
from ..engine import ModernAIDetector
from ..markers import detect_stylistic_markers
from ..parsers import extract_text_from_file
from .schemas import ScoreRequest

router = APIRouter(prefix="/api", tags=["detector"])

# Singleton inference engine instance
engine = ModernAIDetector(device=DEVICE)


@router.get("/health")
async def health_check() -> dict:
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        allocated_mem = f"{torch.cuda.memory_allocated(0) / (1024**2):.2f} MB"
    elif mps_available:
        device_name = "Apple Silicon GPU (Metal / MPS)"
        allocated_mem = "Unified Memory (Mac)"
    else:
        device_name = "CPU (ARM64 / Apple Silicon)"
        allocated_mem = "N/A"

    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device_name": device_name,
        "gpu_memory_allocated": allocated_mem,
        "active_models": list(engine._models.keys()),
        "available_modes": list(MODELS_CATALOG.keys()) + ["auto"],
    }


@router.post("/score")
async def score_text_endpoint(req: ScoreRequest) -> dict:
    res = engine.score_single(req.text, mode=req.mode)
    markers = detect_stylistic_markers(req.text)

    return {
        "label": res["label"],
        "score": res["score"],
        "human_ai_scale": res["human_ai_scale"],
        "mode": req.mode,
        "model_used": res["model_used"],
        "device": DEVICE,
        "markers": markers,
        "windows_analyzed": res.get("windows_analyzed", 1),
    }


@router.post("/score-document")
async def score_document_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    content = await file.read()
    try:
        full_text, paragraphs = extract_text_from_file(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    doc_lang = engine.detect_language(full_text)
    effective_mode = doc_lang if mode == "auto" else mode

    overall_res = engine.score_single(full_text[:30000], mode=effective_mode)

    scored_paragraphs = []
    paras_to_score = paragraphs[:150]

    if paras_to_score:
        batch_results = engine.score_batch(paras_to_score, mode=effective_mode, batch_size=16)
        ai_para_count = sum(1 for r in batch_results if r["label"] == "ai")
        human_para_count = len(batch_results) - ai_para_count

        weights = [max(1, len(p.split())) for p in paras_to_score]
        total_weight = sum(weights)
        weighted_ai_sum = sum(
            res["ai_probability"] * w for res, w in zip(batch_results, weights)
        )
        weighted_ai_pct = (
            round((weighted_ai_sum / total_weight) * 100, 2)
            if total_weight > 0
            else overall_res["human_ai_scale"]
        )

        for idx, (p_text, res) in enumerate(zip(paras_to_score, batch_results)):
            scored_paragraphs.append({
                "index": idx + 1,
                "text": p_text,
                "label": res["label"],
                "score": res["score"],
                "human_ai_scale": res["human_ai_scale"],
            })

        ai_paragraph_pct = round((ai_para_count / len(paras_to_score) * 100), 2)
        calculated_ai_pct = weighted_ai_pct
    else:
        ai_para_count = 1 if overall_res["label"] == "ai" else 0
        human_para_count = 0 if overall_res["label"] == "ai" else 1
        ai_paragraph_pct = overall_res["human_ai_scale"]
        calculated_ai_pct = overall_res["human_ai_scale"]

    words = len(full_text.split())
    chars = len(full_text)
    markers = detect_stylistic_markers(full_text)

    return {
        "filename": file.filename,
        "mode": mode,
        "model_used": overall_res["model_used"],
        "device": DEVICE,
        "summary": {
            "overall_label": "ai" if calculated_ai_pct >= 50.0 else "human",
            "overall_score": round((calculated_ai_pct / 100.0 - 0.5) * 10, 4),
            "overall_scale": calculated_ai_pct,
            "word_count": words,
            "char_count": chars,
            "total_paragraphs": len(paragraphs),
            "analyzed_paragraphs": len(paras_to_score),
            "ai_paragraphs": ai_para_count,
            "human_paragraphs": human_para_count,
            "ai_paragraph_pct": ai_paragraph_pct,
        },
        "paragraphs": scored_paragraphs,
        "full_text": full_text,
        "markers": markers,
    }


@router.post("/score-file")
async def score_file_endpoint(
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None),
    mode: str = Form("auto"),
    batch_size: int = Form(32),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File was not provided.")

    filename_lower = file.filename.lower()
    sep = "\t" if filename_lower.endswith(".tsv") else ","
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".tsv")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Only .csv and .tsv files are supported.",
        )

    content = await file.read()
    try:
        frame = pd.read_csv(io.BytesIO(content), sep=sep, low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read tabular dataset: {str(e)}") from e

    candidate_cols = ("text", "generation", "content", "body", "prompt", "текст", "сообщение")
    resolved_col = text_column
    if not resolved_col:
        for c in candidate_cols:
            if c in frame.columns:
                resolved_col = c
                break
        if not resolved_col:
            resolved_col = str(frame.columns[0])

    texts = frame[resolved_col].fillna("").astype(str).tolist()
    predictions = engine.score_batch(texts, mode=mode, batch_size=batch_size)

    scored_frame = frame.copy()
    scored_frame["ai_detector_label"] = [p["label"] for p in predictions]
    scored_frame["ai_probability_pct"] = [p["human_ai_scale"] for p in predictions]
    scored_frame["ai_score"] = [p["score"] for p in predictions]

    total_rows = len(predictions)
    ai_count = sum(1 for p in predictions if p["label"] == "ai")
    human_count = total_rows - ai_count
    ai_percentage = round((ai_count / total_rows * 100), 2) if total_rows > 0 else 0.0

    out_buffer = io.StringIO()
    scored_frame.to_csv(out_buffer, sep=sep, index=False)
    csv_result = out_buffer.getvalue()

    preview_rows = scored_frame.head(25).to_dict(orient="records")

    return {
        "filename": f"scored_{file.filename}",
        "summary": {
            "total_rows": total_rows,
            "ai_count": ai_count,
            "human_count": human_count,
            "ai_percentage": ai_percentage,
            "text_column": resolved_col,
            "mode": mode,
            "device": DEVICE,
        },
        "preview_columns": list(scored_frame.columns),
        "preview_rows": preview_rows,
        "csv_content": csv_result,
    }
