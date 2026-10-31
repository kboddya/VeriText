from __future__ import annotations

import asyncio
import io
from typing import Optional
import pandas as pd
import torch
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile

from ..cancellation import OperationCancelled, cancellation_manager, track_cancellation
from ..config import DEVICE, MODELS_CATALOG
from ..engine import ModernAIDetector
from ..markers import detect_stylistic_markers
from ..parsers import extract_text_from_file
from ..preprocessing import get_confidence_status
from .schemas import AbortRequest, AbortResponse, ScoreRequest

router = APIRouter(prefix="/api", tags=["detector"])

# Singleton inference engine instance
engine = ModernAIDetector(device=DEVICE)


@router.get("/health")
async def health_check() -> dict:
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_available:
        device_type = "gpu"
        device_name = torch.cuda.get_device_name(0)
        allocated_mem = f"{torch.cuda.memory_allocated(0) / (1024**2):.2f} MB"
    elif mps_available:
        device_type = "gpu"
        device_name = "MPS"
        allocated_mem = "Unified Memory"
    else:
        device_type = "cpu"
        device_name = "CPU"
        allocated_mem = "N/A"

    return {
        "status": "ok",
        "device": DEVICE,
        "device_type": device_type,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device_name": device_name,
        "gpu_memory_allocated": allocated_mem,
        "active_models": list(engine._models.keys()),
        "available_modes": list(MODELS_CATALOG.keys()) + ["auto"],
        "active_requests": cancellation_manager.active_count(),
    }


@router.post("/abort/{request_id}", response_model=AbortResponse)
async def abort_by_id_endpoint(request_id: str) -> dict:
    cancelled = cancellation_manager.cancel(request_id)
    return {
        "request_id": request_id,
        "aborted": cancelled,
        "message": "Cancellation signal sent" if cancelled else "Request not found or already completed",
    }


@router.post("/abort", response_model=AbortResponse)
async def abort_by_body_endpoint(payload: AbortRequest) -> dict:
    cancelled = cancellation_manager.cancel(payload.request_id)
    return {
        "request_id": payload.request_id,
        "aborted": cancelled,
        "message": "Cancellation signal sent" if cancelled else "Request not found or already completed",
    }


@router.get("/status/{request_id}")
async def get_task_status_endpoint(request_id: str) -> dict:
    return cancellation_manager.get_progress(request_id)


@router.post("/score")
async def score_text_endpoint(
    req: ScoreRequest,
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> dict:
    req_id = req.request_id or x_request_id
    async with track_cancellation(request, req_id) as (active_id, cancel_event):
        try:
            full_text = req.text.strip()
            if not full_text:
                raise HTTPException(status_code=400, detail="Text cannot be empty.")

            cancellation_manager.update_progress(
                active_id,
                "Normalizing text and analyzing language...",
                percent=15,
                step=1,
                total_steps=4,
            )

            text_lang = engine.detect_language(full_text)
            effective_mode = text_lang if req.mode == "auto" else req.mode

            cancellation_manager.update_progress(
                active_id,
                f"Evaluating continuous macro attention ({effective_mode.upper()})...",
                percent=35,
                step=2,
                total_steps=4,
            )

            # 1. Monolithic whole-text analysis (continuous context)
            res = await asyncio.to_thread(
                engine.score_single,
                full_text[:30000],
                mode=effective_mode,
                is_cancelled=cancel_event.is_set,
            )

            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

            # 2. Extract paragraphs for granular analysis
            raw_paras = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 20]
            if len(raw_paras) <= 1:
                lines = [line.strip() for line in full_text.split("\n") if len(line.strip()) > 35]
                raw_paras = lines if lines else [full_text]

            paras_to_score = raw_paras[:150]
            scored_paragraphs = []

            if len(paras_to_score) > 1:
                cancellation_manager.update_progress(
                    active_id,
                    f"Scoring paragraphs (0/{len(paras_to_score)})...",
                    percent=55,
                    step=3,
                    total_steps=4,
                )

                def on_text_para_progress(cur: int, tot: int) -> None:
                    pct = 55 + int((cur / max(1, tot)) * 35)
                    cancellation_manager.update_progress(
                        active_id,
                        f"Scoring paragraphs: {cur}/{tot} ({int(cur/tot*100)}%)...",
                        percent=pct,
                        step=3,
                        total_steps=4,
                    )

                batch_results = await asyncio.to_thread(
                    engine.score_batch,
                    paras_to_score,
                    mode=effective_mode,
                    batch_size=16,
                    is_cancelled=cancel_event.is_set,
                    on_progress=on_text_para_progress,
                )
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
                    "text": full_text,
                    "label": res["label"],
                    "score": res["score"],
                    "human_ai_scale": res["human_ai_scale"],
                    "word_count": len(full_text.split()),
                }]

            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

            cancellation_manager.update_progress(
                active_id,
                "Finalizing metrics & confidence status...",
                percent=95,
                step=4,
                total_steps=4,
            )

            words = len(full_text.split())
            chars = len(full_text)
            markers = detect_stylistic_markers(full_text)
            monolithic_scale = res["human_ai_scale"]
            confidence_status = get_confidence_status(monolithic_scale)

            return {
                "label": res["label"],
                "score": res["score"],
                "human_ai_scale": monolithic_scale,
                "monolithic_scale": monolithic_scale,
                "paragraph_scale": paragraph_scale,
                "overall_scale": monolithic_scale,
                "confidence_status": confidence_status,
                "mode": req.mode,
                "model_used": res["model_used"],
                "device": DEVICE,
                "request_id": active_id,
                "markers": markers,
                "windows_analyzed": res.get("windows_analyzed", 1),
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
                "full_text": full_text,
            }
        except OperationCancelled:
            raise HTTPException(status_code=499, detail="Analysis cancelled by user.")


@router.post("/score-document")
async def score_document_endpoint(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    request_id: Optional[str] = Form(None),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    req_id = request_id or x_request_id
    async with track_cancellation(request, req_id) as (active_id, cancel_event):
        try:
            cancellation_manager.update_progress(
                active_id,
                f"Reading and extracting text from {file.filename}...",
                percent=10,
                step=1,
                total_steps=4,
            )

            content = await file.read()
            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

            try:
                full_text, paragraphs = extract_text_from_file(file.filename, content)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

            doc_lang = engine.detect_language(full_text)
            effective_mode = doc_lang if mode == "auto" else mode

            cancellation_manager.update_progress(
                active_id,
                f"Evaluating whole-document continuous context ({effective_mode.upper()})...",
                percent=35,
                step=2,
                total_steps=4,
            )

            # 1. Monolithic whole-document analysis (continuous context)
            overall_res = await asyncio.to_thread(
                engine.score_single,
                full_text[:30000],
                mode=effective_mode,
                is_cancelled=cancel_event.is_set,
            )

            # 2. Paragraph-level batch scoring
            scored_paragraphs = []
            paras_to_score = paragraphs[:150]

            if paras_to_score:
                cancellation_manager.update_progress(
                    active_id,
                    f"Scoring paragraphs (0/{len(paras_to_score)})...",
                    percent=55,
                    step=3,
                    total_steps=4,
                )

                def on_doc_para_progress(cur: int, tot: int) -> None:
                    pct = 55 + int((cur / max(1, tot)) * 35)
                    cancellation_manager.update_progress(
                        active_id,
                        f"Scoring paragraphs: {cur}/{tot} ({int(cur/tot*100)}%)...",
                        percent=pct,
                        step=3,
                        total_steps=4,
                    )

                batch_results = await asyncio.to_thread(
                    engine.score_batch,
                    paras_to_score,
                    mode=effective_mode,
                    batch_size=16,
                    is_cancelled=cancel_event.is_set,
                    on_progress=on_doc_para_progress,
                )
                ai_para_count = sum(1 for r in batch_results if r["label"] == "ai")
                human_para_count = len(batch_results) - ai_para_count

                weights = [max(1, len(p.split())) for p in paras_to_score]
                total_weight = sum(weights)
                weighted_ai_sum = sum(
                    res["ai_probability"] * w for res, w in zip(batch_results, weights)
                )
                paragraph_scale = (
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
                        "word_count": len(p_text.split()),
                    })

                ai_paragraph_pct = round((ai_para_count / len(paras_to_score) * 100), 2)
            else:
                ai_para_count = 1 if overall_res["label"] == "ai" else 0
                human_para_count = 0 if overall_res["label"] == "ai" else 1
                paragraph_scale = overall_res["human_ai_scale"]
                ai_paragraph_pct = 100.0 if overall_res["label"] == "ai" else 0.0
                scored_paragraphs = [{
                    "index": 1,
                    "text": full_text,
                    "label": overall_res["label"],
                    "score": overall_res["score"],
                    "human_ai_scale": overall_res["human_ai_scale"],
                    "word_count": len(full_text.split()),
                }]

            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

            cancellation_manager.update_progress(
                active_id,
                "Finalizing metrics & confidence status...",
                percent=95,
                step=4,
                total_steps=4,
            )

            words = len(full_text.split())
            chars = len(full_text)
            markers = detect_stylistic_markers(full_text)
            monolithic_scale = overall_res["human_ai_scale"]
            confidence_status = get_confidence_status(monolithic_scale)

            return {
                "filename": file.filename,
                "label": overall_res["label"],
                "score": overall_res["score"],
                "human_ai_scale": monolithic_scale,
                "monolithic_scale": monolithic_scale,
                "paragraph_scale": paragraph_scale,
                "overall_scale": monolithic_scale,
                "confidence_status": confidence_status,
                "mode": mode,
                "model_used": overall_res["model_used"],
                "device": DEVICE,
                "request_id": active_id,
                "summary": {
                    "overall_label": overall_res["label"],
                    "overall_scale": monolithic_scale,
                    "monolithic_scale": monolithic_scale,
                    "paragraph_scale": paragraph_scale,
                    "confidence_status": confidence_status,
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
        except OperationCancelled:
            raise HTTPException(status_code=499, detail="Analysis cancelled by user.")


@router.post("/score-file")
async def score_file_endpoint(
    request: Request,
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None),
    mode: str = Form("auto"),
    batch_size: int = Form(32),
    request_id: Optional[str] = Form(None),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
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

    req_id = request_id or x_request_id
    async with track_cancellation(request, req_id) as (active_id, cancel_event):
        try:
            content = await file.read()
            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

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

            predictions = await asyncio.to_thread(
                engine.score_batch,
                texts,
                mode=mode,
                batch_size=batch_size,
                is_cancelled=cancel_event.is_set,
            )

            if cancel_event.is_set():
                raise OperationCancelled("Analysis aborted by user.")

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
                "request_id": active_id,
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
        except OperationCancelled:
            raise HTTPException(status_code=499, detail="Analysis cancelled by user.")
