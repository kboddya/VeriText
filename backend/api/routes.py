from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
import torch
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, Response, UploadFile

from ..cancellation import OperationCancelled, cancellation_manager, track_cancellation
from ..config import DEVICE, MODELS_CATALOG
from ..engine import ModernAIDetector
from ..humanizer import ModernTextHumanizer
from ..markers import detect_stylistic_markers
from ..parsers import extract_text_from_file
from ..preprocessing import detect_dominant_language, get_confidence_status
from ..reports import generate_humanize_markdown_report, generate_markdown_report
from .schemas import (
    AbortRequest,
    AbortResponse,
    HumanizeDocumentResponse,
    HumanizeRequest,
    HumanizeResponse,
    MarkdownReportRequest,
    ScoreRequest,
)

router = APIRouter(prefix="/api", tags=["detector"])

# Singleton inference engine instance
engine = ModernAIDetector(device=DEVICE)
# ModernTextHumanizer instance using detector-guided critic
humanizer = ModernTextHumanizer(detector=engine, device=DEVICE)


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
        "humanizer_info": humanizer.get_llm_info(),
    }


@router.post("/cancel/{request_id}", response_model=AbortResponse)
@router.post("/abort/{request_id}", response_model=AbortResponse)
async def abort_by_id_endpoint(request_id: str) -> dict:
    cancelled = cancellation_manager.cancel(request_id)
    return {
        "request_id": request_id,
        "aborted": cancelled,
        "message": "Cancellation signal sent" if cancelled else "Request not found or already completed",
    }


@router.post("/cancel", response_model=AbortResponse)
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

            text_lang = engine.detect_language(full_text) if hasattr(engine, "detect_language") else detect_dominant_language(full_text)
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

            doc_lang = engine.detect_language(full_text) if hasattr(engine, "detect_language") else detect_dominant_language(full_text)
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


def _format_humanize_response(
    res: dict[str, Any],
    *,
    active_id: str,
    default_mode: str = "auto",
    default_strength: str = "balanced",
) -> dict[str, Any]:
    """Normalize and enrich humanizer output dictionary to conform to HumanizeResponse schema."""
    before_scale = float(res.get("before_scale", 0.0))
    after_scale = float(res.get("after_scale", 0.0))
    delta_scale = float(res.get("delta_scale", round(before_scale - after_scale, 2)))
    before_para_scale = float(res.get("before_paragraph_scale", before_scale))
    after_para_scale = float(res.get("after_paragraph_scale", after_scale))

    before_conf = res.get("before_confidence")
    if not isinstance(before_conf, dict) or "grade" not in before_conf:
        before_conf = get_confidence_status(before_scale)

    after_conf = res.get("after_confidence")
    if not isinstance(after_conf, dict) or "grade" not in after_conf:
        after_conf = get_confidence_status(after_scale)

    orig_full = res.get("original_text", "")
    hum_full = res.get("humanized_text", "")

    paragraphs: list[dict[str, Any]] = []
    for idx, p in enumerate(res.get("paragraphs") or [], 1):
        b_sc = float(p.get("before_scale", 0.0))
        a_sc = float(p.get("after_scale", 0.0))
        d_sc = float(p.get("delta_scale", round(b_sc - a_sc, 2)))
        orig_p = p.get("original_text", "")
        hum_p = p.get("humanized_text", "")
        changed = p.get("changed", orig_p != hum_p)
        stage_applied = p.get("stage_applied", "full" if changed else "unchanged")

        b_conf = p.get("before_confidence")
        if not isinstance(b_conf, dict) or "grade" not in b_conf:
            b_conf = get_confidence_status(b_sc)

        a_conf = p.get("after_confidence")
        if not isinstance(a_conf, dict) or "grade" not in a_conf:
            a_conf = get_confidence_status(a_sc)

        paragraphs.append({
            "index": p.get("index", idx),
            "original_text": orig_p,
            "humanized_text": hum_p,
            "before_scale": b_sc,
            "after_scale": a_sc,
            "delta_scale": d_sc,
            "before_label": p.get("before_label", "ai" if b_sc >= 50.0 else "human"),
            "after_label": p.get("after_label", "ai" if a_sc >= 50.0 else "human"),
            "before_confidence": b_conf,
            "after_confidence": a_conf,
            "changed": changed,
            "stage_applied": stage_applied,
            "is_code": p.get("is_code", False),
        })

    raw_summary = res.get("summary") or {}
    generator_provider = res.get("generator_model", raw_summary.get("generator_provider", "RuT5 / T5-base (Local)"))
    external_llm_active = bool(res.get("external_llm_active", raw_summary.get("external_llm_active", False)))
    local_model_loaded = bool(res.get("local_model_loaded", raw_summary.get("local_model_loaded", False)))

    summary = {
        "word_count_before": raw_summary.get("word_count_before", len(orig_full.split()) if orig_full else 0),
        "word_count_after": raw_summary.get("word_count_after", len(hum_full.split()) if hum_full else 0),
        "char_count_before": raw_summary.get("char_count_before", len(orig_full)),
        "char_count_after": raw_summary.get("char_count_after", len(hum_full)),
        "total_paragraphs": raw_summary.get("total_paragraphs", len(paragraphs) or 1),
        "modified_paragraphs_count": raw_summary.get("modified_paragraphs_count", sum(1 for p in paragraphs if p["changed"])),
        "removed_cliches_count": raw_summary.get("removed_cliches_count", len(res.get("removed_markers", []))),
        "original_ai_paragraphs": raw_summary.get("original_ai_paragraphs", sum(1 for p in paragraphs if p["before_scale"] >= 50.0)),
        "humanized_ai_paragraphs": raw_summary.get("humanized_ai_paragraphs", sum(1 for p in paragraphs if p["after_scale"] >= 50.0)),
        "generator_provider": generator_provider,
        "external_llm_active": external_llm_active,
        "local_model_loaded": local_model_loaded,
    }

    return {
        "original_text": orig_full,
        "humanized_text": hum_full,
        "mode": res.get("mode", default_mode),
        "effective_language": res.get("effective_language", "en"),
        "strength": res.get("strength", default_strength),
        "request_id": active_id,
        "generator_model": generator_provider,
        "external_llm_active": external_llm_active,
        "local_model_loaded": local_model_loaded,
        "before_scale": before_scale,
        "after_scale": after_scale,
        "delta_scale": delta_scale,
        "before_paragraph_scale": before_para_scale,
        "after_paragraph_scale": after_para_scale,
        "before_confidence": before_conf,
        "after_confidence": after_conf,
        "summary": summary,
        "removed_markers": res.get("removed_markers", []),
        "paragraphs": paragraphs,
    }


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize_text_endpoint(
    req: HumanizeRequest,
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> dict:
    full_text = req.text.strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    req_id = req.request_id or x_request_id
    async with track_cancellation(request, req_id) as (active_id, cancel_event):
        try:
            cancellation_manager.update_progress(
                active_id,
                "Analyzing text and detecting language...",
                percent=10,
                step=1,
                total_steps=4,
            )

            if cancel_event.is_set():
                raise OperationCancelled("Humanization was cancelled by user.")

            def on_progress(pct: int, step: int = 1, msg: str = "", cur_p: Optional[int] = None, tot_p: Optional[int] = None) -> None:
                effective_pct = max(5, min(95, pct))
                kwargs: dict[str, Any] = {}
                if cur_p is not None:
                    kwargs["current_paragraph"] = cur_p
                if tot_p is not None:
                    kwargs["total_paragraphs"] = tot_p
                cancellation_manager.update_progress(
                    active_id,
                    message=msg or "Transforming text...",
                    percent=effective_pct,
                    step=step,
                    total_steps=4,
                    **kwargs,
                )

            humanizer.detector = engine
            res = await asyncio.to_thread(
                humanizer.humanize_text,
                text=full_text,
                mode=req.mode,
                strength=req.strength,
                is_cancelled=cancel_event.is_set,
                on_progress=on_progress,
            )

            cancellation_manager.update_progress(
                active_id,
                "Finalizing humanization metrics & verification verdict...",
                percent=95,
                step=4,
                total_steps=4,
            )

            return _format_humanize_response(
                res,
                active_id=active_id,
                default_mode=req.mode,
                default_strength=req.strength,
            )
        except OperationCancelled:
            raise HTTPException(status_code=499, detail="Humanization cancelled by user.")


@router.post("/humanize-document", response_model=HumanizeDocumentResponse)
async def humanize_document_endpoint(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    strength: str = Form("balanced"),
    request_id: Optional[str] = Form(None),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".md", ".markdown", ".docx", ".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported formats: .md, .docx, .pdf, .txt",
        )

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
                raise OperationCancelled("Humanization was cancelled by user.")

            if not content:
                raise HTTPException(status_code=400, detail="Document contains no readable text.")

            try:
                full_text, paragraphs = extract_text_from_file(file.filename, content)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            if cancel_event.is_set():
                raise OperationCancelled("Humanization was cancelled by user.")

            clean_text = full_text.strip()
            if not clean_text:
                raise HTTPException(status_code=400, detail="Document contains no readable text.")

            cancellation_manager.update_progress(
                active_id,
                f"Scrubbing clichés & generating candidate paraphrases ({strength})...",
                percent=35,
                step=2,
                total_steps=4,
            )

            def on_progress(pct: int, step: int = 1, msg: str = "", cur_p: Optional[int] = None, tot_p: Optional[int] = None) -> None:
                effective_pct = max(5, min(95, pct))
                kwargs: dict[str, Any] = {}
                if cur_p is not None:
                    kwargs["current_paragraph"] = cur_p
                if tot_p is not None:
                    kwargs["total_paragraphs"] = tot_p
                cancellation_manager.update_progress(
                    active_id,
                    message=msg or "Transforming document...",
                    percent=effective_pct,
                    step=step,
                    total_steps=4,
                    **kwargs,
                )

            humanizer.detector = engine
            res = await asyncio.to_thread(
                humanizer.humanize_text,
                text=clean_text,
                mode=mode,
                strength=strength,
                is_cancelled=cancel_event.is_set,
                on_progress=on_progress,
            )

            cancellation_manager.update_progress(
                active_id,
                "Finalizing document transformation report...",
                percent=95,
                step=4,
                total_steps=4,
            )

            resp_dict = _format_humanize_response(
                res,
                active_id=active_id,
                default_mode=mode,
                default_strength=strength,
            )
            resp_dict["filename"] = file.filename
            resp_dict["file_type"] = ext.lstrip(".") or "document"
            return resp_dict
        except OperationCancelled:
            raise HTTPException(status_code=499, detail="Humanization cancelled by user.")


@router.post("/report/markdown")
async def generate_markdown_report_endpoint(payload: dict) -> Response:
    """
    Generate and return a downloadable Markdown (.md) inspection or humanization report.
    Accepts either { "data": ..., "report_type": ... } or direct inspection result dictionary.
    """
    if not payload:
        raise HTTPException(status_code=400, detail="Payload data is required to generate report.")

    if "data" in payload and isinstance(payload["data"], dict):
        target_data = payload["data"]
        filename = payload.get("filename")
    else:
        target_data = payload
        filename = payload.get("filename")

    report_type = payload.get("report_type") or target_data.get("report_type", "single")
    base_name = filename or target_data.get("filename") or ("veritext_humanized" if report_type == "humanize" else "veritext_inspection")
    clean_base = base_name.rsplit(".", 1)[0] if "." in base_name else base_name

    if report_type == "humanize":
        md_content = generate_humanize_markdown_report(
            target_data,
            filename=filename or target_data.get("filename"),
            source_name=filename or target_data.get("filename"),
        )
        download_filename = f"report_humanized_{clean_base}.md"
    else:
        md_content = generate_markdown_report(
            target_data,
            source_name=filename or target_data.get("filename"),
        )
        download_filename = f"report_{clean_base}.md"

    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"',
        },
    )

