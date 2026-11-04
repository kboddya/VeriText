from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .cancellation import OperationCancelled
from .cache import load_offline_first
from .config import (
    detect_device,
    PARAPHRASERS_CATALOG,
    HUMANIZER_STRENGTH_PRESETS,
    LENGTH_PARITY_MIN_RATIO,
    LENGTH_PARITY_MAX_RATIO,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_API_KEY,
    LLM_TIMEOUT,
)
from .engine import ModernAIDetector
from .markers import scrub_cliches
from .preprocessing import (
    detect_dominant_language,
    get_confidence_status,
    is_code_or_structure,
    split_paragraphs,
)

logger = logging.getLogger("veritext.humanizer")

# Known abbreviations for sentence segmentation
_ABBREVIATIONS = [
    r'\bт\.е\.', r'\bт\.д\.', r'\bт\.п\.', r'\bруб\.', r'\bкоп\.',
    r'\bг\.', r'\bул\.', r'\bд\.', r'\bкв\.', r'\bстр\.',
    r'\bпроф\.', r'\bакад\.', r'\bим\.', r'\bнапр\.', r'\bсм\.',
    r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bDr\.', r'\bProf\.',
    r'\bSr\.', r'\bJr\.', r'\be\.g\.', r'\bi\.e\.', r'\bvs\.',
    r'\betc\.', r'\bInc\.', r'\bLtd\.', r'\bCorp\.', r'\bapprox\.',
    r'\bfig\.', r'\bal\.',
]
_SENTINEL = "\u2060"


def split_sentences(text: str) -> list[str]:
    """Segment paragraph into sentences without splitting on standard abbreviations."""
    if not text:
        return []
    masked = text
    for abbr in _ABBREVIATIONS:
        masked = re.sub(abbr, lambda m: m.group(0).replace('.', _SENTINEL), masked, flags=re.IGNORECASE)

    parts = re.split(r'(?<=[.!?…])\s+(?=[A-ZА-ЯЁ\d«"\'\(\[])', masked)
    sentences = [p.replace(_SENTINEL, '.').strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]


class ModernTextHumanizer:
    """
    Zero-Training Detector-Guided AI Text Humanizer.
    Implements a 3-stage paragraph-by-paragraph pipeline:
    1. Stage 1: De-cliché and RLHF boilerplate scrubbing with typography normalization.
    2. Stage 2: Multi-candidate neural paraphrasing with burstiness variation and length parity filtering.
    3. Stage 3: Critic-in-the-loop detector selection with mathematical monotonic improvement guarantee.
    """

    def __init__(
        self,
        detector: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device_str = device or detect_device()
        self.device = torch.device(self.device_str)
        self.detector = detector or ModernAIDetector(device=self.device_str)
        self._paraphrasers: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}

    def _load_paraphraser(self, lang: str) -> tuple[object, object]:
        """Load offline-first Seq2Seq paraphraser and tokenizer for the specified language."""
        if lang not in self._paraphrasers:
            cat = PARAPHRASERS_CATALOG.get(lang, PARAPHRASERS_CATALOG.get("en", {}))
            model_id = cat.get("id", "humarin/chatgpt_paraphraser_on_T5_base")
            model, tokenizer, _ = load_offline_first(
                model_id=model_id,
                device=self.device,
                model_cls=AutoModelForSeq2SeqLM,
            )
            self._paraphrasers[lang] = model
            self._tokenizers[lang] = tokenizer
        return self._paraphrasers[lang], self._tokenizers[lang]

    def get_llm_info(self) -> dict[str, Any]:
        """
        Check if an external LLM endpoint (Ollama / OpenAI-compatible API) is configured and responsive.
        If responsive, local T5 paraphrasers will be completely bypassed (never loaded into memory or downloaded).
        """
        import os
        base_url = getattr(self, "_llm_base_url", None) or os.environ.get("VERITEXT_LLM_BASE_URL") or LLM_BASE_URL
        model_name = os.environ.get("VERITEXT_LLM_MODEL", LLM_MODEL)
        if not base_url:
            return {
                "configured": False,
                "active": False,
                "provider": "local_t5",
                "model": "RuT5 / T5-base",
            }

        clean_url = base_url.rstrip("/")
        headers = {}
        api_key = os.environ.get("VERITEXT_LLM_API_KEY", LLM_API_KEY)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                # 1. Try standard OpenAI /models endpoint
                try:
                    r = client.get(f"{clean_url}/models", headers=headers)
                    if r.status_code == 200:
                        return {
                            "configured": True,
                            "active": True,
                            "base_url": clean_url,
                            "model": model_name,
                            "provider": f"External LLM ({model_name})",
                        }
                except Exception:
                    pass

                # 2. Try Ollama native /api/tags if /v1 was used
                if clean_url.endswith("/v1"):
                    root_url = clean_url[:-3]
                    try:
                        r = client.get(f"{root_url}/api/tags")
                        if r.status_code == 200:
                            return {
                                "configured": True,
                                "active": True,
                                "base_url": clean_url,
                                "model": model_name,
                                "provider": f"Ollama ({model_name})",
                            }
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("LLM probe notice: %s", e)

        return {
            "configured": True,
            "active": False,
            "base_url": base_url,
            "model": model_name,
            "provider": "local_t5",
        }

    def _generate_llm_candidates(
        self,
        text: str,
        num_candidates: int,
        temperature: float,
    ) -> list[str]:
        """Attempt candidate generation using an external OpenAI-compatible LLM endpoint."""
        import os
        base_url = getattr(self, "_llm_base_url", None) or os.environ.get("VERITEXT_LLM_BASE_URL") or LLM_BASE_URL
        if not base_url:
            return []

        try:
            import httpx

            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            api_key = os.environ.get("VERITEXT_LLM_API_KEY", LLM_API_KEY)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            model_name = os.environ.get("VERITEXT_LLM_MODEL", LLM_MODEL)
            prompt = (
                "You are an expert editor specializing in humanizing text. "
                "Rewrite the following text into natural, clear, engaging human prose with high burstiness. "
                "Remove formulaic AI boilerplate and synthetic cadence. "
                f"Provide {num_candidates} diverse variations separated by '---' on a new line. "
                "Provide only the rewritten text variations without commentary or numbering."
            )
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                "temperature": temperature,
                "n": num_candidates,
            }

            timeout_sec = float(os.environ.get("VERITEXT_LLM_TIMEOUT", LLM_TIMEOUT))
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                results = []
                for choice in data.get("choices", []):
                    content = choice.get("message", {}).get("content", "").strip()
                    if content:
                        for chunk in content.split("---"):
                            chunk_clean = chunk.strip().lstrip("1234567890.-* ").strip()
                            if chunk_clean:
                                results.append(chunk_clean)
                return results
        except Exception as exc:
            logger.warning("Remote LLM candidate generation failed: %s", exc)
            return []

    def _generate_local_t5_candidates(
        self,
        paragraph: str,
        lang: str,
        num_candidates: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> list[str]:
        """Generate diverse neural paraphrasing candidates using local Seq2Seq model."""
        try:
            model, tokenizer = self._load_paraphraser(lang)
        except Exception as exc:
            logger.warning("Failed to load local paraphraser for '%s': %s", lang, exc)
            return []

        prompt_prefix = "paraphrase: " if lang == "en" else ""
        raw_candidates: list[str] = []

        # 1. Whole paragraph candidate generation
        try:
            full_prompt = f"{prompt_prefix}{paragraph}"
            enc = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
            if hasattr(enc, "to"):
                enc = enc.to(self.device)
            gen_out = model.generate(
                **enc,
                num_return_sequences=num_candidates,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=512,
            )
            if hasattr(tokenizer, "batch_decode"):
                decoded_batch = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
                raw_candidates.extend(decoded_batch)
            if hasattr(tokenizer, "decode"):
                for seq in gen_out:
                    decoded_single = tokenizer.decode(seq, skip_special_tokens=True)
                    if decoded_single not in raw_candidates:
                        raw_candidates.append(decoded_single)
        except Exception as e:
            logger.debug("Full paragraph T5 generation notice: %s", e)

        # 2. Sentence-by-sentence candidate generation with burstiness variation
        sentences = split_sentences(paragraph)
        if len(sentences) > 1:
            try:
                per_sentence_variants: list[list[str]] = []
                for s in sentences:
                    s_prompt = f"{prompt_prefix}{s}"
                    s_enc = tokenizer(s_prompt, return_tensors="pt", truncation=True, max_length=256)
                    if hasattr(s_enc, "to"):
                        s_enc = s_enc.to(self.device)
                    s_gen = model.generate(
                        **s_enc,
                        num_return_sequences=num_candidates,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        max_new_tokens=256,
                    )
                    variants: list[str] = []
                    if hasattr(tokenizer, "batch_decode"):
                        variants.extend(tokenizer.batch_decode(s_gen, skip_special_tokens=True))
                    if hasattr(tokenizer, "decode"):
                        for seq in s_gen:
                            d = tokenizer.decode(seq, skip_special_tokens=True)
                            if d not in variants:
                                variants.append(d)
                    per_sentence_variants.append(variants if variants else [s])

                # Assemble bursty paragraph combinations
                for k in range(num_candidates):
                    assembled: list[str] = []
                    prev_starter = ""
                    for v_list in per_sentence_variants:
                        chosen = v_list[k % len(v_list)]
                        words = chosen.split()
                        starter = words[0].lower() if words else ""
                        # Alternate sentence starters if adjacent sentences start identically
                        if starter == prev_starter and len(v_list) > 1:
                            alt = next((v for v in v_list if v.split() and v.split()[0].lower() != starter), None)
                            if alt:
                                chosen = alt
                        prev_starter = chosen.split()[0].lower() if chosen.split() else ""
                        assembled.append(chosen)
                    cand_text = " ".join(assembled)
                    if cand_text not in raw_candidates:
                        raw_candidates.append(cand_text)
            except Exception as e:
                logger.debug("Sentence-level T5 generation notice: %s", e)

        return raw_candidates

    def humanize_text(
        self,
        text: str,
        mode: str = "auto",
        strength: str = "balanced",
        is_cancelled: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[..., None]] = None,
    ) -> dict[str, Any]:
        """
        Execute full 3-stage humanization pipeline on text.
        Raises:
            ValueError: If text is empty or whitespace-only.
            OperationCancelled: If is_cancelled() returns True.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or contain only whitespace.")

        if is_cancelled and is_cancelled():
            raise OperationCancelled("Humanization was cancelled by user.")

        # Determine language mode
        if mode == "auto":
            effective_lang = detect_dominant_language(text)
        else:
            effective_lang = mode if mode in ("ru", "en") else "en"

        preset = HUMANIZER_STRENGTH_PRESETS.get(
            strength, HUMANIZER_STRENGTH_PRESETS["balanced"]
        )
        num_candidates = int(preset["num_candidates"])
        temperature = float(preset["temperature"])
        top_p = float(preset["top_p"])
        repetition_penalty = float(preset["repetition_penalty"])

        paragraphs_raw = split_paragraphs(text)
        total_paras = len(paragraphs_raw)

        # Detect active external LLM provider (Ollama / OpenAI-compatible endpoint)
        llm_info = self.get_llm_info()
        llm_active = bool(llm_info.get("active", False))

        def report_step(
            step_idx: int,
            total_steps: int,
            msg: str,
            percent: Optional[int] = None,
            current_paragraph: Optional[int] = None,
            total_paragraphs: Optional[int] = None,
        ) -> None:
            if not on_progress:
                return
            pct = percent if percent is not None else int((step_idx / max(1, total_steps)) * 100)
            try:
                on_progress(pct, step_idx, msg, current_paragraph, total_paragraphs)
            except TypeError:
                try:
                    on_progress(pct, step_idx, msg)
                except TypeError:
                    try:
                        on_progress(step_idx, total_steps)
                    except TypeError:
                        on_progress(pct)

        report_step(1, 4, f"Analyzing document structure ({total_paras} paragraphs) and baseline authenticity...", percent=10, total_paragraphs=total_paras)

        # Compute baseline original scores for all paragraphs
        if hasattr(self.detector, "score_batch"):
            baseline_scores = self.detector.score_batch(
                paragraphs_raw,
                mode=effective_lang,
                is_cancelled=is_cancelled,
            )
        else:
            baseline_scores = [
                self.detector.score_single(p, mode=effective_lang, is_cancelled=is_cancelled)
                for p in paragraphs_raw
            ]

        transformed_paragraphs: list[dict[str, Any]] = []
        all_removed_markers: list[dict[str, str]] = []

        report_step(2, 4, f"Starting paragraph transformation (0/{total_paras})...", percent=20, total_paragraphs=total_paras)

        for idx, (orig_p, base_res) in enumerate(zip(paragraphs_raw, baseline_scores)):
            if is_cancelled and is_cancelled():
                raise OperationCancelled("Humanization was cancelled by user.")

            cur_p = idx + 1
            # Step 2 spans percent from 20% to 85% proportionally across paragraphs
            p_start_pct = 20 + int(((cur_p - 1) / max(1, total_paras)) * 65)
            report_step(
                2,
                4,
                f"Humanizing paragraph {cur_p}/{total_paras} ({(cur_p * 100) // total_paras}%)...",
                percent=p_start_pct,
                current_paragraph=cur_p,
                total_paragraphs=total_paras,
            )

            orig_scale = float(base_res["human_ai_scale"])
            orig_prob = float(base_res.get("ai_probability", orig_scale / 100.0))

            # Code / Structured block preservation:
            # Code blocks, scripts, and markdown tables are strictly preserved as-is.
            if is_code_or_structure(orig_p):
                report_step(
                    2,
                    4,
                    f"Paragraph {cur_p}/{total_paras}: code structure detected (preserved intact)",
                    percent=p_start_pct + int(65 / max(1, total_paras)),
                    current_paragraph=cur_p,
                    total_paragraphs=total_paras,
                )
                transformed_paragraphs.append({
                    "index": cur_p,
                    "original_text": orig_p,
                    "humanized_text": orig_p,
                    "before_scale": round(orig_scale, 2),
                    "after_scale": round(orig_scale, 2),
                    "delta_scale": 0.0,
                    "stage_applied": "code",
                    "is_code": True,
                    "changed": False,
                })
                continue

            # Protect inline code snippets (`...`) inside prose from being altered or translated
            inline_re = re.compile(r'(`[^`\n]+`)')
            inlines: list[str] = []

            def _mask_inline(m: re.Match) -> str:
                i = len(inlines)
                inlines.append(m.group(1))
                return f"__VT_INL_{i}__"

            p_to_process = inline_re.sub(_mask_inline, orig_p) if "`" in orig_p else orig_p

            def _restore_inlines(s: str) -> str:
                if not inlines:
                    return s
                res = s
                for i_idx, code_str in enumerate(inlines):
                    res = res.replace(f"__VT_INL_{i_idx}__", code_str)
                return res

            # Stage 1: De-cliché scrubbing on masked text
            p_stage1_masked, markers = scrub_cliches(p_to_process, lang=effective_lang)
            if not p_stage1_masked.strip():
                p_stage1_masked = p_to_process
            all_removed_markers.extend(markers)
            p_stage1 = _restore_inlines(p_stage1_masked)

            # Stage 2: Candidate generation
            candidates: list[str] = []

            # 1. Primary generation via active external LLM (Ollama / OpenAI-compatible endpoint)
            if llm_active:
                llm_cands = self._generate_llm_candidates(p_stage1_masked, num_candidates, temperature)
                if llm_cands:
                    candidates.extend([_restore_inlines(c) for c in llm_cands])

            # 2. Local T5 generation ONLY if external LLM is NOT active.
            # If an external model is connected and active, local T5 is NEVER loaded into memory and NEVER downloaded!
            if not candidates and not llm_active:
                t5_cands = self._generate_local_t5_candidates(
                    paragraph=p_stage1_masked,
                    lang=effective_lang,
                    num_candidates=num_candidates,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                candidates.extend([_restore_inlines(c) for c in t5_cands])

            # Filter candidates by length parity [0.65, 1.40] with graceful buffer for short inputs
            stage1_len = max(1, len(p_stage1))
            min_len = max(3, min(int(stage1_len * LENGTH_PARITY_MIN_RATIO), stage1_len - 25))
            max_len = max(int(stage1_len * LENGTH_PARITY_MAX_RATIO), stage1_len + 50)
            valid_candidates: list[str] = []
            for cand in candidates:
                cand_clean = cand.strip()
                if not cand_clean or cand_clean == p_stage1 or cand_clean == orig_p:
                    continue
                # If original paragraph had inline code, candidate must preserve all of them
                if inlines and not all(code_str in cand_clean for code_str in inlines):
                    continue
                cand_len = len(cand_clean)
                if min_len <= cand_len <= max_len:
                    if cand_clean not in valid_candidates:
                        valid_candidates.append(cand_clean)

            # Stage 3: Detector critic selection
            # Candidate pool: [p_stage1] + valid_candidates
            # Mathematical monotonic improvement guarantee: never selects a candidate worse than Stage 1
            pool: list[str] = [p_stage1]
            for c in valid_candidates:
                if c not in pool:
                    pool.append(c)

            if hasattr(self.detector, "score_batch"):
                pool_results = self.detector.score_batch(
                    pool,
                    mode=effective_lang,
                    is_cancelled=is_cancelled,
                )
            else:
                pool_results = [
                    self.detector.score_single(c, mode=effective_lang, is_cancelled=is_cancelled)
                    for c in pool
                ]

            # Choose candidate with minimum AI probability
            best_idx = 0
            best_prob = float("inf")
            for c_idx, s_info in enumerate(pool_results):
                prob = float(s_info.get("ai_probability", float(s_info["human_ai_scale"]) / 100.0))
                # Monotonic bound: strictly prefer lower or equal AI score
                if prob < best_prob:
                    best_prob = prob
                    best_idx = c_idx

            selected_text = pool[best_idx]
            selected_score_dict = pool_results[best_idx]
            after_p_scale = float(selected_score_dict["human_ai_scale"])

            # Monotonic guarantee clamp: after_p_scale never exceeds orig_scale
            if after_p_scale > orig_scale:
                selected_text = orig_p
                after_p_scale = orig_scale

            # Stage determination
            if selected_text == orig_p:
                stage_applied = "unchanged"
            elif selected_text == p_stage1:
                stage_applied = "stage1"
            else:
                stage_applied = "full"

            delta_p_scale = round(orig_scale - after_p_scale, 2)

            transformed_paragraphs.append({
                "index": cur_p,
                "original_text": orig_p,
                "humanized_text": selected_text,
                "before_scale": round(orig_scale, 2),
                "after_scale": round(after_p_scale, 2),
                "delta_scale": delta_p_scale,
                "stage_applied": stage_applied,
                "is_code": False,
                "changed": (orig_p != selected_text),
            })

            p_done_pct = 20 + int((cur_p / max(1, total_paras)) * 65)
            report_step(
                2,
                4,
                f"Paragraph {cur_p}/{total_paras} complete (AI: {orig_scale:.1f}% → {after_p_scale:.1f}%)",
                percent=p_done_pct,
                current_paragraph=cur_p,
                total_paragraphs=total_paras,
            )

        report_step(3, 4, "Calculating comprehensive document metrics and statistics...")

        # Paragraph-weighted aggregate document score
        total_words = sum(max(1, len(p["original_text"].split())) for p in transformed_paragraphs)
        before_weighted = sum(
            p["before_scale"] * max(1, len(p["original_text"].split()))
            for p in transformed_paragraphs
        ) / max(1, total_words)

        after_weighted = sum(
            p["after_scale"] * max(1, len(p["original_text"].split()))
            for p in transformed_paragraphs
        ) / max(1, total_words)

        before_scale = round(before_weighted, 2)
        after_scale = round(after_weighted, 2)
        delta_scale = round(before_scale - after_scale, 2)

        humanized_full_text = "\n\n".join(p["humanized_text"] for p in transformed_paragraphs)

        generator_provider = llm_info.get("provider", "RuT5 / T5-base (Local)") if llm_active else "RuT5 / T5-base (Local)"
        local_model_loaded = bool(self._paraphrasers)

        summary = {
            "word_count_before": len(text.split()),
            "word_count_after": len(humanized_full_text.split()),
            "char_count_before": len(text),
            "char_count_after": len(humanized_full_text),
            "total_paragraphs": total_paras,
            "modified_paragraphs_count": sum(
                1 for p in transformed_paragraphs if p["original_text"] != p["humanized_text"]
            ),
            "removed_cliches_count": len(all_removed_markers),
            "generator_provider": generator_provider,
            "external_llm_active": llm_active,
            "local_model_loaded": local_model_loaded,
        }

        report_step(4, 4, "Humanization complete.")

        return {
            "original_text": text,
            "humanized_text": humanized_full_text,
            "mode": mode,
            "effective_language": effective_lang,
            "strength": strength,
            "generator_model": generator_provider,
            "external_llm_active": llm_active,
            "local_model_loaded": local_model_loaded,
            "before_scale": before_scale,
            "after_scale": after_scale,
            "delta_scale": delta_scale,
            "before_confidence": get_confidence_status(before_scale),
            "after_confidence": get_confidence_status(after_scale),
            "removed_markers": all_removed_markers,
            "summary": summary,
            "paragraphs": transformed_paragraphs,
        }

    # Method alias
    humanize = humanize_text
