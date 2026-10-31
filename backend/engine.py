from __future__ import annotations

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import (
    detect_device,
    MODELS_CATALOG,
    FALLBACK_EN_MODEL_ID,
    MAX_TOKEN_LENGTH,
    WINDOW_TOKEN_SIZE,
    WINDOW_TOKEN_STRIDE,
    MAX_WINDOWS,
)
from .preprocessing import normalize_text_for_detector, detect_dominant_language


class ModernAIDetector:
    """Universal AI text detection engine powered by fine-tuned Hugging Face transformer models."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device_str = device or detect_device()
        self.device = torch.device(self.device_str)
        self._models: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}

    def _load_model(self, model_key: str) -> tuple[object, object]:
        cfg = MODELS_CATALOG.get(model_key, MODELS_CATALOG["ru"])
        model_id = cfg["id"]

        if model_id not in self._models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
            except Exception as e:
                if model_key == "en":
                    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_EN_MODEL_ID)
                    model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_EN_MODEL_ID)
                    model_id = FALLBACK_EN_MODEL_ID
                else:
                    raise e
            model.to(self.device)
            model.eval()
            self._models[model_id] = model
            self._tokenizers[model_id] = tokenizer

        return self._models[model_id], self._tokenizers[model_id]

    def detect_language(self, text: str) -> str:
        """Determine dominant text language: 'ru' or 'en'."""
        return detect_dominant_language(text)

    def score_single(self, text: str, mode: str = "auto") -> dict:
        """
        Score a single piece of text.
        For long inputs (> 512 tokens), applies an adaptive sliding window mechanism.
        """
        if not text or not text.strip():
            return {
                "label": "human",
                "score": -5.0,
                "human_ai_scale": 0.0,
                "ai_probability": 0.0,
                "model_used": MODELS_CATALOG.get(mode, MODELS_CATALOG["ru"])["name"],
                "windows_analyzed": 0,
            }

        lang = self.detect_language(text) if mode == "auto" else mode
        chosen_key = lang if lang in MODELS_CATALOG else "ru"
        model, tokenizer = self._load_model(chosen_key)
        model_info = MODELS_CATALOG.get(chosen_key, MODELS_CATALOG["ru"])

        normalized = normalize_text_for_detector(text, lang=chosen_key)
        if not normalized:
            normalized = text.strip()

        tokens = tokenizer.encode(normalized, truncation=False, add_special_tokens=True)

        # Single context window
        if len(tokens) <= MAX_TOKEN_LENGTH:
            encoded = tokenizer(
                normalized,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = model(**encoded).logits

            ai_prob = self._extract_ai_prob(logits, model)
            ai_pct = round(ai_prob * 100, 2)
            label = "ai" if ai_pct >= 50.0 else "human"
            logit_approx = round((ai_prob - 0.5) * 10, 4)

            return {
                "label": label,
                "score": logit_approx,
                "human_ai_scale": ai_pct,
                "ai_probability": float(ai_prob),
                "model_used": model_info["name"],
                "windows_analyzed": 1,
            }

        # Long text processing via overlapping sliding windows
        window_tensors = []
        for start_idx in range(0, len(tokens) - 1, WINDOW_TOKEN_STRIDE):
            sub_tokens = tokens[start_idx : start_idx + WINDOW_TOKEN_SIZE]
            if len(sub_tokens) < 15 and window_tensors:
                break
            window_tensors.append(sub_tokens)
            if start_idx + WINDOW_TOKEN_SIZE >= len(tokens):
                break

        probs_list = []
        for w_tokens in window_tensors[:MAX_WINDOWS]:
            w_input = torch.tensor([w_tokens], dtype=torch.long, device=self.device)
            w_mask = torch.ones_like(w_input)
            with torch.no_grad():
                w_logits = model(input_ids=w_input, attention_mask=w_mask).logits
            w_prob = self._extract_ai_prob(w_logits, model)
            probs_list.append(w_prob)

        avg_prob = sum(probs_list) / len(probs_list) if probs_list else 0.5
        max_prob = max(probs_list) if probs_list else avg_prob
        # Blend weighted average with peak anomaly detection (70% avg + 30% peak)
        blended_prob = 0.70 * avg_prob + 0.30 * max_prob
        ai_pct = round(blended_prob * 100, 2)
        label = "ai" if ai_pct >= 50.0 else "human"
        logit_approx = round((blended_prob - 0.5) * 10, 4)

        return {
            "label": label,
            "score": logit_approx,
            "human_ai_scale": ai_pct,
            "ai_probability": float(blended_prob),
            "model_used": model_info["name"],
            "windows_analyzed": len(probs_list),
            "window_scores": [round(p * 100, 2) for p in probs_list],
        }

    def _extract_ai_prob(self, logits: torch.Tensor, model: object) -> float:
        """Extract scalar AI probability from model classification logits."""
        if logits.shape[-1] == 1:
            ai_probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            return float(ai_probs if ai_probs.ndim == 0 else ai_probs[0])

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        id2label = getattr(model.config, "id2label", {0: "human", 1: "ai"})
        ai_idx = 1
        for idx, lbl in id2label.items():
            lbl_str = str(lbl).lower()
            if lbl_str in {"ai", "generated", "fake", "true", "1"}:
                ai_idx = int(idx)
                break
            elif lbl_str in {"human", "real", "false", "0"}:
                ai_idx = 1 - int(idx)
                break
        return float(probs[0, ai_idx])

    def score_batch(
        self,
        texts: list[str],
        mode: str = "auto",
        batch_size: int = 16,
    ) -> list[dict]:
        """Batch score texts with automatic preprocessing and tokenization."""
        if not texts:
            return []

        if mode == "auto":
            sample_text = " ".join([t for t in texts[:5] if t and t.strip()])
            chosen_key = self.detect_language(sample_text)
        elif mode in MODELS_CATALOG:
            chosen_key = mode
        else:
            chosen_key = "ru"

        model, tokenizer = self._load_model(chosen_key)
        model_info = MODELS_CATALOG.get(chosen_key, MODELS_CATALOG["ru"])

        results = []
        raw_texts = [str(t) if t is not None else "" for t in texts]
        normalized_texts = [
            normalize_text_for_detector(t, lang=chosen_key) or t.strip()
            for t in raw_texts
        ]

        for i in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits

            if logits.shape[-1] == 1:
                batch_probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                if batch_probs.ndim == 0:
                    batch_probs = [float(batch_probs)]
                else:
                    batch_probs = batch_probs.tolist()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                id2label = getattr(model.config, "id2label", {0: "human", 1: "ai"})
                ai_idx = 1
                for idx, lbl in id2label.items():
                    lbl_str = str(lbl).lower()
                    if lbl_str in {"ai", "generated", "fake", "true", "1"}:
                        ai_idx = int(idx)
                        break
                    elif lbl_str in {"human", "real", "false", "0"}:
                        ai_idx = 1 - int(idx)
                        break
                batch_probs = probs[:, ai_idx].tolist()

            for ai_prob in batch_probs:
                ai_pct = round(float(ai_prob) * 100, 2)
                label = "ai" if ai_pct >= 50.0 else "human"
                logit_approx = round(float(ai_prob - 0.5) * 10, 4)

                results.append({
                    "label": label,
                    "score": logit_approx,
                    "human_ai_scale": ai_pct,
                    "ai_probability": float(ai_prob),
                    "model_used": model_info["name"],
                })

        return results
