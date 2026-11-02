"""Model cache and offline-first synchronization manager."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("veritext.cache")

CACHE_DIR = Path(os.environ.get("VERITEXT_CACHE_DIR", Path.home() / ".cache" / "veritext"))
SYNC_FILE = CACHE_DIR / "model_sync.json"
CHECK_INTERVAL_SECONDS = 7 * 24 * 3600  # 7 days (1 week)


def get_last_check(model_id: str) -> Optional[float]:
    """Retrieve timestamp of last successful remote sync for model_id."""
    try:
        if SYNC_FILE.exists():
            data = json.loads(SYNC_FILE.read_text(encoding="utf-8"))
            checks = data.get("last_checks", {})
            return checks.get(model_id)
    except Exception as e:
        logger.debug("Failed reading cache sync file: %s", e)
    return None


def record_check(model_id: str, timestamp: Optional[float] = None) -> None:
    """Record verification / sync timestamp for model_id."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {"last_checks": {}}
        if SYNC_FILE.exists():
            try:
                data = json.loads(SYNC_FILE.read_text(encoding="utf-8"))
                if "last_checks" not in data or not isinstance(data["last_checks"], dict):
                    data["last_checks"] = {}
            except Exception:
                data = {"last_checks": {}}
        data["last_checks"][model_id] = timestamp or time.time()
        SYNC_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug("Failed writing cache sync file: %s", e)


def should_check_remote(model_id: str) -> bool:
    """Check if more than 7 days have elapsed since last check for model_id."""
    last = get_last_check(model_id)
    if last is None:
        return True
    return (time.time() - last) > CHECK_INTERVAL_SECONDS


def load_offline_first(
    model_id: str,
    device: torch.device,
    on_step: Optional[Callable[[str], None]] = None,
    fallback_id: Optional[str] = None,
) -> tuple[object, object, str]:
    """
    Load tokenizer and classification model offline-first.
    1. Try to load from local cache (local_files_only=True) without network.
    2. If successful:
       - If more than 7 days passed, do a non-blocking check on Hugging Face.
    3. If local cache missing:
       - Download from Hugging Face (local_files_only=False).
       - Record check timestamp.
    4. Fall back if specified and primary model fails.
    """
    model_name = model_id.split("/")[-1]

    # 1. Attempt local offline load
    if on_step:
        on_step(f"Checking local cache for weights ({model_name})...")

    local_loaded = False
    tokenizer = None
    model = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if on_step:
            on_step(f"Loading model weights ({model_name})...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
        if on_step:
            on_step(f"Transferring weights to device ({device})...")
        model.to(device)
        model.eval()
        local_loaded = True
        logger.info("Model %s loaded successfully from local cache (offline mode)", model_id)
    except Exception as local_err:
        logger.info("Local cache for %s not found: %s", model_id, local_err)

    if local_loaded:
        # Check weekly remote check policy (after first trying local)
        if should_check_remote(model_id):
            logger.info("Over 7 days since last check for %s. Checking Hugging Face for updates...", model_id)
            try:
                from huggingface_hub import model_info

                model_info(model_id, timeout=3.0)
                record_check(model_id)
                logger.info("Hugging Face check for %s completed successfully", model_id)
            except Exception as net_err:
                logger.info("Could not connect to Hugging Face (continuing in offline mode): %s", net_err)
        return model, tokenizer, model_id

    # 2. Local cache was missing or incomplete: try remote download from Hugging Face
    if on_step:
        on_step(f"Downloading model from Hugging Face ({model_name})...")
    logger.info("Downloading %s from Hugging Face Hub (local_files_only=False)...", model_id)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        if on_step:
            on_step(f"Loading model weights ({model_name})...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=False)
        if on_step:
            on_step(f"Transferring weights to device ({device})...")
        model.to(device)
        model.eval()
        record_check(model_id)
        logger.info("Model %s downloaded and cached successfully", model_id)
        return model, tokenizer, model_id
    except Exception as remote_err:
        logger.warning("Failed to download model %s from Hugging Face: %s", model_id, remote_err)
        if fallback_id:
            logger.info("Switching to fallback model: %s", fallback_id)
            if on_step:
                on_step(f"Switching to fallback model ({fallback_id.split('/')[-1]})...")
            return load_offline_first(fallback_id, device, on_step=on_step, fallback_id=None)
        raise RuntimeError(
            f"Failed to load model '{model_id}' (local cache not found, "
            f"and Hugging Face download failed: {remote_err})"
        ) from remote_err
