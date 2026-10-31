from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import threading
from typing import AsyncGenerator, Callable, Optional
import uuid

from starlette.requests import Request


class OperationCancelled(Exception):
    """Raised when an inference operation is aborted by user or client disconnect."""
    pass


class CancellationManager:
    """Thread-safe registry for managing active request cancellation events and live execution progress."""

    def __init__(self) -> None:
        self._tokens: dict[str, threading.Event] = {}
        self._progress: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str) -> threading.Event:
        with self._lock:
            event = threading.Event()
            self._tokens[request_id] = event
            self._progress[request_id] = {
                "request_id": request_id,
                "status": "running",
                "step": 1,
                "total_steps": 4,
                "percent": 5,
                "message": "Initializing inference session...",
            }
            return event

    def update_progress(
        self,
        request_id: str,
        message: str,
        percent: int,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        with self._lock:
            if request_id in self._progress:
                p = self._progress[request_id]
                p["message"] = message
                p["percent"] = max(0, min(100, int(percent)))
                if step is not None:
                    p["step"] = step
                if total_steps is not None:
                    p["total_steps"] = total_steps

    def get_progress(self, request_id: str) -> dict:
        with self._lock:
            if request_id in self._progress:
                return dict(self._progress[request_id])
            return {
                "request_id": request_id,
                "status": "completed",
                "step": 4,
                "total_steps": 4,
                "percent": 100,
                "message": "Analysis completed",
            }

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._tokens.get(request_id)
            if event:
                event.set()
                if request_id in self._progress:
                    self._progress[request_id]["status"] = "cancelled"
                    self._progress[request_id]["message"] = "Analysis cancelled by user"
                return True
            return False

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._tokens.pop(request_id, None)
            if request_id in self._progress:
                self._progress[request_id]["status"] = "completed"
                self._progress[request_id]["percent"] = 100
                self._progress[request_id]["message"] = "Analysis completed"

    def active_count(self) -> int:
        with self._lock:
            return len(self._tokens)


cancellation_manager = CancellationManager()


@asynccontextmanager
async def track_cancellation(
    request: Request,
    request_id: Optional[str] = None,
) -> AsyncGenerator[tuple[str, threading.Event], None]:
    """
    Context manager that registers a cancellation token for the request
    and actively watches for client disconnection.
    """
    header_id = request.headers.get("X-Request-ID")
    req_id = request_id or header_id or f"req_{uuid.uuid4().hex[:12]}"
    event = cancellation_manager.register(req_id)

    async def watch_disconnect() -> None:
        try:
            while not event.is_set():
                if await request.is_disconnected():
                    event.set()
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    watcher_task = asyncio.create_task(watch_disconnect())
    try:
        yield req_id, event
    finally:
        event.set()
        watcher_task.cancel()
        try:
            await watcher_task
        except asyncio.CancelledError:
            pass
        cancellation_manager.unregister(req_id)
