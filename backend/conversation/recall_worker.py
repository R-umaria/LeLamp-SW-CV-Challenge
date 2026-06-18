"""Background recall worker for Milestone 4.3.2.

The worker keeps slow Ollama calls out of the webcam/FSM/Godot command loop.
Main enqueues a bounded RecallWorkItem after it has sent the immediate
"Thinking..." command, then polls the result queue without blocking.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.conversation.recall_agent import RecallAgent, RecallResult


@dataclass(frozen=True)
class RecallWorkItem:
    """One recall question to process in the background."""

    text: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "unknown"
    created_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class RecallWorkResult:
    """Completed recall result with worker timing metadata."""

    request_id: str
    source: str
    text: str
    result: RecallResult | None
    worker_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None

    def response_payload(self) -> dict[str, Any]:
        if self.result is None:
            return {
                "ok": False,
                "request_id": self.request_id,
                "source": self.source,
                "answer": "The recall request failed.",
                "error": self.error or "unknown_worker_error",
                "recall_worker_ms": round(float(self.worker_ms), 3),
            }
        payload = self.result.to_dict()
        payload["ok"] = not self.result.llm_required_failed
        if self.result.llm_required_failed:
            payload["error"] = "llm_required_failed"
        payload["request_id"] = self.request_id
        payload["source"] = self.source
        payload["recall_worker_ms"] = round(float(self.worker_ms), 3)
        return payload


class RecallWorker:
    """Single-consumer background worker for RecallAgent.answer()."""

    def __init__(
        self,
        recall_agent: RecallAgent,
        result_queue: "queue.Queue[RecallWorkResult]",
        logger: logging.Logger | None = None,
    ) -> None:
        self.recall_agent = recall_agent
        self.result_queue = result_queue
        self.logger = logger or logging.getLogger("lelamp")
        self._input_queue: "queue.Queue[RecallWorkItem | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="recall-worker", daemon=True)
        self._thread.start()

    def submit(self, item: RecallWorkItem) -> None:
        self._input_queue.put(item)

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        self._input_queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self._input_queue.get()
            if item is None:
                break

            self.logger.info(
                "recall_worker_started request_id=%s source=%s text=%r",
                item.request_id,
                item.source,
                item.text,
            )
            start = time.perf_counter()
            try:
                result = self.recall_agent.answer(item.text)
                worker_ms = (time.perf_counter() - start) * 1000.0
                self.logger.info(
                    "recall_worker_finished request_id=%s source=%s recall_worker_ms=%.3f llm_used=%s fallback_reason=%s llm_error=%s",
                    item.request_id,
                    item.source,
                    worker_ms,
                    result.llm_used,
                    result.llm_fallback_reason,
                    result.llm_error,
                )
                self.result_queue.put(
                    RecallWorkResult(
                        request_id=item.request_id,
                        source=item.source,
                        text=item.text,
                        result=result,
                        worker_ms=worker_ms,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                worker_ms = (time.perf_counter() - start) * 1000.0
                self.logger.exception(
                    "recall_worker_finished request_id=%s source=%s recall_worker_ms=%.3f llm_used=False fallback_reason=worker_exception error=%s",
                    item.request_id,
                    item.source,
                    worker_ms,
                    exc,
                )
                self.result_queue.put(
                    RecallWorkResult(
                        request_id=item.request_id,
                        source=item.source,
                        text=item.text,
                        result=None,
                        worker_ms=worker_ms,
                        error=f"worker_exception: {exc}",
                    )
                )
