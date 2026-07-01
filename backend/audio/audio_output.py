"""Non-blocking optional sound/TTS output for Lumos behaviors."""

from __future__ import annotations

import logging
import math
import platform
import queue
import threading
import time
from dataclasses import dataclass
from typing import Mapping

from backend.audio.tts_client import LocalTTSClient
from backend.utils.config import AudioOutputConfig


CUE_FREQUENCIES = {
    "happy_ping": 880,
    "gentle_chime": 660,
    "recall_found": 784,
    "recall_not_found": 330,
    "error": 220,
}


@dataclass(frozen=True)
class AudioOutputEvent:
    kind: str
    value: str
    queued_at: float


class AudioOutputWorker:
    """Small fire-and-forget audio worker.

    The main perception loop only enqueues cue/speech requests. All optional
    pyttsx3/winsound work happens in a daemon thread and failures are logged.
    """

    def __init__(self, config: AudioOutputConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.enabled = bool(config.enabled or config.tts_enabled)
        self._queue: "queue.Queue[AudioOutputEvent | None]" = queue.Queue(maxsize=16)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_cue_at: dict[str, float] = {}
        self._last_speech_text = ""
        self._last_speech_at = 0.0
        self._tts = LocalTTSClient(
            enabled=bool(config.tts_enabled),
            rate=int(config.tts_rate),
            volume=float(config.tts_volume),
            logger=self.logger,
        )

    def start(self) -> None:
        if not self.enabled:
            self.logger.info("audio_output_disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="audio-output-worker", daemon=True)
        self._thread.start()
        self.logger.info(
            "audio_output_started cues=%s tts=%s rate=%s volume=%.2f",
            self.config.enabled,
            self.config.tts_enabled,
            self.config.tts_rate,
            self.config.tts_volume,
        )

    def stop(self, timeout_s: float = 1.5) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self.logger.info("audio_output_stopped")

    def submit_behavior(self, behavior: Mapping) -> None:
        cue = behavior.get("sound")
        speech = behavior.get("speech_text")
        if cue:
            self.request_cue(str(cue))
        if speech:
            self.request_speech(str(speech))

    def request_cue(self, cue_name: str) -> bool:
        if not self.config.enabled:
            return False
        cue = str(cue_name or "").strip()
        if not cue or cue.lower() in {"none", "null"}:
            return False
        now = time.monotonic()
        if now - self._last_cue_at.get(cue, 0.0) < float(self.config.cue_cooldown_s):
            self.logger.info("audio_cue_suppressed cue=%s reason=cooldown", cue)
            return False
        self._last_cue_at[cue] = now
        return self._enqueue(AudioOutputEvent("cue", cue, now))

    def request_speech(self, text: str) -> bool:
        if not self.config.tts_enabled:
            return False
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            return False
        now = time.monotonic()
        if cleaned == self._last_speech_text and now - self._last_speech_at < float(self.config.speech_cooldown_s):
            self.logger.info("tts_suppressed reason=duplicate_cooldown text=%r", cleaned)
            return False
        self._last_speech_text = cleaned
        self._last_speech_at = now
        return self._enqueue(AudioOutputEvent("speech", cleaned, now))

    def _enqueue(self, event: AudioOutputEvent) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(event)
            self.logger.info("audio_output_queued kind=%s value=%r", event.kind, event.value)
            return True
        except queue.Full:
            self.logger.warning("audio_output_dropped kind=%s value=%r reason=queue_full", event.kind, event.value)
            return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            event = self._queue.get()
            if event is None:
                break
            start = time.perf_counter()
            success = False
            reason = "unknown"
            if event.kind == "cue":
                success, reason = self._play_cue(event.value)
            elif event.kind == "speech":
                result = self._tts.speak(event.value)
                success = result.success
                reason = result.reason
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.logger.info(
                "audio_output_done kind=%s value=%r success=%s reason=%s latency_ms=%.3f queue_delay_ms=%.3f",
                event.kind,
                event.value,
                success,
                reason,
                latency_ms,
                (time.monotonic() - event.queued_at) * 1000.0,
            )

    def _play_cue(self, cue_name: str) -> tuple[bool, str]:
        freq = CUE_FREQUENCIES.get(cue_name, CUE_FREQUENCIES.get(cue_name.lower(), 520))
        duration_ms = 140 if cue_name != "recall_not_found" else 220
        if platform.system().lower().startswith("win"):
            try:
                import winsound  # type: ignore

                winsound.Beep(int(freq), int(duration_ms))
                return True, "winsound_beep"
            except Exception as exc:  # pragma: no cover - platform specific
                return False, f"winsound_error:{exc}"
        # Cross-platform fallback: terminal bell. It is deliberately best-effort.
        try:
            print("\a", end="", flush=True)
            return True, "terminal_bell"
        except Exception as exc:  # pragma: no cover
            return False, f"terminal_bell_error:{exc}"
