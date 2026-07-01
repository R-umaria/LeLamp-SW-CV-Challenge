"""Optional local text-to-speech wrapper for Lumos.

Uses pyttsx3 when installed. Missing TTS support is not fatal: callers receive a
structured failure and the backend continues with text-only recall/display.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TTSResult:
    requested: bool
    success: bool
    latency_ms: float
    reason: str


class LocalTTSClient:
    def __init__(self, *, enabled: bool, rate: int = 175, volume: float = 0.85, logger: logging.Logger | None = None) -> None:
        self.enabled = bool(enabled)
        self.rate = int(rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.logger = logger or logging.getLogger("lelamp")
        self._engine = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        self._ensure_engine()
        return self._engine is not None

    def speak(self, text: str) -> TTSResult:
        start = time.perf_counter()
        cleaned = " ".join(str(text or "").strip().split())
        if not self.enabled:
            return TTSResult(False, False, 0.0, "tts_disabled")
        if not cleaned:
            return TTSResult(True, False, 0.0, "empty_text")
        if not self._ensure_engine():
            return TTSResult(True, False, (time.perf_counter() - start) * 1000.0, self._load_error or "tts_unavailable")
        try:
            self._engine.say(cleaned)
            self._engine.runAndWait()
            return TTSResult(True, True, (time.perf_counter() - start) * 1000.0, "spoken")
        except Exception as exc:  # pragma: no cover - hardware/backend dependent
            self.logger.warning("tts_failed reason=%s", exc)
            return TTSResult(True, False, (time.perf_counter() - start) * 1000.0, f"tts_runtime_error:{exc}")

    def _ensure_engine(self) -> bool:
        if self._engine is not None:
            return True
        if not self.enabled:
            self._load_error = "tts_disabled"
            return False
        try:
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            self.logger.info("tts_initialized backend=pyttsx3 rate=%s volume=%.2f", self.rate, self.volume)
            return True
        except Exception as exc:  # pragma: no cover - dependency/backend specific
            self._engine = None
            self._load_error = f"pyttsx3_unavailable:{exc}"
            self.logger.warning("tts_unavailable reason=%s", self._load_error)
            return False
