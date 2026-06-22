"""Non-blocking microphone capture for Lumos.

The main webcam/FSM loop must never wait on the audio device. This module wraps
``sounddevice.InputStream`` in a small worker/callback surface and exposes only
"latest block" reads guarded by a lock. If ``sounddevice`` or the microphone is
unavailable, startup logs a warning and the backend continues without speaker
awareness.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.utils.config import AudioConfig


@dataclass(frozen=True)
class AudioChunk:
    timestamp: float
    samples: np.ndarray
    sample_rate: int
    channels: int
    sequence: int
    capture_ms: float = 0.0
    status: str = "ok"


class AudioCaptureWorker:
    """Best-effort non-blocking microphone capture.

    ``start`` returns ``False`` instead of raising when microphone access fails.
    The callback keeps only the newest chunk so stale audio cannot backlog and
    disrupt the visual perception loop.
    """

    def __init__(self, config: AudioConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.enabled = bool(config.enabled)
        self.available = False
        self.disabled_reason = "audio_disabled" if not self.enabled else "not_started"

        self._sounddevice = None
        self._stream = None
        self._lock = threading.Lock()
        self._latest: Optional[AudioChunk] = None
        self._sequence = 0
        self._last_callback_at = 0.0

    def start(self) -> bool:
        if not self.enabled:
            self.disabled_reason = "audio_disabled"
            self.logger.info("speaker_awareness_disabled_reason reason=%s", self.disabled_reason)
            return False

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            self.disabled_reason = f"sounddevice_unavailable: {exc}"
            self.logger.warning("speaker_awareness_disabled_reason reason=%s", self.disabled_reason)
            return False

        self._sounddevice = sd
        blocksize = max(1, int(round(self.config.sample_rate * self.config.block_ms / 1000.0)))
        channels = 2 if self.config.request_stereo else 1
        try:
            self._stream = sd.InputStream(
                device=self.config.device if self.config.device not in (None, "") else None,
                samplerate=int(self.config.sample_rate),
                blocksize=blocksize,
                channels=channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - depends on local hardware
            self.disabled_reason = f"microphone_open_failed: {exc}"
            self.logger.warning("speaker_awareness_disabled_reason reason=%s", self.disabled_reason)
            self.available = False
            return False

        self.available = True
        self.disabled_reason = ""
        self.logger.info(
            "audio_capture_started device=%s sample_rate=%s block_ms=%s channels=%s blocksize=%s",
            self.config.device or "default",
            self.config.sample_rate,
            self.config.block_ms,
            channels,
            blocksize,
        )
        return True

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - callback exercised with hardware
        callback_started = time.perf_counter()
        try:
            samples = np.asarray(indata, dtype=np.float32).copy()
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            channels = int(samples.shape[1]) if samples.ndim == 2 else 1
            timestamp = float(getattr(time_info, "inputBufferAdcTime", 0.0) or time.time())
            status_text = "ok" if not status else str(status)
            capture_ms = (time.perf_counter() - callback_started) * 1000.0
            with self._lock:
                self._sequence += 1
                self._latest = AudioChunk(
                    timestamp=timestamp,
                    samples=samples,
                    sample_rate=int(self.config.sample_rate),
                    channels=channels,
                    sequence=self._sequence,
                    capture_ms=capture_ms,
                    status=status_text,
                )
                self._last_callback_at = time.monotonic()
            if status:
                self.logger.debug("audio_capture_status status=%s", status)
        except Exception as exc:
            self.logger.warning("audio_capture_callback_error error=%s", exc)

    def get_latest(self) -> Optional[AudioChunk]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
            self.logger.info("audio_capture_stopped")
        except Exception as exc:  # pragma: no cover - hardware guard
            self.logger.warning("audio_capture_stop_failed error=%s", exc)
        finally:
            self.available = False
