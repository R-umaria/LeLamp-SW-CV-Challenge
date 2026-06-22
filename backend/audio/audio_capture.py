"""Non-blocking microphone capture for Lumos.

The main webcam/FSM loop must never wait on the audio device. This module wraps
``sounddevice.InputStream`` in a small callback surface and exposes only
"latest block" reads guarded by a lock. If ``sounddevice`` or the microphone is
unavailable, startup logs a clear warning and the backend continues without
crashing.
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
    If DOA requested stereo but the device only opens as mono, VAD still runs and
    DOA reports unavailable instead of disabling the entire audio path.
    """

    def __init__(self, config: AudioConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.enabled = bool(config.enabled)
        self.available = False
        self.disabled_reason = "audio_disabled" if not self.enabled else "not_started"
        self.actual_channels = 0
        self.actual_sample_rate = int(config.sample_rate)

        self._sounddevice = None
        self._stream = None
        self._lock = threading.Lock()
        self._latest: Optional[AudioChunk] = None
        self._sequence = 0
        self._last_callback_at = 0.0

    @staticmethod
    def list_input_devices() -> tuple[bool, str]:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            return False, f"sounddevice is not installed: {exc}\nInstall it with: python -m pip install sounddevice>=0.4.6"
        try:
            devices = sd.query_devices()
            lines = ["Input audio devices visible to sounddevice:"]
            for idx, device in enumerate(devices):
                max_inputs = int(device.get("max_input_channels", 0))
                if max_inputs <= 0:
                    continue
                lines.append(
                    f"  [{idx}] {device.get('name', 'unknown')} | inputs={max_inputs} | default_sr={device.get('default_samplerate', 'unknown')}"
                )
            if len(lines) == 1:
                lines.append("  No input devices reported by PortAudio/sounddevice.")
            return True, "\n".join(lines)
        except Exception as exc:
            return False, f"Failed to list audio devices: {exc}"

    def start(self) -> bool:
        if not self.enabled:
            self.disabled_reason = "audio_disabled"
            self.logger.info("speaker_awareness_disabled_reason reason=%s", self.disabled_reason)
            return False

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            self.disabled_reason = f"sounddevice_unavailable: {exc}"
            self.logger.warning(
                "speaker_awareness_disabled_reason reason=%s install_hint=%s",
                self.disabled_reason,
                "python -m pip install sounddevice>=0.4.6",
            )
            return False

        self._sounddevice = sd
        requested_channels = 2 if self.config.request_stereo else 1
        requested_sample_rate = int(self.config.sample_rate)
        default_sample_rate = requested_sample_rate
        try:
            device_info = sd.query_devices(self.config.device if self.config.device not in (None, "") else None, kind="input")
            default_sample_rate = int(float(device_info.get("default_samplerate", requested_sample_rate)))
        except Exception:
            pass

        open_attempts: list[tuple[int, int, str]] = [(requested_channels, requested_sample_rate, "requested")]
        if requested_channels > 1:
            open_attempts.append((1, requested_sample_rate, "mono_fallback_for_vad"))
        if default_sample_rate != requested_sample_rate:
            open_attempts.append((requested_channels, default_sample_rate, "default_sample_rate"))
            if requested_channels > 1:
                open_attempts.append((1, default_sample_rate, "mono_default_sample_rate"))

        errors: list[str] = []
        for channels, sample_rate, reason in open_attempts:
            if self._try_open_stream(sd, channels, sample_rate, reason):
                return True
            errors.append(f"{reason}: channels={channels} sample_rate={sample_rate} error={self.disabled_reason}")

        self.disabled_reason = "microphone_open_failed: " + " | ".join(errors[-3:])
        self.logger.warning("speaker_awareness_disabled_reason reason=%s", self.disabled_reason)
        self.available = False
        return False

    def _try_open_stream(self, sd, channels: int, sample_rate: int, reason: str) -> bool:
        blocksize = max(1, int(round(sample_rate * self.config.block_ms / 1000.0)))
        try:
            self._stream = sd.InputStream(
                device=self.config.device if self.config.device not in (None, "") else None,
                samplerate=int(sample_rate),
                blocksize=blocksize,
                channels=int(channels),
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:  # pragma: no cover - depends on local hardware
            self.disabled_reason = str(exc)
            self._stream = None
            return False

        self.available = True
        self.disabled_reason = ""
        self.actual_channels = int(channels)
        self.actual_sample_rate = int(sample_rate)
        self.logger.info(
            "audio_capture_started device=%s sample_rate=%s block_ms=%s channels=%s blocksize=%s open_reason=%s doa_possible=%s",
            self.config.device or "default",
            sample_rate,
            self.config.block_ms,
            channels,
            blocksize,
            reason,
            bool(channels >= 2),
        )
        if self.config.request_stereo and channels < 2:
            self.logger.warning(
                "doa_disabled_reason=stereo_input_unavailable_using_mono_vad channels=%s",
                channels,
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
                    sample_rate=int(self.actual_sample_rate),
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
