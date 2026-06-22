"""Lightweight RMS-energy voice activity detector for demo-safe speech cues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from backend.audio.audio_capture import AudioChunk
from backend.utils.config import AudioConfig


@dataclass(frozen=True)
class VoiceActivityResult:
    timestamp: float
    is_speech: bool
    confidence: float
    rms: float
    noise_floor: float
    sample_rate: int
    channels: int

    def to_log_dict(self) -> dict:
        return {
            "timestamp": round(float(self.timestamp), 3),
            "is_speech": bool(self.is_speech),
            "confidence": round(float(self.confidence), 3),
            "rms": round(float(self.rms), 6),
            "noise_floor": round(float(self.noise_floor), 6),
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
        }


class VoiceActivityDetector:
    """Rolling noise-floor VAD with temporal smoothing.

    The threshold is a multiplier over the learned noise floor, plus a small
    absolute floor so an initially silent room does not make numerical noise look
    like speech.
    """

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.noise_floor = max(float(config.vad_min_noise_floor), 1e-8)
        self._history: deque[bool] = deque(maxlen=max(1, int(config.vad_smoothing_blocks)))
        self._confidence_history: deque[float] = deque(maxlen=max(1, int(config.vad_smoothing_blocks)))
        self._last_result = VoiceActivityResult(
            timestamp=0.0,
            is_speech=False,
            confidence=0.0,
            rms=0.0,
            noise_floor=self.noise_floor,
            sample_rate=int(config.sample_rate),
            channels=1,
        )

    def update(self, chunk: AudioChunk | None) -> VoiceActivityResult:
        if chunk is None:
            return self._last_result

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.size == 0:
            result = VoiceActivityResult(chunk.timestamp, False, 0.0, 0.0, self.noise_floor, chunk.sample_rate, chunk.channels)
            self._last_result = result
            return result

        mono = samples.mean(axis=1) if samples.ndim == 2 else samples.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))

        threshold_multiplier = max(1.05, float(self.config.vad_energy_threshold))
        threshold = max(float(self.config.vad_absolute_threshold), self.noise_floor * threshold_multiplier)
        instantaneous_active = rms >= threshold

        if instantaneous_active:
            # During speech, adapt very slowly so speech does not become the floor.
            alpha = float(self.config.vad_noise_update_alpha_speech)
        else:
            alpha = float(self.config.vad_noise_update_alpha)
        self.noise_floor = max(
            float(self.config.vad_min_noise_floor),
            (1.0 - alpha) * self.noise_floor + alpha * rms,
        )

        self._history.append(bool(instantaneous_active))
        active_votes = sum(1 for item in self._history if item)
        required_votes = max(1, int(round(len(self._history) * float(self.config.vad_active_vote_ratio))))
        is_speech = active_votes >= required_votes

        ratio = rms / max(self.noise_floor * threshold_multiplier, float(self.config.vad_absolute_threshold), 1e-8)
        instantaneous_confidence = max(0.0, min(1.0, (ratio - 0.75) / 1.75))
        self._confidence_history.append(float(instantaneous_confidence))
        confidence = instantaneous_confidence
        if is_speech and self._confidence_history:
            # Speech decisions are smoothed over several 30 ms blocks, so carry
            # some recent confidence forward. This avoids confusing states such
            # as is_speech=True with audio_confidence=0.0 during syllable gaps.
            confidence = max(confidence, 0.70 * max(self._confidence_history))
        if not is_speech:
            confidence = min(confidence, 0.42)

        result = VoiceActivityResult(
            timestamp=float(chunk.timestamp),
            is_speech=bool(is_speech),
            confidence=float(confidence),
            rms=rms,
            noise_floor=float(self.noise_floor),
            sample_rate=int(chunk.sample_rate),
            channels=int(chunk.channels),
        )
        self._last_result = result
        return result
