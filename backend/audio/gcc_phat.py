"""Pure NumPy GCC-PHAT direction-of-arrival estimate for two microphones."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionOfArrivalResult:
    timestamp: float
    available: bool
    azimuth_deg: float | None
    confidence: float
    reason: str

    def to_log_dict(self) -> dict:
        return {
            "timestamp": round(float(self.timestamp), 3),
            "available": bool(self.available),
            "azimuth_deg": None if self.azimuth_deg is None else round(float(self.azimuth_deg), 2),
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }


def gcc_phat_delay(left: np.ndarray, right: np.ndarray, sample_rate: int, max_tau: float | None = None) -> tuple[float, float]:
    """Return delay seconds and a simple peak-prominence confidence.

    Positive delay means the left channel lags the right channel. Lumos maps that
    to a positive/right-side azimuth because a sound source on the right reaches
    the right microphone first.
    """

    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    n = int(left.size + right.size)
    if n <= 2 or sample_rate <= 0:
        return 0.0, 0.0

    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    nfft = 1 << (n - 1).bit_length()
    left_fft = np.fft.rfft(left, n=nfft)
    right_fft = np.fft.rfft(right, n=nfft)
    cross_power = left_fft * np.conj(right_fft)
    denom = np.abs(cross_power)
    cross_power = cross_power / np.maximum(denom, 1e-12)
    cc = np.fft.irfft(cross_power, n=nfft)

    max_shift = nfft // 2
    if max_tau is not None:
        max_shift = min(max_shift, int(round(float(max_tau) * float(sample_rate))))
    max_shift = max(1, max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    peak_index = int(np.argmax(np.abs(cc)))
    shift = peak_index - max_shift
    delay_s = float(shift) / float(sample_rate)

    abs_cc = np.abs(cc)
    peak = float(abs_cc[peak_index])
    median = float(np.median(abs_cc)) + 1e-12
    # Peak prominence compressed to [0, 1]. Synthetic clean signals approach 1;
    # diffuse/noisy signals remain low and are rejected by the caller.
    confidence = max(0.0, min(1.0, (peak / median - 1.0) / 9.0))
    return delay_s, confidence


def estimate_direction_of_arrival(
    samples: np.ndarray,
    sample_rate: int,
    mic_distance_m: float,
    *,
    timestamp: float = 0.0,
    speed_of_sound_m_s: float = 343.0,
    min_rms: float = 1e-4,
    min_confidence: float = 0.12,
) -> DirectionOfArrivalResult:
    """Estimate bounded left/right azimuth using two-channel GCC-PHAT."""

    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        return DirectionOfArrivalResult(timestamp, False, None, 0.0, "mono_or_missing_stereo")
    if sample_rate <= 0:
        return DirectionOfArrivalResult(timestamp, False, None, 0.0, "invalid_sample_rate")
    if mic_distance_m <= 0.0:
        return DirectionOfArrivalResult(timestamp, False, None, 0.0, "invalid_mic_distance")

    left = data[:, 0]
    right = data[:, 1]
    rms = float(np.sqrt(np.mean(np.square(data), dtype=np.float64))) if data.size else 0.0
    if rms < min_rms:
        return DirectionOfArrivalResult(timestamp, False, None, 0.0, "signal_too_weak")

    max_tau = float(mic_distance_m) / float(speed_of_sound_m_s)
    delay_s, confidence = gcc_phat_delay(left, right, sample_rate, max_tau=max_tau)
    if confidence < min_confidence:
        return DirectionOfArrivalResult(timestamp, False, None, confidence, "low_correlation_confidence")

    ratio = (float(speed_of_sound_m_s) * delay_s) / float(mic_distance_m)
    ratio = max(-1.0, min(1.0, ratio))
    azimuth_deg = math.degrees(math.asin(ratio))
    azimuth_deg = max(-90.0, min(90.0, azimuth_deg))
    return DirectionOfArrivalResult(timestamp, True, float(azimuth_deg), float(confidence), "gcc_phat")
