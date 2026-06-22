import numpy as np

from backend.audio.gcc_phat import estimate_direction_of_arrival, gcc_phat_delay


def _delayed_pair(delay_samples: int, n: int = 2048):
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.3, n).astype(np.float32)
    left = np.zeros_like(base)
    right = np.zeros_like(base)
    if delay_samples >= 0:
        # Left lags right, interpreted as source on the right.
        left[delay_samples:] = base[:-delay_samples] if delay_samples else base
        right[:] = base
    else:
        d = abs(delay_samples)
        left[:] = base
        right[d:] = base[:-d]
    return np.stack([left, right], axis=1)


def test_gcc_phat_positive_delay_when_left_lags_right():
    samples = _delayed_pair(3)
    delay_s, confidence = gcc_phat_delay(samples[:, 0], samples[:, 1], 16000, max_tau=0.001)
    assert delay_s > 0
    assert confidence > 0.2


def test_doa_estimates_positive_and_negative_direction():
    right_source = _delayed_pair(3)
    left_source = _delayed_pair(-3)
    right = estimate_direction_of_arrival(right_source, 16000, 0.08, timestamp=1.0, min_confidence=0.05)
    left = estimate_direction_of_arrival(left_source, 16000, 0.08, timestamp=1.0, min_confidence=0.05)
    assert right.available is True
    assert left.available is True
    assert right.azimuth_deg is not None and right.azimuth_deg > 0
    assert left.azimuth_deg is not None and left.azimuth_deg < 0


def test_doa_unavailable_for_mono():
    mono = np.zeros((512, 1), dtype=np.float32)
    result = estimate_direction_of_arrival(mono, 16000, 0.08)
    assert result.available is False
    assert result.reason == "mono_or_missing_stereo"
