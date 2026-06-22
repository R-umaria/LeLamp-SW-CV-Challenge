import time

import numpy as np

from backend.audio.audio_capture import AudioChunk
from backend.audio.voice_activity_detector import VoiceActivityDetector
from backend.utils.config import AudioConfig


def _chunk(samples, sequence=1, sample_rate=16000):
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return AudioChunk(timestamp=time.time(), samples=arr, sample_rate=sample_rate, channels=arr.shape[1], sequence=sequence)


def test_vad_inactive_for_silence():
    config = AudioConfig(vad_energy_threshold=2.0, vad_smoothing_blocks=3, vad_absolute_threshold=0.01)
    vad = VoiceActivityDetector(config)
    result = None
    for idx in range(5):
        result = vad.update(_chunk(np.zeros(480), sequence=idx))
    assert result is not None
    assert result.is_speech is False
    assert result.confidence <= 0.42


def test_vad_active_for_high_energy_signal():
    config = AudioConfig(vad_energy_threshold=1.5, vad_smoothing_blocks=3, vad_absolute_threshold=0.01)
    vad = VoiceActivityDetector(config)
    t = np.arange(480) / 16000.0
    signal = 0.12 * np.sin(2 * np.pi * 220 * t)
    result = None
    for idx in range(5):
        result = vad.update(_chunk(signal, sequence=idx))
    assert result is not None
    assert result.is_speech is True
    assert result.confidence > 0.25
