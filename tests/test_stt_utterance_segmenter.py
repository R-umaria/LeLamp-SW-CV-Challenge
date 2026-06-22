import numpy as np

from backend.audio.audio_capture import AudioChunk
from backend.audio.speech_to_text import UtteranceSegmenter, LocalSpeechToTextEngine
from backend.audio.voice_activity_detector import VoiceActivityResult
from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.utils.config import SpeechToTextConfig


def _chunk(seq: int, ts: float, active: bool = True) -> tuple[AudioChunk, VoiceActivityResult]:
    samples = np.ones((480, 1), dtype=np.float32) * (0.05 if active else 0.001)
    chunk = AudioChunk(timestamp=ts, samples=samples, sample_rate=16000, channels=1, sequence=seq)
    voice = VoiceActivityResult(
        timestamp=ts,
        is_speech=active,
        confidence=0.8 if active else 0.0,
        rms=0.05 if active else 0.001,
        noise_floor=0.004,
        sample_rate=16000,
        channels=1,
    )
    return chunk, voice


def _speaker(conf: float = 0.7, to_robot: bool = True) -> ActiveSpeakerResult:
    return ActiveSpeakerResult(
        timestamp=0.0,
        speech_detected=True,
        active_track_id="person_1",
        active_track_location="center",
        speaking_to_robot=to_robot,
        confidence=conf,
        reason="audio_active_mouth_motion_engaged",
        mouth_motion_score=0.3,
        audio_confidence=0.8,
        doa_azimuth_deg=None,
        doa_confidence=None,
    )


def test_segmenter_requires_speaker_talking_to_lumos():
    seg = UtteranceSegmenter(SpeechToTextConfig(enabled=True, min_utterance_s=0.05, end_silence_s=0.05))
    for i in range(5):
        chunk, voice = _chunk(i, ts=i * 0.03, active=True)
        assert seg.update(chunk, voice, _speaker(to_robot=False), now=i * 0.03) is None
    assert not seg.recording


def test_segmenter_emits_after_silence_with_preroll():
    cfg = SpeechToTextConfig(enabled=True, min_utterance_s=0.05, end_silence_s=0.05, cooldown_s=0.0)
    seg = UtteranceSegmenter(cfg)

    for i in range(4):
        chunk, voice = _chunk(i, ts=i * 0.03, active=True)
        assert seg.update(chunk, voice, _speaker(), now=i * 0.03) is None

    # Silence after the minimum duration finalizes the utterance.
    utterance = None
    for i in range(4, 8):
        chunk, voice = _chunk(i, ts=i * 0.03, active=False)
        utterance = seg.update(chunk, voice, _speaker(), now=i * 0.03)
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.speaker_track_id == "person_1"
    assert utterance.duration_s >= 0.05
    assert utterance.samples.ndim == 2


def test_stt_acceptance_filters_tiny_fragments():
    engine = LocalSpeechToTextEngine(SpeechToTextConfig(enabled=True, min_words=2, min_chars=5))
    assert engine._accept_text("") == (False, "empty_transcript")
    assert engine._accept_text("hi") == (False, "too_short")
    assert engine._accept_text("phone") == (False, "too_few_words")
    assert engine._accept_text("did you see my phone") == (True, "accepted")
