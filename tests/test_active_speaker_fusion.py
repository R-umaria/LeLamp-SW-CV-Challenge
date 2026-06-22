from backend.audio.gcc_phat import DirectionOfArrivalResult
from backend.audio.voice_activity_detector import VoiceActivityResult
from backend.perception.active_speaker_detector import fuse_active_speaker
from backend.perception.face_tracker import FaceTrack
from backend.utils.config import SpeakerAwarenessConfig


def voice(active=True, confidence=0.85):
    return VoiceActivityResult(1.0, active, confidence, 0.05, 0.005, 16000, 1)


def track(track_id="person_1", mouth=0.55, engaged=True, location="center"):
    return FaceTrack(track_id, (100, 100, 120, 120), (0.5, 0.5), location, mouth, 0.1, engaged, 0.8, "test", 0.5, 1.0)


def config():
    return SpeakerAwarenessConfig(enabled=True, mouth_motion_threshold=0.18)


def test_audio_active_one_mouth_moving_marks_active_speaker():
    result = fuse_active_speaker([track()], voice(True), None, None, config(), timestamp=1.0)
    assert result.speech_detected is True
    assert result.active_track_id == "person_1"
    assert result.speaking_to_robot is True
    assert result.reason == "audio_active_mouth_motion_engaged"


def test_audio_active_no_face_uses_doa_without_speaker_id():
    doa = DirectionOfArrivalResult(1.0, True, 22.0, 0.7, "gcc_phat")
    result = fuse_active_speaker([], voice(True), doa, None, config(), timestamp=1.0)
    assert result.speech_detected is True
    assert result.active_track_id is None
    assert result.doa_azimuth_deg == 22.0
    assert "no_visible_face" in result.reason


def test_audio_inactive_overrides_mouth_motion():
    result = fuse_active_speaker([track(mouth=0.9)], voice(False, 0.1), None, None, config(), timestamp=1.0)
    assert result.speech_detected is False
    assert result.active_track_id is None
    assert result.reason == "audio_inactive"


def test_multiple_speakers_lowers_confidence_and_marks_ambiguous():
    t1 = track("person_1", mouth=0.50, engaged=True)
    t2 = track("person_2", mouth=0.45, engaged=False, location="right")
    result = fuse_active_speaker([t1, t2], voice(True), None, None, config(), timestamp=1.0)
    assert result.active_track_id == "person_1"
    assert "ambiguous" in result.reason
    assert result.confidence < 0.8
