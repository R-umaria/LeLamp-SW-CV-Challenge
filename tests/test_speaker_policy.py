from backend.behavior.behavior_policy import behavior_for_state
from backend.behavior.speaker_policy import apply_speaker_policy
from backend.behavior.state_machine import LampState
from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.utils.config import SpeakerAwarenessConfig


def speaker(**kwargs):
    data = dict(
        timestamp=1.0,
        speech_detected=True,
        active_track_id="person_1",
        active_track_location="center",
        speaking_to_robot=True,
        confidence=0.8,
        reason="test",
        mouth_motion_score=0.5,
        audio_confidence=0.8,
        doa_azimuth_deg=None,
        doa_confidence=None,
    )
    data.update(kwargs)
    return ActiveSpeakerResult(**data)


def test_policy_does_not_override_low_confidence():
    config = SpeakerAwarenessConfig(enabled=True, policy_min_confidence=0.55)
    base = behavior_for_state(LampState.SEEKING_ATTENTION)
    decision = apply_speaker_policy(base, speaker(confidence=0.2), LampState.SEEKING_ATTENTION, config)
    assert decision.overridden is False
    assert decision.behavior == base


def test_policy_listens_when_user_talks_to_lumos():
    config = SpeakerAwarenessConfig(enabled=True, policy_min_confidence=0.55)
    base = behavior_for_state(LampState.ENGAGED)
    decision = apply_speaker_policy(base, speaker(speaking_to_robot=True), LampState.ENGAGED, config)
    assert decision.overridden is True
    assert decision.behavior["motion"] == "active_listen"
    assert decision.behavior["light"] == "listening_blue"


def test_policy_uses_bounded_sound_seek_when_no_face_with_doa():
    config = SpeakerAwarenessConfig(enabled=True, policy_min_confidence=0.35)
    base = behavior_for_state(LampState.IDLE)
    result = speaker(active_track_id=None, active_track_location=None, speaking_to_robot=None, confidence=0.6, doa_azimuth_deg=25.0)
    decision = apply_speaker_policy(base, result, LampState.IDLE, config)
    assert decision.overridden is True
    assert decision.behavior["motion"] == "sound_seek_right"


def test_gesture_policy_maps_new_emotion_controls():
    from backend.behavior.behavior_policy import behavior_with_gesture_override

    base = behavior_for_state(LampState.ENGAGED)

    thumbs = behavior_with_gesture_override(base, {"status": "thumbs_up", "confidence": 0.86})
    assert thumbs["motion"] == "gesture_thumbs_up"
    assert thumbs["light"] == "happy_gold"

    pinch = behavior_with_gesture_override(base, {"status": "pinch_follow", "confidence": 0.86, "target_x_norm": 0.2, "target_y_norm": 0.4})
    assert pinch["motion"] == "gesture_pinch_follow"
    assert pinch["light"] == "focus_glow"

    heart = behavior_with_gesture_override(base, {"status": "heart", "confidence": 0.86})
    assert heart["motion"] == "gesture_heart_blush"
    assert heart["light"] == "baby_pink_blush"
