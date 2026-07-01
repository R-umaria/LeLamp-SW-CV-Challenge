from backend.behavior.behavior_policy import behavior_for_state
from backend.behavior.speaker_policy import apply_speaker_policy
from backend.behavior.state_machine import LampState
from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.utils.config import SpeakerAwarenessConfig


def speaker(**kwargs):
    data = dict(
        timestamp=1.0,
        speech_detected=True,
        active_track_id="person_2",
        active_track_location="right",
        speaking_to_robot=False,
        confidence=0.8,
        reason="test",
        mouth_motion_score=0.5,
        audio_confidence=0.8,
        doa_azimuth_deg=None,
        doa_confidence=None,
    )
    data.update(kwargs)
    return ActiveSpeakerResult(**data)


def test_attention_seeking_is_suppressed_when_other_person_speaks():
    config = SpeakerAwarenessConfig(enabled=True, policy_min_confidence=0.55)
    base = behavior_for_state(LampState.SEEKING_ATTENTION)
    decision = apply_speaker_policy(base, speaker(), LampState.SEEKING_ATTENTION, config)
    assert decision.overridden is True
    assert decision.do_not_interrupt is True
    assert decision.quiet_listening is True
    assert decision.safe_to_respond is False
    assert decision.behavior["sound"] is None
    assert decision.behavior["motion"] == "listening_attentive"
