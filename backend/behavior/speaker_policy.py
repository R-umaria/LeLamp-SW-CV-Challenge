"""Bounded behavior overrides for active-speaker awareness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from backend.behavior.state_machine import LampState
from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.utils.config import SpeakerAwarenessConfig


@dataclass(frozen=True)
class SpeakerPolicyDecision:
    behavior: dict
    overridden: bool
    reason: str


def apply_speaker_policy(
    base_behavior: Mapping,
    speaker: ActiveSpeakerResult | None,
    state: LampState,
    config: SpeakerAwarenessConfig,
) -> SpeakerPolicyDecision:
    behavior = dict(base_behavior)
    if not config.enabled:
        return SpeakerPolicyDecision(behavior, False, "speaker_awareness_disabled")
    if speaker is None:
        return SpeakerPolicyDecision(behavior, False, "no_speaker_result")
    if not speaker.speech_detected:
        return SpeakerPolicyDecision(behavior, False, "no_speech_active")
    if state == LampState.RECALLING:
        return SpeakerPolicyDecision(behavior, False, "recall_has_priority")
    if speaker.confidence < config.policy_min_confidence:
        return SpeakerPolicyDecision(behavior, False, "speaker_confidence_below_policy_threshold")

    if speaker.active_track_id is not None and speaker.speaking_to_robot is True:
        behavior.update(
            {
                "motion": "active_listen",
                "light": "listening_blue",
                "sound": None,
                "speech_text": None,
            }
        )
        return SpeakerPolicyDecision(behavior, True, "speaker_talking_to_lumos")

    if speaker.active_track_id is not None and speaker.speaking_to_robot is False:
        behavior.update(
            {
                "motion": "listening_attentive",
                "light": "speech_focus",
                "sound": None,
                "speech_text": None,
            }
        )
        return SpeakerPolicyDecision(behavior, True, "speaker_talking_elsewhere_do_not_interrupt")

    if speaker.active_track_id is None:
        azimuth = speaker.doa_azimuth_deg
        if azimuth is not None:
            if azimuth > 12.0:
                motion = "sound_seek_right"
            elif azimuth < -12.0:
                motion = "sound_seek_left"
            else:
                motion = "sound_seek_center"
            reason = "speech_no_face_with_doa"
        else:
            motion = "sound_seek_center"
            reason = "speech_no_face_no_doa"
        behavior.update(
            {
                "motion": motion,
                "light": "listening_blue",
                "sound": None,
                "speech_text": None,
            }
        )
        return SpeakerPolicyDecision(behavior, True, reason)

    return SpeakerPolicyDecision(behavior, False, "speaker_intent_unknown")
