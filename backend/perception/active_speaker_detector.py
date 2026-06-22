"""Audio-visual active speaker fusion for Lumos Milestone 4.11."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from backend.audio.gcc_phat import DirectionOfArrivalResult
from backend.audio.voice_activity_detector import VoiceActivityResult
from backend.perception.engagement_detector import EngagementResult
from backend.perception.face_tracker import FaceTrack
from backend.utils.config import SpeakerAwarenessConfig


@dataclass(frozen=True)
class ActiveSpeakerResult:
    timestamp: float
    speech_detected: bool
    active_track_id: str | None
    active_track_location: str | None
    speaking_to_robot: bool | None
    confidence: float
    reason: str
    mouth_motion_score: float | None
    audio_confidence: float
    doa_azimuth_deg: float | None
    doa_confidence: float | None

    def to_protocol_dict(self) -> dict:
        return {
            "speech_detected": bool(self.speech_detected),
            "active_track_id": self.active_track_id,
            "active_track_location": self.active_track_location,
            "speaking_to_robot": self.speaking_to_robot,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "mouth_motion_score": None if self.mouth_motion_score is None else round(float(self.mouth_motion_score), 3),
            "audio_confidence": round(float(self.audio_confidence), 3),
            "doa_azimuth_deg": None if self.doa_azimuth_deg is None else round(float(self.doa_azimuth_deg), 2),
            "doa_confidence": None if self.doa_confidence is None else round(float(self.doa_confidence), 3),
        }

    def to_log_dict(self) -> dict:
        payload = self.to_protocol_dict()
        payload["timestamp"] = round(float(self.timestamp), 3)
        return payload


class ActiveSpeakerDetector:
    def __init__(self, config: SpeakerAwarenessConfig) -> None:
        self.config = config
        self._last_result = ActiveSpeakerResult(
            timestamp=0.0,
            speech_detected=False,
            active_track_id=None,
            active_track_location=None,
            speaking_to_robot=None,
            confidence=0.0,
            reason="speaker_awareness_disabled" if not config.enabled else "no_audio_yet",
            mouth_motion_score=None,
            audio_confidence=0.0,
            doa_azimuth_deg=None,
            doa_confidence=None,
        )

    @property
    def last_result(self) -> ActiveSpeakerResult:
        return self._last_result

    def update(
        self,
        face_tracks: Iterable[FaceTrack],
        voice: VoiceActivityResult | None,
        doa: DirectionOfArrivalResult | None,
        engagement: EngagementResult | None = None,
        *,
        now: float | None = None,
    ) -> ActiveSpeakerResult:
        timestamp = time.time() if now is None else float(now)
        result = fuse_active_speaker(
            face_tracks=list(face_tracks),
            voice=voice,
            doa=doa,
            engagement=engagement,
            config=self.config,
            timestamp=timestamp,
        )
        self._last_result = result
        return result


def fuse_active_speaker(
    face_tracks: list[FaceTrack],
    voice: VoiceActivityResult | None,
    doa: DirectionOfArrivalResult | None,
    engagement: EngagementResult | None,
    config: SpeakerAwarenessConfig,
    timestamp: float | None = None,
) -> ActiveSpeakerResult:
    ts = time.time() if timestamp is None else float(timestamp)
    doa_azimuth = doa.azimuth_deg if doa is not None and doa.available else None
    doa_confidence = doa.confidence if doa is not None else None

    if not config.enabled:
        return ActiveSpeakerResult(ts, False, None, None, None, 0.0, "speaker_awareness_disabled", None, 0.0, doa_azimuth, doa_confidence)
    if voice is None:
        return ActiveSpeakerResult(ts, False, None, None, None, 0.0, "no_voice_activity_result", None, 0.0, doa_azimuth, doa_confidence)
    if not voice.is_speech:
        return ActiveSpeakerResult(ts, False, None, None, None, min(0.35, voice.confidence), "audio_inactive", None, voice.confidence, doa_azimuth, doa_confidence)

    mouth_threshold = max(0.0, float(config.mouth_motion_threshold))
    candidates = [track for track in face_tracks if track.mouth_motion_score >= mouth_threshold]
    candidates.sort(key=lambda track: track.mouth_motion_score, reverse=True)

    if not face_tracks:
        confidence = min(0.82, 0.52 * voice.confidence + (0.18 * (doa.confidence if doa is not None and doa.available else 0.0)))
        reason = "audio_active_no_visible_face_with_doa" if doa_azimuth is not None else "audio_active_no_visible_face"
        return ActiveSpeakerResult(ts, True, None, None, None, confidence, reason, None, voice.confidence, doa_azimuth, doa_confidence)

    if not candidates:
        confidence = min(0.68, 0.50 * voice.confidence + (0.12 * (doa.confidence if doa is not None and doa.available else 0.0)))
        reason = "audio_active_no_mouth_motion"
        return ActiveSpeakerResult(ts, True, None, None, None, confidence, reason, None, voice.confidence, doa_azimuth, doa_confidence)

    best = candidates[0]
    ambiguous = len(candidates) > 1 and (best.mouth_motion_score - candidates[1].mouth_motion_score) <= config.ambiguous_margin
    visual_engaged = best.engaged
    if visual_engaged is None and engagement is not None:
        visual_engaged = engagement.status == "engaged"

    engaged_bonus = 0.16 if visual_engaged else 0.0
    doa_bonus = 0.05 * (doa.confidence if doa is not None and doa.available else 0.0)
    confidence = 0.44 * voice.confidence + 0.35 * min(1.0, best.mouth_motion_score) + engaged_bonus + doa_bonus
    if ambiguous:
        confidence = max(0.0, confidence - 0.16)
    confidence = min(0.95, confidence)

    if ambiguous:
        reason = "audio_active_multiple_mouth_motion_ambiguous"
    elif visual_engaged is True:
        reason = "audio_active_mouth_motion_engaged"
    elif visual_engaged is False:
        reason = "audio_active_mouth_motion_looking_elsewhere"
    else:
        reason = "audio_active_mouth_motion_unknown_engagement"

    speaking_to_robot: bool | None
    if visual_engaged is True:
        speaking_to_robot = True
    elif visual_engaged is False:
        speaking_to_robot = False
    else:
        speaking_to_robot = None

    return ActiveSpeakerResult(
        timestamp=ts,
        speech_detected=True,
        active_track_id=best.track_id,
        active_track_location=best.location,
        speaking_to_robot=speaking_to_robot,
        confidence=confidence,
        reason=reason,
        mouth_motion_score=best.mouth_motion_score,
        audio_confidence=voice.confidence,
        doa_azimuth_deg=doa_azimuth,
        doa_confidence=doa_confidence,
    )
