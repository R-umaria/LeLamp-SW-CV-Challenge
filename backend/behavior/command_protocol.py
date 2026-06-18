"""JSON command builder matching the backend/frontend protocol shape."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Optional

from backend.behavior.state_machine import LampState
from backend.perception.engagement_detector import EngagementResult


def build_behavior_command(
    state: LampState,
    engagement: EngagementResult,
    behavior: Mapping,
    last_detected_objects: Optional[Iterable[Mapping]] = None,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "state": state.value,
        "engagement": engagement.to_protocol_dict(),
        "behavior": {
            "motion": behavior.get("motion"),
            "light": behavior.get("light"),
            "sound": behavior.get("sound"),
            "speech_text": behavior.get("speech_text"),
        },
        "memory": {
            # This field is stable across Milestones 3-4 and is consumed by the Godot overlay.
            "last_detected_objects": list(last_detected_objects or []),
        },
    }
