"""Backend-to-frontend command builder.

The JSON shape stays bounded and deterministic: Python owns perception, state,
behavior selection, memory, and recall; Godot only renders the named motion/light
skills and optional display hints.
"""

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
    gesture: Optional[Mapping] = None,
    recall_target: Optional[Mapping] = None,
) -> dict:
    command = {
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
            "last_detected_objects": list(last_detected_objects or []),
        },
    }
    if recall_target is not None:
        command["memory"]["recall_target"] = dict(recall_target)
    if gesture is not None:
        command["gesture"] = dict(gesture)
    return command
