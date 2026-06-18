"""Bounded mapping from interaction state to expressive behavior command fields."""

from __future__ import annotations

from backend.behavior.state_machine import LampState


BEHAVIOR_BY_STATE: dict[LampState, dict] = {
    LampState.IDLE: {
        "motion": "idle_breathe",
        "light": "dim_warm",
        "sound": None,
        "speech_text": None,
    },
    LampState.ENGAGED: {
        "motion": "attentive_nod",
        "light": "steady_warm",
        "sound": None,
        "speech_text": None,
    },
    LampState.DISENGAGED: {
        "motion": "searching_glance",
        "light": "slow_pulse",
        "sound": None,
        "speech_text": None,
    },
    LampState.SEEKING_ATTENTION: {
        "motion": "curious_tilt",
        "light": "soft_pulse",
        "sound": "gentle_chime",
        "speech_text": None,
    },
    LampState.RECALLING: {
        "motion": "thinking",
        "light": "focus_glow",
        "sound": None,
        "speech_text": None,
    },
}


def behavior_for_state(state: LampState) -> dict:
    # Return a copy so callers can safely annotate without mutating the policy table.
    return dict(BEHAVIOR_BY_STATE[state])
