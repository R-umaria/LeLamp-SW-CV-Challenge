"""Bounded mapping from perception/state to expressive Lumos behavior.

The backend selects named skills; Godot only renders those skills. This keeps the
real-time behavior deterministic and prevents the LLM or frontend from directly
controlling motion.
"""

from __future__ import annotations

from typing import Mapping

from backend.behavior.state_machine import LampState, StateTransition


STEADY_BEHAVIOR_BY_STATE: dict[LampState, dict] = {
    LampState.IDLE: {
        "motion": "idle_breathe",
        "light": "dim_warm",
        "sound": None,
        "speech_text": None,
    },
    LampState.ENGAGED: {
        "motion": "attentive_follow",
        "light": "steady_warm",
        "sound": None,
        "speech_text": None,
    },
    LampState.DISENGAGED: {
        "motion": "searching_glance_slow",
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
        "motion": "thinking_slow",
        "light": "focus_glow",
        "sound": None,
        "speech_text": None,
    },
}


def behavior_for_state(state: LampState) -> dict:
    """Compatibility wrapper for older call sites and tests."""

    return dict(STEADY_BEHAVIOR_BY_STATE[state])


def behavior_for_transition(transition: StateTransition) -> dict:
    """Return the default expressive behavior for the current FSM state."""

    state = transition.current_state
    behavior = behavior_for_state(state)

    if state == LampState.ENGAGED:
        if transition.changed or transition.state_elapsed_s < 2.6:
            behavior.update(
                {
                    "motion": "excited_greeting",
                    "light": "excited_pink",
                    "sound": "happy_ping",
                    "speech_text": None,
                }
            )
        return behavior

    if state == LampState.IDLE:
        if transition.state_elapsed_s < 6.4:
            behavior.update(
                {
                    "motion": "sleepy_search_then_rest",
                    "light": "dim_warm",
                    "sound": None,
                    "speech_text": None,
                }
            )
        else:
            behavior.update(
                {
                    "motion": "sleep_rest",
                    "light": "sleep_red",
                    "sound": None,
                    "speech_text": None,
                }
            )
        return behavior

    if state == LampState.SEEKING_ATTENTION and transition.state_elapsed_s >= 4.0:
        behavior.update(
            {
                "motion": "gentle_wave",
                "light": "soft_pulse",
                "sound": "gentle_chime",
                "speech_text": None,
            }
        )
        return behavior

    return behavior


def behavior_with_gesture_override(base_behavior: Mapping, gesture: Mapping | None) -> dict:
    """Overlay deliberate hand-gesture control on top of normal behavior.

    Gesture override is intentionally narrow: a beckon gesture maps to one known
    approach skill, and an open palm maps to one known retreat skill. Memory,
    recall, and engagement state remain owned by the backend pipeline.
    """

    behavior = dict(base_behavior)
    if not gesture:
        return behavior

    status = str(gesture.get("status", "none"))
    confidence = float(gesture.get("confidence", 0.0) or 0.0)
    if confidence < 0.64:
        return behavior

    if status == "beckon":
        behavior.update(
            {
                "motion": "gesture_approach",
                "light": "happy_gold",
                "sound": None,
                "speech_text": None,
            }
        )
    elif status == "palm_push":
        behavior.update(
            {
                "motion": "gesture_retreat",
                "light": "soft_pulse",
                "sound": None,
                "speech_text": None,
            }
        )
    return behavior
