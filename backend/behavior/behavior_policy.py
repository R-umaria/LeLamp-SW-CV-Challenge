"""Bounded mapping from interaction state transitions to expressive Lumos behavior.

The backend still owns behavior selection. The Godot frontend only renders the
named motion/light skills it receives through the stable command protocol.
"""

from __future__ import annotations

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
    """Return an expressive behavior using only bounded state-transition context.

    This adds personality without letting an LLM or the frontend make autonomous
    motion decisions. Timing windows are deliberately simple and explainable.
    """

    state = transition.current_state
    behavior = behavior_for_state(state)

    if state == LampState.ENGAGED:
        # Eye contact moment: Lumos briefly stands taller and turns pink, then
        # falls back to calm attentive following after the first few seconds.
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
        # When no one is present, do a slow room check before returning to the
        # user-authored sleep pose. This keeps the demo expressive but bounded.
        if transition.state_elapsed_s < 4.2:
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
