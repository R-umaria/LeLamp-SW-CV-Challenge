"""Finite state machine for lamp interaction state.

Milestone 1.5 adds hysteresis and dwell-time gating. The FSM consumes the
smoothed engagement signal, not the raw per-frame detector output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.perception.engagement_detector import EngagementResult
from backend.utils.config import StateMachineConfig


class LampState(str, Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"
    SEEKING_ATTENTION = "seeking_attention"
    RECALLING = "recalling"


@dataclass(frozen=True)
class StateTransition:
    previous_state: LampState
    current_state: LampState
    changed: bool
    reason: str
    state_elapsed_s: float
    consecutive_engaged: int
    consecutive_disengaged: int
    consecutive_absent: int


class InteractionStateMachine:
    def __init__(self, config: StateMachineConfig) -> None:
        self.config = config
        self.state = LampState.IDLE
        self._state_started_at = time.monotonic()
        self._disengaged_started_at: Optional[float] = None
        self._consecutive_engaged = 0
        self._consecutive_disengaged = 0
        self._consecutive_absent = 0

    def update(self, engagement: EngagementResult, now: Optional[float] = None) -> StateTransition:
        now = now if now is not None else time.monotonic()
        previous = self.state
        target = self.state
        reason = engagement.reason

        self._update_consecutive_counts(engagement.status)
        dwell_s = now - self._state_started_at
        clear_engaged_recovery = self._is_clear_engaged_recovery(engagement)

        if engagement.status == "engaged":
            self._disengaged_started_at = None
            if clear_engaged_recovery:
                target = LampState.ENGAGED
                reason = "clear_engaged_recovery"
            else:
                reason = "holding_until_engaged_stable"

        elif engagement.status == "disengaged":
            if self._disengaged_started_at is None:
                self._disengaged_started_at = now

            if self.state == LampState.ENGAGED:
                if self._consecutive_disengaged >= self.config.exit_engaged_disengaged_frames:
                    target = LampState.DISENGAGED
                    reason = f"stable_disengaged_{self._consecutive_disengaged}_frames"
                else:
                    reason = f"hold_engaged_disengaged_count_{self._consecutive_disengaged}"
            elif self.state in (LampState.IDLE, LampState.DISENGAGED):
                target = LampState.DISENGAGED
                reason = engagement.reason
            elif self.state == LampState.SEEKING_ATTENTION:
                target = LampState.SEEKING_ATTENTION
                reason = "continue_seeking_attention"

            disengaged_elapsed = now - self._disengaged_started_at
            if target == LampState.DISENGAGED and disengaged_elapsed >= self.config.seek_attention_after_s:
                target = LampState.SEEKING_ATTENTION
                reason = f"stable_disengaged_for_{disengaged_elapsed:.1f}s"

        elif engagement.status == "absent":
            if self.state == LampState.ENGAGED:
                if self._consecutive_absent >= self.config.exit_engaged_absent_frames:
                    target = LampState.DISENGAGED
                    reason = f"stable_absent_{self._consecutive_absent}_frames_after_engaged"
                else:
                    reason = f"hold_engaged_absent_count_{self._consecutive_absent}"
            elif self.state in (LampState.DISENGAGED, LampState.SEEKING_ATTENTION):
                if self._consecutive_absent >= self.config.absent_to_idle_frames:
                    self._disengaged_started_at = None
                    target = LampState.IDLE
                    reason = f"stable_absent_{self._consecutive_absent}_frames_idle"
                else:
                    target = self.state
                    reason = f"hold_active_absent_count_{self._consecutive_absent}"
            else:
                target = LampState.IDLE
                reason = "no_face_idle"

        if target != previous and not clear_engaged_recovery and dwell_s < self.config.min_state_dwell_s:
            target = previous
            reason = f"min_dwell_hold_{dwell_s:.2f}s"

        changed = target != previous
        if changed:
            self.state = target
            self._state_started_at = now
            dwell_s = 0.0
        else:
            dwell_s = now - self._state_started_at

        return StateTransition(
            previous_state=previous,
            current_state=self.state,
            changed=changed,
            reason=reason,
            state_elapsed_s=dwell_s,
            consecutive_engaged=self._consecutive_engaged,
            consecutive_disengaged=self._consecutive_disengaged,
            consecutive_absent=self._consecutive_absent,
        )

    def _update_consecutive_counts(self, status: str) -> None:
        if status == "engaged":
            self._consecutive_engaged += 1
            self._consecutive_disengaged = 0
            self._consecutive_absent = 0
        elif status == "disengaged":
            self._consecutive_engaged = 0
            self._consecutive_disengaged += 1
            self._consecutive_absent = 0
        elif status == "absent":
            self._consecutive_engaged = 0
            self._consecutive_disengaged = 0
            self._consecutive_absent += 1
        else:
            self._consecutive_engaged = 0
            self._consecutive_disengaged = 0
            self._consecutive_absent = 0

    def _is_clear_engaged_recovery(self, engagement: EngagementResult) -> bool:
        return (
            engagement.status == "engaged"
            and engagement.confidence >= self.config.clear_engaged_confidence
            and self._consecutive_engaged >= self.config.engaged_recovery_frames
        )
