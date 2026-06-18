"""Finite state machine for lamp interaction state."""

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


@dataclass(frozen=True)
class StateTransition:
    previous_state: LampState
    current_state: LampState
    changed: bool
    reason: str
    state_elapsed_s: float


class InteractionStateMachine:
    def __init__(self, config: StateMachineConfig) -> None:
        self.config = config
        self.state = LampState.IDLE
        self._state_started_at = time.monotonic()
        self._disengaged_started_at: Optional[float] = None
        self._absent_started_at: Optional[float] = None

    def update(self, engagement: EngagementResult, now: Optional[float] = None) -> StateTransition:
        now = now if now is not None else time.monotonic()
        previous = self.state
        target = self.state
        reason = engagement.reason

        if engagement.status == "engaged":
            self._disengaged_started_at = None
            self._absent_started_at = None
            target = LampState.ENGAGED
            reason = "engaged_face_centered"

        elif engagement.status == "disengaged":
            self._absent_started_at = None
            target, reason = self._handle_disengaged(now, reason)

        elif engagement.status == "absent":
            if self._absent_started_at is None:
                self._absent_started_at = now

            absent_elapsed = now - self._absent_started_at
            if self.state == LampState.IDLE or absent_elapsed >= self.config.absent_grace_s:
                self._disengaged_started_at = None
                target = LampState.IDLE
                reason = "no_face_idle"
            else:
                # Brief disappearance after engagement behaves like disengagement to avoid flicker.
                target, reason = self._handle_disengaged(now, "brief_no_face_after_active_state")

        if target != previous:
            self.state = target
            self._state_started_at = now

        return StateTransition(
            previous_state=previous,
            current_state=self.state,
            changed=(target != previous),
            reason=reason,
            state_elapsed_s=now - self._state_started_at,
        )

    def _handle_disengaged(self, now: float, reason: str) -> tuple[LampState, str]:
        if self._disengaged_started_at is None:
            self._disengaged_started_at = now

        disengaged_elapsed = now - self._disengaged_started_at
        if disengaged_elapsed >= self.config.seek_attention_after_s:
            return LampState.SEEKING_ATTENTION, f"disengaged_for_{disengaged_elapsed:.1f}s"
        return LampState.DISENGAGED, reason
