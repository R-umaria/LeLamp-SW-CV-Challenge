"""Shared labels for engagement reliability evaluation."""

from __future__ import annotations

VALID_ENGAGEMENT_LABELS = (
    "engaged",
    "looking_left",
    "looking_right",
    "looking_down",
    "looking_away",
    "no_person",
    "partial_face",
)

LABEL_TO_STATUS = {
    "engaged": "engaged",
    "looking_left": "disengaged",
    "looking_right": "disengaged",
    "looking_down": "disengaged",
    "looking_away": "disengaged",
    "no_person": "absent",
    "partial_face": "disengaged",
}
