"""Recall-result feedback mapping for Lumos.

The recall agent decides only whether a grounded memory exists. This module turns
that bounded result into a small frontend hint: point toward the remembered
screen region, or show a disappointed no-memory motion. The LLM never selects
motion names or coordinates.
"""

from __future__ import annotations

from typing import Any, Mapping


def behavior_for_recall_result(result: Any, answer_text: str | None = None) -> dict:
    """Return the controlled behavior skill for a completed recall answer."""
    if result is not None and getattr(result, "answer_type", None) == "memory_answer" and getattr(result, "memory_record", None) is not None:
        return {
            "motion": "recall_point",
            "light": "pointer_spot",
            "sound": None,
            "speech_text": answer_text,
        }

    return {
        "motion": "recall_not_found",
        "light": "sad_dim",
        "sound": None,
        "speech_text": answer_text,
    }


def recall_target_for_result(result: Any) -> dict:
    """Return an optional command payload describing where Lumos should point."""
    parsed_object = getattr(result, "parsed_object", None) if result is not None else None
    memory_record = getattr(result, "memory_record", None) if result is not None else None

    if result is None or getattr(result, "answer_type", None) != "memory_answer" or memory_record is None:
        return {
            "found": False,
            "object_label": parsed_object,
            "reason": "no_recent_memory",
        }

    location_label = str(getattr(memory_record, "location_label", "center of view") or "center of view")
    point = point_from_location_label(location_label)
    return {
        "found": True,
        "object_label": getattr(memory_record, "normalized_label", None) or getattr(memory_record, "object_label", None),
        "location_label": location_label,
        "confidence": round(float(getattr(memory_record, "confidence", 0.0)), 3),
        "timestamp": getattr(memory_record, "timestamp", None),
        "bbox": list(getattr(memory_record, "bbox", ()) or ()),
        "point_x_norm": point["x"],
        "point_y_norm": point["y"],
    }


def point_from_location_label(location_label: str) -> dict[str, float]:
    """Map the stored coarse location text back to a normalized screen target.

    Object memory intentionally stores approximate labels, not calibrated 3D
    coordinates. These values are therefore expressive pointing hints for Godot,
    not claims about exact metric position.
    """
    lowered = " ".join(str(location_label or "").lower().split())

    if "left" in lowered:
        x = 0.18
    elif "right" in lowered:
        x = 0.82
    else:
        x = 0.50

    if "upper" in lowered or "top" in lowered:
        y = 0.18
    elif "lower" in lowered or "bottom" in lowered:
        y = 0.82
    else:
        y = 0.50

    return {"x": x, "y": y}


def is_recall_target_found(memory_payload: Mapping[str, Any]) -> bool:
    """Small helper for callers/tests that inspect protocol memory payloads."""
    recall_target = memory_payload.get("recall_target")
    return isinstance(recall_target, Mapping) and bool(recall_target.get("found", False))
