"""Optional YOLO-based object detection for Milestone 3 scene memory.

The detector is deliberately optional. If Ultralytics is not installed, the
model file is missing, or the model fails to load, the backend keeps running
with object detection disabled. This preserves the stable engagement/FSM/Godot
pipeline from Milestone 2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.utils.config import ObjectDetectionConfig


BBox = tuple[int, int, int, int]


# COCO labels that are useful for a desk/lamp demo. The detector may see many
# COCO classes, but memory is intentionally scoped to common objects to keep the
# demo explainable and reduce noisy writes.
DEFAULT_DESK_OBJECTS = {
    "cell phone",
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "cup",
    "bottle",
    "remote",
    "scissors",
    "clock",
    "vase",
    "backpack",
    "handbag",
    "chair",
    "tv",
    "monitor",
}


NORMALIZATION_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "phone": "phone",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "book": "book",
    "cup": "cup",
    "bottle": "bottle",
    "remote": "remote",
    "scissors": "scissors",
    "clock": "clock",
    "vase": "vase",
    "backpack": "backpack",
    "handbag": "bag",
    "chair": "chair",
    "tv": "monitor",
    "monitor": "monitor",
}


@dataclass(frozen=True)
class ObjectDetection:
    """One object detection in image coordinates."""

    label: str
    normalized_label: str
    bbox: BBox  # x, y, w, h in pixels
    confidence: float
    location_label: str
    center_x_norm: float | None = None
    center_y_norm: float | None = None
    zone_x: str | None = None
    zone_y: str | None = None
    distance_hint: str | None = None
    pointing_target: dict | None = None
    class_id: Optional[int] = None

    def to_memory_command_dict(self) -> dict:
        """Return the compact memory payload used by the Godot protocol."""
        payload = {
            "label": self.label,
            "location": self.location_label,
            "confidence": round(float(self.confidence), 3),
        }
        if self.center_x_norm is not None:
            payload["center_x_norm"] = round(float(self.center_x_norm), 3)
        if self.center_y_norm is not None:
            payload["center_y_norm"] = round(float(self.center_y_norm), 3)
        if self.zone_x:
            payload["zone_x"] = self.zone_x
        if self.zone_y:
            payload["zone_y"] = self.zone_y
        if self.pointing_target:
            payload["pointing_target"] = dict(self.pointing_target)
        return payload

    def to_log_dict(self) -> dict:
        return {
            "label": self.label,
            "normalized_label": self.normalized_label,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "location_label": self.location_label,
            "center_x_norm": None if self.center_x_norm is None else round(float(self.center_x_norm), 3),
            "center_y_norm": None if self.center_y_norm is None else round(float(self.center_y_norm), 3),
            "zone_x": self.zone_x,
            "zone_y": self.zone_y,
            "distance_hint": self.distance_hint,
            "pointing_target": self.pointing_target,
            "class_id": self.class_id,
        }


def normalize_label(label: str) -> str:
    """Normalize detector labels for queryable memory records."""
    cleaned = re.sub(r"\s+", " ", label.strip().lower())
    return NORMALIZATION_ALIASES.get(cleaned, cleaned)


def normalized_bbox_center(bbox: BBox, frame_width: int, frame_height: int) -> tuple[float, float]:
    x, y, w, h = bbox
    cx = (x + (w / 2.0)) / max(frame_width, 1)
    cy = (y + (h / 2.0)) / max(frame_height, 1)
    return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))


def spatial_zones(cx: float, cy: float) -> tuple[str, str]:
    if cx < 0.33:
        zone_x = "left"
    elif cx > 0.67:
        zone_x = "right"
    else:
        zone_x = "center"

    if cy < 0.33:
        zone_y = "upper"
    elif cy > 0.67:
        zone_y = "lower"
    else:
        zone_y = "middle"
    return zone_x, zone_y


def distance_hint_from_bbox(bbox: BBox, frame_width: int, frame_height: int) -> str:
    _, _, w, h = bbox
    area_ratio = (float(w) * float(h)) / max(1.0, float(frame_width * frame_height))
    if area_ratio >= 0.20:
        return "very close in camera view"
    if area_ratio >= 0.08:
        return "near in camera view"
    if area_ratio <= 0.015:
        return "far or small in camera view"
    return "medium distance in camera view"


def pointing_target_for_bbox(bbox: BBox, frame_width: int, frame_height: int) -> dict:
    cx, cy = normalized_bbox_center(bbox, frame_width, frame_height)
    return {"type": "point_to_memory", "x_norm": round(cx, 3), "y_norm": round(cy, 3)}


def estimate_location_label(bbox: BBox, frame_width: int, frame_height: int) -> str:
    """Estimate an honest image-space semantic location from a bounding box."""
    cx, cy = normalized_bbox_center(bbox, frame_width, frame_height)
    zone_x, zone_y = spatial_zones(cx, cy)
    if zone_x == "center" and zone_y == "middle":
        return "center of the camera view"
    if zone_y == "middle":
        return f"{zone_x} side of the camera view"
    if zone_x == "center":
        return f"{zone_y} center of the camera view"
    return f"{zone_y} {zone_x} side of the camera view"


class YoloObjectDetector:
    """Best-effort Ultralytics YOLO wrapper.

    Object detection is enabled only if requested by CLI/config and the model can
    load successfully. The rest of the backend should treat ``detector.enabled``
    as the source of truth.
    """

    def __init__(self, config: ObjectDetectionConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.model: Any | None = None
        self.names: dict[int, str] = {}
        self.enabled = False
        self.allowed_labels = {label.strip().lower() for label in (config.allowed_labels or DEFAULT_DESK_OBJECTS)}

        if not config.enabled:
            self.logger.info("Object detection disabled by config")
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            self.logger.warning(
                "Object detector disabled: failed to import ultralytics (%s). Install with: pip install ultralytics",
                exc,
            )
            return

        try:
            model_path = str(config.model_path)
            self.model = YOLO(model_path)
            raw_names = getattr(self.model, "names", {}) or {}
            self.names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else {}
            self.enabled = True
            self.logger.info(
                "Object detector loaded successfully model=%s confidence=%.2f allowed_labels=%s",
                model_path,
                config.confidence,
                sorted(self.allowed_labels),
            )
        except Exception as exc:  # pragma: no cover - model/runtime guard
            self.model = None
            self.enabled = False
            self.logger.warning(
                "Object detector disabled: failed to load model '%s' (%s)",
                config.model_path,
                exc,
            )

    def detect(self, frame) -> list[ObjectDetection]:
        """Run object detection on one BGR OpenCV frame."""
        if not self.enabled or self.model is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        detections: list[ObjectDetection] = []

        try:
            results = self.model.predict(
                source=frame,
                conf=float(self.config.confidence),
                verbose=False,
            )
        except Exception as exc:
            self.logger.warning("Object detection failed for current frame: %s", exc)
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        for box in boxes:
            parsed = self._parse_box(box, frame_width, frame_height)
            if parsed is None:
                continue
            detections.append(parsed)

        detections.sort(key=lambda item: item.confidence, reverse=True)
        if self.config.max_objects_per_frame > 0:
            detections = detections[: self.config.max_objects_per_frame]
        return detections

    def _parse_box(self, box: Any, frame_width: int, frame_height: int) -> ObjectDetection | None:
        try:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = self.names.get(cls_id, str(cls_id)).strip().lower()
            if self.allowed_labels and label not in self.allowed_labels:
                return None

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            x1 = max(0.0, min(float(frame_width - 1), x1))
            y1 = max(0.0, min(float(frame_height - 1), y1))
            x2 = max(0.0, min(float(frame_width - 1), x2))
            y2 = max(0.0, min(float(frame_height - 1), y2))
            w = max(1, int(round(x2 - x1)))
            h = max(1, int(round(y2 - y1)))
            bbox: BBox = (int(round(x1)), int(round(y1)), w, h)
            normalized_label = normalize_label(label)
            location_label = estimate_location_label(bbox, frame_width, frame_height)
            cx, cy = normalized_bbox_center(bbox, frame_width, frame_height)
            zone_x, zone_y = spatial_zones(cx, cy)
            return ObjectDetection(
                label=label,
                normalized_label=normalized_label,
                bbox=bbox,
                confidence=confidence,
                location_label=location_label,
                center_x_norm=cx,
                center_y_norm=cy,
                zone_x=zone_x,
                zone_y=zone_y,
                distance_hint=distance_hint_from_bbox(bbox, frame_width, frame_height),
                pointing_target=pointing_target_for_bbox(bbox, frame_width, frame_height),
                class_id=cls_id,
            )
        except Exception as exc:
            self.logger.debug("Skipping malformed object detection box: %s", exc)
            return None


def draw_object_overlay(frame, detections: Iterable[ObjectDetection]) -> None:
    """Draw object boxes and labels in-place for the OpenCV preview window."""
    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        text = f"{detection.label} {detection.confidence:.2f} | {detection.location_label}"
        cv2.putText(
            frame,
            text,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
        )
