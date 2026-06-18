"""Simple face-presence and face-position engagement detector.

Milestone 1 deliberately uses OpenCV's built-in Haar face detector instead of a
heavier gaze/head-pose model. This is less accurate than MediaPipe landmarks, but
it is fast, dependency-light, explainable, and enough to validate the backend
state-command loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "OpenCV is required for engagement detection. Install with: pip install opencv-python"
    ) from exc

from backend.utils.config import EngagementConfig


@dataclass(frozen=True)
class EngagementResult:
    status: str  # "engaged", "disengaged", or "absent"
    confidence: float
    reason: str
    face_bbox: Optional[tuple[int, int, int, int]] = None
    face_center_norm: Optional[tuple[float, float]] = None
    face_area_ratio: float = 0.0

    def to_protocol_dict(self) -> dict:
        return {
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 3)
        data["face_area_ratio"] = round(float(self.face_area_ratio), 4)
        return data


class FaceEngagementDetector:
    def __init__(self, config: EngagementConfig) -> None:
        self.config = config
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from: {cascade_path}")

    def detect(self, frame) -> EngagementResult:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.cascade_scale_factor,
            minNeighbors=self.config.cascade_min_neighbors,
            minSize=self.config.cascade_min_size,
        )

        if len(faces) == 0:
            return EngagementResult(
                status="absent",
                confidence=0.0,
                reason="no_face_detected",
            )

        # Choose the largest detected face as the primary user.
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        area_ratio = (w * h) / float(width * height)

        x_offset = abs(cx - 0.5)
        y_offset = abs(cy - 0.5)

        horizontally_centered = x_offset <= self.config.center_tolerance_x
        vertically_centered = y_offset <= self.config.center_tolerance_y
        face_large_enough = area_ratio >= self.config.min_face_area_ratio

        if horizontally_centered and vertically_centered and face_large_enough:
            confidence = self._engaged_confidence(x_offset, y_offset, area_ratio)
            return EngagementResult(
                status="engaged",
                confidence=confidence,
                reason="face_centered",
                face_bbox=(int(x), int(y), int(w), int(h)),
                face_center_norm=(round(cx, 3), round(cy, 3)),
                face_area_ratio=area_ratio,
            )

        reason_parts: list[str] = []
        if not horizontally_centered:
            reason_parts.append("face_left_or_right")
        if not vertically_centered:
            reason_parts.append("face_high_or_low")
        if not face_large_enough:
            reason_parts.append("face_too_small")

        confidence = self._disengaged_confidence(x_offset, y_offset, area_ratio)
        return EngagementResult(
            status="disengaged",
            confidence=confidence,
            reason="+".join(reason_parts) or "face_not_centered",
            face_bbox=(int(x), int(y), int(w), int(h)),
            face_center_norm=(round(cx, 3), round(cy, 3)),
            face_area_ratio=area_ratio,
        )

    def _engaged_confidence(self, x_offset: float, y_offset: float, area_ratio: float) -> float:
        x_score = max(0.0, 1.0 - x_offset / max(self.config.center_tolerance_x, 1e-6))
        y_score = max(0.0, 1.0 - y_offset / max(self.config.center_tolerance_y, 1e-6))
        size_score = min(1.0, area_ratio / max(self.config.min_face_area_ratio * 3.0, 1e-6))
        return min(0.99, 0.45 + 0.25 * x_score + 0.20 * y_score + 0.10 * size_score)

    def _disengaged_confidence(self, x_offset: float, y_offset: float, area_ratio: float) -> float:
        # Confidence rises as the face is farther from the central engagement zone.
        x_excess = max(0.0, x_offset - self.config.center_tolerance_x)
        y_excess = max(0.0, y_offset - self.config.center_tolerance_y)
        position_score = min(1.0, (x_excess + y_excess) * 3.0)
        size_penalty_score = 1.0 if area_ratio < self.config.min_face_area_ratio else 0.0
        return min(0.95, 0.55 + 0.30 * position_score + 0.10 * size_penalty_score)


def draw_engagement_overlay(frame, result: EngagementResult, state: str) -> None:
    """Draw debug overlay in-place for the optional OpenCV preview window."""
    label = f"state={state} engagement={result.status} conf={result.confidence:.2f}"
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"reason={result.reason}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if result.face_bbox is not None:
        x, y, w, h = result.face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    height, width = frame.shape[:2]
    # Draw engagement zone.
    x_tol = int(width * 0.22)
    y_tol = int(height * 0.28)
    cx, cy = width // 2, height // 2
    cv2.rectangle(frame, (cx - x_tol, cy - y_tol), (cx + x_tol, cy + y_tol), (255, 255, 255), 1)
