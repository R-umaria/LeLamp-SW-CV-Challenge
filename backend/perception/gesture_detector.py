"""Optional MediaPipe hand gesture detector for bounded Lumos motion control.

The detector recognizes two deliberate, demo-friendly gestures:

* ``beckon``: an index-finger calling gesture, preferably with a small repeated
  curl/wag motion. This maps to Lumos moving closer to the camera.
* ``palm_push``: an open palm facing the camera. This maps to Lumos backing away.

The module degrades safely when MediaPipe is unavailable. It returns a structured
``HandGestureResult`` with ``status='unavailable'`` or ``status='none'`` instead
of raising during the live camera loop.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required for gesture overlays. Install with: pip install opencv-python") from exc

try:  # MediaPipe is optional so the rest of Lumos can still run without it.
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    mp = None

from backend.utils.config import HandGestureConfig


@dataclass(frozen=True)
class HandGestureResult:
    """Frame-level hand gesture classification consumed by behavior policy."""

    status: str  # "beckon", "palm_push", "none", or "unavailable"
    confidence: float
    reason: str
    hand_label: Optional[str] = None
    hand_bbox: Optional[tuple[int, int, int, int]] = None
    hand_center_norm: Optional[tuple[float, float]] = None
    palm_area_ratio: float = 0.0

    def to_protocol_dict(self) -> dict:
        payload = {
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }
        if self.hand_label is not None:
            payload["hand_label"] = self.hand_label
        if self.hand_center_norm is not None:
            payload["hand_x_norm"] = round(float(self.hand_center_norm[0]), 3)
            payload["hand_y_norm"] = round(float(self.hand_center_norm[1]), 3)
        return payload

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 3)
        data["palm_area_ratio"] = round(float(self.palm_area_ratio), 4)
        return data


class HandGestureDetector:
    """MediaPipe Hands wrapper with deterministic gesture rules.

    The rules are intentionally simple and explainable for the challenge demo.
    We do not expose raw hand landmarks to the LLM or Godot; only a bounded
    gesture label is emitted.
    """

    def __init__(self, config: HandGestureConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.enabled = bool(config.enabled and mp is not None)
        self._hands = None
        self._index_history: Deque[tuple[float, float]] = deque(maxlen=18)
        self._last_active_result: Optional[HandGestureResult] = None
        self._last_active_at = 0.0

        if config.enabled and mp is None:
            self.logger.warning("Hand gestures requested but mediapipe is not installed; gesture control is disabled")
            return

        if self.enabled:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=config.max_num_hands,
                model_complexity=0,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            self.logger.info("MediaPipe hand gesture detector initialized")

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def detect(self, frame, now: Optional[float] = None) -> HandGestureResult:
        now = time.monotonic() if now is None else now
        if not self.config.enabled:
            return HandGestureResult("unavailable", 0.0, "gesture_detection_disabled")
        if not self.enabled or self._hands is None:
            return HandGestureResult("unavailable", 0.0, "mediapipe_unavailable")

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return self._held_or_none(now, "no_hand_detected")

        best_result: Optional[HandGestureResult] = None
        handedness_items = results.multi_handedness or []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = None
            if index < len(handedness_items):
                hand_label = handedness_items[index].classification[0].label
            candidate = self._classify_landmarks(hand_landmarks.landmark, width, height, now, hand_label)
            if best_result is None or candidate.confidence > best_result.confidence:
                best_result = candidate

        if best_result is None:
            return self._held_or_none(now, "no_classifiable_hand")

        if best_result.status in {"beckon", "palm_push"} and best_result.confidence >= self.config.min_gesture_confidence:
            self._last_active_result = best_result
            self._last_active_at = now
            return best_result

        return self._held_or_none(now, best_result.reason)

    def _classify_landmarks(
        self,
        landmarks,
        width: int,
        height: int,
        now: float,
        hand_label: Optional[str],
    ) -> HandGestureResult:
        bbox = self._bbox_from_landmarks(landmarks, width, height)
        x, y, w, h = bbox
        palm_area_ratio = (w * h) / float(max(width * height, 1))
        center_norm = (round((x + w * 0.5) / width, 3), round((y + h * 0.5) / height, 3))

        if palm_area_ratio < self.config.min_hand_area_ratio:
            return HandGestureResult(
                "none",
                0.0,
                "hand_too_small",
                hand_label=hand_label,
                hand_bbox=bbox,
                hand_center_norm=center_norm,
                palm_area_ratio=palm_area_ratio,
            )

        extended = {
            "index": self._finger_extended(landmarks, 8, 6),
            "middle": self._finger_extended(landmarks, 12, 10),
            "ring": self._finger_extended(landmarks, 16, 14),
            "pinky": self._finger_extended(landmarks, 20, 18),
        }
        folded = {
            "middle": self._finger_folded(landmarks, 12, 10),
            "ring": self._finger_folded(landmarks, 16, 14),
            "pinky": self._finger_folded(landmarks, 20, 18),
        }

        open_finger_count = sum(1 for is_extended in extended.values() if is_extended)
        if open_finger_count >= 4:
            confidence = min(0.96, 0.62 + palm_area_ratio * 3.6)
            return HandGestureResult(
                "palm_push",
                confidence,
                "open_palm_push",
                hand_label=hand_label,
                hand_bbox=bbox,
                hand_center_norm=center_norm,
                palm_area_ratio=palm_area_ratio,
            )

        self._index_history.append((now, float(landmarks[8].y)))
        index_prominent = landmarks[8].y < landmarks[5].y - self.config.index_prominence_margin
        other_folded_count = sum(1 for value in folded.values() if value)
        oscillating = self._index_tip_oscillating(now)

        if index_prominent and other_folded_count >= 2:
            confidence = 0.72
            reason = "index_call_pose"
            if oscillating:
                confidence = 0.90
                reason = "index_beckon_motion"
            return HandGestureResult(
                "beckon",
                confidence,
                reason,
                hand_label=hand_label,
                hand_bbox=bbox,
                hand_center_norm=center_norm,
                palm_area_ratio=palm_area_ratio,
            )

        return HandGestureResult(
            "none",
            0.0,
            "no_supported_gesture",
            hand_label=hand_label,
            hand_bbox=bbox,
            hand_center_norm=center_norm,
            palm_area_ratio=palm_area_ratio,
        )

    def _held_or_none(self, now: float, reason: str) -> HandGestureResult:
        if self._last_active_result is not None and now - self._last_active_at <= self.config.hold_s:
            return HandGestureResult(
                self._last_active_result.status,
                max(0.0, self._last_active_result.confidence - 0.08),
                "held_previous_gesture",
                hand_label=self._last_active_result.hand_label,
                hand_bbox=self._last_active_result.hand_bbox,
                hand_center_norm=self._last_active_result.hand_center_norm,
                palm_area_ratio=self._last_active_result.palm_area_ratio,
            )
        return HandGestureResult("none", 0.0, reason)

    @staticmethod
    def _bbox_from_landmarks(landmarks, width: int, height: int) -> tuple[int, int, int, int]:
        xs = [float(lm.x) for lm in landmarks]
        ys = [float(lm.y) for lm in landmarks]
        x_min = max(0, int(min(xs) * width))
        y_min = max(0, int(min(ys) * height))
        x_max = min(width - 1, int(max(xs) * width))
        y_max = min(height - 1, int(max(ys) * height))
        return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

    def _finger_extended(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        return float(landmarks[tip_idx].y) < float(landmarks[pip_idx].y) - self.config.finger_extension_margin

    def _finger_folded(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        return float(landmarks[tip_idx].y) > float(landmarks[pip_idx].y) - self.config.finger_fold_margin

    def _index_tip_oscillating(self, now: float) -> bool:
        recent = [(t, y) for t, y in self._index_history if now - t <= self.config.motion_window_s]
        if len(recent) < 5:
            return False

        y_values = [value for _, value in recent]
        amplitude = max(y_values) - min(y_values)
        if amplitude < self.config.beckon_motion_min_amplitude:
            return False

        direction_changes = 0
        previous_sign = 0
        for idx in range(1, len(y_values)):
            diff = y_values[idx] - y_values[idx - 1]
            sign = 1 if diff > 0.003 else -1 if diff < -0.003 else 0
            if sign != 0 and previous_sign != 0 and sign != previous_sign:
                direction_changes += 1
            if sign != 0:
                previous_sign = sign
        return direction_changes >= self.config.beckon_min_direction_changes


def draw_hand_gesture_overlay(frame, gesture: HandGestureResult | None) -> None:
    """Draw a compact hand gesture overlay on the OpenCV preview frame."""

    if gesture is None or gesture.status == "unavailable":
        return

    if gesture.hand_bbox is not None:
        x, y, w, h = gesture.hand_bbox
        color = (120, 220, 120) if gesture.status == "beckon" else (80, 180, 255)
        if gesture.status == "none":
            color = (160, 160, 160)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"gesture={gesture.status} conf={gesture.confidence:.2f}"
        cv2.putText(frame, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"Hand: {gesture.status} ({gesture.reason})",
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
