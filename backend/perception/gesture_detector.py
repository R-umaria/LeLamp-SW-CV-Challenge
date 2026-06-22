"""Optional MediaPipe hand gesture detector for bounded Lumos motion control.

The detector recognizes deliberate, demo-friendly gestures:

* ``beckon``: index-finger calling pose/motion; Lumos leans closer.
* ``palm_push``: open palm; Lumos backs away.
* ``thumbs_up``: affirmative approval; Lumos performs a happy response.
* ``pinch_follow``: thumb-index pinch; Lumos follows the pinch point.
* ``heart``: two-hand heart / affection cue; Lumos blushes baby pink.

The module degrades safely when MediaPipe is unavailable. It returns a structured
``HandGestureResult`` with ``status='unavailable'`` or ``status='none'`` instead
of raising during the live camera loop. Python emits only bounded gesture labels
and normalized target hints; Godot remains a renderer for named motion/light
skills.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Iterable, Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required for gesture overlays. Install with: pip install opencv-python") from exc

try:  # MediaPipe is optional so the rest of Lumos can still run without it.
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    mp = None

from backend.utils.config import HandGestureConfig


ACTIVE_GESTURES = {"beckon", "palm_push", "thumbs_up", "pinch_follow", "heart"}


@dataclass(frozen=True)
class HandGestureResult:
    """Frame-level hand gesture classification consumed by behavior policy."""

    status: str  # active label, "none", or "unavailable"
    confidence: float
    reason: str
    hand_label: Optional[str] = None
    hand_bbox: Optional[tuple[int, int, int, int]] = None
    hand_center_norm: Optional[tuple[float, float]] = None
    palm_area_ratio: float = 0.0
    hand_count: int = 0

    def to_protocol_dict(self) -> dict:
        payload = {
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }
        if self.hand_label is not None:
            payload["hand_label"] = self.hand_label
        if self.hand_center_norm is not None:
            # Keep the legacy hand_x/y fields and add explicit target aliases for
            # new pinch-follow behavior. Existing Godot/debug consumers stay valid.
            payload["hand_x_norm"] = round(float(self.hand_center_norm[0]), 3)
            payload["hand_y_norm"] = round(float(self.hand_center_norm[1]), 3)
            payload["target_x_norm"] = round(float(self.hand_center_norm[0]), 3)
            payload["target_y_norm"] = round(float(self.hand_center_norm[1]), 3)
        if self.hand_count:
            payload["hand_count"] = int(self.hand_count)
        return payload

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 3)
        data["palm_area_ratio"] = round(float(self.palm_area_ratio), 4)
        return data


@dataclass(frozen=True)
class _HandCandidate:
    landmarks: object
    hand_label: Optional[str]
    bbox: tuple[int, int, int, int]
    center_norm: tuple[float, float]
    palm_area_ratio: float


class HandGestureDetector:
    """MediaPipe Hands wrapper with deterministic gesture rules.

    The rules are intentionally simple and explainable for the challenge demo.
    We do not expose raw hand landmarks to the LLM or Godot; only a bounded
    gesture label and normalized target hint are emitted.
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
            self.logger.info("MediaPipe hand gesture detector initialized max_hands=%s", config.max_num_hands)

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

        candidates = self._make_candidates(results, width, height)
        if not candidates:
            return self._held_or_none(now, "no_classifiable_hand")

        # Two-hand heart is evaluated before single-hand gestures because a heart
        # can contain curled fingers that would otherwise look like pinches.
        if len(candidates) >= 2:
            heart = self._classify_two_hand_heart(candidates, width, height)
            if heart.status == "heart" and heart.confidence >= self.config.min_gesture_confidence:
                self._remember_active(heart, now)
                return heart

        best_result: Optional[HandGestureResult] = None
        for candidate in candidates:
            result = self._classify_single_hand(candidate, now)
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        if best_result is None:
            return self._held_or_none(now, "no_supported_gesture")

        if best_result.status in ACTIVE_GESTURES and best_result.confidence >= self.config.min_gesture_confidence:
            self._remember_active(best_result, now)
            return best_result

        return self._held_or_none(now, best_result.reason)

    def _make_candidates(self, results, width: int, height: int) -> list[_HandCandidate]:
        candidates: list[_HandCandidate] = []
        handedness_items = results.multi_handedness or []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = None
            if index < len(handedness_items):
                hand_label = handedness_items[index].classification[0].label
            landmarks = hand_landmarks.landmark
            bbox = self._bbox_from_landmarks(landmarks, width, height)
            x, y, w, h = bbox
            palm_area_ratio = (w * h) / float(max(width * height, 1))
            center_norm = (round((x + w * 0.5) / width, 3), round((y + h * 0.5) / height, 3))
            candidates.append(_HandCandidate(landmarks, hand_label, bbox, center_norm, palm_area_ratio))
        candidates.sort(key=lambda item: item.palm_area_ratio, reverse=True)
        return candidates

    def _classify_single_hand(self, hand: _HandCandidate, now: float) -> HandGestureResult:
        if hand.palm_area_ratio < self.config.min_hand_area_ratio:
            return self._single_result("none", 0.0, "hand_too_small", hand)

        landmarks = hand.landmarks
        extended = {
            "index": self._finger_extended(landmarks, 8, 6),
            "middle": self._finger_extended(landmarks, 12, 10),
            "ring": self._finger_extended(landmarks, 16, 14),
            "pinky": self._finger_extended(landmarks, 20, 18),
        }
        folded = {
            "index": self._finger_folded(landmarks, 8, 6),
            "middle": self._finger_folded(landmarks, 12, 10),
            "ring": self._finger_folded(landmarks, 16, 14),
            "pinky": self._finger_folded(landmarks, 20, 18),
        }

        open_finger_count = sum(1 for is_extended in extended.values() if is_extended)
        if open_finger_count >= 4:
            confidence = min(0.96, 0.62 + hand.palm_area_ratio * 3.6)
            return self._single_result("palm_push", confidence, "open_palm_push", hand)

        if self._is_thumbs_up(landmarks, folded):
            confidence = min(0.94, 0.72 + hand.palm_area_ratio * 2.4)
            return self._single_result("thumbs_up", confidence, "thumb_extended_fingers_folded", hand)

        pinch_result = self._classify_pinch(hand)
        if pinch_result.status == "pinch_follow":
            return pinch_result

        self._index_history.append((now, float(landmarks[8].y)))
        index_prominent = landmarks[8].y < landmarks[5].y - self.config.index_prominence_margin
        other_folded_count = sum(1 for name in ("middle", "ring", "pinky") if folded[name])
        oscillating = self._index_tip_oscillating(now)

        if index_prominent and other_folded_count >= 2:
            confidence = 0.72
            reason = "index_call_pose"
            if oscillating:
                confidence = 0.90
                reason = "index_beckon_motion"
            return self._single_result("beckon", confidence, reason, hand)

        return self._single_result("none", 0.0, "no_supported_gesture", hand)

    def _classify_pinch(self, hand: _HandCandidate) -> HandGestureResult:
        landmarks = hand.landmarks
        pinch_dist = self._distance(landmarks[4], landmarks[8])
        palm_span = max(self._distance(landmarks[0], landmarks[9]), 1e-6)
        pinch_ratio = pinch_dist / palm_span
        close_by_absolute = pinch_dist <= self.config.pinch_tip_max_distance
        close_by_relative = pinch_ratio <= self.config.pinch_tip_max_distance_ratio
        if not (close_by_absolute or close_by_relative):
            return self._single_result("none", 0.0, "pinch_not_closed", hand)

        midpoint = self._midpoint(landmarks[4], landmarks[8])
        target = (round(float(midpoint[0]), 3), round(float(midpoint[1]), 3))
        # Give more confidence when the fingertips are very close, but keep the
        # range bounded so one noisy frame does not dominate all other gestures.
        closeness = 1.0 - min(1.0, pinch_ratio / max(self.config.pinch_tip_max_distance_ratio, 1e-6))
        confidence = min(0.94, 0.72 + 0.16 * closeness + hand.palm_area_ratio * 1.1)
        return HandGestureResult(
            "pinch_follow",
            confidence,
            "thumb_index_pinch",
            hand_label=hand.hand_label,
            hand_bbox=hand.bbox,
            hand_center_norm=target,
            palm_area_ratio=hand.palm_area_ratio,
            hand_count=1,
        )

    def _classify_two_hand_heart(self, candidates: list[_HandCandidate], width: int, height: int) -> HandGestureResult:
        first, second = candidates[0], candidates[1]
        c1x, c1y = first.center_norm
        c2x, c2y = second.center_norm
        center_gap = abs(float(c1x) - float(c2x))
        vertical_gap = abs(float(c1y) - float(c2y))
        if center_gap > self.config.heart_center_max_gap or vertical_gap > self.config.heart_vertical_max_gap:
            return HandGestureResult("none", 0.0, "hands_too_far_for_heart", hand_count=2)

        lms_a = first.landmarks
        lms_b = second.landmarks
        paired_same = self._distance(lms_a[8], lms_b[8]) + self._distance(lms_a[4], lms_b[4])
        paired_cross = self._distance(lms_a[8], lms_b[4]) + self._distance(lms_a[4], lms_b[8])
        paired_gap = min(paired_same, paired_cross)
        if paired_gap > self.config.heart_tip_pair_max_distance:
            return HandGestureResult("none", 0.0, "heart_fingertips_not_paired", hand_count=2)

        bbox = self._combined_bbox([first.bbox, second.bbox], width, height)
        x, y, w, h = bbox
        combined_area = (w * h) / float(max(width * height, 1))
        if combined_area < self.config.min_hand_area_ratio * 1.35:
            return HandGestureResult("none", 0.0, "heart_hands_too_small", hand_count=2)

        center = (round((x + w * 0.5) / width, 3), round((y + h * 0.5) / height, 3))
        closeness = 1.0 - min(1.0, paired_gap / max(self.config.heart_tip_pair_max_distance, 1e-6))
        confidence = min(0.96, 0.70 + 0.18 * closeness + combined_area * 0.85)
        hand_label = "+".join(label for label in [first.hand_label, second.hand_label] if label) or None
        return HandGestureResult(
            "heart",
            confidence,
            "two_hand_heart",
            hand_label=hand_label,
            hand_bbox=bbox,
            hand_center_norm=center,
            palm_area_ratio=combined_area,
            hand_count=2,
        )

    def _is_thumbs_up(self, landmarks, folded: dict[str, bool]) -> bool:
        folded_count = sum(1 for name in ("index", "middle", "ring", "pinky") if folded[name])
        if folded_count < 3:
            return False
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        # In image coordinates, smaller y is higher. This works for a demo-friendly
        # upright thumbs-up without relying on left/right handedness.
        thumb_vertical = float(thumb_tip.y) < float(thumb_ip.y) - self.config.thumb_up_margin
        thumb_high = float(thumb_tip.y) < min(float(landmarks[idx].y) for idx in (6, 10, 14, 18)) - self.config.thumb_up_margin
        thumb_not_collapsed = self._distance(thumb_tip, thumb_mcp) > self._distance(thumb_ip, thumb_mcp) * 1.15
        hand_not_sideways = abs(float(landmarks[9].x) - float(wrist.x)) < 0.28
        return bool(thumb_vertical and thumb_high and thumb_not_collapsed and hand_not_sideways)

    def _single_result(self, status: str, confidence: float, reason: str, hand: _HandCandidate) -> HandGestureResult:
        return HandGestureResult(
            status,
            confidence,
            reason,
            hand_label=hand.hand_label,
            hand_bbox=hand.bbox,
            hand_center_norm=hand.center_norm,
            palm_area_ratio=hand.palm_area_ratio,
            hand_count=1,
        )

    def _remember_active(self, result: HandGestureResult, now: float) -> None:
        self._last_active_result = result
        self._last_active_at = now

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
                hand_count=self._last_active_result.hand_count,
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

    @staticmethod
    def _combined_bbox(boxes: Iterable[tuple[int, int, int, int]], width: int, height: int) -> tuple[int, int, int, int]:
        boxes = list(boxes)
        x_min = max(0, min(x for x, _y, _w, _h in boxes))
        y_min = max(0, min(y for _x, y, _w, _h in boxes))
        x_max = min(width - 1, max(x + w for x, _y, w, _h in boxes))
        y_max = min(height - 1, max(y + h for _x, y, _w, h in boxes))
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

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    @staticmethod
    def _midpoint(a, b) -> tuple[float, float]:
        return (float(a.x + b.x) * 0.5, float(a.y + b.y) * 0.5)


def draw_hand_gesture_overlay(frame, gesture: HandGestureResult | None) -> None:
    """Draw a compact hand gesture overlay on the OpenCV preview frame."""

    if gesture is None or gesture.status == "unavailable":
        return

    color_by_status = {
        "beckon": (120, 220, 120),
        "palm_push": (80, 180, 255),
        "thumbs_up": (80, 220, 255),
        "pinch_follow": (255, 210, 80),
        "heart": (210, 120, 255),
        "none": (160, 160, 160),
    }
    color = color_by_status.get(gesture.status, (220, 220, 220))

    if gesture.hand_bbox is not None:
        x, y, w, h = gesture.hand_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"gesture={gesture.status} conf={gesture.confidence:.2f}"
        cv2.putText(frame, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    if gesture.hand_center_norm is not None:
        px = int(float(gesture.hand_center_norm[0]) * frame.shape[1])
        py = int(float(gesture.hand_center_norm[1]) * frame.shape[0])
        cv2.circle(frame, (px, py), 6, color, 2)

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
