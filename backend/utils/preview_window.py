"""Small, display-only OpenCV preview window for Lumos.

This module intentionally keeps preview rendering separate from perception.
The backend should run detection on the original camera frame, then call this
helper only after perception, FSM, memory, and command generation are finished.
That keeps display resizing or mirroring from changing robot behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Iterable

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.perception.engagement_detector import EngagementResult, draw_engagement_overlay
from backend.perception.gesture_detector import HandGestureResult, draw_hand_gesture_overlay
from backend.perception.object_detector import ObjectDetection, draw_object_overlay
from backend.utils.config import EngagementConfig


@dataclass(frozen=True)
class PreviewWindowConfig:
    """Display settings for the optional OpenCV preview window.

    ``flip_horizontal`` is display-only. It is useful when the preview looks
    mirrored or reversed, but it never changes the frame used for perception,
    memory, or Godot face-follow commands.
    """

    title: str = "Lumos - Preview"
    scale: float = 0.50
    width: int = 0
    height: int = 0
    x: int = 24
    y: int = 24
    flip_horizontal: bool = False


class PreviewWindow:
    """A small resizable OpenCV preview that does not affect perception."""

    def __init__(self, config: PreviewWindowConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self._initialized = False
        self._last_size: tuple[int, int] | None = None

    def show(
        self,
        frame,
        *,
        raw_result: EngagementResult,
        smoothed_result: EngagementResult,
        state: str,
        state_elapsed_s: float,
        fps: float,
        engagement_config: EngagementConfig,
        object_detections: Iterable[ObjectDetection] | None = None,
        object_overlay_enabled: bool = False,
        gesture_result: HandGestureResult | None = None,
        speaker_result: ActiveSpeakerResult | None = None,
    ) -> int:
        """Render a scaled preview and return the OpenCV key code.

        The input ``frame`` is copied before display edits. The caller can keep
        using the original frame for detection, logging, and memory writes.
        """

        display_frame = frame.copy()
        display_raw = raw_result
        display_smoothed = smoothed_result
        display_detections = list(object_detections or [])
        display_gesture = gesture_result

        if self.config.flip_horizontal:
            display_frame = cv2.flip(display_frame, 1)
            frame_width = int(frame.shape[1])
            display_raw = _mirror_engagement_result(raw_result, frame_width)
            display_smoothed = _mirror_engagement_result(smoothed_result, frame_width)
            display_detections = [_mirror_object_detection(item, frame_width) for item in display_detections]
            display_gesture = _mirror_gesture_result(gesture_result, frame_width)

        draw_engagement_overlay(
            display_frame,
            raw_result=display_raw,
            smoothed_result=display_smoothed,
            state=state,
            state_elapsed_s=state_elapsed_s,
            fps=fps,
            config=engagement_config,
        )
        if object_overlay_enabled:
            draw_object_overlay(display_frame, display_detections)
        draw_hand_gesture_overlay(display_frame, display_gesture)
        draw_speaker_overlay(display_frame, speaker_result)

        preview_frame = self._resize_for_preview(display_frame)
        preview_h, preview_w = preview_frame.shape[:2]
        self._ensure_window(preview_w, preview_h)
        cv2.imshow(self.config.title, preview_frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if self._initialized:
            try:
                cv2.destroyWindow(self.config.title)
            except Exception:
                pass
            self._initialized = False
            self._last_size = None

    def _resize_for_preview(self, frame):
        source_h, source_w = frame.shape[:2]
        target_w, target_h = self._target_size(source_w, source_h)
        if target_w == source_w and target_h == source_h:
            return frame
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _target_size(self, source_w: int, source_h: int) -> tuple[int, int]:
        if self.config.width > 0 and self.config.height > 0:
            return max(64, int(self.config.width)), max(48, int(self.config.height))

        if self.config.width > 0:
            target_w = max(64, int(self.config.width))
            ratio = target_w / max(1, source_w)
            return target_w, max(48, int(round(source_h * ratio)))

        if self.config.height > 0:
            target_h = max(48, int(self.config.height))
            ratio = target_h / max(1, source_h)
            return max(64, int(round(source_w * ratio))), target_h

        scale = max(0.10, min(1.0, float(self.config.scale)))
        return max(64, int(round(source_w * scale))), max(48, int(round(source_h * scale)))

    def _ensure_window(self, width: int, height: int) -> None:
        requested_size = (int(width), int(height))
        if not self._initialized:
            cv2.namedWindow(self.config.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.config.title, requested_size[0], requested_size[1])
            cv2.moveWindow(self.config.title, int(self.config.x), int(self.config.y))
            self._initialized = True
            self._last_size = requested_size
            self.logger.info(
                "Preview window initialized title=%r size=%sx%s position=(%s,%s) flip_horizontal=%s",
                self.config.title,
                requested_size[0],
                requested_size[1],
                self.config.x,
                self.config.y,
                self.config.flip_horizontal,
            )
            return

        if requested_size != self._last_size:
            cv2.resizeWindow(self.config.title, requested_size[0], requested_size[1])
            self._last_size = requested_size


def _mirror_bbox(bbox: tuple[int, int, int, int] | None, frame_width: int) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    x, y, w, h = bbox
    mirrored_x = max(0, int(frame_width) - int(x) - int(w))
    return mirrored_x, int(y), int(w), int(h)


def _mirror_center(center: tuple[float, float] | None) -> tuple[float, float] | None:
    if center is None:
        return None
    x, y = center
    return round(1.0 - float(x), 3), round(float(y), 3)


def _mirror_engagement_result(result: EngagementResult, frame_width: int) -> EngagementResult:
    return replace(
        result,
        face_bbox=_mirror_bbox(result.face_bbox, frame_width),
        face_center_norm=_mirror_center(result.face_center_norm),
    )


def _mirror_object_detection(detection: ObjectDetection, frame_width: int) -> ObjectDetection:
    mirrored_bbox = _mirror_bbox(detection.bbox, frame_width)
    return replace(
        detection,
        bbox=detection.bbox if mirrored_bbox is None else mirrored_bbox,
        location_label=_mirror_location_label(detection.location_label),
    )


def _mirror_location_label(label: str) -> str:
    # Keep text labels readable when the displayed preview is flipped.
    temp = "__LUMOS_LEFT__"
    return label.replace("left", temp).replace("right", "left").replace(temp, "right")


def _mirror_gesture_result(result: HandGestureResult | None, frame_width: int) -> HandGestureResult | None:
    if result is None:
        return None
    return replace(
        result,
        hand_bbox=_mirror_bbox(result.hand_bbox, frame_width),
        hand_center_norm=_mirror_center(result.hand_center_norm),
    )


def draw_speaker_overlay(frame, speaker_result: ActiveSpeakerResult | None) -> None:
    if speaker_result is None:
        return
    to_robot_value = speaker_result.speaking_to_robot
    if to_robot_value is True:
        to_robot = "yes"
    elif to_robot_value is False:
        to_robot = "no"
    else:
        to_robot = "unknown"
    doa = "unavailable" if speaker_result.doa_azimuth_deg is None else f"{speaker_result.doa_azimuth_deg:+.0f} deg"
    lines = [
        f"speech={'active' if speaker_result.speech_detected else 'inactive'} speaker={speaker_result.active_track_id or 'none'} conf={speaker_result.confidence:.2f}",
        f"to_lumos={to_robot} doa={doa}",
        f"speaker_reason={speaker_result.reason}",
    ]
    height, _width = frame.shape[:2]
    base_y = max(24, height - 74)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, base_y + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
