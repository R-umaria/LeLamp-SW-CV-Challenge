"""Practical webcam head-pose approximation for Lumos engagement.

The estimator is deliberately best-effort. It uses MediaPipe Face Mesh when the
package is available, estimates yaw/pitch/roll with OpenCV solvePnP, and returns
"unknown" instead of raising when landmarks fail. The engagement detector can
then safely fall back to its existing face-center/size heuristics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, asin, degrees
from typing import Any, Optional

import numpy as np

try:  # OpenCV is already required by the engagement detector.
    import cv2
except Exception as exc:  # pragma: no cover
    raise ImportError("OpenCV is required for head-pose estimation") from exc


@dataclass(frozen=True)
class HeadPoseResult:
    available: bool
    yaw: float | None
    pitch: float | None
    roll: float | None
    classification: str
    confidence: float
    reason: str
    fallback_used: bool = False

    def to_log_dict(self) -> dict:
        return {
            "head_pose_available": bool(self.available),
            "yaw": None if self.yaw is None else round(float(self.yaw), 2),
            "pitch": None if self.pitch is None else round(float(self.pitch), 2),
            "roll": None if self.roll is None else round(float(self.roll), 2),
            "head_pose_classification": self.classification,
            "head_pose_confidence": round(float(self.confidence), 3),
            "head_pose_reason": self.reason,
            "head_pose_fallback_used": bool(self.fallback_used),
        }


class HeadPoseEstimator:
    """MediaPipe Face Mesh + solvePnP head-pose estimator.

    Yaw is positive when the face turns toward camera-right. Pitch is positive
    when the face tilts upward. The thresholds are intentionally conservative so
    normal webcam movement does not flicker engagement.
    """

    LANDMARK_IDS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),       # nose tip
            (0.0, -63.6, -12.5),   # chin
            (-43.3, 32.7, -26.0),  # left eye outer corner
            (43.3, 32.7, -26.0),   # right eye outer corner
            (-28.9, -28.9, -24.1), # left mouth corner
            (28.9, -28.9, -24.1),  # right mouth corner
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_history: int = 5,
        yaw_at_lamp_deg: float = 24.0,
        pitch_at_lamp_deg: float = 22.0,
        yaw_away_deg: float = 38.0,
        pitch_away_deg: float = 32.0,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        self.enabled = bool(enabled)
        self.yaw_at_lamp_deg = float(yaw_at_lamp_deg)
        self.pitch_at_lamp_deg = float(pitch_at_lamp_deg)
        self.yaw_away_deg = float(yaw_away_deg)
        self.pitch_away_deg = float(pitch_away_deg)
        self._history: deque[tuple[float, float, float, str, float]] = deque(maxlen=max(1, int(max_history)))
        self._mesh: Any | None = None
        self._load_error: str | None = None

        if not self.enabled:
            self._load_error = "head_pose_disabled"
            return
        try:
            import mediapipe as mp  # type: ignore

            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=float(min_detection_confidence),
                min_tracking_confidence=float(min_tracking_confidence),
            )
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            self._mesh = None
            self._load_error = f"mediapipe_face_mesh_unavailable:{exc}"

    @property
    def available(self) -> bool:
        return self._mesh is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def estimate(self, frame_bgr: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None) -> HeadPoseResult:
        if not self.enabled:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, "head_pose_disabled", True)
        if self._mesh is None:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, self._load_error or "face_mesh_unavailable", True)
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, "empty_frame", True)

        height, width = frame_bgr.shape[:2]
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._mesh.process(rgb)
        except Exception as exc:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, f"face_mesh_runtime_error:{exc}", True)

        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, "no_face_mesh_landmarks", True)

        # FaceEngagementDetector already selected a primary bbox. Face Mesh returns
        # one face; for the MVP this is sufficiently stable and avoids brittle ID
        # matching between two separate detectors.
        landmarks = faces[0].landmark
        try:
            image_points = np.array(
                [
                    self._landmark_xy(landmarks[self.LANDMARK_IDS["nose_tip"]], width, height),
                    self._landmark_xy(landmarks[self.LANDMARK_IDS["chin"]], width, height),
                    self._landmark_xy(landmarks[self.LANDMARK_IDS["left_eye_outer"]], width, height),
                    self._landmark_xy(landmarks[self.LANDMARK_IDS["right_eye_outer"]], width, height),
                    self._landmark_xy(landmarks[self.LANDMARK_IDS["left_mouth"]], width, height),
                    self._landmark_xy(landmarks[self.LANDMARK_IDS["right_mouth"]], width, height),
                ],
                dtype=np.float64,
            )
        except Exception as exc:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, f"landmark_parse_error:{exc}", True)

        focal = float(width)
        camera_matrix = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        try:
            ok, rvec, _tvec = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return HeadPoseResult(False, None, None, None, "unknown", 0.0, "solvepnp_failed", True)
            rotation, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = self._rotation_matrix_to_euler(rotation)
        except Exception as exc:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, f"solvepnp_error:{exc}", True)

        raw_classification, raw_confidence, reason = self.classify_pose(yaw, pitch, roll)
        self._history.append((yaw, pitch, roll, raw_classification, raw_confidence))
        smoothed = self._smoothed_result(reason)
        return smoothed

    def classify_pose(self, yaw: float, pitch: float, roll: float) -> tuple[str, float, str]:
        abs_yaw = abs(float(yaw))
        abs_pitch = abs(float(pitch))
        if abs_yaw <= self.yaw_at_lamp_deg and abs_pitch <= self.pitch_at_lamp_deg:
            margin = min(
                1.0 - abs_yaw / max(self.yaw_at_lamp_deg, 1e-6),
                1.0 - abs_pitch / max(self.pitch_at_lamp_deg, 1e-6),
            )
            return "looking_at_lamp", min(0.98, 0.62 + 0.34 * max(0.0, margin)), "head_pose_forward"
        if pitch < -self.pitch_away_deg:
            return "looking_down", min(0.95, 0.55 + abs(pitch) / 100.0), "pitch_down"
        if pitch > self.pitch_away_deg:
            return "looking_away", min(0.90, 0.50 + abs(pitch) / 120.0), "pitch_up_away"
        if yaw < -self.yaw_away_deg:
            return "looking_left", min(0.95, 0.55 + abs(yaw) / 100.0), "yaw_left"
        if yaw > self.yaw_away_deg:
            return "looking_right", min(0.95, 0.55 + abs(yaw) / 100.0), "yaw_right"
        return "unknown", 0.45, "pose_between_thresholds"

    def close(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
            self._mesh = None

    def _smoothed_result(self, reason: str) -> HeadPoseResult:
        if not self._history:
            return HeadPoseResult(False, None, None, None, "unknown", 0.0, "no_history", True)
        yaws = [item[0] for item in self._history]
        pitches = [item[1] for item in self._history]
        rolls = [item[2] for item in self._history]
        labels = [item[3] for item in self._history]
        confidences = [item[4] for item in self._history]
        label = max(set(labels), key=labels.count)
        if labels.count(label) < max(1, int(round(len(labels) * 0.45))):
            label = labels[-1]
        confidence = sum(conf for lab, conf in zip(labels, confidences) if lab == label) / max(1, labels.count(label))
        return HeadPoseResult(
            True,
            float(sum(yaws) / len(yaws)),
            float(sum(pitches) / len(pitches)),
            float(sum(rolls) / len(rolls)),
            label,
            confidence,
            reason,
            False,
        )

    @staticmethod
    def _landmark_xy(landmark: Any, width: int, height: int) -> tuple[float, float]:
        return (float(landmark.x) * float(width), float(landmark.y) * float(height))

    @staticmethod
    def _rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
        # OpenCV camera coordinates. This returns intuitive approximate degrees
        # for webcam usage; signs are normalized in classify_pose.
        sy = (rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]) ** 0.5
        singular = sy < 1e-6
        if not singular:
            x = atan2(rotation[2, 1], rotation[2, 2])
            y = atan2(-rotation[2, 0], sy)
            z = atan2(rotation[1, 0], rotation[0, 0])
        else:
            x = atan2(-rotation[1, 2], rotation[1, 1])
            y = atan2(-rotation[2, 0], sy)
            z = 0.0
        pitch = degrees(x)
        yaw = degrees(y)
        roll = degrees(z)
        return yaw, pitch, roll
