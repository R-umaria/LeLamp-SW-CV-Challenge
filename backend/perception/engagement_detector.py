"""Primary-face engagement detector built on top of multi-face detection.

Milestone 1.5 still avoids heavy gaze/head-pose dependencies. It improves the
Milestone 1 OpenCV-only detector by:

1. applying low-light preprocessing,
2. filtering tiny face candidates,
3. tracking the primary face by bbox continuity, size, and center proximity, and
4. exposing richer debug fields for overlays and logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Optional, Sequence

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "OpenCV is required for engagement detection. Install with: pip install opencv-python"
    ) from exc

from backend.utils.config import EngagementConfig
from backend.perception.multi_face_detector import FaceDetection, MultiFaceDetector
from backend.perception.head_pose_estimator import HeadPoseEstimator, HeadPoseResult


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceCandidate:
    bbox: BBox
    center_norm: tuple[float, float]
    area_ratio: float
    selection_score: float = 0.0


@dataclass(frozen=True)
class EngagementResult:
    status: str  # "engaged", "disengaged", or "absent"
    confidence: float
    reason: str
    face_bbox: Optional[BBox] = None
    face_center_norm: Optional[tuple[float, float]] = None
    face_area_ratio: float = 0.0
    raw_face_count: int = 0
    candidate_count: int = 0
    selected_face_score: float = 0.0
    head_pose_status: str = "unknown"
    head_pose_confidence: float = 0.0
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    head_pose_reason: str = "not_evaluated"
    fallback_mode_used: bool = True

    def to_protocol_dict(self) -> dict:
        # Preserve the existing command protocol and add optional normalized face
        # coordinates only when the detector has a stable face. Godot treats these
        # as hints for subtle face-following, not as perception decisions.
        payload = {
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
        }
        if self.face_center_norm is not None:
            payload["face_x_norm"] = round(float(self.face_center_norm[0]), 3)
            payload["face_y_norm"] = round(float(self.face_center_norm[1]), 3)
            payload["face_area_ratio"] = round(float(self.face_area_ratio), 4)
        if self.yaw is not None:
            payload["yaw"] = round(float(self.yaw), 2)
        if self.pitch is not None:
            payload["pitch"] = round(float(self.pitch), 2)
        if self.roll is not None:
            payload["roll"] = round(float(self.roll), 2)
        payload["head_pose_status"] = self.head_pose_status
        payload["head_pose_confidence"] = round(float(self.head_pose_confidence), 3)
        payload["fallback_mode_used"] = bool(self.fallback_mode_used)
        return payload

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 3)
        data["face_area_ratio"] = round(float(self.face_area_ratio), 4)
        data["selected_face_score"] = round(float(self.selected_face_score), 3)
        data["head_pose_confidence"] = round(float(self.head_pose_confidence), 3)
        if self.yaw is not None:
            data["yaw"] = round(float(self.yaw), 2)
        if self.pitch is not None:
            data["pitch"] = round(float(self.pitch), 2)
        if self.roll is not None:
            data["roll"] = round(float(self.roll), 2)
        return data


class FaceEngagementDetector:
    def __init__(self, config: EngagementConfig) -> None:
        self.config = config
        self.face_detector = MultiFaceDetector(self.config)
        self.head_pose_estimator = HeadPoseEstimator(
            enabled=getattr(config, "head_pose_enabled", True),
            max_history=getattr(config, "head_pose_history", 5),
            yaw_at_lamp_deg=getattr(config, "head_pose_yaw_at_lamp_deg", 24.0),
            pitch_at_lamp_deg=getattr(config, "head_pose_pitch_at_lamp_deg", 22.0),
            yaw_away_deg=getattr(config, "head_pose_yaw_away_deg", 38.0),
            pitch_away_deg=getattr(config, "head_pose_pitch_away_deg", 32.0),
        )
        self._primary_bbox: Optional[BBox] = None

    def detect(self, frame) -> EngagementResult:
        height, width = frame.shape[:2]
        detections = self.face_detector.detect(frame)

        candidates = self._build_candidates_from_detections(detections)
        if not candidates:
            return EngagementResult(
                status="absent",
                confidence=0.0,
                reason="no_stable_face_candidate" if len(detections) else "no_face_detected",
                raw_face_count=int(len(detections)),
                candidate_count=0,
            )

        selected = self._select_primary_candidate(candidates)
        self._primary_bbox = selected.bbox
        head_pose = self.head_pose_estimator.estimate(frame, selected.bbox)
        return self._classify_candidate(
            selected,
            width,
            height,
            raw_face_count=int(len(detections)),
            candidate_count=len(candidates),
            head_pose=head_pose,
        )

    def close(self) -> None:
        self.face_detector.close()
        self.head_pose_estimator.close()

    def _build_candidates_from_detections(self, detections: Sequence[FaceDetection]) -> list[FaceCandidate]:
        return [
            FaceCandidate(
                bbox=detection.bbox,
                center_norm=detection.center_norm,
                area_ratio=detection.area_ratio,
            )
            for detection in detections
        ]

    def _build_candidates(self, faces: Sequence, width: int, height: int) -> list[FaceCandidate]:
        candidates: list[FaceCandidate] = []
        frame_area = float(width * height)
        for face in faces:
            x, y, w, h = [int(v) for v in face]
            area_ratio = (w * h) / frame_area
            if area_ratio < self.config.min_candidate_area_ratio:
                continue
            if area_ratio > self.config.max_candidate_area_ratio:
                continue

            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            candidates.append(
                FaceCandidate(
                    bbox=(x, y, w, h),
                    center_norm=(round(cx, 3), round(cy, 3)),
                    area_ratio=area_ratio,
                )
            )
        return candidates

    def _select_primary_candidate(self, candidates: list[FaceCandidate]) -> FaceCandidate:
        scored = []
        for candidate in candidates:
            score = self._candidate_score(candidate)
            scored.append(
                FaceCandidate(
                    bbox=candidate.bbox,
                    center_norm=candidate.center_norm,
                    area_ratio=candidate.area_ratio,
                    selection_score=score,
                )
            )
        return max(scored, key=lambda c: c.selection_score)

    def _candidate_score(self, candidate: FaceCandidate) -> float:
        size_score = min(1.0, candidate.area_ratio / max(self.config.min_face_area_ratio * 4.0, 1e-6))
        center_score = self._center_score(candidate.center_norm)
        continuity_score = self._continuity_score(candidate)
        return (
            self.config.primary_size_weight * size_score
            + self.config.primary_center_weight * center_score
            + self.config.primary_continuity_weight * continuity_score
        )

    def _center_score(self, center_norm: tuple[float, float]) -> float:
        cx, cy = center_norm
        dx = abs(cx - 0.5) / max(self.config.center_tolerance_x, 1e-6)
        dy = abs(cy - 0.5) / max(self.config.center_tolerance_y, 1e-6)
        return max(0.0, 1.0 - min(1.0, (dx + dy) / 2.0))

    def _continuity_score(self, candidate: FaceCandidate) -> float:
        if self._primary_bbox is None:
            return 0.0

        px, py, pw, ph = self._primary_bbox
        x, y, w, h = candidate.bbox
        prev_cx = px + pw / 2.0
        prev_cy = py + ph / 2.0
        curr_cx = x + w / 2.0
        curr_cy = y + h / 2.0
        prev_diag = max(sqrt(pw * pw + ph * ph), 1.0)
        center_distance = sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) / prev_diag
        center_score = max(0.0, 1.0 - center_distance / max(self.config.max_primary_center_distance, 1e-6))

        prev_area = max(float(pw * ph), 1.0)
        curr_area = max(float(w * h), 1.0)
        area_ratio = max(curr_area / prev_area, prev_area / curr_area)
        if area_ratio > self.config.max_primary_area_change_ratio:
            area_score = 0.0
        else:
            area_score = 1.0 - ((area_ratio - 1.0) / max(self.config.max_primary_area_change_ratio - 1.0, 1e-6))

        return 0.70 * center_score + 0.30 * area_score

    def _classify_candidate(
        self,
        candidate: FaceCandidate,
        width: int,
        height: int,
        raw_face_count: int,
        candidate_count: int,
        head_pose: HeadPoseResult | None = None,
    ) -> EngagementResult:
        x, y, w, h = candidate.bbox
        cx, cy = candidate.center_norm
        area_ratio = candidate.area_ratio

        x_offset = abs(cx - 0.5)
        y_offset = abs(cy - 0.5)

        horizontally_centered = x_offset <= self.config.center_tolerance_x
        vertically_centered = y_offset <= self.config.center_tolerance_y
        face_large_enough = area_ratio >= self.config.min_face_area_ratio

        pose_status = "unknown" if head_pose is None else head_pose.classification
        pose_confidence = 0.0 if head_pose is None else float(head_pose.confidence)
        pose_available = bool(head_pose is not None and head_pose.available)
        pose_reason = "not_evaluated" if head_pose is None else head_pose.reason
        fallback_mode_used = not pose_available or pose_status == "unknown"

        # Head pose is the stronger signal when available. This addresses the
        # challenge requirement for practical gaze/head-pose approximation while
        # preserving the old geometry fallback when MediaPipe/solvePnP fails.
        if pose_available and pose_status in {"looking_left", "looking_right", "looking_down", "looking_away"}:
            confidence = max(0.55, min(0.96, pose_confidence))
            return self._result_from_candidate(
                "disengaged",
                confidence,
                pose_status,
                candidate,
                raw_face_count,
                candidate_count,
                head_pose,
                fallback_mode_used=False,
            )

        if horizontally_centered and vertically_centered and face_large_enough:
            confidence = self._engaged_confidence(x_offset, y_offset, area_ratio, candidate.selection_score)
            reason = "face_centered"
            if pose_available and pose_status == "looking_at_lamp":
                confidence = min(0.99, max(confidence, 0.58 + 0.35 * pose_confidence))
                reason = "head_pose_looking_at_lamp+face_centered"
                fallback_mode_used = False
            return self._result_from_candidate(
                "engaged",
                confidence,
                reason,
                candidate,
                raw_face_count,
                candidate_count,
                head_pose,
                fallback_mode_used=fallback_mode_used,
            )

        # A confident forward head pose can rescue slightly off-center faces, but
        # not tiny/far-away faces. This makes engagement less dependent on exact
        # face-center placement without making every visible face "engaged".
        if pose_available and pose_status == "looking_at_lamp" and face_large_enough:
            confidence = min(0.94, max(0.64, 0.50 + 0.30 * pose_confidence + 0.10 * candidate.selection_score))
            return self._result_from_candidate(
                "engaged",
                confidence,
                "head_pose_looking_at_lamp",
                candidate,
                raw_face_count,
                candidate_count,
                head_pose,
                fallback_mode_used=False,
            )

        reason_parts: list[str] = []
        if not horizontally_centered:
            reason_parts.append("face_left_or_right")
        if not vertically_centered:
            reason_parts.append("face_high_or_low")
        if not face_large_enough:
            reason_parts.append("face_too_small")

        confidence = self._disengaged_confidence(x_offset, y_offset, area_ratio, candidate.selection_score)
        reason = "+".join(reason_parts) or "face_not_centered"
        if head_pose is not None and head_pose.reason and fallback_mode_used:
            reason = f"{reason}+pose_fallback:{head_pose.reason}"
        return self._result_from_candidate(
            "disengaged",
            confidence,
            reason,
            candidate,
            raw_face_count,
            candidate_count,
            head_pose,
            fallback_mode_used=fallback_mode_used,
        )

    def _result_from_candidate(
        self,
        status: str,
        confidence: float,
        reason: str,
        candidate: FaceCandidate,
        raw_face_count: int,
        candidate_count: int,
        head_pose: HeadPoseResult | None,
        *,
        fallback_mode_used: bool,
    ) -> EngagementResult:
        x, y, w, h = candidate.bbox
        cx, cy = candidate.center_norm
        return EngagementResult(
            status=status,
            confidence=float(confidence),
            reason=reason,
            face_bbox=(int(x), int(y), int(w), int(h)),
            face_center_norm=(round(cx, 3), round(cy, 3)),
            face_area_ratio=candidate.area_ratio,
            raw_face_count=raw_face_count,
            candidate_count=candidate_count,
            selected_face_score=candidate.selection_score,
            head_pose_status="unknown" if head_pose is None else head_pose.classification,
            head_pose_confidence=0.0 if head_pose is None else float(head_pose.confidence),
            yaw=None if head_pose is None else head_pose.yaw,
            pitch=None if head_pose is None else head_pose.pitch,
            roll=None if head_pose is None else head_pose.roll,
            head_pose_reason="not_evaluated" if head_pose is None else head_pose.reason,
            fallback_mode_used=bool(fallback_mode_used),
        )

    def _engaged_confidence(
        self,
        x_offset: float,
        y_offset: float,
        area_ratio: float,
        selection_score: float,
    ) -> float:
        x_score = max(0.0, 1.0 - x_offset / max(self.config.center_tolerance_x, 1e-6))
        y_score = max(0.0, 1.0 - y_offset / max(self.config.center_tolerance_y, 1e-6))
        size_score = min(1.0, area_ratio / max(self.config.min_face_area_ratio * 4.0, 1e-6))
        return min(0.99, 0.40 + 0.24 * x_score + 0.18 * y_score + 0.10 * size_score + 0.08 * selection_score)

    def _disengaged_confidence(
        self,
        x_offset: float,
        y_offset: float,
        area_ratio: float,
        selection_score: float,
    ) -> float:
        x_excess = max(0.0, x_offset - self.config.center_tolerance_x)
        y_excess = max(0.0, y_offset - self.config.center_tolerance_y)
        position_score = min(1.0, (x_excess + y_excess) * 3.0)
        size_penalty_score = 1.0 if area_ratio < self.config.min_face_area_ratio else 0.0
        return min(0.95, 0.50 + 0.25 * position_score + 0.12 * size_penalty_score + 0.08 * selection_score)


def draw_engagement_overlay(
    frame,
    raw_result: EngagementResult,
    smoothed_result: EngagementResult,
    state: str,
    state_elapsed_s: float,
    fps: float,
    config: EngagementConfig,
) -> None:
    """Draw debug overlay in-place for the optional OpenCV preview window."""
    lines = [
        f"state={state} dwell={state_elapsed_s:.1f}s fps={fps:.1f}",
        f"raw={raw_result.status} smoothed={smoothed_result.status} conf={smoothed_result.confidence:.2f}",
        f"reason={smoothed_result.reason}",
        f"pose={smoothed_result.head_pose_status} yaw={smoothed_result.yaw if smoothed_result.yaw is not None else 'n/a'} pitch={smoothed_result.pitch if smoothed_result.pitch is not None else 'n/a'}",
        f"area={smoothed_result.face_area_ratio:.3f} raw_faces={raw_result.raw_face_count} candidates={raw_result.candidate_count}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 26 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2 if idx == 0 else 1,
        )

    if smoothed_result.face_bbox is not None:
        x, y, w, h = smoothed_result.face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"bbox=({x},{y},{w},{h})",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    height, width = frame.shape[:2]
    x_tol = int(width * config.center_tolerance_x)
    y_tol = int(height * config.center_tolerance_y)
    cx, cy = width // 2, height // 2
    cv2.rectangle(frame, (cx - x_tol, cy - y_tol), (cx + x_tol, cy + y_tol), (255, 255, 255), 1)
