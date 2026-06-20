"""Face-presence and face-position engagement detector.

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
        return payload

    def to_log_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 3)
        data["face_area_ratio"] = round(float(self.face_area_ratio), 4)
        data["selected_face_score"] = round(float(self.selected_face_score), 3)
        return data


class FaceEngagementDetector:
    def __init__(self, config: EngagementConfig) -> None:
        self.config = config
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from: {cascade_path}")

        self._primary_bbox: Optional[BBox] = None
        self._clahe = None
        if self.config.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size,
            )

    def detect(self, frame) -> EngagementResult:
        height, width = frame.shape[:2]
        gray = self._preprocess(frame)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.cascade_scale_factor,
            minNeighbors=self.config.cascade_min_neighbors,
            minSize=self.config.cascade_min_size,
        )

        candidates = self._build_candidates(faces, width, height)
        if not candidates:
            return EngagementResult(
                status="absent",
                confidence=0.0,
                reason="no_stable_face_candidate" if len(faces) else "no_face_detected",
                raw_face_count=int(len(faces)),
                candidate_count=0,
            )

        selected = self._select_primary_candidate(candidates)
        self._primary_bbox = selected.bbox
        return self._classify_candidate(selected, width, height, raw_face_count=int(len(faces)), candidate_count=len(candidates))

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.blur_kernel_size and self.config.blur_kernel_size > 1:
            k = self.config.blur_kernel_size
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        else:
            gray = cv2.equalizeHist(gray)
        return gray

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
    ) -> EngagementResult:
        x, y, w, h = candidate.bbox
        cx, cy = candidate.center_norm
        area_ratio = candidate.area_ratio

        x_offset = abs(cx - 0.5)
        y_offset = abs(cy - 0.5)

        horizontally_centered = x_offset <= self.config.center_tolerance_x
        vertically_centered = y_offset <= self.config.center_tolerance_y
        face_large_enough = area_ratio >= self.config.min_face_area_ratio

        if horizontally_centered and vertically_centered and face_large_enough:
            confidence = self._engaged_confidence(x_offset, y_offset, area_ratio, candidate.selection_score)
            return EngagementResult(
                status="engaged",
                confidence=confidence,
                reason="face_centered",
                face_bbox=(int(x), int(y), int(w), int(h)),
                face_center_norm=(round(cx, 3), round(cy, 3)),
                face_area_ratio=area_ratio,
                raw_face_count=raw_face_count,
                candidate_count=candidate_count,
                selected_face_score=candidate.selection_score,
            )

        reason_parts: list[str] = []
        if not horizontally_centered:
            reason_parts.append("face_left_or_right")
        if not vertically_centered:
            reason_parts.append("face_high_or_low")
        if not face_large_enough:
            reason_parts.append("face_too_small")

        confidence = self._disengaged_confidence(x_offset, y_offset, area_ratio, candidate.selection_score)
        return EngagementResult(
            status="disengaged",
            confidence=confidence,
            reason="+".join(reason_parts) or "face_not_centered",
            face_bbox=(int(x), int(y), int(w), int(h)),
            face_center_norm=(round(cx, 3), round(cy, 3)),
            face_area_ratio=area_ratio,
            raw_face_count=raw_face_count,
            candidate_count=candidate_count,
            selected_face_score=candidate.selection_score,
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
