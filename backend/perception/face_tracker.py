"""Temporary visual face tracks for active-speaker awareness.

This is not identity recognition. Track ids such as ``person_1`` are short-lived
continuity labels based on bounding-box overlap and centroid distance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required for face tracking. Install opencv-python.") from exc

from backend.perception.engagement_detector import BBox, EngagementResult
from backend.utils.config import EngagementConfig, SpeakerAwarenessConfig


@dataclass(frozen=True)
class FaceTrack:
    track_id: str
    bbox: BBox
    center_norm: tuple[float, float]
    location: str
    mouth_motion_score: float
    mouth_opening: float | None
    engaged: bool | None
    confidence: float
    source: str
    age_s: float
    last_seen_s: float

    def to_log_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "center_norm": [round(float(self.center_norm[0]), 3), round(float(self.center_norm[1]), 3)],
            "location": self.location,
            "mouth_motion_score": round(float(self.mouth_motion_score), 3),
            "mouth_opening": None if self.mouth_opening is None else round(float(self.mouth_opening), 3),
            "engaged": self.engaged,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "age_s": round(float(self.age_s), 3),
        }


@dataclass
class _MutableTrack:
    track_id: str
    bbox: BBox
    center_norm: tuple[float, float]
    created_s: float
    last_seen_s: float
    mouth_motion_score: float = 0.0
    mouth_opening: float | None = None
    lower_face_patch: np.ndarray | None = None
    source: str = "lower_face_motion"


class FaceTracker:
    def __init__(
        self,
        engagement_config: EngagementConfig,
        speaker_config: SpeakerAwarenessConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engagement_config = engagement_config
        self.speaker_config = speaker_config
        self.logger = logger or logging.getLogger("lelamp")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from: {cascade_path}")
        self._tracks: dict[str, _MutableTrack] = {}
        self._next_id = 1
        self._facemesh = None
        self._facemesh_available = False
        if self.speaker_config.enabled:
            self._try_init_facemesh()

    def _try_init_facemesh(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            self._facemesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=False,
                min_detection_confidence=0.50,
                min_tracking_confidence=0.50,
            )
            self._facemesh_available = True
            self.logger.info("face_tracker_facemesh_available=true")
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._facemesh = None
            self._facemesh_available = False
            self.logger.info("face_tracker_facemesh_available=false reason=%s", exc)

    def update(self, frame, engagement_result: EngagementResult | None = None, now: float | None = None) -> list[FaceTrack]:
        now_s = time.monotonic() if now is None else float(now)
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.engagement_config.cascade_scale_factor,
            minNeighbors=max(3, self.engagement_config.cascade_min_neighbors - 1),
            minSize=self.engagement_config.cascade_min_size,
        )
        bboxes = self._filter_faces(faces, width, height)
        mesh_estimates = self._facemesh_mouth_estimates(frame) if self._facemesh_available else []

        matched_ids: set[str] = set()
        for bbox in bboxes:
            center_norm = _bbox_center_norm(bbox, width, height)
            track = self._match_existing_track(bbox, center_norm, now_s)
            if track is None:
                track_id = f"person_{self._next_id}"
                self._next_id += 1
                track = _MutableTrack(track_id=track_id, bbox=bbox, center_norm=center_norm, created_s=now_s, last_seen_s=now_s)
                self._tracks[track_id] = track
            else:
                track.bbox = bbox
                track.center_norm = center_norm
                track.last_seen_s = now_s
            matched_ids.add(track.track_id)
            self._update_mouth_motion(track, frame, gray, mesh_estimates)

        self._drop_stale(now_s)
        results: list[FaceTrack] = []
        for track in self._tracks.values():
            if track.track_id not in matched_ids:
                continue
            results.append(self._to_face_track(track, width, height, engagement_result, now_s))
        results.sort(key=lambda t: (t.center_norm[0], t.track_id))
        return results

    def close(self) -> None:
        if self._facemesh is not None:
            try:
                self._facemesh.close()
            except Exception:
                pass
            self._facemesh = None

    def _filter_faces(self, faces, width: int, height: int) -> list[BBox]:
        frame_area = float(width * height)
        out: list[BBox] = []
        for face in faces:
            x, y, w, h = [int(v) for v in face]
            area_ratio = (w * h) / max(frame_area, 1.0)
            if area_ratio < self.engagement_config.min_candidate_area_ratio:
                continue
            if area_ratio > self.engagement_config.max_candidate_area_ratio:
                continue
            out.append((x, y, w, h))
        return out

    def _match_existing_track(self, bbox: BBox, center_norm: tuple[float, float], now_s: float) -> _MutableTrack | None:
        best_track = None
        best_score = -1.0
        for track in self._tracks.values():
            age_since_seen = now_s - track.last_seen_s
            if age_since_seen > self.speaker_config.face_track_ttl_s:
                continue
            iou = _bbox_iou(bbox, track.bbox)
            distance = _center_distance(center_norm, track.center_norm)
            distance_score = max(0.0, 1.0 - distance / max(self.speaker_config.max_face_match_distance_norm, 1e-6))
            score = 0.65 * iou + 0.35 * distance_score
            if score > best_score:
                best_score = score
                best_track = track
        if best_track is None:
            return None
        if best_score < 0.20:
            return None
        return best_track

    def _drop_stale(self, now_s: float) -> None:
        stale = [tid for tid, track in self._tracks.items() if now_s - track.last_seen_s > self.speaker_config.face_track_ttl_s]
        for tid in stale:
            self._tracks.pop(tid, None)

    def _facemesh_mouth_estimates(self, frame) -> list[dict]:
        if self._facemesh is None:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._facemesh.process(rgb)
        except Exception as exc:  # pragma: no cover - optional runtime guard
            self.logger.debug("face_tracker_facemesh_failed error=%s", exc)
            return []
        estimates: list[dict] = []
        if not getattr(result, "multi_face_landmarks", None):
            return estimates
        height, width = frame.shape[:2]
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x0 = max(0, int(min(xs) * width))
            y0 = max(0, int(min(ys) * height))
            x1 = min(width - 1, int(max(xs) * width))
            y1 = min(height - 1, int(max(ys) * height))
            face_h = max(1.0, float(y1 - y0))
            # MediaPipe Face Mesh approximate lip landmarks: upper/lower inner lip.
            try:
                upper = landmarks[13]
                lower = landmarks[14]
                mouth_opening = abs(float(lower.y - upper.y)) * height / face_h
            except Exception:
                mouth_opening = None
            estimates.append(
                {
                    "bbox": (x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
                    "center_norm": ((x0 + x1) / 2.0 / width, (y0 + y1) / 2.0 / height),
                    "mouth_opening": mouth_opening,
                }
            )
        return estimates

    def _update_mouth_motion(self, track: _MutableTrack, frame, gray, mesh_estimates: list[dict]) -> None:
        mesh = self._match_mesh_estimate(track.bbox, track.center_norm, mesh_estimates)
        mesh_score = None
        if mesh is not None and mesh.get("mouth_opening") is not None:
            current_opening = float(mesh["mouth_opening"])
            previous_opening = track.mouth_opening
            delta = 0.0 if previous_opening is None else abs(current_opening - float(previous_opening))
            # Both visible openness and frame-to-frame change contribute. Speech
            # often alternates mouth opening rather than staying wide open.
            mesh_score = min(1.0, current_opening * 3.8 + delta * 12.0)
            track.mouth_opening = current_opening
            track.source = "mediapipe_facemesh"

        fallback_score = self._lower_face_motion_score(track, gray)
        if mesh_score is None:
            score = fallback_score
            track.source = "lower_face_motion"
        else:
            score = max(mesh_score, fallback_score * 0.65)
        track.mouth_motion_score = 0.72 * float(track.mouth_motion_score) + 0.28 * float(score)

    def _match_mesh_estimate(self, bbox: BBox, center_norm: tuple[float, float], estimates: list[dict]) -> dict | None:
        best = None
        best_score = -1.0
        for estimate in estimates:
            iou = _bbox_iou(bbox, estimate["bbox"])
            distance = _center_distance(center_norm, estimate["center_norm"])
            score = 0.70 * iou + 0.30 * max(0.0, 1.0 - distance / 0.20)
            if score > best_score:
                best_score = score
                best = estimate
        return best if best_score >= 0.20 else None

    def _lower_face_motion_score(self, track: _MutableTrack, gray) -> float:
        x, y, w, h = track.bbox
        y0 = int(y + h * 0.55)
        y1 = int(y + h * 0.95)
        x0 = int(x + w * 0.18)
        x1 = int(x + w * 0.82)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = gray[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
        if patch.size == 0:
            return 0.0
        patch = cv2.resize(patch, (64, 32), interpolation=cv2.INTER_AREA)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        if track.lower_face_patch is None:
            track.lower_face_patch = patch
            return 0.0
        diff = float(np.mean(np.abs(patch.astype(np.float32) - track.lower_face_patch.astype(np.float32)))) / 255.0
        track.lower_face_patch = patch
        return min(1.0, max(0.0, diff * 8.0))

    def _to_face_track(
        self,
        track: _MutableTrack,
        width: int,
        height: int,
        engagement_result: EngagementResult | None,
        now_s: float,
    ) -> FaceTrack:
        location = _screen_location(track.center_norm[0])
        engaged = self._track_engaged(track, width, height, engagement_result)
        area_ratio = (track.bbox[2] * track.bbox[3]) / max(float(width * height), 1.0)
        confidence = min(0.95, 0.40 + 0.25 * min(1.0, area_ratio / 0.08) + 0.30 * min(1.0, track.mouth_motion_score))
        return FaceTrack(
            track_id=track.track_id,
            bbox=track.bbox,
            center_norm=(round(track.center_norm[0], 3), round(track.center_norm[1], 3)),
            location=location,
            mouth_motion_score=float(track.mouth_motion_score),
            mouth_opening=track.mouth_opening,
            engaged=engaged,
            confidence=float(confidence),
            source=track.source,
            age_s=max(0.0, now_s - track.created_s),
            last_seen_s=track.last_seen_s,
        )

    def _track_engaged(self, track: _MutableTrack, width: int, height: int, engagement_result: EngagementResult | None) -> bool | None:
        if engagement_result is not None and engagement_result.face_bbox is not None:
            if _bbox_iou(track.bbox, engagement_result.face_bbox) >= 0.25:
                return engagement_result.status == "engaged"
        cx, cy = track.center_norm
        area_ratio = (track.bbox[2] * track.bbox[3]) / max(float(width * height), 1.0)
        centered = (
            abs(cx - 0.5) <= self.engagement_config.center_tolerance_x
            and abs(cy - 0.5) <= self.engagement_config.center_tolerance_y
        )
        if area_ratio < self.engagement_config.min_candidate_area_ratio:
            return None
        return bool(centered and area_ratio >= self.engagement_config.min_face_area_ratio)


def _bbox_center_norm(bbox: BBox, width: int, height: int) -> tuple[float, float]:
    x, y, w, h = bbox
    return ((x + w / 2.0) / max(width, 1), (y + h / 2.0) / max(height, 1))


def _screen_location(cx: float) -> str:
    if cx < 0.36:
        return "left"
    if cx > 0.64:
        return "right"
    return "center"


def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = float(inter_w * inter_h)
    union = float(aw * ah + bw * bh - inter)
    return 0.0 if union <= 0.0 else inter / union
