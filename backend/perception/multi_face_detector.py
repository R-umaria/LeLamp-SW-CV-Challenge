"""Multi-face detection utilities for Lumos perception.

This module deliberately separates *detecting every visible face* from
*choosing the primary engaged face*.  Engagement can still select one face for
lamp-facing behavior, but active-speaker awareness needs the full set of visible
people so it can assign speech to a non-primary person.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required for multi-face detection. Install opencv-python.") from exc

from backend.utils.config import EngagementConfig

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceDetection:
    bbox: BBox
    center_norm: tuple[float, float]
    area_ratio: float
    confidence: float
    source: str

    def to_log_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "center_norm": [round(float(self.center_norm[0]), 3), round(float(self.center_norm[1]), 3)],
            "area_ratio": round(float(self.area_ratio), 4),
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
        }


class MultiFaceDetector:
    """Detect all credible visible faces using MediaPipe when available.

    MediaPipe FaceDetection is better than Haar cascades for multiple people and
    non-perfect frontal poses.  Haar frontal/profile cascades remain as an
    offline fallback so the demo still runs when MediaPipe is unavailable.
    """

    def __init__(
        self,
        config: EngagementConfig,
        *,
        logger: logging.Logger | None = None,
        enable_mediapipe: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self._mp_face_detection = None
        self._mp_available = False

        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self._frontal_cascade = cv2.CascadeClassifier(frontal_path)
        self._profile_cascade = cv2.CascadeClassifier(profile_path)
        if self._frontal_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV frontal face cascade from: {frontal_path}")
        if self._profile_cascade.empty():
            self.logger.info("multi_face_profile_cascade_available=false path=%s", profile_path)

        self._clahe = None
        if self.config.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size,
            )

        if enable_mediapipe:
            self._try_init_mediapipe()

    @property
    def mediapipe_available(self) -> bool:
        return self._mp_available

    def close(self) -> None:
        if self._mp_face_detection is not None:
            try:
                self._mp_face_detection.close()
            except Exception:
                pass
            self._mp_face_detection = None
            self._mp_available = False

    def detect(self, frame) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        detections: list[FaceDetection] = []
        if self._mp_available:
            detections.extend(self._detect_mediapipe(frame, width, height))
        detections.extend(self._detect_haar(frame, width, height))
        return self._dedupe_and_sort(detections)

    def _try_init_mediapipe(self) -> None:
        try:
            try:
                import mediapipe as mp  # type: ignore
                face_detection_module = mp.solutions.face_detection
            except Exception:
                from mediapipe.python.solutions import face_detection as face_detection_module  # type: ignore

            # model_selection=1 is the full-range detector, which is better for
            # second users who are not as close to the camera as the primary user.
            self._mp_face_detection = face_detection_module.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.45,
            )
            self._mp_available = True
            self.logger.info("multi_face_mediapipe_available=true")
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._mp_face_detection = None
            self._mp_available = False
            self.logger.info("multi_face_mediapipe_available=false reason=%s", exc)

    def _detect_mediapipe(self, frame, width: int, height: int) -> list[FaceDetection]:
        if self._mp_face_detection is None:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mp_face_detection.process(rgb)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.logger.debug("multi_face_mediapipe_failed error=%s", exc)
            return []

        out: list[FaceDetection] = []
        for detection in getattr(result, "detections", None) or []:
            loc = getattr(detection, "location_data", None)
            rel = getattr(loc, "relative_bounding_box", None)
            if rel is None:
                continue
            x = int(round(float(rel.xmin) * width))
            y = int(round(float(rel.ymin) * height))
            w = int(round(float(rel.width) * width))
            h = int(round(float(rel.height) * height))
            bbox = _clip_bbox((x, y, w, h), width, height)
            if bbox is None:
                continue
            score_values = getattr(detection, "score", None) or []
            confidence = float(score_values[0]) if score_values else 0.70
            candidate = _make_detection(bbox, width, height, confidence, "mediapipe_face_detection")
            if self._accept(candidate):
                out.append(candidate)
        return out

    def _detect_haar(self, frame, width: int, height: int) -> list[FaceDetection]:
        gray = self._preprocess(frame)
        min_neighbors = max(3, int(self.config.cascade_min_neighbors) - 1)
        faces = list(
            self._frontal_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.cascade_scale_factor,
                minNeighbors=min_neighbors,
                minSize=self.config.cascade_min_size,
            )
        )
        out = [
            _make_detection((int(x), int(y), int(w), int(h)), width, height, 0.62, "haar_frontal")
            for x, y, w, h in faces
        ]

        if not self._profile_cascade.empty():
            out.extend(self._detect_profile_faces(gray, width, height, flipped=False))
            flipped_gray = cv2.flip(gray, 1)
            out.extend(self._detect_profile_faces(flipped_gray, width, height, flipped=True))

        return [candidate for candidate in out if self._accept(candidate)]

    def _detect_profile_faces(self, gray, width: int, height: int, *, flipped: bool) -> list[FaceDetection]:
        min_neighbors = max(3, int(self.config.cascade_min_neighbors) - 1)
        faces = self._profile_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.cascade_scale_factor,
            minNeighbors=min_neighbors,
            minSize=self.config.cascade_min_size,
        )
        out: list[FaceDetection] = []
        for face in faces:
            x, y, w, h = [int(v) for v in face]
            if flipped:
                x = width - x - w
            out.append(_make_detection((x, y, w, h), width, height, 0.54, "haar_profile"))
        return out

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.blur_kernel_size and self.config.blur_kernel_size > 1:
            k = int(self.config.blur_kernel_size)
            if k % 2 == 0:
                k += 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        if self._clahe is not None:
            return self._clahe.apply(gray)
        return cv2.equalizeHist(gray)

    def _accept(self, detection: FaceDetection) -> bool:
        if detection.area_ratio < float(self.config.min_candidate_area_ratio):
            return False
        if detection.area_ratio > float(self.config.max_candidate_area_ratio):
            return False
        return True

    def _dedupe_and_sort(self, detections: Iterable[FaceDetection]) -> list[FaceDetection]:
        # Prefer higher confidence and MediaPipe detections when boxes overlap.
        source_rank = {
            "mediapipe_face_detection": 3,
            "haar_frontal": 2,
            "haar_profile": 1,
        }
        ordered = sorted(
            detections,
            key=lambda d: (float(d.confidence), source_rank.get(d.source, 0), float(d.area_ratio)),
            reverse=True,
        )
        kept: list[FaceDetection] = []
        for candidate in ordered:
            duplicate = False
            for existing in kept:
                if _bbox_iou(candidate.bbox, existing.bbox) >= 0.25:
                    duplicate = True
                    break
                if _center_distance(candidate.center_norm, existing.center_norm) <= 0.055:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        kept.sort(key=lambda d: (d.center_norm[0], d.center_norm[1]))
        return kept


def _make_detection(bbox: BBox, width: int, height: int, confidence: float, source: str) -> FaceDetection:
    clipped = _clip_bbox(bbox, width, height)
    if clipped is None:
        clipped = (0, 0, 1, 1)
    x, y, w, h = clipped
    frame_area = max(float(width * height), 1.0)
    cx = (x + w / 2.0) / max(float(width), 1.0)
    cy = (y + h / 2.0) / max(float(height), 1.0)
    return FaceDetection(
        bbox=(int(x), int(y), int(w), int(h)),
        center_norm=(round(cx, 3), round(cy, 3)),
        area_ratio=float(w * h) / frame_area,
        confidence=max(0.0, min(1.0, float(confidence))),
        source=source,
    )


def _clip_bbox(bbox: BBox, width: int, height: int) -> BBox | None:
    x, y, w, h = [int(v) for v in bbox]
    if w <= 1 or h <= 1:
        return None
    x0 = max(0, min(width - 1, x))
    y0 = max(0, min(height - 1, y))
    x1 = max(0, min(width, x + w))
    y1 = max(0, min(height, y + h))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


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
