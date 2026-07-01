import logging
from dataclasses import dataclass

import numpy as np

from backend.perception.engagement_detector import EngagementResult
from backend.perception.face_tracker import FaceTracker
from backend.perception.multi_face_detector import FaceDetection, MultiFaceDetector
from backend.utils.config import EngagementConfig, SpeakerAwarenessConfig


@dataclass(frozen=True)
class _StubDetection:
    bbox: tuple[int, int, int, int]


class _StubFaceDetector:
    def __init__(self, bboxes):
        self._detections = [_StubDetection(bbox) for bbox in bboxes]

    def detect(self, frame):
        return list(self._detections)

    def close(self):
        pass


def test_face_tracker_keeps_secondary_non_primary_face_by_default():
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.engagement_config = EngagementConfig()
    tracker.speaker_config = SpeakerAwarenessConfig(enabled=True)
    tracker.logger = logging.getLogger("test")
    tracker.face_detector = _StubFaceDetector(
        [
            (250, 140, 72, 72),  # primary duplicate from detector path
            (35, 150, 60, 60),   # smaller non-primary speaker candidate
        ]
    )
    tracker._tracks = {}
    tracker._next_id = 1
    tracker._facemesh = None
    tracker._facemesh_available = False

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    engagement = EngagementResult(
        status="engaged",
        confidence=0.8,
        reason="face_centered",
        face_bbox=(250, 140, 72, 72),
        face_center_norm=(0.447, 0.367),
        face_area_ratio=(72 * 72) / (640 * 480),
        raw_face_count=2,
        candidate_count=2,
    )

    tracks = tracker.update(frame, engagement_result=engagement, now=1.0)

    assert len(tracks) == 2
    assert [track.track_id for track in tracks] == ["person_1", "person_2"]
    assert any(track.bbox == (35, 150, 60, 60) for track in tracks)


def test_multi_face_detector_dedupes_overlapping_detector_outputs():
    detector = MultiFaceDetector.__new__(MultiFaceDetector)
    detector.config = EngagementConfig()

    detections = [
        FaceDetection((100, 100, 80, 80), (0.219, 0.292), 0.0208, 0.62, "haar_frontal"),
        FaceDetection((103, 101, 82, 82), (0.225, 0.296), 0.0219, 0.91, "mediapipe_face_detection"),
        FaceDetection((360, 115, 75, 75), (0.621, 0.318), 0.0183, 0.89, "mediapipe_face_detection"),
    ]

    kept = detector._dedupe_and_sort(detections)

    assert len(kept) == 2
    assert kept[0].bbox == (103, 101, 82, 82)
    assert kept[1].bbox == (360, 115, 75, 75)
