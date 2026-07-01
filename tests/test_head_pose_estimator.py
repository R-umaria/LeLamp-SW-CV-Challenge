from backend.perception.head_pose_estimator import HeadPoseEstimator, HeadPoseResult
from backend.perception.engagement_detector import FaceCandidate, FaceEngagementDetector
from backend.utils.config import EngagementConfig


def test_head_pose_classification_forward_and_away():
    estimator = HeadPoseEstimator(enabled=False)
    label, conf, reason = estimator.classify_pose(0.0, 0.0, 0.0)
    assert label == "looking_at_lamp"
    assert conf > 0.8
    label, conf, reason = estimator.classify_pose(-50.0, 0.0, 0.0)
    assert label == "looking_left"
    label, conf, reason = estimator.classify_pose(0.0, -40.0, 0.0)
    assert label == "looking_down"


def test_engagement_detector_fuses_pose_over_center_heuristic():
    detector = FaceEngagementDetector(EngagementConfig(head_pose_enabled=False))
    candidate = FaceCandidate(bbox=(200, 120, 160, 160), center_norm=(0.5, 0.5), area_ratio=0.08, selection_score=0.8)
    pose = HeadPoseResult(True, 52.0, 0.0, 0.0, "looking_right", 0.9, "yaw_right", False)
    result = detector._classify_candidate(candidate, 640, 480, raw_face_count=1, candidate_count=1, head_pose=pose)
    assert result.status == "disengaged"
    assert result.reason == "looking_right"
    assert result.fallback_mode_used is False
    detector.close()
