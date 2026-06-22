from backend.perception.face_tracker import FaceTrack, FaceTracker


def _track(track_id, x):
    return FaceTrack(
        track_id=track_id,
        bbox=(int(x * 100), 10, 40, 40),
        center_norm=(x, 0.5),
        location="center",
        mouth_motion_score=0.3,
        mouth_opening=None,
        engaged=True,
        confidence=0.7,
        source="test",
        age_s=0.0,
        last_seen_s=0.0,
    )


def test_single_visible_track_is_presented_as_person_1_even_if_internal_id_is_later():
    tracker = FaceTracker.__new__(FaceTracker)
    result = tracker._with_presentation_ids([_track("person_5", 0.5)])
    assert [t.track_id for t in result] == ["person_1"]


def test_two_visible_tracks_are_presented_left_to_right():
    tracker = FaceTracker.__new__(FaceTracker)
    result = tracker._with_presentation_ids([_track("person_8", 0.2), _track("person_3", 0.8)])
    assert [t.track_id for t in result] == ["person_1", "person_2"]
