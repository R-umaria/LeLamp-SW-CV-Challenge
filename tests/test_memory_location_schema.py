from backend.memory.memory_store import MemoryRecord, MemoryStore
from backend.perception.object_detector import estimate_location_label, normalized_bbox_center, spatial_zones, pointing_target_for_bbox
from backend.behavior.recall_feedback import recall_target_for_result


class DummyResult:
    answer_type = "memory_answer"
    parsed_object = "phone"

    def __init__(self, record):
        self.memory_record = record


def test_location_helpers_produce_structured_camera_space():
    bbox = (10, 340, 80, 80)
    cx, cy = normalized_bbox_center(bbox, 640, 480)
    assert cx < 0.33
    assert cy > 0.67
    assert spatial_zones(cx, cy) == ("left", "lower")
    assert "lower left" in estimate_location_label(bbox, 640, 480)
    assert pointing_target_for_bbox(bbox, 640, 480)["type"] == "point_to_memory"


def test_memory_store_migrates_and_roundtrips_new_fields(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    record = MemoryRecord.create(
        object_label="cell phone",
        normalized_label="phone",
        location_label="lower left side of the camera view",
        bbox=(10, 340, 80, 80),
        confidence=0.88,
        center_x_norm=0.078,
        center_y_norm=0.792,
        zone_x="left",
        zone_y="lower",
        distance_hint="medium distance in camera view",
        pointing_target={"type": "point_to_memory", "x_norm": 0.078, "y_norm": 0.792},
    )
    store.insert(record)
    loaded = store.find_latest_by_normalized_label("phone")
    assert loaded is not None
    assert loaded.center_x_norm == 0.078
    assert loaded.zone_x == "left"
    assert loaded.pointing_target["type"] == "point_to_memory"


def test_recall_target_uses_normalized_memory_point():
    record = MemoryRecord.create(
        object_label="cup",
        normalized_label="cup",
        location_label="right side of the camera view",
        bbox=(500, 200, 60, 60),
        confidence=0.7,
        center_x_norm=0.82,
        center_y_norm=0.48,
    )
    target = recall_target_for_result(DummyResult(record))
    assert target["found"] is True
    assert target["point_x_norm"] == 0.82
    assert target["target"]["type"] == "point_to_memory"
