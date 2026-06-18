# Milestone 3 Complete Code Listing

## `backend/perception/object_detector.py`

```python
"""Optional YOLO-based object detection for Milestone 3 scene memory.

The detector is deliberately optional. If Ultralytics is not installed, the
model file is missing, or the model fails to load, the backend keeps running
with object detection disabled. This preserves the stable engagement/FSM/Godot
pipeline from Milestone 2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.utils.config import ObjectDetectionConfig


BBox = tuple[int, int, int, int]


# COCO labels that are useful for a desk/lamp demo. The detector may see many
# COCO classes, but memory is intentionally scoped to common objects to keep the
# demo explainable and reduce noisy writes.
DEFAULT_DESK_OBJECTS = {
    "cell phone",
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "cup",
    "bottle",
    "remote",
    "scissors",
    "clock",
    "vase",
    "backpack",
    "handbag",
    "chair",
    "tv",
    "monitor",
}


NORMALIZATION_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "phone": "phone",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "book": "book",
    "cup": "cup",
    "bottle": "bottle",
    "remote": "remote",
    "scissors": "scissors",
    "clock": "clock",
    "vase": "vase",
    "backpack": "backpack",
    "handbag": "bag",
    "chair": "chair",
    "tv": "monitor",
    "monitor": "monitor",
}


@dataclass(frozen=True)
class ObjectDetection:
    """One object detection in image coordinates."""

    label: str
    normalized_label: str
    bbox: BBox  # x, y, w, h in pixels
    confidence: float
    location_label: str
    class_id: Optional[int] = None

    def to_memory_command_dict(self) -> dict:
        """Return the compact memory payload used by the Godot protocol."""
        return {
            "label": self.label,
            "location": self.location_label,
            "confidence": round(float(self.confidence), 3),
        }

    def to_log_dict(self) -> dict:
        return {
            "label": self.label,
            "normalized_label": self.normalized_label,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "location_label": self.location_label,
            "class_id": self.class_id,
        }


def normalize_label(label: str) -> str:
    """Normalize detector labels for queryable memory records."""
    cleaned = re.sub(r"\s+", " ", label.strip().lower())
    return NORMALIZATION_ALIASES.get(cleaned, cleaned)


def estimate_location_label(bbox: BBox, frame_width: int, frame_height: int) -> str:
    """Estimate an approximate semantic location from a bounding box.

    The location is intentionally coarse: it should be useful for recall without
    pretending to know exact 3D coordinates.
    """
    x, y, w, h = bbox
    cx = (x + (w / 2.0)) / max(frame_width, 1)
    cy = (y + (h / 2.0)) / max(frame_height, 1)

    if cx < 0.33:
        horizontal = "left side of view"
    elif cx > 0.67:
        horizontal = "right side of view"
    else:
        horizontal = "center of view"

    # Add vertical detail only when the object is clearly high/low. This avoids
    # overly specific labels for objects near the middle of the desk view.
    if cy < 0.25:
        return f"upper {horizontal}"
    if cy > 0.75:
        return f"lower {horizontal}"
    return horizontal


class YoloObjectDetector:
    """Best-effort Ultralytics YOLO wrapper.

    Object detection is enabled only if requested by CLI/config and the model can
    load successfully. The rest of the backend should treat ``detector.enabled``
    as the source of truth.
    """

    def __init__(self, config: ObjectDetectionConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.model: Any | None = None
        self.names: dict[int, str] = {}
        self.enabled = False
        self.allowed_labels = {label.strip().lower() for label in (config.allowed_labels or DEFAULT_DESK_OBJECTS)}

        if not config.enabled:
            self.logger.info("Object detection disabled by config")
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            self.logger.warning(
                "Object detector disabled: failed to import ultralytics (%s). Install with: pip install ultralytics",
                exc,
            )
            return

        try:
            model_path = str(config.model_path)
            self.model = YOLO(model_path)
            raw_names = getattr(self.model, "names", {}) or {}
            self.names = {int(k): str(v) for k, v in raw_names.items()} if isinstance(raw_names, dict) else {}
            self.enabled = True
            self.logger.info(
                "Object detector loaded successfully model=%s confidence=%.2f allowed_labels=%s",
                model_path,
                config.confidence,
                sorted(self.allowed_labels),
            )
        except Exception as exc:  # pragma: no cover - model/runtime guard
            self.model = None
            self.enabled = False
            self.logger.warning(
                "Object detector disabled: failed to load model '%s' (%s)",
                config.model_path,
                exc,
            )

    def detect(self, frame) -> list[ObjectDetection]:
        """Run object detection on one BGR OpenCV frame."""
        if not self.enabled or self.model is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        detections: list[ObjectDetection] = []

        try:
            results = self.model.predict(
                source=frame,
                conf=float(self.config.confidence),
                verbose=False,
            )
        except Exception as exc:
            self.logger.warning("Object detection failed for current frame: %s", exc)
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        for box in boxes:
            parsed = self._parse_box(box, frame_width, frame_height)
            if parsed is None:
                continue
            detections.append(parsed)

        detections.sort(key=lambda item: item.confidence, reverse=True)
        if self.config.max_objects_per_frame > 0:
            detections = detections[: self.config.max_objects_per_frame]
        return detections

    def _parse_box(self, box: Any, frame_width: int, frame_height: int) -> ObjectDetection | None:
        try:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = self.names.get(cls_id, str(cls_id)).strip().lower()
            if self.allowed_labels and label not in self.allowed_labels:
                return None

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            x1 = max(0.0, min(float(frame_width - 1), x1))
            y1 = max(0.0, min(float(frame_height - 1), y1))
            x2 = max(0.0, min(float(frame_width - 1), x2))
            y2 = max(0.0, min(float(frame_height - 1), y2))
            w = max(1, int(round(x2 - x1)))
            h = max(1, int(round(y2 - y1)))
            bbox: BBox = (int(round(x1)), int(round(y1)), w, h)
            normalized_label = normalize_label(label)
            location_label = estimate_location_label(bbox, frame_width, frame_height)
            return ObjectDetection(
                label=label,
                normalized_label=normalized_label,
                bbox=bbox,
                confidence=confidence,
                location_label=location_label,
                class_id=cls_id,
            )
        except Exception as exc:
            self.logger.debug("Skipping malformed object detection box: %s", exc)
            return None


def draw_object_overlay(frame, detections: Iterable[ObjectDetection]) -> None:
    """Draw object boxes and labels in-place for the OpenCV preview window."""
    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        text = f"{detection.label} {detection.confidence:.2f} | {detection.location_label}"
        cv2.putText(
            frame,
            text,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
        )

```

## `backend/memory/__init__.py`

```python
"""Scene memory modules for Milestone 3."""

```

## `backend/memory/memory_store.py`

```python
"""SQLite-backed object memory store for Milestone 3.

The schema is intentionally small and explainable. Records represent what the
lamp saw, where it approximately appeared in the camera frame, when it was seen,
and optional evidence through a saved frame path.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    object_label: str
    normalized_label: str
    location_label: str
    bbox: BBox
    confidence: float
    timestamp: str
    source: str = "webcam"
    frame_path: Optional[str] = None

    @classmethod
    def create(
        cls,
        object_label: str,
        normalized_label: str,
        location_label: str,
        bbox: BBox,
        confidence: float,
        source: str = "webcam",
        frame_path: str | None = None,
        timestamp: str | None = None,
    ) -> "MemoryRecord":
        return cls(
            id=str(uuid.uuid4()),
            object_label=object_label,
            normalized_label=normalized_label,
            location_label=location_label,
            bbox=tuple(int(v) for v in bbox),
            confidence=float(confidence),
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            source=source,
            frame_path=frame_path,
        )

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "normalized_label": self.normalized_label,
            "location_label": self.location_label,
            "bbox": json.dumps(list(self.bbox)),
            "confidence": float(self.confidence),
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "normalized_label": self.normalized_label,
            "location_label": self.location_label,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
        }


class MemoryStore:
    """SQLite store for object-memory records."""

    def __init__(self, db_path: str | Path = "data/scene_memory.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS object_memory (
                    id TEXT PRIMARY KEY,
                    object_label TEXT NOT NULL,
                    normalized_label TEXT NOT NULL,
                    location_label TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    frame_path TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_label_time ON object_memory(normalized_label, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_location_time ON object_memory(location_label, timestamp)"
            )

    def insert(self, record: MemoryRecord) -> None:
        row = record.to_row()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO object_memory (
                    id,
                    object_label,
                    normalized_label,
                    location_label,
                    bbox,
                    confidence,
                    timestamp,
                    source,
                    frame_path
                ) VALUES (
                    :id,
                    :object_label,
                    :normalized_label,
                    :location_label,
                    :bbox,
                    :confidence,
                    :timestamp,
                    :source,
                    :frame_path
                )
                """,
                row,
            )

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        normalized_query = query.strip().lower()
        like = f"%{normalized_query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) LIKE ? OR lower(object_label) LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (like, like, int(limit)),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_recent_duplicate(
        self,
        normalized_label: str,
        location_label: str,
        within_seconds: float,
        now: datetime | None = None,
    ) -> MemoryRecord | None:
        """Return a same-label/same-location record inside the dedupe window."""
        cutoff = (now or datetime.now()) - timedelta(seconds=float(within_seconds))
        cutoff_text = cutoff.isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE normalized_label = ?
                  AND location_label = ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized_label, location_label, cutoff_text),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM object_memory")
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM object_memory").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        raw_bbox = row["bbox"]
        bbox_values = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
        bbox: BBox = tuple(int(v) for v in bbox_values)  # type: ignore[assignment]
        return MemoryRecord(
            id=str(row["id"]),
            object_label=str(row["object_label"]),
            normalized_label=str(row["normalized_label"]),
            location_label=str(row["location_label"]),
            bbox=bbox,
            confidence=float(row["confidence"]),
            timestamp=str(row["timestamp"]),
            source=str(row["source"]),
            frame_path=row["frame_path"],
        )


def records_to_table(records: Iterable[MemoryRecord]) -> str:
    """Render records as a readable fixed-width CLI table."""
    rows = list(records)
    if not rows:
        return "No memory records found."

    headers = ["timestamp", "object", "normalized", "location", "conf", "bbox", "frame"]
    data = []
    for record in rows:
        data.append(
            [
                record.timestamp,
                record.object_label,
                record.normalized_label,
                record.location_label,
                f"{record.confidence:.2f}",
                str(list(record.bbox)),
                record.frame_path or "",
            ]
        )

    widths = [len(header) for header in headers]
    for row in data:
        for idx, cell in enumerate(row):
            widths[idx] = min(max(widths[idx], len(cell)), 42)

    def fit(value: str, width: int) -> str:
        return value if len(value) <= width else value[: max(0, width - 1)] + "…"

    lines = []
    lines.append(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in data:
        lines.append(" | ".join(fit(cell, widths[idx]).ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)

```

## `backend/memory/scene_memory.py`

```python
"""Scene-memory orchestration and inspection CLI for Milestone 3.

Runtime use:
    SceneMemory.observe(...) writes deduplicated object memories and returns the
    compact object list that should be attached to command["memory"].

Inspection CLI:
    python -m backend.memory.scene_memory --recent
    python -m backend.memory.scene_memory --find phone
    python -m backend.memory.scene_memory --clear
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import cv2
except ImportError:  # pragma: no cover - optional for CLI-only use
    cv2 = None  # type: ignore

from backend.memory.memory_store import MemoryRecord, MemoryStore, records_to_table
from backend.perception.object_detector import ObjectDetection, normalize_label
from backend.utils.config import MemoryConfig


@dataclass(frozen=True)
class SceneMemoryWriteResult:
    command_objects: list[dict]
    written_records: list[MemoryRecord]
    skipped_duplicates: int
    memory_write_ms: float


class SceneMemory:
    """Writes deduplicated object observations into SQLite."""

    def __init__(self, config: MemoryConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.store = MemoryStore(config.db_path)
        self.frame_dir = Path(config.frame_dir)
        if config.save_object_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Scene memory initialized db=%s dedupe_window=%.1fs save_frames=%s",
            config.db_path,
            config.dedupe_window_s,
            config.save_object_frames,
        )

    def observe(
        self,
        detections: Iterable[ObjectDetection],
        frame=None,
        frame_index: int | None = None,
        source: str = "webcam",
    ) -> SceneMemoryWriteResult:
        start = time.perf_counter()
        detections_list = list(detections)
        command_objects = [detection.to_memory_command_dict() for detection in detections_list]
        written: list[MemoryRecord] = []
        skipped_duplicates = 0

        for detection in detections_list:
            duplicate = self.store.find_recent_duplicate(
                normalized_label=detection.normalized_label,
                location_label=detection.location_label,
                within_seconds=self.config.dedupe_window_s,
            )
            if duplicate is not None:
                skipped_duplicates += 1
                self.logger.info(
                    "Skipped duplicate memory label=%s location=%s existing_id=%s window=%.1fs",
                    detection.normalized_label,
                    detection.location_label,
                    duplicate.id,
                    self.config.dedupe_window_s,
                )
                continue

            frame_path = None
            if self.config.save_object_frames and frame is not None:
                frame_path = self._save_frame(frame, detection, frame_index)

            record = MemoryRecord.create(
                object_label=detection.label,
                normalized_label=detection.normalized_label,
                location_label=detection.location_label,
                bbox=detection.bbox,
                confidence=detection.confidence,
                source=source,
                frame_path=frame_path,
            )
            self.store.insert(record)
            written.append(record)
            self.logger.info(
                "Memory write id=%s label=%s normalized=%s location=%s conf=%.2f bbox=%s frame=%s",
                record.id,
                record.object_label,
                record.normalized_label,
                record.location_label,
                record.confidence,
                list(record.bbox),
                record.frame_path,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SceneMemoryWriteResult(
            command_objects=command_objects,
            written_records=written,
            skipped_duplicates=skipped_duplicates,
            memory_write_ms=elapsed_ms,
        )

    def _save_frame(self, frame, detection: ObjectDetection, frame_index: int | None) -> str | None:
        if cv2 is None:
            self.logger.warning("Cannot save object frame because OpenCV is unavailable")
            return None

        safe_label = re.sub(r"[^a-z0-9_-]+", "_", detection.normalized_label.lower()).strip("_") or "object"
        timestamp_ms = int(time.time() * 1000)
        frame_suffix = "unknown" if frame_index is None else str(frame_index)
        output_path = self.frame_dir / f"frame_{frame_suffix}_{safe_label}_{timestamp_ms}.jpg"

        snapshot = frame.copy()
        x, y, w, h = detection.bbox
        cv2.rectangle(snapshot, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            snapshot,
            f"{detection.label} {detection.confidence:.2f} | {detection.location_label}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
        )
        ok = cv2.imwrite(str(output_path), snapshot)
        if not ok:
            self.logger.warning("Failed to save object frame to %s", output_path)
            return None
        return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or clear LeLamp scene memory")
    parser.add_argument("--db", type=str, default="data/scene_memory.sqlite", help="SQLite memory database path")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--recent", action="store_true", help="Show recent memory records")
    action.add_argument("--find", type=str, help="Find memories by object label, e.g. phone")
    action.add_argument("--clear", action="store_true", help="Delete all memory records from the database")

    parser.add_argument("--limit", type=int, default=20, help="Maximum records to print for --recent or --find")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = MemoryStore(args.db)

    if args.clear:
        deleted = store.clear()
        print(f"Cleared {deleted} memory record(s) from {args.db}")
        return 0

    if args.recent:
        records = store.recent(limit=args.limit)
        print(records_to_table(records))
        return 0

    if args.find:
        query = normalize_label(args.find)
        records = store.find(query, limit=args.limit)
        print(records_to_table(records))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/main.py`

```python
"""LeLamp backend vertical slice with optional Milestone 3 scene memory.

Run from the project root with:
    python -m backend.main --show-window

Milestone 3 adds optional object detection and SQLite scene memory while
preserving the stable engagement detector, temporal smoothing, FSM, isolated run
logs, command JSON shape, and Godot UDP frontend bridge.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.behavior.behavior_policy import behavior_for_state
from backend.behavior.command_protocol import build_behavior_command
from backend.behavior.godot_udp_sender import GodotUdpSender
from backend.behavior.state_machine import InteractionStateMachine
from backend.evaluation.latency_logger import LatencyLogger
from backend.memory.scene_memory import SceneMemory
from backend.perception.camera import OpenCVCamera
from backend.perception.engagement_detector import FaceEngagementDetector, draw_engagement_overlay
from backend.perception.object_detector import YoloObjectDetector, draw_object_overlay
from backend.perception.temporal_smoother import EngagementSmoother
from backend.utils.config import (
    CameraConfig,
    EngagementConfig,
    GodotUdpConfig,
    MemoryConfig,
    ObjectDetectionConfig,
    RuntimeConfig,
    SmoothingConfig,
    StateMachineConfig,
)
from backend.utils.logging_utils import setup_logging
from backend.utils.run_paths import create_run_paths, write_latest_pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeLamp Milestone 3 backend: engagement + optional object memory")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seek-after", type=float, default=5.0)
    parser.add_argument("--emit-interval", type=float, default=1.0)
    parser.add_argument("--godot-udp", action="store_true", help="Enable best-effort UDP command streaming to the Godot frontend.")
    parser.add_argument("--godot-host", type=str, default="127.0.0.1", help="Godot UDP host. Use 127.0.0.1 for local demo.")
    parser.add_argument("--godot-port", type=int, default=4242, help="Godot UDP listen port.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Root log directory. Each run writes under <log-dir>/runs/<run_id>/")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id. Defaults to timestamp YYYY-MM-DD_HH-MM-SS.")
    parser.add_argument("--no-latest", action="store_true", help="Do not mirror this run into logs/latest.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Esc/Ctrl-C")

    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--min-state-dwell", type=float, default=0.75)
    parser.add_argument("--exit-disengaged-frames", type=int, default=5)
    parser.add_argument("--exit-absent-frames", type=int, default=8)
    parser.add_argument("--engaged-recovery-frames", type=int, default=2)
    parser.add_argument("--clear-engaged-confidence", type=float, default=0.78)

    parser.add_argument("--center-tolerance-x", type=float, default=0.24)
    parser.add_argument("--center-tolerance-y", type=float, default=0.30)
    parser.add_argument("--min-face-area-ratio", type=float, default=0.020)
    parser.add_argument("--min-candidate-area-ratio", type=float, default=0.010)
    parser.add_argument("--cascade-min-neighbors", type=int, default=6)

    parser.add_argument("--enable-objects", action="store_true", help="Enable optional YOLO object detection and scene-memory writes.")
    parser.add_argument("--object-model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model path/name, e.g. yolov8n.pt")
    parser.add_argument("--object-interval", type=float, default=2.0, help="Seconds between object-detection passes.")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path.")
    parser.add_argument("--save-object-frames", action="store_true", help="Save annotated evidence frames when a memory record is written.")
    parser.add_argument("--object-confidence", type=float, default=0.35, help="YOLO confidence threshold for object detection.")
    parser.add_argument("--memory-dedupe-window", type=float, default=8.0, help="Seconds to suppress repeated same-object/same-location memory writes.")
    parser.add_argument("--object-frame-dir", type=str, default="data/object_frames", help="Directory for saved object evidence frames.")

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--show-window", action="store_true", default=True)
    window_group.add_argument("--no-window", action="store_false", dest="show_window")
    return parser.parse_args()


def append_jsonl(paths: Path | list[Path] | tuple[Path, ...], payload: dict) -> None:
    if isinstance(paths, Path):
        output_paths = [paths]
    else:
        output_paths = list(paths)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_configs(
    args: argparse.Namespace,
) -> tuple[
    CameraConfig,
    EngagementConfig,
    SmoothingConfig,
    StateMachineConfig,
    RuntimeConfig,
    GodotUdpConfig,
    ObjectDetectionConfig,
    MemoryConfig,
]:
    camera_config = CameraConfig(index=args.camera_index, width=args.width, height=args.height)
    engagement_config = EngagementConfig(
        center_tolerance_x=args.center_tolerance_x,
        center_tolerance_y=args.center_tolerance_y,
        min_face_area_ratio=args.min_face_area_ratio,
        min_candidate_area_ratio=args.min_candidate_area_ratio,
        cascade_min_neighbors=args.cascade_min_neighbors,
    )
    smoothing_config = SmoothingConfig(
        window_size=args.smoothing_window,
        clear_engaged_confidence=args.clear_engaged_confidence,
    )
    state_config = StateMachineConfig(
        seek_attention_after_s=args.seek_after,
        min_state_dwell_s=args.min_state_dwell,
        exit_engaged_disengaged_frames=args.exit_disengaged_frames,
        exit_engaged_absent_frames=args.exit_absent_frames,
        engaged_recovery_frames=args.engaged_recovery_frames,
        clear_engaged_confidence=args.clear_engaged_confidence,
    )
    runtime_config = RuntimeConfig(
        command_emit_interval_s=args.emit_interval,
        log_dir=args.log_dir,
        show_window=args.show_window,
    )
    godot_udp_config = GodotUdpConfig(
        enabled=args.godot_udp,
        host=args.godot_host,
        port=args.godot_port,
    )
    object_config = ObjectDetectionConfig(
        enabled=args.enable_objects,
        model_path=args.object_model,
        interval_s=max(0.1, args.object_interval),
        confidence=args.object_confidence,
    )
    memory_config = MemoryConfig(
        db_path=args.memory_db,
        dedupe_window_s=max(0.0, args.memory_dedupe_window),
        save_object_frames=args.save_object_frames,
        frame_dir=args.object_frame_dir,
    )
    return (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        memory_config,
    )


def main() -> int:
    args = parse_args()
    (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        memory_config,
    ) = build_configs(args)

    run_paths = create_run_paths(
        log_root=runtime_config.log_dir,
        run_id=args.run_id,
        mirror_latest=not args.no_latest,
    )
    write_latest_pointer(run_paths.log_root, run_paths.run_dir)

    latest_latency_path = None if args.no_latest else run_paths.latest_latency_path
    latest_commands_path = None if args.no_latest else run_paths.latest_commands_path
    latest_runtime_log_paths = [] if args.no_latest else [run_paths.latest_runtime_log_path]

    logger = setup_logging(run_paths.run_dir, extra_runtime_log_paths=latest_runtime_log_paths)
    latency_paths = [run_paths.latency_path] + ([latest_latency_path] if latest_latency_path else [])
    latency_logger = LatencyLogger(latency_paths)
    command_paths = [run_paths.commands_path] + ([latest_commands_path] if latest_commands_path else [])
    commands_path = run_paths.commands_path
    godot_sender = GodotUdpSender(godot_udp_config, logger=logger)

    camera = OpenCVCamera(camera_config.index, camera_config.width, camera_config.height)
    detector = FaceEngagementDetector(engagement_config)
    smoother = EngagementSmoother(smoothing_config)
    fsm = InteractionStateMachine(state_config)
    object_detector = YoloObjectDetector(object_config, logger=logger)
    scene_memory = SceneMemory(memory_config, logger=logger)

    logger.info("Starting Milestone 3 backend with isolated run logging")
    logger.info("Run id=%s", run_paths.run_id)
    logger.info("Run directory=%s", run_paths.run_dir)
    if not args.no_latest:
        logger.info("Latest mirror directory=%s", run_paths.latest_dir)
    logger.info("Camera index=%s size=%sx%s", camera_config.index, camera_config.width, camera_config.height)
    logger.info(
        "Stability config smoothing_window=%s min_dwell=%.2fs exit_disengaged_frames=%s exit_absent_frames=%s min_face_area=%.3f min_candidate_area=%.3f",
        smoothing_config.window_size,
        state_config.min_state_dwell_s,
        state_config.exit_engaged_disengaged_frames,
        state_config.exit_engaged_absent_frames,
        engagement_config.min_face_area_ratio,
        engagement_config.min_candidate_area_ratio,
    )
    logger.info(
        "Object config enabled=%s active=%s model=%s interval=%.2fs confidence=%.2f",
        object_config.enabled,
        object_detector.enabled,
        object_config.model_path,
        object_config.interval_s,
        object_config.confidence,
    )
    logger.info(
        "Memory config db=%s save_frames=%s dedupe_window=%.1fs",
        memory_config.db_path,
        memory_config.save_object_frames,
        memory_config.dedupe_window_s,
    )
    logger.info("Commands will be saved to %s", commands_path)
    if godot_udp_config.enabled:
        logger.info("Commands will also be streamed to Godot via udp://%s:%s", godot_udp_config.host, godot_udp_config.port)

    frame_count = 0
    last_emit_at = 0.0
    last_object_detection_at = 0.0
    fps_ema = 0.0
    last_godot_udp_send_ms = None
    last_detected_objects: list[dict] = []
    display_detections = []

    try:
        camera.open()
        while True:
            loop_start = time.perf_counter()

            camera_frame = camera.read()
            frame = camera_frame.frame

            t0 = time.perf_counter()
            raw_engagement = detector.detect(frame)
            engagement_ms = (time.perf_counter() - t0) * 1000.0

            object_detection_ms = ""
            memory_write_ms = ""
            memory_write_count = 0
            memory_duplicate_skip_count = 0
            now = time.monotonic()
            should_detect_objects = object_detector.enabled and (
                last_object_detection_at == 0.0 or (now - last_object_detection_at >= object_config.interval_s)
            )
            if should_detect_objects:
                t0 = time.perf_counter()
                display_detections = object_detector.detect(frame)
                object_detection_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                last_object_detection_at = now

                if display_detections:
                    logger.info(
                        "Detected objects count=%s objects=%s latency_ms=%.3f",
                        len(display_detections),
                        [d.to_log_dict() for d in display_detections],
                        object_detection_ms,
                    )
                else:
                    logger.info("Detected objects count=0 latency_ms=%.3f", object_detection_ms)

                memory_result = scene_memory.observe(
                    display_detections,
                    frame=frame,
                    frame_index=frame_count,
                    source="webcam",
                )
                last_detected_objects = memory_result.command_objects
                memory_write_ms = round(memory_result.memory_write_ms, 3)
                memory_write_count = len(memory_result.written_records)
                memory_duplicate_skip_count = memory_result.skipped_duplicates
                logger.info(
                    "Object memory update detections=%s writes=%s duplicates=%s memory_write_ms=%.3f",
                    len(display_detections),
                    memory_write_count,
                    memory_duplicate_skip_count,
                    memory_result.memory_write_ms,
                )

            t0 = time.perf_counter()
            smoothed_engagement = smoother.update(raw_engagement)
            smoothing_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            transition = fsm.update(smoothed_engagement)
            state_machine_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = behavior_for_state(transition.current_state)
            command = build_behavior_command(
                state=transition.current_state,
                engagement=smoothed_engagement,
                behavior=behavior,
                last_detected_objects=last_detected_objects,
            )
            command_ms = (time.perf_counter() - t0) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            instantaneous_fps = 1000.0 / max(total_ms, 1e-6)
            fps_ema = instantaneous_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * instantaneous_fps)

            now = time.monotonic()
            should_emit = transition.changed or (now - last_emit_at >= runtime_config.command_emit_interval_s)

            if transition.changed:
                logger.info(
                    "State transition %s -> %s | reason=%s | engagement=%s conf=%.2f",
                    transition.previous_state.value,
                    transition.current_state.value,
                    transition.reason,
                    smoothed_engagement.status,
                    smoothed_engagement.confidence,
                )

            if should_emit:
                print(json.dumps(command, ensure_ascii=False), flush=True)
                append_jsonl(command_paths, command)
                last_godot_udp_send_ms = godot_sender.send(command)
                last_emit_at = now

            latency_logger.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame_index": frame_count,
                    "capture_ms": round(camera_frame.capture_latency_ms, 3),
                    "engagement_detection_ms": round(engagement_ms, 3),
                    "object_detection_ms": object_detection_ms,
                    "memory_write_ms": memory_write_ms,
                    "smoothing_ms": round(smoothing_ms, 3),
                    "state_machine_ms": round(state_machine_ms, 3),
                    "command_build_ms": round(command_ms, 3),
                    "godot_udp_send_ms": "" if last_godot_udp_send_ms is None else round(last_godot_udp_send_ms, 3),
                    "total_loop_ms": round(total_ms, 3),
                    "fps": round(fps_ema, 2),
                    "state": transition.current_state.value,
                    "state_elapsed_s": round(transition.state_elapsed_s, 3),
                    "raw_engagement_status": raw_engagement.status,
                    "raw_engagement_confidence": round(raw_engagement.confidence, 3),
                    "raw_engagement_reason": raw_engagement.reason,
                    "smoothed_engagement_status": smoothed_engagement.status,
                    "smoothed_engagement_confidence": round(smoothed_engagement.confidence, 3),
                    "smoothed_engagement_reason": smoothed_engagement.reason,
                    "face_bbox": smoothed_engagement.face_bbox or "",
                    "face_area_ratio": round(smoothed_engagement.face_area_ratio, 4),
                    "raw_face_count": raw_engagement.raw_face_count,
                    "candidate_count": raw_engagement.candidate_count,
                    "selected_face_score": round(smoothed_engagement.selected_face_score, 3),
                    "object_count": len(display_detections) if object_detector.enabled else 0,
                    "memory_write_count": memory_write_count,
                    "memory_duplicate_skip_count": memory_duplicate_skip_count,
                    "consecutive_engaged": transition.consecutive_engaged,
                    "consecutive_disengaged": transition.consecutive_disengaged,
                    "consecutive_absent": transition.consecutive_absent,
                }
            )

            if runtime_config.show_window:
                draw_engagement_overlay(
                    frame,
                    raw_result=raw_engagement,
                    smoothed_result=smoothed_engagement,
                    state=transition.current_state.value,
                    state_elapsed_s=transition.state_elapsed_s,
                    fps=fps_ema,
                    config=engagement_config,
                )
                if object_detector.enabled:
                    draw_object_overlay(frame, display_detections)
                cv2.imshow("LeLamp Milestone 3 - Engagement/FSM/Object Memory", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("Quit requested from preview window")
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                logger.info("Reached --max-frames=%s", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception("Backend stopped due to error: %s", exc)
        return 1
    finally:
        godot_sender.close()
        camera.release()
        if runtime_config.show_window:
            cv2.destroyAllWindows()
        logger.info("Stopped Milestone 3 backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/utils/config.py`

```python
"""Configuration defaults for the LeLamp backend.

Milestone 3 preserves the stable engagement/FSM/Godot path and adds optional
object detection plus SQLite scene memory. Object detection is disabled unless
explicitly enabled from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class EngagementConfig:
    # Engagement geometry. A face must be central and large enough to count as engaged.
    center_tolerance_x: float = 0.24
    center_tolerance_y: float = 0.30
    min_face_area_ratio: float = 0.020

    # Candidate filtering. This rejects many tiny false positives such as faces in
    # posters/photo frames while keeping a near-desk user detectable.
    min_candidate_area_ratio: float = 0.010
    max_candidate_area_ratio: float = 0.60

    # OpenCV Haar cascade settings. Higher min_neighbors reduces false positives
    # in low light at the cost of missing some weak faces.
    cascade_scale_factor: float = 1.08
    cascade_min_neighbors: int = 6
    cascade_min_size: tuple[int, int] = (64, 64)

    # Dark-room preprocessing.
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    blur_kernel_size: int = 3

    # Primary-face tracking. The detector scores candidates by continuity, size,
    # and center proximity instead of blindly picking the largest face every frame.
    primary_continuity_weight: float = 0.45
    primary_size_weight: float = 0.35
    primary_center_weight: float = 0.20
    max_primary_center_distance: float = 0.35
    max_primary_area_change_ratio: float = 2.75


@dataclass(frozen=True)
class SmoothingConfig:
    # Sliding window over raw frame-level predictions.
    window_size: int = 7
    engaged_vote_ratio: float = 0.55
    disengaged_vote_ratio: float = 0.60
    absent_vote_ratio: float = 0.75

    # Used by the FSM for fast but safe recovery when the user clearly returns.
    clear_engaged_confidence: float = 0.78


@dataclass(frozen=True)
class StateMachineConfig:
    # A user must remain disengaged before attention seeking begins.
    seek_attention_after_s: float = 5.0

    # State changes are suppressed until the current state has lasted this long,
    # except for clear engaged recovery.
    min_state_dwell_s: float = 0.75

    # Hysteresis: engaged -> disengaged/idle requires several consecutive smoothed
    # predictions rather than one noisy frame.
    exit_engaged_disengaged_frames: int = 5
    exit_engaged_absent_frames: int = 8

    # Recovery is deliberately quicker than disengagement so the lamp feels responsive.
    engaged_recovery_frames: int = 2
    clear_engaged_confidence: float = 0.78

    # No-face behavior after active states.
    absent_to_idle_frames: int = 12


@dataclass(frozen=True)
class RuntimeConfig:
    command_emit_interval_s: float = 1.0
    log_dir: str = "logs"
    show_window: bool = True


@dataclass(frozen=True)
class GodotUdpConfig:
    # Milestone 2 local embodiment bridge. Disabled by default so Milestone 1.5.1
    # behavior remains unchanged unless explicitly enabled.
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 4242
    max_packet_bytes: int = 8192


@dataclass(frozen=True)
class ObjectDetectionConfig:
    # Disabled by default to preserve the stable engagement-only run path.
    enabled: bool = False
    model_path: str = "yolov8n.pt"
    interval_s: float = 2.0
    confidence: float = 0.35
    max_objects_per_frame: int = 8
    allowed_labels: set[str] = field(
        default_factory=lambda: {
            "cell phone",
            "laptop",
            "keyboard",
            "mouse",
            "book",
            "cup",
            "bottle",
            "remote",
            "scissors",
            "clock",
            "vase",
            "backpack",
            "handbag",
            "chair",
            "tv",
            "monitor",
        }
    )


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str = "data/scene_memory.sqlite"
    dedupe_window_s: float = 8.0
    save_object_frames: bool = False
    frame_dir: str = "data/object_frames"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    engagement: EngagementConfig = EngagementConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    state_machine: StateMachineConfig = StateMachineConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    godot_udp: GodotUdpConfig = GodotUdpConfig()
    objects: ObjectDetectionConfig = ObjectDetectionConfig()
    memory: MemoryConfig = MemoryConfig()


DEFAULT_CONFIG = AppConfig()

```

## `backend/evaluation/latency_logger.py`

```python
"""CSV latency logger for Milestone 1.5+ / Milestone 3 evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


class LatencyLogger:
    FIELDNAMES = [
        "timestamp",
        "frame_index",
        "capture_ms",
        "engagement_detection_ms",
        "object_detection_ms",
        "memory_write_ms",
        "smoothing_ms",
        "state_machine_ms",
        "command_build_ms",
        "godot_udp_send_ms",
        "total_loop_ms",
        "fps",
        "state",
        "state_elapsed_s",
        "raw_engagement_status",
        "raw_engagement_confidence",
        "raw_engagement_reason",
        "smoothed_engagement_status",
        "smoothed_engagement_confidence",
        "smoothed_engagement_reason",
        "face_bbox",
        "face_area_ratio",
        "raw_face_count",
        "candidate_count",
        "selected_face_score",
        "object_count",
        "memory_write_count",
        "memory_duplicate_skip_count",
        "consecutive_engaged",
        "consecutive_disengaged",
        "consecutive_absent",
    ]

    def __init__(self, paths: str | Path | Iterable[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            self.paths = [Path(paths)]
        else:
            self.paths = [Path(path) for path in paths]

        if not self.paths:
            raise ValueError("LatencyLogger requires at least one output path")

        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append(self, row: Mapping) -> None:
        clean_row = {field: row.get(field, "") for field in self.FIELDNAMES}
        for path in self.paths:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(clean_row)

```

## `backend/evaluation/analyze_logs.py`

```python
"""Analyze one isolated LeLamp backend run.

Preferred usage:
    python -m backend.evaluation.analyze_logs --latest
    python -m backend.evaluation.analyze_logs --run-dir logs/runs/2026-06-17_23-10-46

Milestone 1.5.1 deliberately avoids aggregating the root logs directory because
that mixed Milestone 1 and Milestone 1.5 runs and corrupted evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


VALID_STATES = {"idle", "engaged", "disengaged", "seeking_attention"}
VALID_ENGAGEMENT_STATUSES = {"engaged", "disengaged", "absent"}

TRANSITION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*State transition "
    r"(?P<from>[a-z_]+) -> (?P<to>[a-z_]+) \| reason=(?P<reason>.*?) \| "
    r"engagement=(?P<engagement>[a-z_]+) conf=(?P<confidence>[0-9.]+)"
)
START_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Starting Milestone")
STOP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Stopped Milestone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one LeLamp backend run")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-dir", type=str, help="Analyze a specific logs/runs/<run_id> directory")
    selection.add_argument("--latest", action="store_true", help="Analyze logs/latest")
    parser.add_argument("--log-root", type=str, default="logs", help="Root log directory used with --latest")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Legacy alias for --run-dir. Use only for an isolated run folder, not the root logs folder.",
    )
    parser.add_argument("--flicker-window-s", type=float, default=1.0)
    parser.add_argument("--max-state-gap-s", type=float, default=5.0)
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.latest:
        return Path(args.log_root) / "latest"
    if args.log_dir:
        candidate = Path(args.log_dir)
        if candidate.name == "logs" or candidate == Path("logs"):
            raise SystemExit(
                "Refusing to analyze the root logs directory because it can mix multiple runs. "
                "Use --latest or --run-dir logs/runs/<run_id>."
            )
        return candidate

    latest = Path(args.log_root) / "latest"
    if latest.exists():
        return latest

    raise SystemExit("No run selected. Use --latest or --run-dir logs/runs/<run_id>.")


def parse_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in (None, "%Y-%m-%d %H:%M:%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(value)
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_runtime(path: Path) -> tuple[list[dict], Optional[datetime], Optional[datetime]]:
    if not path.exists():
        return [], None, None

    transitions: list[dict] = []
    start_dt: Optional[datetime] = None
    stop_dt: Optional[datetime] = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if start_dt is None and (m := START_RE.match(line)):
            start_dt = parse_dt(m.group("ts"))
        if m := STOP_RE.match(line):
            stop_dt = parse_dt(m.group("ts"))
        if m := TRANSITION_RE.match(line):
            data = m.groupdict()
            data["dt"] = parse_dt(data["ts"])
            data["confidence"] = float(data["confidence"])
            transitions.append(data)

    transitions = [t for t in transitions if t.get("dt") is not None]
    return transitions, start_dt, stop_dt


def parse_commands(path: Path) -> list[dict]:
    if not path.exists():
        return []
    commands: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(command, dict):
            commands.append(command)
    return commands


def parse_latency(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def as_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
    return values[idx]


def summarize_latency(rows: list[dict], field: str) -> dict[str, float]:
    values = [v for row in rows if (v := as_float(row.get(field))) is not None]
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "avg": sum(values) / len(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 95),
    }


def valid_state(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value in VALID_STATES else None


def timestamped_states_from_latency(rows: list[dict]) -> list[tuple[datetime, str]]:
    parsed: list[tuple[datetime, str]] = []
    for row in rows:
        dt = parse_dt(row.get("timestamp"))
        state = valid_state(row.get("state"))
        if dt is not None and state is not None:
            parsed.append((dt, state))
    return sorted(parsed, key=lambda item: item[0])


def timestamped_states_from_commands(commands: list[dict]) -> list[tuple[datetime, str]]:
    parsed: list[tuple[datetime, str]] = []
    for command in commands:
        dt = parse_dt(command.get("timestamp"))
        state = valid_state(command.get("state"))
        if dt is not None and state is not None:
            parsed.append((dt, state))
    return sorted(parsed, key=lambda item: item[0])


def estimate_sample_interval_s(points: list[tuple[datetime, str]]) -> float:
    deltas = [
        (b[0] - a[0]).total_seconds()
        for a, b in zip(points, points[1:])
        if 0.0 < (b[0] - a[0]).total_seconds() <= 5.0
    ]
    if not deltas:
        return 0.0
    return min(1.0, statistics.median(deltas))


def state_time_from_points(points: list[tuple[datetime, str]], max_gap_s: float) -> dict[str, float]:
    buckets: dict[str, float] = defaultdict(float)
    if not points:
        return {state: 0.0 for state in sorted(VALID_STATES)}

    for (t1, state), (t2, _) in zip(points, points[1:]):
        dt = (t2 - t1).total_seconds()
        if 0.0 <= dt <= max_gap_s:
            buckets[state] += dt

    # Credit the last observed state with one typical sample interval so short
    # runs do not undercount the final state entirely.
    last_interval = estimate_sample_interval_s(points)
    if last_interval > 0.0:
        buckets[points[-1][1]] += last_interval

    return {state: buckets.get(state, 0.0) for state in sorted(VALID_STATES)}


def count_flickers(transitions: list[dict], flicker_window_s: float) -> int:
    flickers = 0
    for prev, curr in zip(transitions, transitions[1:]):
        prev_dt = prev.get("dt")
        curr_dt = curr.get("dt")
        if prev_dt is None or curr_dt is None:
            continue
        dt = (curr_dt - prev_dt).total_seconds()
        if 0.0 <= dt <= flicker_window_s and prev.get("to") != curr.get("to"):
            flickers += 1
    return flickers


def run_time_bounds(
    runtime_start: Optional[datetime],
    runtime_stop: Optional[datetime],
    latency_points: list[tuple[datetime, str]],
    command_points: list[tuple[datetime, str]],
    transitions: list[dict],
) -> tuple[Optional[datetime], Optional[datetime]]:
    starts = [runtime_start]
    ends = [runtime_stop]

    if latency_points:
        starts.append(latency_points[0][0])
        ends.append(latency_points[-1][0])
    if command_points:
        starts.append(command_points[0][0])
        ends.append(command_points[-1][0])
    if transitions:
        starts.append(transitions[0]["dt"])
        ends.append(transitions[-1]["dt"])

    start_values = [dt for dt in starts if dt is not None]
    end_values = [dt for dt in ends if dt is not None]
    return (min(start_values) if start_values else None, max(end_values) if end_values else None)


def top_reasons_from_latency(rows: list[dict], statuses: set[str]) -> list[tuple[str, int]]:
    reasons: Counter[str] = Counter()
    for row in rows:
        status = row.get("smoothed_engagement_status") or row.get("engagement_status")
        reason = row.get("smoothed_engagement_reason") or row.get("engagement_reason")
        if status in statuses and reason:
            reasons[reason] += 1
    return reasons.most_common(10)


def top_reasons_from_commands(commands: list[dict], statuses: set[str]) -> list[tuple[str, int]]:
    reasons: Counter[str] = Counter()
    for command in commands:
        engagement = command.get("engagement", {})
        if not isinstance(engagement, dict):
            continue
        if engagement.get("status") in statuses:
            reasons[engagement.get("reason", "unknown")] += 1
    return reasons.most_common(10)


def top_transition_reasons(transitions: list[dict], statuses: set[str]) -> list[tuple[str, int]]:
    reasons: Counter[str] = Counter()
    for transition in transitions:
        if transition.get("engagement") in statuses:
            reasons[transition.get("reason", "unknown")] += 1
    return reasons.most_common(10)


def top_reasons(rows: list[dict], commands: list[dict], transitions: list[dict], statuses: set[str]) -> list[tuple[str, int]]:
    return (
        top_reasons_from_latency(rows, statuses)
        or top_reasons_from_commands(commands, statuses)
        or top_transition_reasons(transitions, statuses)
    )


def format_dt(dt: Optional[datetime]) -> str:
    return dt.isoformat(sep=" ", timespec="seconds") if dt else "unavailable"


def format_seconds(seconds: Optional[float]) -> str:
    return f"{seconds:.1f}s" if seconds is not None else "unavailable"


def format_ms(summary: dict[str, float]) -> str:
    return f"avg={summary['avg']:.2f} ms, p50={summary['p50']:.2f} ms, p95={summary['p95']:.2f} ms"


def print_reason_block(title: str, reasons: list[tuple[str, int]]) -> None:
    print(f"\n{title}:")
    if reasons:
        for reason, count in reasons:
            print(f"  {reason}: {count}")
    else:
        print("  unavailable")


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    runtime_path = run_dir / "runtime.log"
    commands_path = run_dir / "commands.jsonl"
    latency_path = run_dir / "latency.csv"

    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    transitions, runtime_start, runtime_stop = parse_runtime(runtime_path)
    commands = parse_commands(commands_path)
    latency_rows = parse_latency(latency_path)
    latency_points = timestamped_states_from_latency(latency_rows)
    command_points = timestamped_states_from_commands(commands)

    state_points = latency_points or command_points
    state_times = state_time_from_points(state_points, max_gap_s=args.max_state_gap_s)
    start_dt, end_dt = run_time_bounds(runtime_start, runtime_stop, latency_points, command_points, transitions)
    duration_s = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None

    print("LeLamp Milestone 2-Compatible Run Summary")
    print("===================================")
    print(f"run_dir: {run_dir}")
    print(f"run_start: {format_dt(start_dt)}")
    print(f"run_end: {format_dt(end_dt)}")
    print(f"duration: {format_seconds(duration_s)}")
    print(f"total_frames: {len(latency_rows)}")
    print(f"commands_emitted: {len(commands)}")
    print(f"state_transition_count: {len(transitions)}")
    print(f"flicker_count_<={args.flicker_window_s:.1f}s: {count_flickers(transitions, args.flicker_window_s)}")

    print("\nState transition targets:")
    target_counts = Counter(t["to"] for t in transitions if valid_state(t.get("to")))
    if target_counts:
        for state in sorted(VALID_STATES):
            print(f"  {state}: {target_counts.get(state, 0)}")
    else:
        print("  unavailable")

    print("\nTime spent in each state, estimated from isolated run data:")
    for state in sorted(VALID_STATES):
        print(f"  {state}: {state_times.get(state, 0.0):.1f}s")

    print("\nLatency:")
    latency_fields = [
        "capture_ms",
        "engagement_detection_ms",
        "object_detection_ms",
        "memory_write_ms",
        "smoothing_ms",
        "state_machine_ms",
        "command_build_ms",
        "godot_udp_send_ms",
        "total_loop_ms",
    ]
    printed_latency = False
    for field in latency_fields:
        if any(field in row for row in latency_rows):
            print(f"  {field}: {format_ms(summarize_latency(latency_rows, field))}")
            printed_latency = True
    if not printed_latency:
        print("  unavailable")

    print_reason_block(
        "Top engaged reasons",
        top_reasons(latency_rows, commands, transitions, {"engaged"}),
    )
    print_reason_block(
        "Top disengagement/absence reasons",
        top_reasons(latency_rows, commands, transitions, {"disengaged", "absent"}),
    )

    print("\nPowerShell log viewing commands for this run:")
    print(f"  Get-Content {runtime_path} -TotalCount 40")
    print(f"  Get-Content {runtime_path} -Tail 40")
    print(f"  Import-Csv {latency_path} | Select-Object -First 10 | Format-Table")
    print(f"  Get-Content {commands_path} -Tail 5")
    print("\nAnalyze latest run:")
    print("  python -m backend.evaluation.analyze_logs --latest")
    print("Analyze selected run:")
    print(f"  python -m backend.evaluation.analyze_logs --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/requirements.txt`

```text
opencv-python>=4.9.0
numpy>=1.26.0
ultralytics>=8.2.0

```
