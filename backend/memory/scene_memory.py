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
    parser = argparse.ArgumentParser(description="Inspect or clear Lumos scene memory")
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
