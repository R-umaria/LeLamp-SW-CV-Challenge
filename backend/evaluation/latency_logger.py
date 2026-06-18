"""CSV latency logger for engagement, memory, and recall evaluation."""

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
        "memory_retrieval_ms",
        "llm_response_ms",
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
