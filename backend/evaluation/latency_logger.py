"""CSV latency logger for Milestone 1 evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping


class LatencyLogger:
    FIELDNAMES = [
        "timestamp",
        "capture_ms",
        "engagement_detection_ms",
        "state_machine_ms",
        "command_build_ms",
        "total_loop_ms",
        "state",
        "engagement_status",
        "engagement_confidence",
        "engagement_reason",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append(self, row: Mapping) -> None:
        clean_row = {field: row.get(field, "") for field in self.FIELDNAMES}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(clean_row)
