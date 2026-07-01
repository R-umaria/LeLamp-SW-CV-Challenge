"""SQLite-backed object memory store for Milestone 3.

The schema is intentionally small and explainable. Records represent what the
lamp saw, where it approximately appeared in the camera frame, when it was seen,
and optional evidence through a saved frame path.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Iterable, Optional


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
    center_x_norm: Optional[float] = None
    center_y_norm: Optional[float] = None
    zone_x: Optional[str] = None
    zone_y: Optional[str] = None
    distance_hint: Optional[str] = None
    pointing_target: Optional[dict] = None

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
        center_x_norm: float | None = None,
        center_y_norm: float | None = None,
        zone_x: str | None = None,
        zone_y: str | None = None,
        distance_hint: str | None = None,
        pointing_target: dict | None = None,
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
            center_x_norm=None if center_x_norm is None else float(center_x_norm),
            center_y_norm=None if center_y_norm is None else float(center_y_norm),
            zone_x=zone_x,
            zone_y=zone_y,
            distance_hint=distance_hint,
            pointing_target=pointing_target,
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
            "center_x_norm": self.center_x_norm,
            "center_y_norm": self.center_y_norm,
            "zone_x": self.zone_x,
            "zone_y": self.zone_y,
            "distance_hint": self.distance_hint,
            "pointing_target": json.dumps(self.pointing_target) if self.pointing_target is not None else None,
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
            "center_x_norm": None if self.center_x_norm is None else round(float(self.center_x_norm), 3),
            "center_y_norm": None if self.center_y_norm is None else round(float(self.center_y_norm), 3),
            "zone_x": self.zone_x,
            "zone_y": self.zone_y,
            "distance_hint": self.distance_hint,
            "pointing_target": self.pointing_target,
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

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
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
                    frame_path TEXT,
                    center_x_norm REAL,
                    center_y_norm REAL,
                    zone_x TEXT,
                    zone_y TEXT,
                    distance_hint TEXT,
                    pointing_target TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_label_time ON object_memory(normalized_label, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_location_time ON object_memory(location_label, timestamp)"
            )
            self._ensure_optional_columns(conn)

    def _ensure_optional_columns(self, conn: sqlite3.Connection) -> None:
        existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(object_memory)").fetchall()}
        optional_columns = {
            "center_x_norm": "REAL",
            "center_y_norm": "REAL",
            "zone_x": "TEXT",
            "zone_y": "TEXT",
            "distance_hint": "TEXT",
            "pointing_target": "TEXT",
        }
        for name, ddl_type in optional_columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE object_memory ADD COLUMN {name} {ddl_type}")

    def insert(self, record: MemoryRecord) -> None:
        row = record.to_row()
        with self._connection() as conn:
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
                    frame_path,
                    center_x_norm,
                    center_y_norm,
                    zone_x,
                    zone_y,
                    distance_hint,
                    pointing_target
                ) VALUES (
                    :id,
                    :object_label,
                    :normalized_label,
                    :location_label,
                    :bbox,
                    :confidence,
                    :timestamp,
                    :source,
                    :frame_path,
                    :center_x_norm,
                    :center_y_norm,
                    :zone_x,
                    :zone_y,
                    :distance_hint,
                    :pointing_target
                )
                """,
                row,
            )

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]


    def recent_unique_by_normalized_label(self, limit: int = 8, row_limit: int = 200) -> list[MemoryRecord]:
        """Return latest records for recent unique normalized labels.

        SQLite remains the source of truth. This method is used for questions like
        "What objects did you detect?" so the answer is grounded in stored rows,
        not in the object detector vocabulary or LLM guesses.
        """

        records = self.recent(limit=max(int(row_limit), int(limit)))
        seen: set[str] = set()
        unique: list[MemoryRecord] = []
        for record in records:
            label = record.normalized_label.strip().lower()
            if not label or label in seen:
                continue
            seen.add(label)
            unique.append(record)
            if len(unique) >= int(limit):
                break
        return unique

    def find(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        normalized_query = query.strip().lower()
        like = f"%{normalized_query}%"
        with self._connection() as conn:
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

    def find_latest_by_normalized_label(self, normalized_label: str) -> MemoryRecord | None:
        """Return the latest exact normalized-label match for grounded recall."""
        normalized = normalized_label.strip().lower()
        if not normalized:
            return None

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_latest_by_normalized_label_since(
        self,
        normalized_label: str,
        since: datetime,
    ) -> MemoryRecord | None:
        """Return the latest exact label match at or after ``since``.

        Timestamps are stored as ISO-8601 strings, so lexicographic ordering works
        for records produced by this project. This keeps the 24-hour recall rule
        in SQLite instead of asking the LLM to decide whether a memory is recent.
        """
        normalized = normalized_label.strip().lower()
        if not normalized:
            return None

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) = ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized, since.isoformat(timespec="seconds")),
            ).fetchone()
        return self._row_to_record(row) if row else None

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
        with self._connection() as conn:
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
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM object_memory")
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def count(self) -> int:
        with self._connection() as conn:
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
            center_x_norm=self._optional_float(row, "center_x_norm"),
            center_y_norm=self._optional_float(row, "center_y_norm"),
            zone_x=self._optional_str(row, "zone_x"),
            zone_y=self._optional_str(row, "zone_y"),
            distance_hint=self._optional_str(row, "distance_hint"),
            pointing_target=self._optional_json(row, "pointing_target"),
        )

    @staticmethod
    def _row_has(row: sqlite3.Row, key: str) -> bool:
        return key in row.keys()

    @classmethod
    def _optional_float(cls, row: sqlite3.Row, key: str) -> float | None:
        if not cls._row_has(row, key) or row[key] is None:
            return None
        return float(row[key])

    @classmethod
    def _optional_str(cls, row: sqlite3.Row, key: str) -> str | None:
        if not cls._row_has(row, key) or row[key] is None:
            return None
        value = str(row[key])
        return value if value else None

    @classmethod
    def _optional_json(cls, row: sqlite3.Row, key: str) -> dict | None:
        if not cls._row_has(row, key) or row[key] in (None, ""):
            return None
        try:
            parsed = json.loads(str(row[key]))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


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
