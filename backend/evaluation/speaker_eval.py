"""Small utilities for summarizing speaker-awareness event logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_speaker_events(path: str | Path) -> list[dict]:
    events: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def summarize_speaker_events(events: Iterable[dict]) -> dict:
    counter = Counter()
    overrides = Counter()
    for event in events:
        event_type = str(event.get("event", "unknown"))
        counter[event_type] += 1
        if event_type == "speaker_behavior_override":
            overrides[str(event.get("reason", "unknown"))] += 1
    return {"event_counts": dict(counter), "override_reasons": dict(overrides)}
