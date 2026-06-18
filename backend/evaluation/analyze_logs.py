"""Analyze Milestone 1/1.5 runtime, command, and latency logs.

Run from the project root:
    python -m backend.evaluation.analyze_logs --log-dir logs
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


TRANSITION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*State transition "
    r"(?P<from>[a-z_]+) -> (?P<to>[a-z_]+) \| reason=(?P<reason>.*?) \| "
    r"engagement=(?P<engagement>[a-z_]+) conf=(?P<confidence>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LeLamp backend logs")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--flicker-window-s", type=float, default=1.0)
    return parser.parse_args()


def parse_runtime_transitions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    transitions: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRANSITION_RE.match(line)
        if not match:
            continue
        data = match.groupdict()
        data["dt"] = datetime.strptime(data["ts"], "%Y-%m-%d %H:%M:%S")
        data["confidence"] = float(data["confidence"])
        transitions.append(data)
    return transitions


def parse_commands(path: Path) -> list[dict]:
    if not path.exists():
        return []
    commands: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            commands.append(json.loads(line))
        except json.JSONDecodeError:
            continue
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


def summarize_latency(rows: list[dict], field: str) -> dict:
    values = [v for row in rows if (v := as_float(row.get(field))) is not None]
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "avg": sum(values) / len(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 95),
    }


def state_time_from_latency(rows: list[dict]) -> dict[str, float]:
    buckets: dict[str, float] = defaultdict(float)
    parsed: list[tuple[datetime, str]] = []
    for row in rows:
        ts = row.get("timestamp")
        state = row.get("state")
        if not ts or not state:
            continue
        try:
            parsed.append((datetime.fromisoformat(ts), state))
        except ValueError:
            continue
    for (t1, state), (t2, _) in zip(parsed, parsed[1:]):
        dt = max(0.0, min(5.0, (t2 - t1).total_seconds()))
        buckets[state] += dt
    return dict(sorted(buckets.items()))


def count_flickers(transitions: list[dict], flicker_window_s: float) -> int:
    flickers = 0
    for prev, curr in zip(transitions, transitions[1:]):
        dt = (curr["dt"] - prev["dt"]).total_seconds()
        if 0.0 <= dt <= flicker_window_s and prev["to"] != curr["to"]:
            flickers += 1
    return flickers


def top_disengagement_reasons(rows: list[dict], commands: list[dict], transitions: list[dict]) -> list[tuple[str, int]]:
    reasons: Counter[str] = Counter()
    for row in rows:
        status = row.get("smoothed_engagement_status") or row.get("engagement_status")
        reason = row.get("smoothed_engagement_reason") or row.get("engagement_reason")
        if status in {"disengaged", "absent"} and reason:
            reasons[reason] += 1
    if not reasons:
        for command in commands:
            engagement = command.get("engagement", {})
            if engagement.get("status") in {"disengaged", "absent"}:
                reasons[engagement.get("reason", "unknown")] += 1
    if not reasons:
        for transition in transitions:
            if transition.get("engagement") in {"disengaged", "absent"}:
                reasons[transition.get("reason", "unknown")] += 1
    return reasons.most_common(10)


def format_ms(summary: dict) -> str:
    return f"avg={summary['avg']:.2f} ms, p50={summary['p50']:.2f} ms, p95={summary['p95']:.2f} ms"


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    runtime_path = log_dir / "runtime.log"
    commands_path = log_dir / "commands.jsonl"
    latency_path = log_dir / "latency.csv"

    transitions = parse_runtime_transitions(runtime_path)
    commands = parse_commands(commands_path)
    latency_rows = parse_latency(latency_path)

    print("LeLamp Milestone 1.5 Log Summary")
    print("================================")
    print(f"log_dir: {log_dir}")
    print(f"total_frames: {len(latency_rows)}")
    print(f"commands_emitted: {len(commands)}")
    print(f"state_transition_count: {len(transitions)}")
    print(f"flicker_count_<={args.flicker_window_s:.1f}s: {count_flickers(transitions, args.flicker_window_s)}")

    print("\nState transition targets:")
    target_counts = Counter(t["to"] for t in transitions)
    for state, count in sorted(target_counts.items()):
        print(f"  {state}: {count}")

    print("\nTime spent in each state, estimated from latency.csv:")
    state_times = state_time_from_latency(latency_rows)
    if state_times:
        for state, seconds in state_times.items():
            print(f"  {state}: {seconds:.1f}s")
    else:
        print("  unavailable")

    print("\nLatency:")
    for field in [
        "capture_ms",
        "engagement_detection_ms",
        "smoothing_ms",
        "state_machine_ms",
        "command_build_ms",
        "total_loop_ms",
    ]:
        if latency_rows and field in latency_rows[0]:
            print(f"  {field}: {format_ms(summarize_latency(latency_rows, field))}")

    print("\nTop disengagement/absence reasons:")
    reasons = top_disengagement_reasons(latency_rows, commands, transitions)
    if reasons:
        for reason, count in reasons:
            print(f"  {reason}: {count}")
    else:
        print("  unavailable")

    print("\nPowerShell log viewing commands:")
    print("  Get-Content logs\\runtime.log -TotalCount 40")
    print("  Get-Content logs\\runtime.log -Tail 40")
    print("  Import-Csv logs\\latency.csv | Select-Object -First 10 | Format-Table")
    print("  Get-Content logs\\commands.jsonl -Tail 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
