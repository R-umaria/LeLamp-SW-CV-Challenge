"""Analyze one isolated Lumos backend run.

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
    parser = argparse.ArgumentParser(description="Analyze one Lumos backend run")
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

    print("Lumos Run Summary")
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
