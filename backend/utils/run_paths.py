"""Run-directory management for LeLamp backend logs.

Each backend invocation writes into a unique run folder under:
    logs/runs/YYYY-MM-DD_HH-MM-SS/

For PowerShell convenience, the same files are also mirrored into:
    logs/latest/

This avoids mixing old Milestone 1 logs with new Milestone 1.5+ runs while
keeping an easy path for quick inspection and analysis.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunPaths:
    log_root: Path
    run_id: str
    run_dir: Path
    latest_dir: Path
    runtime_log_path: Path
    commands_path: Path
    latency_path: Path
    latest_runtime_log_path: Path
    latest_commands_path: Path
    latest_latency_path: Path


def _safe_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def make_run_id(now: datetime | None = None) -> str:
    """Return a filesystem-friendly timestamp run id."""
    return (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")


def create_run_paths(log_root: str | Path = "logs", run_id: str | None = None, mirror_latest: bool = True) -> RunPaths:
    """Create a fresh run directory and optionally reset logs/latest.

    If a run with the same timestamp already exists, a numeric suffix is added.
    This makes repeated test starts in the same second safe.
    """
    root = Path(log_root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_run_id = run_id or make_run_id()
    candidate = base_run_id
    suffix = 1
    run_dir = runs_dir / candidate
    while run_dir.exists():
        candidate = f"{base_run_id}_{suffix:02d}"
        run_dir = runs_dir / candidate
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)

    latest_dir = root / "latest"
    if mirror_latest:
        _safe_remove(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
    else:
        latest_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        log_root=root,
        run_id=candidate,
        run_dir=run_dir,
        latest_dir=latest_dir,
        runtime_log_path=run_dir / "runtime.log",
        commands_path=run_dir / "commands.jsonl",
        latency_path=run_dir / "latency.csv",
        latest_runtime_log_path=latest_dir / "runtime.log",
        latest_commands_path=latest_dir / "commands.jsonl",
        latest_latency_path=latest_dir / "latency.csv",
    )


def write_latest_pointer(log_root: str | Path, run_dir: str | Path) -> None:
    """Write a small text pointer for tools or humans that want the run id."""
    root = Path(log_root)
    pointer = root / "latest_run.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(Path(run_dir)), encoding="utf-8")


def mirror_file_to_latest(source: Path, latest_path: Path) -> None:
    """Copy a completed source file to latest.

    The main backend normally writes to latest live, but this helper is useful if
    a caller wants to refresh latest after a run has completed.
    """
    if not source.exists():
        return
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, latest_path)


def mirror_run_to_latest(run_paths: RunPaths) -> None:
    """Refresh logs/latest from the selected run directory."""
    _safe_remove(run_paths.latest_dir)
    run_paths.latest_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("runtime.log", "commands.jsonl", "latency.csv"):
        mirror_file_to_latest(run_paths.run_dir / filename, run_paths.latest_dir / filename)
    write_latest_pointer(run_paths.log_root, run_paths.run_dir)


def existing_run_dirs(log_root: str | Path = "logs") -> Iterable[Path]:
    runs_dir = Path(log_root) / "runs"
    if not runs_dir.exists():
        return []
    return sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
