"""Logging helpers for runtime diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable


def setup_logging(
    log_dir: str | Path = "logs",
    extra_runtime_log_paths: Iterable[str | Path] | None = None,
) -> logging.Logger:
    """Configure the Lumos logger.

    Parameters
    ----------
    log_dir:
        Directory that receives the primary ``runtime.log``.
    extra_runtime_log_paths:
        Optional additional files that receive the same runtime log. Milestone
        1.5.1 uses this to mirror logs into ``logs/latest/runtime.log`` while
        preserving the isolated timestamped run directory.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lelamp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    runtime_paths: list[Path] = [log_dir / "runtime.log"]
    if extra_runtime_log_paths:
        runtime_paths.extend(Path(p) for p in extra_runtime_log_paths)

    # Preserve order but avoid duplicate handlers for identical paths.
    seen: set[Path] = set()
    for path in runtime_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger
