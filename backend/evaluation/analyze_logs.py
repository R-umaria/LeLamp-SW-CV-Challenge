"""Convenience wrapper for producing a latency summary from Lumos logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Lumos latency.csv without labels.")
    parser.add_argument("--latency-log", default="logs/latest/latency.csv")
    parser.add_argument("--latest", action="store_true", help="Use logs/latest/latency.csv. Kept for README compatibility.")
    parser.add_argument("--output-dir", "--out-dir", default="logs/evaluation")
    parser.add_argument("--out", default=None, help="Optional direct path for latency_summary.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latency_log = "logs/latest/latency.csv" if args.latest else args.latency_log
    rows = list(csv.DictReader(Path(latency_log).open("r", encoding="utf-8", newline="")))
    fields = [field for field in rows[0].keys() if field.endswith("_ms")] if rows else []
    output_path = Path(args.out) if args.out else Path(args.output_dir) / "latency_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["latency_field", "count", "avg_ms", "max_ms"])
        writer.writeheader()
        for field in fields:
            values = []
            for row in rows:
                raw = row.get(field, "")
                if raw in ("", None):
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
            writer.writerow({"latency_field": field, "count": len(values), "avg_ms": "" if not values else f"{mean(values):.3f}", "max_ms": "" if not values else f"{max(values):.3f}"})
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
