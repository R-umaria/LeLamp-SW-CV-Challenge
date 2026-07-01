"""Analyze labeled Lumos engagement logs and write CSV evaluation outputs.

Expected labels CSV columns:
    frame_index,label

Runtime log CSV can be logs/latest/latency.csv or any latency CSV that includes
frame_index and raw/smoothed engagement columns.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from backend.evaluation.label_schema import LABEL_TO_STATUS, VALID_ENGAGEMENT_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Lumos engagement predictions from labeled logs.")
    parser.add_argument("--labels", required=True, help="CSV with frame_index,label.")
    parser.add_argument("--latency-log", default="logs/latest/latency.csv", help="Lumos latency.csv to analyze.")
    parser.add_argument("--output-dir", "--out-dir", default="logs/evaluation", help="Directory for summary/confusion/latency CSVs.")
    parser.add_argument("--prediction-column", default="smoothed_engagement_status", help="Prediction column in latency log.")
    return parser.parse_args()


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_labels(path: str | Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for row in read_csv_rows(path):
        if not row.get("frame_index"):
            continue
        label = str(row.get("label", "")).strip().lower()
        if label not in VALID_ENGAGEMENT_LABELS:
            raise ValueError(f"invalid label {label!r}; expected one of {VALID_ENGAGEMENT_LABELS}")
        labels[int(float(row["frame_index"]))] = label
    return labels


def analyze(labels_path: str | Path, latency_log_path: str | Path, output_dir: str | Path, prediction_column: str = "smoothed_engagement_status") -> dict:
    labels = load_labels(labels_path)
    rows = read_csv_rows(latency_log_path)
    by_frame = {int(float(row["frame_index"])): row for row in rows if row.get("frame_index")}
    y_true: list[str] = []
    y_pred: list[str] = []
    matched_rows: list[dict] = []
    for frame_index, label in sorted(labels.items()):
        row = by_frame.get(frame_index)
        if row is None:
            continue
        expected = LABEL_TO_STATUS[label]
        predicted = str(row.get(prediction_column, "")).strip().lower() or "unknown"
        y_true.append(expected)
        y_pred.append(predicted)
        merged = dict(row)
        merged["label"] = label
        merged["expected_status"] = expected
        merged["predicted_status"] = predicted
        matched_rows.append(merged)

    classes = sorted(set(y_true) | set(y_pred) | {"engaged", "disengaged", "absent", "unknown"})
    confusion = {actual: Counter() for actual in classes}
    for actual, pred in zip(y_true, y_pred):
        confusion[actual][pred] += 1

    total = len(y_true)
    correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    accuracy = correct / total if total else 0.0
    metrics = []
    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[a][cls] for a in classes if a != cls)
        fn = sum(confusion[cls][p] for p in classes if p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics.append({"class": cls, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall})

    flicker_count = count_flickers(y_pred)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_summary(out / "engagement_eval_summary.csv", total, correct, accuracy, flicker_count, metrics)
    write_confusion(out / "engagement_confusion_matrix.csv", classes, confusion)
    write_latency_summary(out / "latency_summary.csv", matched_rows)
    return {"total": total, "correct": correct, "accuracy": accuracy, "flicker_count": flicker_count, "output_dir": str(out)}


def count_flickers(predictions: list[str]) -> int:
    if len(predictions) < 3:
        return 0
    flickers = 0
    for a, b, c in zip(predictions, predictions[1:], predictions[2:]):
        if a == c and b != a:
            flickers += 1
    return flickers


def write_summary(path: Path, total: int, correct: int, accuracy: float, flicker_count: int, metrics: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "class", "value"])
        writer.writeheader()
        writer.writerow({"metric": "total_labeled_frames", "class": "all", "value": total})
        writer.writerow({"metric": "correct_frames", "class": "all", "value": correct})
        writer.writerow({"metric": "accuracy", "class": "all", "value": f"{accuracy:.6f}"})
        writer.writerow({"metric": "flicker_count", "class": "all", "value": flicker_count})
        for item in metrics:
            writer.writerow({"metric": "precision", "class": item["class"], "value": f"{item['precision']:.6f}"})
            writer.writerow({"metric": "recall", "class": item["class"], "value": f"{item['recall']:.6f}"})
            writer.writerow({"metric": "false_positives", "class": item["class"], "value": item["fp"]})
            writer.writerow({"metric": "false_negatives", "class": item["class"], "value": item["fn"]})


def write_confusion(path: Path, classes: list[str], confusion: dict[str, Counter]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *classes])
        for actual in classes:
            writer.writerow([actual, *[confusion[actual][pred] for pred in classes]])


def write_latency_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "capture_ms", "engagement_detection_ms", "object_detection_ms", "state_machine_ms",
        "godot_udp_send_ms", "memory_write_ms", "memory_retrieval_ms", "llm_response_ms", "total_loop_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
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
                    pass
            writer.writerow({
                "latency_field": field,
                "count": len(values),
                "avg_ms": "" if not values else f"{mean(values):.3f}",
                "max_ms": "" if not values else f"{max(values):.3f}",
            })


def main() -> int:
    args = parse_args()
    result = analyze(args.labels, args.latency_log, args.output_dir, args.prediction_column)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
