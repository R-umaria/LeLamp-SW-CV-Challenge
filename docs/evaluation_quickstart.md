# Engagement Evaluation Quickstart

This milestone adds tools only. It does **not** fabricate evaluation results.

## 1. Run Lumos and collect a latency log

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --preview-flip-horizontal
```

The latest run writes `logs/latest/latency.csv`.

## 2. Create a labels CSV

Create a CSV such as `logs/evaluation/engagement_labels.csv`:

```csv
frame_index,label
12,engaged
42,looking_left
78,looking_down
110,no_person
```

Allowed labels:

- `engaged`
- `looking_left`
- `looking_right`
- `looking_down`
- `looking_away`
- `no_person`
- `partial_face`

## 3. Run the evaluator

```powershell
python -m backend.evaluation.engagement_eval --labels logs/evaluation/engagement_labels.csv --latency-log logs/latest/latency.csv
```

Outputs:

- `logs/evaluation/engagement_eval_summary.csv`
- `logs/evaluation/engagement_confusion_matrix.csv`
- `logs/evaluation/latency_summary.csv`

## 4. Latency-only summary

```powershell
python -m backend.evaluation.analyze_logs --latency-log logs/latest/latency.csv
```
