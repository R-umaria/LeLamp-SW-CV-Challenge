# Milestone 3 — Object Detection and Scene Memory

## Files created

- `backend/perception/object_detector.py`
- `backend/memory/__init__.py`
- `backend/memory/memory_store.py`
- `backend/memory/scene_memory.py`

## Files modified

- `backend/main.py`
- `backend/utils/config.py`
- `backend/evaluation/latency_logger.py`
- `backend/evaluation/analyze_logs.py`
- `backend/requirements.txt`

## Design summary

Milestone 3 keeps Python as the intelligence layer and Godot as the embodiment layer. YOLO object detection is optional and disabled by default. When enabled, the backend detects common desk objects every `--object-interval` seconds, estimates an approximate location from each bounding box, writes deduplicated SQLite memory records, and includes compact object summaries in `command["memory"]["last_detected_objects"]`.

If Ultralytics or the requested model fails to load, the backend logs a warning and continues with object detection disabled. Engagement detection, smoothing, state transitions, isolated logs, `commands.jsonl`, latency logging, analyzer behavior, and Godot UDP streaming remain preserved.

## Setup

```powershell
cd path\to\lelamp_challenge
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

The first YOLO run may download `yolov8n.pt` if it is not already present and network access is available. To avoid runtime download uncertainty, place `yolov8n.pt` in the project root or pass an explicit local path with `--object-model`.

## Smoke tests

Engagement-only regression path:

```powershell
python -m backend.main --no-window --max-frames 120
python -m backend.evaluation.analyze_logs --latest
```

Milestone 3 object-memory path without Godot:

```powershell
python -m backend.main --enable-objects --object-model yolov8n.pt --object-interval 2.0 --object-confidence 0.35 --memory-db data\scene_memory.sqlite --save-object-frames --max-frames 300 --show-window
python -m backend.memory.scene_memory --recent
python -m backend.memory.scene_memory --find phone
```

Full demo path with Godot already open:

```powershell
python -m backend.main --godot-udp --enable-objects --object-model yolov8n.pt --object-interval 2.0 --object-confidence 0.35 --memory-db data\scene_memory.sqlite --save-object-frames --show-window
```

Clear memory before a clean demo run:

```powershell
python -m backend.memory.scene_memory --clear
```

Inspect generated logs:

```powershell
Get-Content logs\latest\runtime.log -Tail 80
Get-Content logs\latest\commands.jsonl -Tail 5
Import-Csv logs\latest\latency.csv | Select-Object -First 10 | Format-Table
python -m backend.evaluation.analyze_logs --latest
```

## Milestone 3 test procedure

1. Start Godot and confirm it is listening on UDP port `4242`.
2. Clear memory: `python -m backend.memory.scene_memory --clear`.
3. Run the backend with `--godot-udp --enable-objects --save-object-frames --show-window`.
4. Place 2–4 desk objects in view, such as phone, cup, bottle, book, laptop, keyboard, or mouse.
5. Look at the camera/lamp until the state becomes engaged.
6. Look away until the state becomes disengaged, then wait until `seeking_attention` triggers.
7. Confirm object boxes, labels, confidence values, and approximate locations appear in the OpenCV overlay.
8. Stop the run with `q` or Esc.
9. Run `python -m backend.memory.scene_memory --recent` and verify records were stored.
10. Run `python -m backend.memory.scene_memory --find phone` or another visible object.
11. Run `python -m backend.evaluation.analyze_logs --latest` and confirm object/memory latency fields are included.
12. Open `logs\latest\commands.jsonl` and verify the `memory.last_detected_objects` field is populated when objects are detected.

## Pass/fail criteria for moving to Milestone 4

Pass when all of these are true:

- Engagement detection and FSM still work with `--enable-objects` off.
- If YOLO or the model cannot load, backend continues running and logs object detection as disabled.
- With `--enable-objects`, common desk objects are detected at the configured interval.
- OpenCV overlay shows object bounding boxes, labels, confidence, and approximate location.
- SQLite database contains records with object label, normalized label, location, bbox, confidence, timestamp, source, and optional frame path.
- Duplicate same-object/same-location observations are skipped inside the dedupe window.
- `commands.jsonl` preserves the existing protocol shape and fills `memory.last_detected_objects`.
- `latency.csv` includes `object_detection_ms` and `memory_write_ms`.
- Godot still receives and responds to state commands.
- `python -m backend.memory.scene_memory --recent`, `--find`, and `--clear` work.

Do not move to Milestone 4 if object detection causes frequent backend crashes, command emission stalls, or the memory store repeatedly writes the same object every detection cycle.
