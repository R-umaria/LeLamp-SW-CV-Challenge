# Lumos Current Patch — Hand Gestures + Motion Polish

This version adds optional MediaPipe hand gesture control, wider face-follow stability, Pixar-style elbow-hump motion targets, and richer pre-sleep scanning.

Recommended demo run:

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --enable-objects --enable-web-chat --preview-flip-horizontal
```

Gesture controls:

- Index-finger beckon/call gesture: Lumos uses `gesture_approach` and shifts closer to the camera.
- Open palm facing the camera: Lumos uses `gesture_retreat` and shifts away.

See `docs/milestone4_6_hand_gesture_motion.md` for implementation details and tuning notes.

---

# Lumos Challenge - Milestone 1.5.1 Backend

Vertical slice: webcam capture -> face-based engagement estimate -> temporal smoothing -> hysteresis finite state machine -> protocol-shaped JSON commands -> isolated run logs -> run-specific evaluation.

Milestone 1.5.1 is a logging/evaluation cleanup. It does **not** add Godot, object detection, memory, or LLM recall, and it does **not** change the command JSON protocol.

## Setup

```powershell
cd lumos_challenge
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

macOS/Linux:

```bash
cd lumos_challenge
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Run

```powershell
python -m backend.main --show-window
```

Press `q` or `Esc` in the preview window to quit.

Terminal-only mode:

```powershell
python -m backend.main --no-window
```

## Run isolation

Every backend invocation creates a fresh timestamped run folder:

```text
logs/runs/YYYY-MM-DD_HH-MM-SS/
  runtime.log
  commands.jsonl
  latency.csv
```

The current run is also mirrored to:

```text
logs/latest/
  runtime.log
  commands.jsonl
  latency.csv
```

A pointer is written to:

```text
logs/latest_run.txt
```

This prevents old Milestone 1 logs and newer Milestone 1.5 logs from being mixed in evaluation.

## Tuned dark-room run

```powershell
python -m backend.main --show-window --smoothing-window 9 --min-state-dwell 1.0 --exit-disengaged-frames 7 --exit-absent-frames 10 --min-candidate-area-ratio 0.012 --min-face-area-ratio 0.022
```

## Analyze logs

Analyze the latest run only:

```powershell
python -m backend.evaluation.analyze_logs --latest
```

Analyze one selected run:

```powershell
python -m backend.evaluation.analyze_logs --run-dir logs\runs\2026-06-17_23-10-46
```

The analyzer reports:

- run start/end time
- duration
- total frames
- commands emitted
- state transition count
- flicker count within the selected run only
- time spent in `idle`, `engaged`, `disengaged`, and `seeking_attention`
- latency avg/p50/p95
- top engaged and disengaged/absence reasons

## PowerShell-friendly log viewing

Latest run:

```powershell
Get-Content logs\latest\runtime.log -TotalCount 40
Get-Content logs\latest\runtime.log -Tail 40
Import-Csv logs\latest\latency.csv | Select-Object -First 10 | Format-Table
Get-Content logs\latest\commands.jsonl -Tail 5
Get-Content logs\latest_run.txt
```

Selected run:

```powershell
$RunDir = "logs\runs\2026-06-17_23-10-46"
Get-Content "$RunDir\runtime.log" -Tail 40
Import-Csv "$RunDir\latency.csv" | Select-Object -First 10 | Format-Table
Get-Content "$RunDir\commands.jsonl" -Tail 5
```

## Validation checklist before Godot

1. Clear the latest run mirror if desired:

```powershell
Remove-Item -Recurse -Force logs\latest -ErrorAction SilentlyContinue
Remove-Item -Force logs\latest_run.txt -ErrorAction SilentlyContinue
```

2. Run the tuned command:

```powershell
python -m backend.main --show-window --smoothing-window 9 --min-state-dwell 1.0 --exit-disengaged-frames 7 --exit-absent-frames 10 --min-candidate-area-ratio 0.012 --min-face-area-ratio 0.022
```

3. Test these cases:

```text
centered face
looking away
absent user
photo-frame-only background
```

4. Analyze only the latest isolated run:

```powershell
python -m backend.evaluation.analyze_logs --latest
```

5. Move to Godot only if flicker is low, centered engagement is stable, absent/photo-frame-only does not repeatedly trigger engaged, and `seeking_attention` appears only after sustained disengagement.

---

# Milestone 2 Add-on: Godot UDP Frontend

Milestone 2 adds a lightweight Godot 3D embodiment layer while preserving the Milestone 1.5.1 backend protocol and isolated logging.

## What changed

Added:

```text
backend/behavior/godot_udp_sender.py
backend/tools/send_test_commands.py
frontend_godot/
  project.godot
  scenes/Main.tscn
  scenes/LampRig.tscn
  scripts/Main.gd
  scripts/UdpCommandReceiver.gd
  scripts/LampController.gd
  assets/
  README.md
```

Modified:

```text
backend/main.py
backend/utils/config.py
backend/evaluation/latency_logger.py
backend/evaluation/analyze_logs.py
```

The backend still emits and saves the same command JSON shape:

```text
timestamp, state, engagement, behavior, memory
```

## Godot setup

Use Godot 4.6.3 stable or newer Godot 4.x stable. Open:

```text
frontend_godot/project.godot
```

Run the project. The debug panel should show UDP listening on port `4242`.

## Webcam-free frontend test

```powershell
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

## Real backend + Godot

Start Godot first, then run:

```powershell
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242
```

Tuned command with Godot enabled:

```powershell
python -m backend.main --show-window --godot-udp --smoothing-window 9 --min-state-dwell 1.0 --exit-disengaged-frames 7 --exit-absent-frames 10 --min-candidate-area-ratio 0.012 --min-face-area-ratio 0.022
```

## Milestone 2 pass/fail

Pass if:

- Godot receives test packets and updates the debug UI.
- The lamp cycles through `idle_breathe`, `attentive_nod`, `searching_glance`, `curious_tilt`, `scanning`, and `thinking`.
- The real backend streams commands when `--godot-udp` is enabled.
- The real backend still writes `logs/runs/<run_id>/commands.jsonl` and `latency.csv`.
- The real backend does not crash when Godot is closed.

Fail if:

- Godot makes engagement/memory/behavior decisions.
- The backend command JSON shape changes.
- Command logging breaks.
- The frontend requires object detection, memory, LLM, IK, or imported models to run.

## Milestone 4.5 preview-window controls

The Python OpenCV preview window is now smaller by default and handled through `backend/utils/preview_window.py`. The preview is display-only, so resizing or flipping it does not change engagement detection, object memory, or Godot face-follow commands.

Common commands:

```bash
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-scale 0.50
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-scale 0.35
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-width 320 --preview-height 240
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-flip-horizontal
python -m backend.main --godot-udp --enable-web-chat --no-window
```

For a true single-window future version, stream Python's processed preview into Godot instead of letting Godot open the webcam directly. Python should remain the perception owner for the challenge demo.

## Milestone 4.7 patch

Adds vertical face-following and embodied recall feedback. Lumos now uses both `face_x_norm` and `face_y_norm` to track the user across the webcam frame. Recall answers now drive a bounded Godot motion: found objects trigger `recall_point` + `pointer_spot`, while no recent memory triggers `recall_not_found` + `sad_dim`.

Recommended command:

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --enable-objects --enable-web-chat --preview-flip-horizontal --recall-lookback-hours 24
```

See `docs/milestone4_7_vertical_follow_recall_pointing.md` for implementation details.

### Milestone 4.8 recall hold + distance-aware tracking

This version adds a 5-second recall feedback hold and distance-aware attentive tracking. When Lumos remembers an object, it keeps pointing toward the remembered region before returning to attentive face tracking. When it does not remember the object within the configured lookback window, it holds the sad/no-memory motion before returning to normal behavior.

Recommended command:

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --enable-objects --enable-web-chat --preview-flip-horizontal --recall-lookback-hours 24 --recall-point-hold-seconds 5
```

See `docs/milestone4_8_recall_hold_distance_follow.md` for implementation details.
