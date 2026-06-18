# LeLamp Challenge - Milestone 1.5.1 Backend

Vertical slice: webcam capture -> face-based engagement estimate -> temporal smoothing -> hysteresis finite state machine -> protocol-shaped JSON commands -> isolated run logs -> run-specific evaluation.

Milestone 1.5.1 is a logging/evaluation cleanup. It does **not** add Godot, object detection, memory, or LLM recall, and it does **not** change the command JSON protocol.

## Setup

```powershell
cd lelamp_challenge
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

macOS/Linux:

```bash
cd lelamp_challenge
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
