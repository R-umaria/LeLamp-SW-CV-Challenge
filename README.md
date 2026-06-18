# LeLamp Challenge - Milestone 1.5 Backend

Vertical slice: webcam capture -> face-based engagement estimate -> temporal smoothing -> hysteresis finite state machine -> protocol-shaped JSON commands -> latency/state logs.

Milestone 1.5 stabilizes the original Milestone 1 loop before Godot, object detection, memory, or LLM recall are added.

## Run

```bash
cd lelamp_challenge
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python -m backend.main --show-window
```

Press `q` or `Esc` to quit.

For terminal-only mode:

```bash
python -m backend.main --no-window
```

## Dark-room tuning run

```powershell
python -m backend.main --show-window --smoothing-window 9 --min-state-dwell 1.0 --exit-disengaged-frames 7 --exit-absent-frames 10 --min-candidate-area-ratio 0.012 --min-face-area-ratio 0.022
```

## Analyze logs

```powershell
python -m backend.evaluation.analyze_logs --log-dir logs
```

## PowerShell-friendly log viewing

```powershell
Get-Content logs\runtime.log -TotalCount 40
Get-Content logs\runtime.log -Tail 40
Import-Csv logs\latency.csv | Select-Object -First 10 | Format-Table
Get-Content logs\commands.jsonl -Tail 5
```

Logs are written to `logs/runtime.log`, `logs/latency.csv`, and `logs/commands.jsonl`.
