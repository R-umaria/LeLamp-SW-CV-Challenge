# LeLamp Challenge - Milestone 1 Backend

Vertical slice: webcam capture -> face-based engagement estimate -> finite state machine -> protocol-shaped JSON commands -> latency/state logs.

## Run

```bash
cd lelamp_challenge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r backend/requirements.txt
python -m backend.main --show-window
```

Press `q` or `Esc` to quit.

For headless/terminal-only mode:

```bash
python -m backend.main --no-window
```

Logs are written to `logs/runtime.log`, `logs/latency.csv`, and `logs/commands.jsonl`.
