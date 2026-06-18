# Milestone 2: Godot UDP Frontend

## Objective

Add a lightweight Godot 3D frontend for the LeLamp-inspired 6-DOF lamp avatar. Python remains responsible for perception, state, memory, logging, and behavior selection. Godot only receives command JSON and renders bounded expressive animations.

## Files added

```text
backend/behavior/godot_udp_sender.py
backend/tools/__init__.py
backend/tools/send_test_commands.py
frontend_godot/project.godot
frontend_godot/scenes/Main.tscn
frontend_godot/scenes/LampRig.tscn
frontend_godot/scripts/Main.gd
frontend_godot/scripts/UdpCommandReceiver.gd
frontend_godot/scripts/LampController.gd
frontend_godot/assets/
frontend_godot/README.md
```

## Files modified

```text
backend/main.py
backend/utils/config.py
backend/evaluation/latency_logger.py
backend/evaluation/analyze_logs.py
README.md
```

## Backend command path

1. Existing backend builds the same protocol-shaped command JSON.
2. Existing command JSONL logging remains unchanged.
3. If `--godot-udp` is enabled, the same command dict is serialized and sent to Godot via UDP.
4. UDP errors are logged and suppressed so the backend keeps running if Godot is closed.
5. `latency.csv` now includes `godot_udp_send_ms`, which measures the local UDP send call duration. UDP has no acknowledgment, so this is not a visual animation latency measurement.

## Test command

```bash
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

## Real backend command

```bash
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242
```

## Pass criteria

- Godot project opens without missing scripts or scenes.
- Debug panel shows UDP listening status.
- Test sender cycles through all placeholder behaviors.
- Real backend still writes isolated `commands.jsonl` and `latency.csv`.
- Real backend does not crash when Godot is not running.
- Godot does not implement engagement, memory, behavior policy, object detection, or LLM logic.
