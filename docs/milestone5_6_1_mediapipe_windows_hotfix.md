# Milestone 5.6.1: MediaPipe Windows Gesture Hotfix

## Problem

Running Lumos 5.6 with gestures enabled could fail during startup on some Windows MediaPipe installations:

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

The object detector, Godot UDP sender, and other Lumos modules were loading correctly. The crash happened only when `backend/perception/gesture_detector.py` initialized MediaPipe Hands through `mp.solutions.hands.Hands`.

## Fix

`backend/perception/gesture_detector.py` now uses the same compatibility pattern already used by the face tracker:

1. Try the common public path: `mediapipe.solutions.hands`.
2. If that is unavailable, fall back to: `mediapipe.python.solutions.hands`.
3. If both paths fail, gesture control disables itself cleanly and the rest of Lumos continues running.

## Files changed

- `backend/perception/gesture_detector.py`
- `tests/test_gesture_mediapipe_compat.py`
- `docs/milestone5_6_1_mediapipe_windows_hotfix.md`

## Validation

```text
python -m compileall -q backend tests
PYTHONPATH=/mnt/data/lumos55_work pytest -q tests backend/conversation/test_*.py
60 passed
```

## Recommended command

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --gesture-max-hands 2 --enable-objects --enable-web-chat --preview-flip-horizontal
```
