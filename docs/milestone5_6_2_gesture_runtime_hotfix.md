# Milestone 5.6.2 Gesture Runtime Hotfix

## Problem observed

The runtime logs showed gesture control was enabled from the CLI, but the live command payload repeatedly reported:

```json
"gesture": {"status": "unavailable", "confidence": 0.0, "reason": "mediapipe_unavailable"}
```

That means Lumos was not failing to classify a thumbs-up/heart/pinch; it was not running MediaPipe Hands at all. YOLO object detection was still active, so hand shapes could be misclassified as normal desk objects such as `remote` or `cell phone` and written into object memory.

## Fix

- Strengthened MediaPipe Hands import resolution in `backend/perception/gesture_detector.py`.
- Added detailed unavailable reasons instead of the generic `mediapipe_unavailable` result.
- Added `backend/tools/check_mediapipe_hands.py` to diagnose the active venv.
- Runs gesture detection before object detection in the main loop.
- Suppresses YOLO object detections that overlap an active hand gesture, preventing heart/pinch/thumbs-up poses from being remembered as phones/remotes.

## Diagnostic command

Run this in the same activated `.venv`:

```powershell
python -m backend.tools.check_mediapipe_hands
```

Expected success line:

```text
PASS: MediaPipe Hands is available for Lumos gesture detection.
```

If it fails, repair MediaPipe inside the active venv:

```powershell
python -m pip uninstall -y mediapipe
python -m pip install --no-cache-dir mediapipe==0.10.14
```

Then rerun the diagnostic and start Lumos again.

## Full feature run command

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --save-object-frames --enable-gestures --gesture-max-hands 2 --enable-audio --enable-speaker-awareness --enable-doa --enable-stt --stt-backend faster_whisper --stt-model-size tiny.en --stt-device cpu --stt-compute-type int8 --enable-web-chat --use-llm --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --preview-flip-horizontal
```

## What to look for

Good gesture startup:

```text
MediaPipe hand gesture detector initialized max_hands=2
```

Good gesture recognition:

```text
Detected gesture status=heart confidence=... reason=two_hand_heart
```

Bad startup:

```text
Hand gestures requested but MediaPipe Hands is unavailable; gesture control is disabled reason=...
```

If bad startup appears, the diagnostic command will show which MediaPipe import path failed.
