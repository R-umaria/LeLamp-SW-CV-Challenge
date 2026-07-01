# Milestone 5.7: Multi-Face Speaker Awareness Fix

## Problem fixed

The previous speaker-awareness path could effectively collapse the visible scene to the primary engagement face. That made active-speaker awareness unreliable in two-person scenes: if the speaking person was not the selected engagement face, the system often had no valid face track to assign speech to.

## Architecture change

Added `backend/perception/multi_face_detector.py` as a shared perception layer for all visible face candidates.

The new flow is:

1. `MultiFaceDetector` detects all credible faces in the frame.
2. `FaceEngagementDetector` selects one primary face for the existing engagement state machine.
3. `FaceTracker` keeps all visible face tracks for active-speaker fusion.
4. `ActiveSpeakerDetector` chooses the likely speaker from the full face-track list using audio activity plus mouth-motion evidence.

This preserves the existing single-user engagement behavior while allowing multi-person speaker awareness.

## Detection strategy

`MultiFaceDetector` uses:

- MediaPipe FaceDetection when available, which is better for multiple people and off-center faces.
- OpenCV Haar frontal and profile cascades as an offline fallback.
- Deduplication across detector outputs using IoU and normalized center distance.
- Candidate filtering using the existing engagement area thresholds.

## Key files changed

- `backend/perception/multi_face_detector.py`
- `backend/perception/engagement_detector.py`
- `backend/perception/face_tracker.py`
- `backend/utils/config.py`
- `backend/main.py`
- `tests/test_multi_face_speaker_tracking.py`

## Important default change

`--speaker-secondary-min-area` now defaults to `0.006` instead of `0.018`.

Reason: the old value matched the primary engagement face-size threshold, which was too strict for secondary speakers. A second person standing farther from the camera could be a valid speaker but still be dropped before speaker fusion.

You can still raise this value if false `person_2` tracks appear in a noisy room.

## Recommended test command

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --save-object-frames --enable-gestures --gesture-max-hands 2 --enable-audio --enable-speaker-awareness --enable-doa --enable-stt --stt-backend faster_whisper --stt-model-size tiny.en --stt-device cpu --stt-compute-type int8 --enable-web-chat --use-llm --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --preview-flip-horizontal --speaker-debug
```

Watch `logs/latest/runtime.log` for lines like:

```text
speaker_debug face_tracks=[{'track_id': 'person_1', ...}, {'track_id': 'person_2', ...}]
speaker_result speech=True track=person_2 ...
```

## Validation

Automated tests pass:

```text
62 passed
```
