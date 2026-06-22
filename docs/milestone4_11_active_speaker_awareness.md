# Milestone 4.11: Active Speaker Awareness + Directional Listening

## What was implemented

Milestone 4.11 adds an optional, demo-safe active-speaker-awareness layer to Lumos while preserving the existing engagement, object memory, recall, browser chat, gesture, and Godot UDP paths.

The new pipeline is:

```text
Microphone audio -> RMS VAD -> optional stereo GCC-PHAT DOA
Webcam frame -> temporary face tracks -> mouth-motion estimate
VAD + face tracks + engagement + DOA -> active speaker result
Active speaker result -> bounded behavior override -> existing command JSON + optional speaker field
Godot -> listening / sound-seeking embodiment only
```

The backend remains the intelligence layer. Godot only renders named motion and light skills.

## New backend modules

```text
backend/audio/audio_capture.py
backend/audio/voice_activity_detector.py
backend/audio/gcc_phat.py
backend/perception/face_tracker.py
backend/perception/active_speaker_detector.py
backend/behavior/speaker_policy.py
backend/evaluation/speaker_eval.py
```

## Heuristic vs. model-based behavior

This patch intentionally uses pragmatic heuristics rather than a heavyweight active speaker model:

- Speech activity is RMS energy above a rolling noise floor.
- Mouth activity is estimated from MediaPipe Face Mesh lip opening when available, with a lower-face frame-difference fallback.
- Person IDs are temporary visual tracks such as `person_1` and `person_2`, maintained by bounding-box overlap and centroid continuity.
- Speaking-to-Lumos is inferred only when the likely active speaker is also visually engaged with the camera/lamp.
- Stereo left/right direction uses GCC-PHAT TDOA when stereo input is available and `--enable-doa` is set.

This is sufficient for a reliable challenge demo without claiming identity recognition, perfect gaze tracking, or research-grade active speaker detection.

## Why TalkNet / AV-HuBERT / Light-ASD are not required

TalkNet, AV-HuBERT, Light-ASD, ODAS, HARK, ROS2, and similar systems are valid future upgrades, but they introduce model downloads, hardware assumptions, runtime cost, and integration risk. The challenge already rewards clean real-time perception-to-action architecture. For this milestone, the safer engineering tradeoff is a deterministic local heuristic pipeline that fails gracefully and keeps the main webcam/FSM/Godot loop responsive.

## Single-microphone mode vs. stereo mode

Single-microphone mode supports:

- speech active/inactive detection,
- active visible-speaker association using mouth motion,
- speaking-to-Lumos vs. speaking-elsewhere behavior,
- interruption-aware suppression of attention seeking.

Stereo mode additionally supports:

- bounded left/right speech direction hints using GCC-PHAT,
- `sound_seek_left`, `sound_seek_right`, and `sound_seek_center` Godot motions when speech is heard but no face is visible.

Mono mode never claims a left/right direction. Missing stereo input returns `available=false` with a clear reason.

## Command protocol extension

The existing command shape is preserved:

```json
{
  "timestamp": "...",
  "state": "...",
  "engagement": {},
  "behavior": {},
  "memory": {}
}
```

Milestone 4.11 adds an optional top-level field:

```json
"speaker": {
  "speech_detected": true,
  "active_track_id": "person_1",
  "active_track_location": "center",
  "speaking_to_robot": true,
  "confidence": 0.78,
  "reason": "audio_active_mouth_motion_engaged",
  "mouth_motion_score": 0.62,
  "audio_confidence": 0.84,
  "doa_azimuth_deg": 18.5,
  "doa_confidence": 0.55
}
```

Consumers that ignore `speaker` continue to work.

## Behavior policy

When confidence is above the policy threshold:

1. User speaking to Lumos:
   - motion: `active_listen`
   - light: `listening_blue`
   - attention seeking is not triggered while the user is actively talking to Lumos.

2. Visible person speaking elsewhere:
   - motion: `listening_attentive`
   - light: `speech_focus`
   - Lumos stays aware but non-intrusive.

3. Speech heard but no visible face:
   - stereo DOA available: `sound_seek_left`, `sound_seek_right`, or `sound_seek_center`
   - no DOA: `sound_seek_center`

4. No speech or low confidence:
   - existing behavior policy continues unchanged.

## Godot embodiment updates

New motion skills:

```text
active_listen
listening_attentive
sound_seek_left
sound_seek_right
sound_seek_center
```

New light skills:

```text
listening_blue
speech_focus
```

The DOA value is used only as a bounded yaw hint. Godot clamps the head/base-yaw contribution to a safe range and smooths it through the existing physical motion controller. The rig root/base remains planted and does not spin or translate.

The debug overlay now shows:

```text
Speech: active/inactive
Speaker: person_1 / none
To Lumos: yes/no/unknown
DOA: +18 deg / unavailable
Speaker reason: ...
```

## How to run

Single-microphone demo:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --preview-flip-horizontal
```

Stereo direction demo:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --enable-doa --mic-distance-m 0.08 --preview-flip-horizontal
```

Minimal speaker-awareness test without object detection:

```powershell
python -m backend.main --show-window --godot-udp --enable-audio --enable-speaker-awareness --speaker-debug
```

## Logs and evaluation

New runtime event log:

```text
logs/runs/<run_id>/speaker_events.jsonl
logs/latest/speaker_events.jsonl
```

New latency columns:

```text
audio_capture_ms
vad_ms
doa_ms
active_speaker_fusion_ms
speaker_policy_ms
```

Event types include:

```text
voice_activity
doa_result
speaker_result
speaker_behavior_override
speaker_awareness_disabled_reason
```

Pure logic tests are in:

```text
tests/test_voice_activity_detector.py
tests/test_gcc_phat.py
tests/test_active_speaker_fusion.py
tests/test_speaker_policy.py
```

## Known limitations

- The VAD is energy-based, so loud non-speech sounds may trigger speech.
- Mouth motion is heuristic and can be confused by lighting changes, chewing, or camera motion.
- Temporary track IDs are not identity recognition and may swap under occlusion or crossing faces.
- Speaking-to-Lumos is an engagement approximation, not true intent recognition.
- DOA requires a real stereo microphone geometry and a correct `--mic-distance-m` value.
- A laptop “stereo mix” or beamforming mic may not expose true raw left/right channels.

## Future upgrades

Recommended upgrade path for true ML-based active speaker detection:

1. Record synchronized webcam/audio clips from the Lumos demo setup.
2. Add an offline evaluation harness with frame-level labels: speaker/no speaker, speaking person, looking at Lumos, looking elsewhere.
3. Integrate a lightweight pre-trained active speaker detector behind the same `ActiveSpeakerResult` interface.
4. Keep the current heuristic path as fallback for missing models or low-confidence ML predictions.
5. Add calibrated thresholds and confusion matrix reporting for single-user and two-user scenes.
