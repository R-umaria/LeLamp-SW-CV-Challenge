# Milestone 5.0: Final Challenge Hardening + IK Embodiment

Milestone 5.0 hardens Lumos for the SW/CV Challenge without changing the core architecture: Python remains the perception, memory, behavior, evaluation, and recall layer; Godot remains the expressive embodiment layer.

## What changed

### Engagement detection

- Added `backend/perception/head_pose_estimator.py`.
- Uses MediaPipe Face Mesh when available and OpenCV `solvePnP` to estimate approximate yaw, pitch, and roll.
- Classifies head pose as `looking_at_lamp`, `looking_left`, `looking_right`, `looking_down`, `looking_away`, or `unknown`.
- Fuses head-pose classification with the existing face-center and face-size heuristic.
- Preserves safe fallback mode when Face Mesh, landmarks, or solvePnP are unavailable.
- Adds temporal smoothing and logs yaw, pitch, roll, confidence, reason, and fallback mode.

### Audio output and TTS

- Added `backend/audio/audio_output.py` and `backend/audio/tts_client.py`.
- Optional non-blocking sound cue worker for engagement, attention seeking, recall found/not found, and error events.
- Optional offline/local TTS through `pyttsx3` when available.
- Missing audio/TTS dependencies are warnings, not fatal errors.

New CLI flags:

```powershell
--enable-audio-output
--enable-tts
--tts-rate 175
--tts-volume 0.85
```

### Voice/STT recall gating

- Added `backend/audio/stt_intent_gate.py`.
- Voice recall now requires either a configured wake phrase or a clear object-memory query.
- Rejects random background speech, incomplete fragments, and duplicate/cooldown transcripts.
- Browser chat remains the most reliable recall path.

New CLI flags:

```powershell
--stt-require-wake-word
--stt-wake-words Lumos,"hey Lumos"
--stt-min-confidence 0.0
--stt-cooldown-sec 1.25
```

### Structured object memory localization

- Extended object memory records with normalized image-space fields:
  - `center_x_norm`
  - `center_y_norm`
  - `zone_x`
  - `zone_y`
  - `distance_hint`
  - `pointing_target`
- Existing SQLite rows remain compatible through safe migration.
- Recall answers now use more precise but honest camera-view wording.
- Recall pointing uses normalized target coordinates when available.

### Godot IK / reverse kinematics embodiment

- Added a bounded analytic IK layer in `frontend_godot/scripts/lumos/IkSolver.gd`.
- Added a compatibility wrapper at `frontend_godot/scripts/IkSolver.gd`.
- Updated `frontend_godot/scripts/LampController.gd` so look/follow/point targets are solved through base yaw, shoulder, elbow, wrist, and head joints.
- The rig root/base remains planted; distance changes are produced by folding/unfolding the arm chain, not by sliding the lamp.
- Added debug values for IK target position, workspace clamping, and joint clamp indicators.

Supported target command extension:

```json
{
  "target": {
    "type": "point_to_memory",
    "x_norm": 0.25,
    "y_norm": 0.65,
    "hold_sec": 5.0
  }
}
```

Older commands that only send `state`, `behavior.motion`, `behavior.light`, `sound`, and `speech_text` still work.

### Interruption awareness

- Extended `backend/behavior/speaker_policy.py`.
- If speech is detected but is not directed to Lumos, the behavior policy marks:
  - `do_not_interrupt`
  - `quiet_listening`
  - `safe_to_respond`
- Attention-seeking speech/sound is suppressed during non-directed speech.

### Evaluation tooling

- Added labeled engagement evaluation utilities:
  - `backend/evaluation/label_schema.py`
  - `backend/evaluation/engagement_eval.py`
  - `backend/evaluation/analyze_logs.py`
- Added `docs/evaluation_quickstart.md`.
- Outputs are written to:
  - `logs/evaluation/engagement_eval_summary.csv`
  - `logs/evaluation/engagement_confusion_matrix.csv`
  - `logs/evaluation/latency_summary.csv`

No evaluation results are fabricated. These scripts only analyze collected labels/logs.

## Recommended demo commands

Core reliable demo path:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --save-object-frames --enable-web-chat --use-llm --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --preview-flip-horizontal
```

Optional voice/STT/TTS path:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --save-object-frames --enable-web-chat --enable-audio --enable-stt --stt-backend faster_whisper --stt-model-size tiny.en --stt-device cpu --stt-compute-type int8 --stt-require-wake-word --stt-wake-words Lumos,"hey Lumos" --enable-audio-output --enable-tts --use-llm --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --preview-flip-horizontal
```

## Evaluation commands

Run all tests:

```powershell
python -m pytest
```

Run targeted Milestone 5.0 tests:

```powershell
python -m pytest tests/test_head_pose_estimator.py tests/test_stt_intent_gate.py tests/test_memory_location_schema.py tests/test_interruption_policy.py tests/test_audio_output.py
```

Analyze labeled engagement logs:

```powershell
python -m backend.evaluation.engagement_eval --labels path\to\labels.csv --latency-log logs\latest\latency.csv --output-dir logs\evaluation
```

Analyze latency only:

```powershell
python -m backend.evaluation.analyze_logs --latest --out logs\evaluation\latency_summary.csv
```

## Dependency notes

- Head pose uses MediaPipe Face Mesh if `mediapipe` is installed; otherwise Lumos falls back to the existing face heuristic.
- TTS uses `pyttsx3` only when installed and enabled. Without it, Lumos still runs with text-only responses.
- Audio cues are best effort. On Windows, simple beeps use `winsound`; on non-Windows platforms, a terminal bell fallback is used.
- Ollama remains optional. Deterministic recall fallback remains available.

## Known limitations

- Head-pose estimation is an approximate webcam-based signal, not calibrated eye tracking.
- Structured object localization is image-space only; it does not claim true 3D world coordinates.
- Godot IK is an embodiment controller, not a physics-accurate robotics simulator.
- STT is still optional and should be treated as less reliable than browser chat for final demo safety.
