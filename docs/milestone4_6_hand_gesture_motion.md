# Milestone 4.6 — Hand Gesture Control + Pixar-Style Motion Polish

## Goal

This patch adds deliberate hand gesture control while preserving the project boundary:

- Python performs perception and behavior selection.
- Godot remains a bounded expressive renderer.
- The command protocol stays deterministic and gains an optional `gesture` display/debug field.

## New gesture controls

Gesture detection is optional and uses MediaPipe Hands when enabled.

| Gesture | Backend label | Godot motion | Intended behavior |
|---|---|---|---|
| Index-finger calling / beckon | `beckon` | `gesture_approach` | Lumos leans and slides closer toward the camera. |
| Open palm facing camera | `palm_push` | `gesture_retreat` | Lumos recoils/slides away from the camera. |

The detector degrades safely. If `mediapipe` is not installed or `--enable-gestures` is not passed, the backend continues with normal engagement, memory, recall, and Godot UDP behavior.

## Run command

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --enable-objects --enable-web-chat --preview-flip-horizontal
```

The preview overlay shows the detected hand gesture, confidence, and reason. Latency logs now include:

```text
gesture_detection_ms
gesture_status
gesture_confidence
gesture_reason
```

## Tracking stability changes

Defaults were made less brittle so Lumos does not enter attention-seeking just because the user's face is slightly near the edge of the camera frame:

- `--center-tolerance-x` default increased to `0.36`
- `--center-tolerance-y` default increased to `0.34`
- `--smoothing-window` default increased to `9`
- `--exit-disengaged-frames` default increased to `9`
- `--exit-absent-frames` default increased to `12`
- `--clear-engaged-confidence` default lowered to `0.72`

## Motion changes

Most Godot motion skills now use a negative elbow target so the elbow forms a rear hump opposite the lamp head. This makes the silhouette closer to a Pixar-style lamp instead of a generic robot arm.

New or changed frontend behavior:

- `gesture_approach`: beckon response with forward slide and eager posture.
- `gesture_retreat`: open-palm response with backward slide and recoil posture.
- `sleepy_search_then_rest`: longer pre-sleep scanning sequence before returning to the sleep pose.
- `attentive_follow`: wider face-follow range with slower smoothing.
- Existing light skills are reused; no LLM or frontend autonomy was added.

## Files changed

```text
backend/main.py
backend/requirements.txt
backend/behavior/behavior_policy.py
backend/behavior/command_protocol.py
backend/perception/gesture_detector.py
backend/utils/config.py
backend/utils/preview_window.py
frontend_godot/scripts/LampController.gd
frontend_godot/scripts/Main.gd
frontend_godot/scripts/lumos/MotionSkillLibrary.gd
docs/milestone4_6_hand_gesture_motion.md
```

## Tuning notes

For the final demo, keep `--enable-gestures` on only when you are intentionally testing hand controls. Gesture detection adds another real-time CV pass, so measure latency with and without it.

Useful knobs:

```powershell
--gesture-confidence 0.70
--gesture-hold 1.25
--gesture-interval 0.12
--center-tolerance-x 0.40
--exit-disengaged-frames 11
```

The next feature, interruption awareness, should be implemented as a separate module after this patch is stable. It should not be mixed with gesture control because it likely needs audio activity, face orientation, and possibly multi-person context.
