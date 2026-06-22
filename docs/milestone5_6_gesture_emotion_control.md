# Milestone 5.6 — Gesture Emotion Control

## Goal

This patch expands Lumos' existing MediaPipe hand-gesture layer from simple approach/retreat control into a small, bounded gesture language for expressive robot control.

Python remains the intelligence layer:

- MediaPipe Hands detects landmarks in the backend.
- `backend/perception/gesture_detector.py` classifies landmarks into discrete labels.
- `backend/behavior/behavior_policy.py` maps labels to known behavior names.
- Godot only renders named motion/light skills and normalized target hints.

No LLM is allowed to directly control motion.

## Supported gestures

| Gesture | Backend label | Godot motion | Light | Behavior |
|---|---|---|---|---|
| Index beckon | `beckon` | `gesture_approach` | `happy_gold` | Lumos leans closer using arm joints, not base translation. |
| Open palm push | `palm_push` | `gesture_retreat` | `soft_pulse` | Lumos recoils/back-away posture. |
| Thumbs up | `thumbs_up` | `gesture_thumbs_up` | `happy_gold` | Lumos performs an affirmative happy bounce. |
| Thumb-index pinch | `pinch_follow` | `gesture_pinch_follow` | `focus_glow` | Lumos tracks the pinch target like face-follow tracking. |
| Two-hand heart | `heart` | `gesture_heart_blush` | `baby_pink_blush` | Lumos acts shy/delighted and blushes baby pink. |

## Run command

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --gesture-max-hands 2 --enable-objects --enable-web-chat --preview-flip-horizontal
```

For the full voice + perception demo, keep the existing STT/speaker flags and add the gesture flags:

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --gesture-max-hands 2 --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --enable-stt --preview-flip-horizontal
```

## Implementation details

### Backend files

- `backend/perception/gesture_detector.py`
  - Adds deterministic labels: `thumbs_up`, `pinch_follow`, and `heart`.
  - Keeps safe fallback when MediaPipe is unavailable.
  - Adds `target_x_norm` / `target_y_norm` to the gesture payload for pinch tracking.
  - Supports up to two hands for heart detection.

- `backend/utils/config.py`
  - Sets `max_num_hands=2` by default.
  - Adds thresholds for thumbs-up, pinch, and two-hand heart detection.

- `backend/main.py`
  - Adds `--gesture-max-hands`.
  - Logs all active gesture labels and their normalized target point.
  - Gives deliberate hand gestures priority over active-speaker motion overrides, except during recall feedback.

- `backend/behavior/behavior_policy.py`
  - Maps gestures to bounded Godot behavior names.

### Frontend files

- `frontend_godot/scripts/LampController.gd`
  - Parses gesture target coordinates.
  - Uses gesture target coordinates for `gesture_pinch_follow`.
  - Keeps the base planted; distance and gesture expression are arm-joint based.

- `frontend_godot/scripts/lumos/MotionSkillLibrary.gd`
  - Adds `gesture_thumbs_up`, `gesture_pinch_follow`, and `gesture_heart_blush`.

- `frontend_godot/scripts/lumos/LightSkillLibrary.gd`
  - Adds `baby_pink_blush`, synchronized across emissive head, visible light cone, and SpotLight3D.

- `frontend_godot/scripts/Main.gd`
  - Displays gesture label, confidence, and target point in the debug overlay.

## Tuning notes

Useful knobs:

```powershell
--gesture-confidence 0.66
--gesture-hold 1.10
--gesture-interval 0.10
--gesture-max-hands 2
```

If FPS drops, increase `--gesture-interval` to `0.15` or temporarily set `--gesture-max-hands 1`. Heart detection requires two hands, so it needs `--gesture-max-hands 2`.

## Limitations

- This is practical webcam gesture recognition, not perfect sign-language understanding.
- The two-hand heart is a heuristic and may need tuning for camera distance and lighting.
- Pinch follow tracks the normalized thumb-index midpoint from the camera frame; it does not estimate 3D hand depth.
- Gesture priority is deliberately bounded so recall, memory, and safety-critical logic stay deterministic.
