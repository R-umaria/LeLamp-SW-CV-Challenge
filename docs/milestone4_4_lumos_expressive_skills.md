# Milestone 4.4 — Lumos Expressive Skills Patch

## Purpose

This patch renames the demo personality to **Lumos** and makes the Godot embodiment more expressive while preserving the existing software-first architecture:

- Python still owns perception, state, memory, recall, and behavior selection.
- Godot still owns bounded rendering of named motion/light skills.
- The existing backend-to-Godot JSON command shape is preserved.

## Research inspiration

The patch follows the ELEGNT design direction: non-anthropomorphic robots should balance functional objectives with expressive objectives such as attention, intention, and emotion. In Lumos, this is implemented as a small bounded skill vocabulary rather than free-form motion generation.

## New frontend structure

New files:

- `frontend_godot/scripts/lumos/MotionSkillLibrary.gd`
- `frontend_godot/scripts/lumos/LightSkillLibrary.gd`

Updated file:

- `frontend_godot/scripts/LampController.gd`

`LampController.gd` now handles only command parsing, face-follow smoothing, joint interpolation, and material/light synchronization. New expressive motions should be added to `MotionSkillLibrary.gd`; new light moods should be added to `LightSkillLibrary.gd`.

## Motion timing changes

All existing motions are intentionally slower:

- Global motion time scale is reduced through `motion_time_scale`.
- Joint interpolation speeds are lower and centralized in `MotionSkillLibrary.smooth_speed_for()`.
- Face following is wider but smoother: max yaw is increased, smoothing speed is reduced.

## New bounded motion skills

| Skill | Trigger / use | Description |
|---|---|---|
| `excited_greeting` | first seconds after engaged transition | Lumos stands more upright and glows pink after eye contact. |
| `attentive_follow` | stable engaged state | Lumos calmly tracks the face hint from the backend. |
| `searching_glance_slow` | disengaged state | Slow look-around motion when the user looks away. |
| `curious_tilt` | early seeking-attention state | Subtle attention-seeking tilt and pulse. |
| `gentle_wave` | sustained seeking-attention state | Slight wave-like motion after longer disengagement. |
| `sleepy_search_then_rest` | idle/stale/sleep transition | Looks around briefly, then settles into the saved sleep pose. |
| `sleep_rest` | sleep state | Preserves the user-updated sleeping pose. |
| `thinking_slow` | recall in progress | Slow contemplative motion for memory recall. |
| `happy_bounce` | optional/test command | Happy upward bounce. |
| `dance_loop` | optional/test command, future music trigger | Rhythmic bounded dance motion. |
| `upset_turn` | optional/test command, future disagreement trigger | Turns away with blue/dim light. |

## Backend behavior policy changes

Updated file:

- `backend/behavior/behavior_policy.py`

The new `behavior_for_transition()` function selects short-lived expressive behaviors using the FSM transition context:

- New eye contact / engaged transition → `excited_greeting` + `excited_pink` for about 2.6 seconds.
- Stable engaged → `attentive_follow`.
- Idle/no face → `sleepy_search_then_rest`, then `sleep_rest`.
- Sustained seeking attention → `gentle_wave`.
- Recall → `thinking_slow`.

This keeps behavior selection deterministic and explainable.

## Manual skill test

Start Godot, then run:

```bash
python -m backend.tools.send_test_commands --interval 1.2 --count 13 --print-json
```

This cycles through eye-contact excitement, face following, disengagement, seeking attention, scanning, recall thinking, happy, dance, upset, and sleep transition skills without requiring the webcam pipeline.

## Limitations

- `dance_loop`, `happy_bounce`, and `upset_turn` are implemented as bounded skills and included in the UDP test tool, but they are not yet connected to real audio detection or robust natural-language sentiment detection.
- Face following still uses backend face-center hints from practical CV, not full 3D gaze estimation.
- Godot does not autonomously decide emotions; it only renders named skills from Python commands.
