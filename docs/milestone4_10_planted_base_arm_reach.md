# Milestone 4.10 — Planted Base Arm Reach Distance Control

## Problem fixed

The previous distance-follow behavior used the detected face box size to move the whole Godot `LampRig` root forward or backward. That made Lumos look as if it had a mobile base or wheels. A real desk-lamp/robot-arm embodiment should keep its base planted and change reach by articulating the arm joints.

## Design decision

Lumos now treats distance maintenance as a lightweight 2D IK-style reach offset layered on top of the existing bounded motion skills:

- **Base/root position stays fixed** at the scene-authored home position.
- **Shoulder and elbow** handle reach/fold behavior.
- **Wrist pitch and head tilt** compensate so the lamp keeps looking at the detected face.
- Existing S-curve, velocity, acceleration, and jerk limits still smooth the resulting joint targets.

This keeps the same Python perception/state/command protocol. The change is entirely in the Godot embodiment layer.

## Files changed

- `frontend_godot/scripts/LampController.gd`
- `frontend_godot/scripts/lumos/MotionSkillLibrary.gd`
- `frontend_godot/scripts/Main.gd`
- `frontend_godot/README.md`

## Runtime behavior

When the face box is small, Lumos assumes the user is farther away:

- shoulder rotates into a more forward/reaching posture;
- elbow extends;
- wrist/head compensate to maintain focus.

When the face box is too large, Lumos assumes the user is too close:

- shoulder folds backward/upward;
- elbow flexes inward;
- wrist/head compensate to keep the lamp aimed at the user.

The root `position` is no longer used for face-distance correction.

## Inspector parameters

Useful values on `LampRig`:

- `planted_base_enabled`: keep this enabled for the physical-robot demo.
- `distance_follow_enabled`: enables/disables face-box-based reach compensation.
- `desired_face_area_ratio`: target apparent face size.
- `close_face_area_ratio`: threshold at which Lumos starts folding back.
- `distance_follow_max_forward`: max forward reach blend.
- `distance_follow_max_back`: max fold-back blend.
- `distance_reach_smooth_speed`: low-pass speed for reach changes.
- `distance_reach_shoulder_forward_degrees`
- `distance_reach_elbow_forward_degrees`
- `distance_reach_wrist_forward_degrees`
- `distance_reach_head_forward_degrees`
- `distance_reach_shoulder_back_degrees`
- `distance_reach_elbow_back_degrees`
- `distance_reach_wrist_back_degrees`
- `distance_reach_head_back_degrees`

## Tuning guidance

If Lumos does not appear to reach forward enough when you move away, increase:

- `distance_follow_max_forward`
- `distance_reach_elbow_forward_degrees`
- the magnitude of `distance_reach_shoulder_forward_degrees`

If Lumos folds too aggressively when you move close, reduce:

- `distance_follow_max_back`
- `distance_reach_elbow_back_degrees` magnitude
- `distance_reach_shoulder_back_degrees`

If the head drifts off the face during reach/fold, tune:

- `distance_reach_wrist_forward_degrees`
- `distance_reach_head_forward_degrees`
- `distance_reach_wrist_back_degrees`
- `distance_reach_head_back_degrees`

## Why this is better for the challenge

This keeps the simulator aligned with the intended 6-DOF robotic-arm abstraction: the first joints position the head/end-effector, and the wrist/head joints preserve orientation. It also makes the demo more credible if the same behavior is later ported to servo hardware.
