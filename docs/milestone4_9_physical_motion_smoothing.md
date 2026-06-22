# Milestone 4.9 Physical Motion Smoothing

## Goal

Make Lumos' Godot embodiment behave more like a physical 6-DOF robot instead of a visual-only animation rig. The Python backend still owns perception, state, memory, recall, and behavior selection. Godot still receives bounded behavior commands, but now converts motion targets into actuator-safe profiles before moving the lamp joints.

## Problem Fixed

The previous controller used direct exponential interpolation toward the latest target pose. That looked smooth at low speed, but when the backend switched from one behavior state to another, the target pose could jump instantly. If this were mapped to servos, those discontinuities would create unnecessary velocity and acceleration spikes.

## What Changed

### 1. S-curve transition blending

When the backend switches motion names, `LampController.gd` captures the current actual joint pose and blends toward the next skill target using a quintic smoothstep curve. This gives the transition zero initial and final slope/curvature, reducing hard starts and stops.

Relevant exports:

- `transition_blend_seconds`
- `min_transition_blend_seconds`
- `max_transition_blend_seconds`

Large pose changes automatically receive more transition time, capped by `max_transition_blend_seconds`.

### 2. Velocity, acceleration, and jerk limits

Each of the six joints now has an internal servo profile state:

- joint position
- joint velocity
- joint acceleration

The controller computes a desired velocity toward the target, clamps that velocity, clamps acceleration, and then slew-limits acceleration change. The acceleration slew limit is a jerk limit.

Relevant exports:

- `physical_motion_enabled`
- `servo_response_gain`
- `servo_max_velocity_degrees_per_second`
- `servo_max_acceleration_degrees_per_second2`
- `servo_max_jerk_degrees_per_second3`
- `servo_snap_error_degrees`

### 3. Root movement profiling

The forward/back distance-follow behavior also now uses velocity, acceleration, and jerk limits instead of direct position lerp. This keeps the whole lamp base from sliding abruptly when face size changes.

Relevant exports:

- `root_response_gain`
- `root_max_speed_units_per_second`
- `root_max_acceleration_units_per_second2`
- `root_max_jerk_units_per_second3`

### 4. Sensor/control slew limiting

Face-follow target inputs are still low-pass filtered, but now also have an explicit maximum angular slew rate. This prevents sudden face detection jumps from causing sharp yaw/pitch changes.

Relevant export:

- `face_follow_max_slew_degrees_per_second`

### 5. Light slew limiting

Spotlight, head emission, and visible cone colors/brightness now slew toward their targets instead of hard-switching. This makes expressive light changes feel like a real lamp driver ramping output.

Relevant exports:

- `smooth_light_enabled`
- `light_color_slew_per_second`
- `light_energy_slew_per_second`

## Files Changed

- `frontend_godot/scripts/LampController.gd`
- `frontend_godot/scripts/Main.gd`
- `docs/milestone4_9_physical_motion_smoothing.md`

## Design Rationale

This patch keeps the intelligence layer unchanged. The backend can still issue simple behavior commands such as `attentive_follow`, `recall_point`, or `sleepy_search_then_rest`. The Godot embodiment layer is responsible for making those commands safe, continuous, and expressive.

This is the right separation for a software-first prototype that may later be mapped to physical NOVA hardware or Raspberry Pi servo control.

## Tuning Notes

If Lumos feels too slow, increase:

- `servo_max_velocity_degrees_per_second`
- `servo_max_acceleration_degrees_per_second2`
- `servo_max_jerk_degrees_per_second3`

If Lumos still feels jerky, decrease velocity/acceleration slightly or increase:

- `transition_blend_seconds`
- `min_transition_blend_seconds`

For the final demo, keep `physical_motion_enabled = true` so the behavior clearly demonstrates physical-robot awareness.
