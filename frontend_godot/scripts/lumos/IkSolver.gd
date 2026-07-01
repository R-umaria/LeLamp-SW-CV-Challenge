extends RefCounted

# Deterministic bounded 2-link IK for the expressive lamp embodiment.
# Python sends semantic/normalized targets; Godot only solves geometry for the
# already-selected embodiment target. The rig root/base stays fixed on the desk.

const RAD_TO_DEG: float = 180.0 / PI

const MIN_BASE_YAW_DEG: float = -155.0
const MAX_BASE_YAW_DEG: float = 155.0
const MIN_SHOULDER_DEG: float = -100.0
const MAX_SHOULDER_DEG: float = 100.0
const MIN_ELBOW_DEG: float = -150.0
const MAX_ELBOW_DEG: float = 115.0
const MIN_WRIST_DEG: float = -85.0
const MAX_WRIST_DEG: float = 85.0
const MIN_WRIST_YAW_DEG: float = -95.0
const MAX_WRIST_YAW_DEG: float = 95.0
const MIN_HEAD_TILT_DEG: float = -62.0
const MAX_HEAD_TILT_DEG: float = 46.0

static func solve_normalized_target(target_type: String, x_norm: float, y_norm: float, reach_shift: float) -> Dictionary:
	var x: float = clampf(x_norm, 0.0, 1.0)
	var y: float = clampf(y_norm, 0.0, 1.0)
	var x_offset: float = clampf(0.5 - x, -0.5, 0.5)
	var y_offset: float = clampf(0.5 - y, -0.5, 0.5)

	var base_yaw_deg: float = clampf(x_offset * 164.0, MIN_BASE_YAW_DEG, MAX_BASE_YAW_DEG)
	var target_radius: float = 1.08 + clampf(reach_shift, -0.72, 0.70) * 0.42
	if target_type == "point_to_memory":
		target_radius += 0.16
	elif target_type == "sleep_rest":
		target_radius = 0.58
	var target_height: float = 0.20 + y_offset * 0.88
	if target_type == "scanning":
		target_height += 0.10

	var link_a: float = 0.86
	var link_b: float = 0.78
	var distance: float = sqrt(target_radius * target_radius + target_height * target_height)
	var min_reach: float = absf(link_a - link_b) + 0.04
	var max_reach: float = link_a + link_b - 0.05
	var clamped: bool = false
	if distance < min_reach:
		distance = min_reach
		clamped = true
	elif distance > max_reach:
		distance = max_reach
		clamped = true

	var cos_elbow: float = clampf((distance * distance - link_a * link_a - link_b * link_b) / (2.0 * link_a * link_b), -1.0, 1.0)
	var elbow_inner: float = acos(cos_elbow)
	var elbow_pitch_deg: float = -(180.0 - elbow_inner * RAD_TO_DEG)
	var shoulder_angle: float = atan2(target_height, target_radius) - atan2(link_b * sin(elbow_inner), link_a + link_b * cos(elbow_inner))
	var shoulder_pitch_deg: float = -shoulder_angle * RAD_TO_DEG + 12.0

	var desired_head_tilt_deg: float = clampf(y_offset * 84.0, MIN_HEAD_TILT_DEG, MAX_HEAD_TILT_DEG)
	var wrist_pitch_deg: float = clampf(-(shoulder_pitch_deg + elbow_pitch_deg) * 0.32 - desired_head_tilt_deg * 0.18, MIN_WRIST_DEG, MAX_WRIST_DEG)
	var wrist_yaw_deg: float = clampf(base_yaw_deg * 0.34, MIN_WRIST_YAW_DEG, MAX_WRIST_YAW_DEG)

	if target_type == "sleep_rest":
		base_yaw_deg = -85.0
		shoulder_pitch_deg = 85.0
		elbow_pitch_deg = -140.0
		wrist_pitch_deg = 20.0
		wrist_yaw_deg = 0.0
		desired_head_tilt_deg = -20.0

	return {
		"base_yaw_deg": _clamp_marked(base_yaw_deg, MIN_BASE_YAW_DEG, MAX_BASE_YAW_DEG),
		"shoulder_pitch_deg": _clamp_marked(shoulder_pitch_deg, MIN_SHOULDER_DEG, MAX_SHOULDER_DEG),
		"elbow_pitch_deg": _clamp_marked(elbow_pitch_deg, MIN_ELBOW_DEG, MAX_ELBOW_DEG),
		"wrist_pitch_deg": _clamp_marked(wrist_pitch_deg, MIN_WRIST_DEG, MAX_WRIST_DEG),
		"wrist_yaw_deg": _clamp_marked(wrist_yaw_deg, MIN_WRIST_YAW_DEG, MAX_WRIST_YAW_DEG),
		"head_tilt_deg": _clamp_marked(desired_head_tilt_deg, MIN_HEAD_TILT_DEG, MAX_HEAD_TILT_DEG),
		"target_position": Vector3(x_offset, target_height, target_radius),
		"workspace_clamped": clamped,
	}

static func _clamp_marked(value: float, min_value: float, max_value: float) -> Dictionary:
	var clamped_value: float = clampf(value, min_value, max_value)
	return {"value": clamped_value, "clamped": absf(clamped_value - value) > 0.001}
