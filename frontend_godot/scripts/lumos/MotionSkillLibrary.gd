extends RefCounted

# MotionSkillLibrary is the bounded expression/motion vocabulary for Lumos.
# Add new skills here, not inside LampController.gd.
#
# Target array shape:
# [base_yaw_deg, shoulder_pitch_deg, elbow_pitch_deg, wrist_pitch_deg, wrist_yaw_deg, head_tilt_deg, face_follow_weight]
# Keep this as target poses rather than direct joint writes so the controller can
# apply one global smoothing policy and preserve the 6-DOF embodiment boundary.

const TARGET_SIZE: int = 7


static func normalize_motion(motion: String) -> String:
	match motion:
		"idle", "idle_breathe":
			return "idle_breathe"
		"attentive_nod", "attentive_follow":
			return "attentive_follow"
		"searching_glance", "searching_glance_slow":
			return "searching_glance_slow"
		"thinking", "recalling", "thinking_slow":
			return "thinking_slow"
		"sleep", "sleep_rest":
			return "sleep_rest"
		"sleepy_search", "sleepy_search_then_rest", "sleep_scan_then_rest":
			return "sleepy_search_then_rest"
		"happy", "happy_bounce":
			return "happy_bounce"
		"dance", "dance_loop", "music_dance":
			return "dance_loop"
		"upset", "upset_turn", "sad_turn_away":
			return "upset_turn"
		_:
			return motion


static func target_for(motion: String, t: float, elapsed_s: float) -> PackedFloat32Array:
	var skill: String = normalize_motion(motion)
	match skill:
		"sleep_rest":
			return _sleep_rest(t)
		"sleepy_search_then_rest":
			return _sleepy_search_then_rest(t, elapsed_s)
		"idle_breathe":
			return _idle_breathe(t)
		"excited_greeting":
			return _excited_greeting(t, elapsed_s)
		"attentive_follow":
			return _attentive_follow(t)
		"searching_glance_slow":
			return _searching_glance_slow(t)
		"curious_tilt":
			return _curious_tilt(t)
		"gentle_wave":
			return _gentle_wave(t)
		"scanning":
			return _scanning(t)
		"thinking_slow":
			return _thinking_slow(t)
		"happy_bounce":
			return _happy_bounce(t)
		"dance_loop":
			return _dance_loop(t)
		"upset_turn":
			return _upset_turn(t)
		_:
			return _idle_breathe(t)


static func smooth_speed_for(motion: String) -> float:
	var skill: String = normalize_motion(motion)
	match skill:
		"sleep_rest":
			return 1.35
		"sleepy_search_then_rest":
			return 0.95
		"excited_greeting":
			return 2.35
		"attentive_follow":
			return 2.45
		"thinking_slow":
			return 2.65
		"happy_bounce", "dance_loop":
			return 3.10
		"upset_turn":
			return 1.90
		_:
			return 2.15


static func blocks_face_follow(motion: String) -> bool:
	var skill: String = normalize_motion(motion)
	return skill == "sleep_rest" or skill == "sleepy_search_then_rest" or skill == "upset_turn"


static func _target(
	base_yaw_deg: float,
	shoulder_pitch_deg: float,
	elbow_pitch_deg: float,
	wrist_pitch_deg: float,
	wrist_yaw_deg: float,
	head_tilt_deg: float,
	face_follow_weight: float
) -> PackedFloat32Array:
	return PackedFloat32Array([
		base_yaw_deg,
		shoulder_pitch_deg,
		elbow_pitch_deg,
		wrist_pitch_deg,
		wrist_yaw_deg,
		head_tilt_deg,
		face_follow_weight,
	])


static func _sleep_rest(_t: float) -> PackedFloat32Array:
	# User-authored sleeping pose preserved from the uploaded project.
	return _target(-85.0, 85.0, -140.0, 20.0, 0.0, -20.0, 0.0)


static func _sleepy_search_then_rest(t: float, elapsed_s: float) -> PackedFloat32Array:
	# Before sleeping, Lumos slowly checks the room as if waiting for someone.
	# The skill then converges into the exact sleep_rest pose.
	if elapsed_s < 4.0:
		var fade: float = clampf(1.0 - (elapsed_s / 3.6), 0.0, 1.0)
		var scan: float = sin(t * 4.15) * fade
		var lift: float = sin(t * 2.85) * fade
		return _target(-160.0 * scan, 10.0 + 8.0 * lift, -34.0 - 8.0 * lift, -8.0, 60.0 * scan, -10.0 + 4.0 * lift, 0.0)
	return _sleep_rest(t)


static func _idle_breathe(t: float) -> PackedFloat32Array:
	var breathe: float = sin(t * 0.62)
	var sway: float = sin(t * 0.34)
	return _target(-3 * sway, 20.0 + 9 * breathe, -40.0 + 0.8 * breathe, 10.0 + 1.1 * sin(t * 0.70), 1.2 * sway, 1.0 * sin(t * 0.52), 0.42)


static func _excited_greeting(t: float, elapsed_s: float) -> PackedFloat32Array:
	# Eye-contact response: upright, pink, delighted, then backend policy returns to attentive_follow.
	var pulse: float = maxf(0.0, sin(t * 2.60))
	var settle: float = clampf(elapsed_s / 2.4, 0.0, 1.0)
	var shoulder: float = lerpf(-4.0, -10.0, settle) - 1.8 * pulse
	var elbow: float = lerpf(16.0, 26.0, settle) + 1.4 * pulse
	var wrist: float = lerpf(-9.0, -18.0, settle) - 2.0 * pulse
	return _target(0.0, shoulder, elbow, wrist, 0.0, 7.0 + 2.4 * pulse, 0.78)


static func _attentive_follow(t: float) -> PackedFloat32Array:
	var nod: float = 1.4 * sin(t * 0.90)
	var micro_sway: float = 0.9 * sin(t * 0.45)
	return _target(micro_sway, -12.0 + 0.6 * nod, 31.0 + 0.5 * nod, -23.0 - 1.2 * nod, 0.0, 2.0 + 1.6 * nod, 1.0)


static func _searching_glance_slow(t: float) -> PackedFloat32Array:
	var scan: float = sin(t * 0.44)
	return _target(24.0 * scan, -16.0, 39.0, -18.0, 17.0 * sin(t * 0.55), -5.0 + 2.0 * sin(t * 0.40), 0.18)


static func _curious_tilt(t: float) -> PackedFloat32Array:
	var anticipation: float = maxf(0.0, sin(t * 1.12))
	var tiny_bounce: float = 0.8 * sin(t * 1.70)
	return _target(4.0 * sin(t * 0.46), -13.0 - tiny_bounce, 41.0 + tiny_bounce, -25.0 - 1.6 * anticipation, 13.0, -18.0 + 2.8 * sin(t * 1.04), 0.58)


static func _gentle_wave(t: float) -> PackedFloat32Array:
	var wave: float = sin(t * 1.08)
	return _target(8.0 * wave, -10.0, 35.0, -27.0, 20.0 * wave, -12.0 + 2.0 * sin(t * 0.80), 0.40)


static func _scanning(t: float) -> PackedFloat32Array:
	return _target(36.0 * sin(t * 0.38), -11.0 + 2.4 * sin(t * 0.58), 42.0, -21.0, 22.0 * sin(t * 0.70), -8.0, 0.10)


static func _thinking_slow(t: float) -> PackedFloat32Array:
	return _target(-1.5 + 2.0 * sin(t * 0.44), -14.5, 42.0, -30.0 + 1.6 * sin(t * 1.05), -7.0 + 2.2 * sin(t * 0.55), -17.0 + 1.8 * sin(t * 0.92), 0.32)


static func _happy_bounce(t: float) -> PackedFloat32Array:
	var bounce: float = maxf(0.0, sin(t * 1.85))
	var sway: float = sin(t * 0.82)
	return _target(10.0 * sway, -8.0 - 3.0 * bounce, 27.0 + 2.0 * bounce, -18.0 - 3.0 * bounce, 8.0 * sway, 8.0 + 3.2 * bounce, 0.65)


static func _dance_loop(t: float) -> PackedFloat32Array:
	var groove: float = sin(t * 1.18)
	var counter: float = sin(t * 1.18 + PI * 0.5)
	return _target(22.0 * groove, -9.0 - 3.0 * maxf(0.0, counter), 32.0 + 3.2 * maxf(0.0, -counter), -24.0 - 4.0 * maxf(0.0, groove), 25.0 * counter, 2.0 + 4.0 * groove, 0.35)


static func _upset_turn(t: float) -> PackedFloat32Array:
	var droop: float = 1.4 * sin(t * 0.36)
	return _target(118.0, 34.0 + droop, -66.0, 8.0, -28.0, -35.0 + droop, 0.0)
