extends RefCounted

# Bounded motion vocabulary for Lumos. Backend commands select one of these named
# skills; this library converts the name into target joint poses for the renderer.
#
# Target array shape:
# [base_yaw_deg, shoulder_pitch_deg, elbow_pitch_deg, wrist_pitch_deg, wrist_yaw_deg, head_tilt_deg, face_follow_weight]
# Negative elbow pitch is used as the default "hump" posture so the elbow bends
# away from the lamp face, closer to Pixar-style desk-lamp body language.

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
		"gesture_approach", "beckon", "come_closer":
			return "gesture_approach"
		"gesture_retreat", "palm_push", "move_away":
			return "gesture_retreat"
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
		"gesture_approach":
			return _gesture_approach(t)
		"gesture_retreat":
			return _gesture_retreat(t)
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


static func root_shift_for(motion: String, t: float, _elapsed_s: float) -> float:
	# Unitless value consumed by LampController. Positive moves toward the camera;
	# negative moves away. The small oscillation makes gesture control feel alive
	# without introducing unbounded translation.
	var skill: String = normalize_motion(motion)
	match skill:
		"gesture_approach":
			return 1.0 + 0.05 * sin(t * 1.8)
		"gesture_retreat":
			return -0.85 + 0.04 * sin(t * 1.2)
		_:
			return 0.0


static func smooth_speed_for(motion: String) -> float:
	var skill: String = normalize_motion(motion)
	match skill:
		"sleep_rest":
			return 1.05
		"sleepy_search_then_rest":
			return 0.78
		"excited_greeting":
			return 1.70
		"attentive_follow":
			return 1.85
		"gesture_approach", "gesture_retreat":
			return 2.10
		"thinking_slow":
			return 1.95
		"happy_bounce", "dance_loop":
			return 2.35
		"upset_turn":
			return 1.35
		_:
			return 1.55


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
	return _target(-85.0, 85.0, -140.0, 20.0, 0.0, -20.0, 0.0)


static func _sleepy_search_then_rest(t: float, elapsed_s: float) -> PackedFloat32Array:
	# Longer pre-sleep scan: Lumos checks left, right, high, and low before folding
	# back into the preserved sleep pose.
	if elapsed_s < 5.8:
		var fade: float = clampf(1.0 - (elapsed_s / 5.8), 0.18, 1.0)
		var scan: float = (0.72 * sin(t * 0.72) + 0.28 * sin(t * 1.55)) * fade
		var lift: float = sin(t * 0.95 + 0.6) * fade
		return _target(58.0 * scan, 48.0 + 8.0 * lift, -104.0 - 12.0 * lift, 26.0 + 6.0 * lift, -32.0 * scan, -12.0 + 7.0 * lift, 0.0)
	return _sleep_rest(t)


static func _idle_breathe(t: float) -> PackedFloat32Array:
	var breathe: float = sin(t * 0.52)
	var sway: float = sin(t * 0.30)
	return _target(-4.0 * sway, 34.0 + 6.0 * breathe, -78.0 - 3.0 * breathe, 22.0 + 2.0 * sin(t * 0.62), 3.0 * sway, -3.0 + 2.0 * sin(t * 0.46), 0.48)


static func _excited_greeting(t: float, elapsed_s: float) -> PackedFloat32Array:
	var pulse: float = maxf(0.0, sin(t * 2.05))
	var settle: float = clampf(elapsed_s / 2.4, 0.0, 1.0)
	var shoulder: float = lerpf(16.0, 26.0, settle) - 2.0 * pulse
	var elbow: float = lerpf(-42.0, -60.0, settle) - 3.0 * pulse
	var wrist: float = lerpf(6.0, 15.0, settle) + 3.5 * pulse
	return _target(0.0, shoulder, elbow, wrist, 0.0, 9.0 + 3.0 * pulse, 0.86)


static func _attentive_follow(t: float) -> PackedFloat32Array:
	var nod: float = sin(t * 0.76)
	var micro_sway: float = sin(t * 0.38)
	return _target(1.4 * micro_sway, 28.0 + 1.4 * nod, -70.0 - 2.0 * nod, 19.0 + 1.2 * nod, 0.0, 2.0 + 1.6 * nod, 1.0)


static func _searching_glance_slow(t: float) -> PackedFloat32Array:
	var scan: float = sin(t * 0.36)
	var lift: float = sin(t * 0.52 + 0.4)
	return _target(34.0 * scan, 31.0 + 5.0 * lift, -82.0 - 6.0 * lift, 22.0 + 4.0 * lift, -20.0 * scan, -7.0 + 3.0 * lift, 0.24)


static func _curious_tilt(t: float) -> PackedFloat32Array:
	var anticipation: float = maxf(0.0, sin(t * 0.92))
	var tiny_bounce: float = 0.8 * sin(t * 1.10)
	return _target(6.0 * sin(t * 0.38), 29.0 - tiny_bounce, -84.0 - 3.0 * tiny_bounce, 27.0 + 2.0 * anticipation, 16.0, -18.0 + 3.0 * sin(t * 0.84), 0.62)


static func _gentle_wave(t: float) -> PackedFloat32Array:
	var wave: float = sin(t * 0.92)
	return _target(10.0 * wave, 27.0, -76.0, 24.0, 24.0 * wave, -12.0 + 2.0 * sin(t * 0.72), 0.46)


static func _gesture_approach(t: float) -> PackedFloat32Array:
	var eagerness: float = maxf(0.0, sin(t * 1.25))
	return _target(2.0 * sin(t * 0.42), 18.0 - 2.0 * eagerness, -50.0 - 4.0 * eagerness, 10.0 + 4.0 * eagerness, 0.0, 8.0 + 2.0 * eagerness, 0.92)


static func _gesture_retreat(t: float) -> PackedFloat32Array:
	var recoil: float = maxf(0.0, sin(t * 0.95))
	return _target(-8.0 * sin(t * 0.34), 48.0 + 5.0 * recoil, -112.0 - 5.0 * recoil, 30.0, -10.0, -18.0 - 3.0 * recoil, 0.18)


static func _scanning(t: float) -> PackedFloat32Array:
	return _target(42.0 * sin(t * 0.32), 30.0 + 3.5 * sin(t * 0.50), -86.0, 24.0, -26.0 * sin(t * 0.58), -8.0, 0.14)


static func _thinking_slow(t: float) -> PackedFloat32Array:
	return _target(-2.0 + 2.0 * sin(t * 0.34), 31.0, -86.0, 32.0 + 2.0 * sin(t * 0.82), -8.0 + 2.0 * sin(t * 0.44), -17.0 + 2.0 * sin(t * 0.70), 0.36)


static func _happy_bounce(t: float) -> PackedFloat32Array:
	var bounce: float = maxf(0.0, sin(t * 1.42))
	var sway: float = sin(t * 0.68)
	return _target(12.0 * sway, 21.0 - 4.0 * bounce, -58.0 - 5.0 * bounce, 14.0 + 5.0 * bounce, 9.0 * sway, 8.0 + 3.2 * bounce, 0.70)


static func _dance_loop(t: float) -> PackedFloat32Array:
	var groove: float = sin(t * 0.95)
	var counter: float = sin(t * 0.95 + PI * 0.5)
	return _target(24.0 * groove, 24.0 - 4.0 * maxf(0.0, counter), -64.0 - 6.0 * maxf(0.0, -counter), 18.0 + 5.0 * maxf(0.0, groove), 26.0 * counter, 2.0 + 4.0 * groove, 0.40)


static func _upset_turn(t: float) -> PackedFloat32Array:
	var droop: float = 1.4 * sin(t * 0.30)
	return _target(118.0, 42.0 + droop, -98.0, 28.0, -28.0, -35.0 + droop, 0.0)
