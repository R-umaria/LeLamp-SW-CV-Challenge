extends Node3D

const DEG: float = PI / 180.0
const RAD_TO_DEG: float = 180.0 / PI
const JOINT_COUNT: int = 6
const JOINT_BASE_YAW: int = 0
const JOINT_SHOULDER_PITCH: int = 1
const JOINT_ELBOW_PITCH: int = 2
const JOINT_WRIST_PITCH: int = 3
const JOINT_WRIST_YAW: int = 4
const JOINT_HEAD_TILT: int = 5
const MotionSkillLibrary = preload("res://scripts/lumos/MotionSkillLibrary.gd")
const LightSkillLibrary = preload("res://scripts/lumos/LightSkillLibrary.gd")

@export var face_follow_max_degrees: float = 82.0
@export var face_follow_vertical_max_degrees: float = 46.0
@export var face_follow_smooth_speed: float = 1.85
@export var face_follow_max_slew_degrees_per_second: float = 78.0
@export var motion_time_scale: float = 0.48
@export var smooth_speed_scale: float = 0.72
@export var camera_approach_offset: Vector3 = Vector3(0.24, 0.0, -0.26)
@export var root_shift_smooth_speed: float = 1.55
@export var distance_follow_enabled: bool = true
@export var desired_face_area_ratio: float = 0.070
@export var close_face_area_ratio: float = 0.150
@export var distance_follow_max_forward: float = 0.70
@export var distance_follow_max_back: float = 0.72

# Physical-style embodiment smoothing. Python still sends bounded behavior names;
# Godot turns those behavior targets into actuator-safe motion profiles.
@export var physical_motion_enabled: bool = true
@export var transition_blend_seconds: float = 0.95
@export var min_transition_blend_seconds: float = 0.45
@export var max_transition_blend_seconds: float = 2.40
@export var servo_response_gain: float = 4.2
@export var servo_max_velocity_degrees_per_second: float = 82.0
@export var servo_max_acceleration_degrees_per_second2: float = 145.0
@export var servo_max_jerk_degrees_per_second3: float = 360.0
@export var servo_snap_error_degrees: float = 0.05
@export var root_response_gain: float = 2.6
@export var root_max_speed_units_per_second: float = 0.36
@export var root_max_acceleration_units_per_second2: float = 0.72
@export var root_max_jerk_units_per_second3: float = 1.45
@export var smooth_light_enabled: bool = true
@export var light_color_slew_per_second: float = 1.90
@export var light_energy_slew_per_second: float = 1.35

@onready var base_yaw: Node3D = $BaseYaw_DOF1
@onready var shoulder_pitch: Node3D = $BaseYaw_DOF1/ShoulderPitch_DOF2
@onready var elbow_pitch: Node3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3
@onready var wrist_pitch: Node3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4
@onready var wrist_yaw: Node3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4/WristYaw_DOF5
@onready var lamp_head_tilt: Node3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4/WristYaw_DOF5/LampHeadTilt_DOF6
@onready var spot_light: SpotLight3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4/WristYaw_DOF5/LampHeadTilt_DOF6/LampSpotLight
@onready var head_mesh: MeshInstance3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4/WristYaw_DOF5/LampHeadTilt_DOF6/LampHead
@onready var shade_mesh: MeshInstance3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4/WristYaw_DOF5/LampHeadTilt_DOF6/LampShade
@onready var cone_mesh: MeshInstance3D = $BaseYaw_DOF1/ShoulderPitch_DOF2/ElbowPitch_DOF3/WristPitch_DOF4/WristYaw_DOF5/LampHeadTilt_DOF6/VisibleLightCone

var head_material: StandardMaterial3D
var cone_material: StandardMaterial3D

var current_state: String = "idle"
var current_motion: String = "idle_breathe"
var current_light: String = "dim_warm"
var current_sound: String = "none"
var current_speech_text: String = ""
var engagement_status: String = "absent"
var engagement_confidence: float = 0.0
var face_x_norm: float = -1.0
var face_y_norm: float = -1.0
var face_area_ratio: float = 0.0
var gesture_status: String = "none"
var gesture_confidence: float = 0.0
var recall_target_found: bool = false
var recall_point_x_norm: float = -1.0
var recall_point_y_norm: float = -1.0
var recall_location_label: String = ""

var _time: float = 0.0
var _motion_started_at: float = 0.0
var _smoothed_face_follow_x_deg: float = 0.0
var _smoothed_face_follow_y_deg: float = 0.0
var _home_position: Vector3 = Vector3.ZERO
var _target_root_position: Vector3 = Vector3.ZERO
var _smoothed_distance_shift: float = 0.0
var _target: Dictionary = {
	"base_yaw": 0.0,
	"shoulder_pitch": -18.0 * DEG,
	"elbow_pitch": -70.0 * DEG,
	"wrist_pitch": 18.0 * DEG,
	"wrist_yaw": 0.0,
	"head_tilt": 0.0,
}

var _joint_position_rad: PackedFloat32Array = PackedFloat32Array()
var _joint_velocity_rad_s: PackedFloat32Array = PackedFloat32Array()
var _joint_acceleration_rad_s2: PackedFloat32Array = PackedFloat32Array()
var _transition_from_degrees: PackedFloat32Array = PackedFloat32Array()
var _transition_started_at: float = -1000.0
var _transition_duration_s: float = 0.0
var _servo_initialized: bool = false
var _root_velocity: Vector3 = Vector3.ZERO
var _root_acceleration: Vector3 = Vector3.ZERO
var _smoothed_light_color: Color = Color(1.0, 0.86, 0.48)
var _smoothed_light_energy: float = 0.30
var _light_initialized: bool = false


func _ready() -> void:
	_home_position = position
	_target_root_position = _home_position
	_bind_runtime_materials()
	_init_motion_profile_state()
	apply_command({
		"state": "idle",
		"engagement": {"status": "absent", "confidence": 0.0},
		"gesture": {"status": "none", "confidence": 0.0},
		"behavior": {"motion": "idle_breathe", "light": "dim_warm", "sound": null, "speech_text": null},
		"memory": {"last_detected_objects": []},
	})


func _bind_runtime_materials() -> void:
	# Scene geometry is editor-authored in LampRig.tscn. Runtime code only binds
	# mutable material instances for behavior-driven light and emission changes.
	head_material = _make_material(Color(1.0, 0.86, 0.48), true)
	cone_material = _make_transparent_material(Color(1.0, 0.82, 0.30, 0.18))

	head_mesh.material_override = head_material
	shade_mesh.material_override = head_material
	cone_mesh.material_override = cone_material
	spot_light.light_color = head_material.albedo_color


func _init_motion_profile_state() -> void:
	_joint_position_rad = PackedFloat32Array([
		base_yaw.rotation.y,
		shoulder_pitch.rotation.x,
		elbow_pitch.rotation.x,
		wrist_pitch.rotation.x,
		wrist_yaw.rotation.y,
		lamp_head_tilt.rotation.x,
	])
	_joint_velocity_rad_s = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	_joint_acceleration_rad_s2 = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	_transition_from_degrees = _current_joint_degrees_array()
	_root_velocity = Vector3.ZERO
	_root_acceleration = Vector3.ZERO
	_servo_initialized = true


func apply_command(command: Dictionary) -> void:
	current_state = str(command.get("state", current_state))

	var behavior: Dictionary = _dictionary_value(command, "behavior")
	var next_motion: String = MotionSkillLibrary.normalize_motion(str(behavior.get("motion", current_motion)))
	if next_motion != current_motion:
		_begin_motion_transition(next_motion)
	current_motion = next_motion
	current_light = LightSkillLibrary.normalize_light(str(behavior.get("light", current_light)))
	current_sound = str(behavior.get("sound", "none"))
	var speech_value: Variant = behavior.get("speech_text", "")
	current_speech_text = "" if speech_value == null else str(speech_value)

	var engagement: Dictionary = _dictionary_value(command, "engagement")
	engagement_status = str(engagement.get("status", engagement_status))
	engagement_confidence = float(engagement.get("confidence", engagement_confidence))
	face_x_norm = _optional_norm_float(engagement, "face_x_norm", -1.0)
	face_y_norm = _optional_norm_float(engagement, "face_y_norm", -1.0)
	face_area_ratio = maxf(0.0, float(engagement.get("face_area_ratio", 0.0)))

	var gesture: Dictionary = _dictionary_value(command, "gesture")
	gesture_status = str(gesture.get("status", gesture_status))
	gesture_confidence = float(gesture.get("confidence", gesture_confidence))

	var memory: Dictionary = _dictionary_value(command, "memory")
	var recall_target: Dictionary = _dictionary_value(memory, "recall_target")
	recall_target_found = bool(recall_target.get("found", false))
	recall_point_x_norm = _optional_norm_float(recall_target, "point_x_norm", -1.0)
	recall_point_y_norm = _optional_norm_float(recall_target, "point_y_norm", -1.0)
	recall_location_label = str(recall_target.get("location_label", ""))


func _process(delta: float) -> void:
	_time += delta
	_update_face_follow(delta)
	_update_motion_targets()
	_update_root_shift(delta)
	_smooth_to_targets(delta)
	_update_light(delta)


func _begin_motion_transition(next_motion: String) -> void:
	if not _servo_initialized:
		return

	_transition_from_degrees = _current_joint_degrees_array()
	_transition_started_at = _time

	var preview_target: PackedFloat32Array = MotionSkillLibrary.target_for(next_motion, _time * motion_time_scale, 0.0)
	var max_delta_deg: float = 0.0
	var preview_limit: int = mini(preview_target.size(), JOINT_COUNT)
	for joint_index: int in range(preview_limit):
		max_delta_deg = maxf(max_delta_deg, absf(preview_target[joint_index] - _transition_from_degrees[joint_index]))

	var velocity_based_time: float = max_delta_deg / maxf(servo_max_velocity_degrees_per_second * 0.72, 1.0)
	_transition_duration_s = clampf(
		maxf(transition_blend_seconds, min_transition_blend_seconds + velocity_based_time),
		min_transition_blend_seconds,
		max_transition_blend_seconds
	)
	_motion_started_at = _time
	print("Lumos smooth transition: %s -> %s over %.2fs" % [current_motion, next_motion, _transition_duration_s])


func _update_face_follow(delta: float) -> void:
	var source_x: float = face_x_norm
	var source_y: float = face_y_norm
	if current_motion == "recall_point" and recall_target_found:
		source_x = recall_point_x_norm
		source_y = recall_point_y_norm

	var desired_x_deg: float = 0.0
	var desired_y_deg: float = 0.0
	if not MotionSkillLibrary.blocks_face_follow(current_motion):
		if source_x >= 0.0 and source_x <= 1.0:
			desired_x_deg = clampf((0.5 - source_x) * face_follow_max_degrees * 2.0, -face_follow_max_degrees, face_follow_max_degrees)
		if source_y >= 0.0 and source_y <= 1.0:
			desired_y_deg = clampf((0.5 - source_y) * face_follow_vertical_max_degrees * 2.0, -face_follow_vertical_max_degrees, face_follow_vertical_max_degrees)

	var lowpass_weight: float = clampf(delta * face_follow_smooth_speed, 0.0, 1.0)
	var filtered_x_deg: float = lerpf(_smoothed_face_follow_x_deg, desired_x_deg, lowpass_weight)
	var filtered_y_deg: float = lerpf(_smoothed_face_follow_y_deg, desired_y_deg, lowpass_weight)
	var slew_step_deg: float = maxf(0.0, face_follow_max_slew_degrees_per_second) * delta
	_smoothed_face_follow_x_deg = move_toward(_smoothed_face_follow_x_deg, filtered_x_deg, slew_step_deg)
	_smoothed_face_follow_y_deg = move_toward(_smoothed_face_follow_y_deg, filtered_y_deg, slew_step_deg)


func _update_motion_targets() -> void:
	var scaled_time: float = _time * motion_time_scale
	var motion_elapsed_s: float = maxf(0.0, _time - _motion_started_at)
	var target_degrees: PackedFloat32Array = MotionSkillLibrary.target_for(current_motion, scaled_time, motion_elapsed_s)
	_set_target_from_degrees(target_degrees)


func _update_root_shift(delta: float) -> void:
	var scaled_time: float = _time * motion_time_scale
	var motion_elapsed_s: float = maxf(0.0, _time - _motion_started_at)
	var behavior_shift: float = MotionSkillLibrary.root_shift_for(current_motion, scaled_time, motion_elapsed_s)
	var distance_shift: float = _distance_shift_for_face(delta)
	var shift: float = clampf(behavior_shift + distance_shift, -1.0, 1.0)
	_target_root_position = _home_position + camera_approach_offset * shift
	if physical_motion_enabled:
		position = _smooth_vector_profile(position, _target_root_position, delta)
	else:
		var weight: float = clampf(delta * root_shift_smooth_speed, 0.0, 1.0)
		position = position.lerp(_target_root_position, weight)
		_root_velocity = Vector3.ZERO
		_root_acceleration = Vector3.ZERO


func _distance_shift_for_face(delta: float) -> float:
	# Keep a comfortable apparent distance during attentive tracking. A small face
	# box means the user is far away, so Lumos slides forward. A very large face
	# box means the user is close, so Lumos gives them space. Gesture and recall
	# motions own root movement and therefore suppress this automatic correction.
	var desired_shift: float = 0.0
	if distance_follow_enabled and _can_use_distance_follow():
		if face_area_ratio < desired_face_area_ratio:
			var far_error: float = (desired_face_area_ratio - face_area_ratio) / maxf(desired_face_area_ratio, 0.001)
			desired_shift = clampf(far_error, 0.0, distance_follow_max_forward)
		elif face_area_ratio > close_face_area_ratio:
			var close_error: float = (face_area_ratio - close_face_area_ratio) / maxf(close_face_area_ratio, 0.001)
			desired_shift = -clampf(close_error, 0.0, distance_follow_max_back)

	var weight: float = clampf(delta * root_shift_smooth_speed * 0.70, 0.0, 1.0)
	_smoothed_distance_shift = lerpf(_smoothed_distance_shift, desired_shift, weight)
	return _smoothed_distance_shift


func _can_use_distance_follow() -> bool:
	if face_area_ratio <= 0.0:
		return false
	if engagement_status == "absent":
		return false
	if current_motion in ["gesture_approach", "gesture_retreat", "recall_point", "recall_not_found", "sleep_rest", "sleepy_search_then_rest"]:
		return false
	return true


func _set_target_from_degrees(target_degrees: PackedFloat32Array) -> void:
	if target_degrees.size() < MotionSkillLibrary.TARGET_SIZE:
		return
	var follow_weight: float = clampf(target_degrees[6], 0.0, 1.0)
	var follow_x_deg: float = _smoothed_face_follow_x_deg * follow_weight
	var follow_y_deg: float = _smoothed_face_follow_y_deg * follow_weight

	var base_yaw_deg: float = clampf(target_degrees[0] + follow_x_deg, -155.0, 155.0)
	var shoulder_pitch_deg: float = clampf(target_degrees[1] - follow_y_deg * 0.28, -100.0, 100.0)
	var elbow_pitch_deg: float = clampf(target_degrees[2], -150.0, 115.0)
	var wrist_pitch_deg: float = clampf(target_degrees[3] - follow_y_deg * 0.34, -85.0, 85.0)
	var wrist_yaw_deg: float = clampf(target_degrees[4] + follow_x_deg * 0.46, -95.0, 95.0)
	var head_tilt_deg: float = clampf(target_degrees[5] + follow_y_deg * 0.88, -62.0, 46.0)
	var degrees: PackedFloat32Array = PackedFloat32Array([
		base_yaw_deg,
		shoulder_pitch_deg,
		elbow_pitch_deg,
		wrist_pitch_deg,
		wrist_yaw_deg,
		head_tilt_deg,
	])

	if _is_transition_active():
		var blend_weight: float = _active_transition_weight()
		for joint_index: int in range(JOINT_COUNT):
			degrees[joint_index] = lerpf(_transition_from_degrees[joint_index], degrees[joint_index], blend_weight)

	_target["base_yaw"] = degrees[JOINT_BASE_YAW] * DEG
	_target["shoulder_pitch"] = degrees[JOINT_SHOULDER_PITCH] * DEG
	_target["elbow_pitch"] = degrees[JOINT_ELBOW_PITCH] * DEG
	_target["wrist_pitch"] = degrees[JOINT_WRIST_PITCH] * DEG
	_target["wrist_yaw"] = degrees[JOINT_WRIST_YAW] * DEG
	_target["head_tilt"] = degrees[JOINT_HEAD_TILT] * DEG


func _smooth_to_targets(delta: float) -> void:
	if not physical_motion_enabled:
		var speed: float = MotionSkillLibrary.smooth_speed_for(current_motion) * smooth_speed_scale
		var weight: float = clampf(delta * speed, 0.0, 1.0)
		base_yaw.rotation.y = lerp_angle(base_yaw.rotation.y, float(_target["base_yaw"]), weight)
		shoulder_pitch.rotation.x = lerp_angle(shoulder_pitch.rotation.x, float(_target["shoulder_pitch"]), weight)
		elbow_pitch.rotation.x = lerp_angle(elbow_pitch.rotation.x, float(_target["elbow_pitch"]), weight)
		wrist_pitch.rotation.x = lerp_angle(wrist_pitch.rotation.x, float(_target["wrist_pitch"]), weight)
		wrist_yaw.rotation.y = lerp_angle(wrist_yaw.rotation.y, float(_target["wrist_yaw"]), weight)
		lamp_head_tilt.rotation.x = lerp_angle(lamp_head_tilt.rotation.x, float(_target["head_tilt"]), weight)
		_sync_servo_state_from_nodes()
		return

	var max_velocity_rad_s: float = servo_max_velocity_degrees_per_second * DEG
	var max_acceleration_rad_s2: float = servo_max_acceleration_degrees_per_second2 * DEG
	var max_jerk_rad_s3: float = servo_max_jerk_degrees_per_second3 * DEG
	var snap_error_rad: float = servo_snap_error_degrees * DEG
	var target_rad: PackedFloat32Array = PackedFloat32Array([
		float(_target["base_yaw"]),
		float(_target["shoulder_pitch"]),
		float(_target["elbow_pitch"]),
		float(_target["wrist_pitch"]),
		float(_target["wrist_yaw"]),
		float(_target["head_tilt"]),
	])

	for joint_index: int in range(JOINT_COUNT):
		var current_rad: float = _joint_position_rad[joint_index]
		var error_rad: float = _shortest_angle_error(target_rad[joint_index], current_rad)
		var desired_velocity_rad_s: float = clampf(error_rad * servo_response_gain, -max_velocity_rad_s, max_velocity_rad_s)
		var desired_acceleration_rad_s2: float = clampf(
			(desired_velocity_rad_s - _joint_velocity_rad_s[joint_index]) / maxf(delta, 0.001),
			-max_acceleration_rad_s2,
			max_acceleration_rad_s2
		)
		var next_acceleration_rad_s2: float = move_toward(_joint_acceleration_rad_s2[joint_index], desired_acceleration_rad_s2, max_jerk_rad_s3 * delta)
		var next_velocity_rad_s: float = clampf(_joint_velocity_rad_s[joint_index] + next_acceleration_rad_s2 * delta, -max_velocity_rad_s, max_velocity_rad_s)
		var step_rad: float = next_velocity_rad_s * delta
		var next_position_rad: float = current_rad + step_rad
		var next_error_rad: float = _shortest_angle_error(target_rad[joint_index], next_position_rad)

		if absf(error_rad) <= snap_error_rad and absf(next_velocity_rad_s) < max_velocity_rad_s * 0.04:
			next_position_rad = target_rad[joint_index]
			next_velocity_rad_s = 0.0
			next_acceleration_rad_s2 = 0.0
		elif error_rad * next_error_rad < 0.0 and absf(error_rad) < max_velocity_rad_s * delta * 1.5:
			next_position_rad = target_rad[joint_index]
			next_velocity_rad_s = 0.0
			next_acceleration_rad_s2 = 0.0

		_joint_position_rad[joint_index] = next_position_rad
		_joint_velocity_rad_s[joint_index] = next_velocity_rad_s
		_joint_acceleration_rad_s2[joint_index] = next_acceleration_rad_s2

	_apply_servo_state_to_nodes()


func _smooth_vector_profile(current: Vector3, target: Vector3, delta: float) -> Vector3:
	var error: Vector3 = target - current
	var desired_velocity: Vector3 = _clamp_vector_length(error * root_response_gain, root_max_speed_units_per_second)
	var desired_acceleration: Vector3 = _clamp_vector_length(
		(desired_velocity - _root_velocity) / maxf(delta, 0.001),
		root_max_acceleration_units_per_second2
	)
	_root_acceleration = _move_vector_toward(_root_acceleration, desired_acceleration, root_max_jerk_units_per_second3 * delta)
	_root_velocity = _clamp_vector_length(_root_velocity + _root_acceleration * delta, root_max_speed_units_per_second)

	var step: Vector3 = _root_velocity * delta
	if error.length() <= 0.002 and _root_velocity.length() <= 0.010:
		_root_velocity = Vector3.ZERO
		_root_acceleration = Vector3.ZERO
		return target
	if step.length() > error.length() and error.length() < 0.050:
		_root_velocity = Vector3.ZERO
		_root_acceleration = Vector3.ZERO
		return target
	return current + step


func _update_light(delta: float) -> void:
	var scaled_time: float = _time * motion_time_scale
	var light_values: PackedFloat32Array = LightSkillLibrary.values_for(current_light, scaled_time)
	if light_values.size() < 4:
		return
	var target_color: Color = Color(light_values[0], light_values[1], light_values[2])
	var target_energy: float = light_values[3]

	if not _light_initialized:
		_smoothed_light_color = target_color
		_smoothed_light_energy = target_energy
		_light_initialized = true
	elif smooth_light_enabled:
		var color_step: float = maxf(0.0, light_color_slew_per_second) * delta
		_smoothed_light_color = Color(
			move_toward(_smoothed_light_color.r, target_color.r, color_step),
			move_toward(_smoothed_light_color.g, target_color.g, color_step),
			move_toward(_smoothed_light_color.b, target_color.b, color_step),
			1.0
		)
		_smoothed_light_energy = move_toward(_smoothed_light_energy, target_energy, maxf(0.0, light_energy_slew_per_second) * delta)
	else:
		_smoothed_light_color = target_color
		_smoothed_light_energy = target_energy

	if spot_light != null:
		spot_light.light_energy = _smoothed_light_energy
		spot_light.light_color = _smoothed_light_color

	if head_material != null:
		head_material.albedo_color = _smoothed_light_color
		head_material.emission = _smoothed_light_color
		head_material.emission_energy_multiplier = _smoothed_light_energy * 0.55

	if cone_material != null:
		var alpha: float = clampf(0.075 + _smoothed_light_energy * 0.065, 0.07, 0.24)
		cone_material.albedo_color = Color(_smoothed_light_color.r, _smoothed_light_color.g, _smoothed_light_color.b, alpha)


func _is_transition_active() -> bool:
	if _transition_duration_s <= 0.0:
		return false
	return _time - _transition_started_at < _transition_duration_s


func _active_transition_weight() -> float:
	var progress: float = clampf((_time - _transition_started_at) / maxf(_transition_duration_s, 0.001), 0.0, 1.0)
	return _s_curve(progress)


func _s_curve(x: float) -> float:
	# Quintic smoothstep: zero first and second derivative at both endpoints.
	var p: float = clampf(x, 0.0, 1.0)
	return p * p * p * (p * (p * 6.0 - 15.0) + 10.0)


func _current_joint_degrees_array() -> PackedFloat32Array:
	if _servo_initialized and _joint_position_rad.size() >= JOINT_COUNT:
		return PackedFloat32Array([
			_joint_position_rad[JOINT_BASE_YAW] * RAD_TO_DEG,
			_joint_position_rad[JOINT_SHOULDER_PITCH] * RAD_TO_DEG,
			_joint_position_rad[JOINT_ELBOW_PITCH] * RAD_TO_DEG,
			_joint_position_rad[JOINT_WRIST_PITCH] * RAD_TO_DEG,
			_joint_position_rad[JOINT_WRIST_YAW] * RAD_TO_DEG,
			_joint_position_rad[JOINT_HEAD_TILT] * RAD_TO_DEG,
		])
	return PackedFloat32Array([
		base_yaw.rotation.y * RAD_TO_DEG,
		shoulder_pitch.rotation.x * RAD_TO_DEG,
		elbow_pitch.rotation.x * RAD_TO_DEG,
		wrist_pitch.rotation.x * RAD_TO_DEG,
		wrist_yaw.rotation.y * RAD_TO_DEG,
		lamp_head_tilt.rotation.x * RAD_TO_DEG,
	])


func _sync_servo_state_from_nodes() -> void:
	_joint_position_rad = PackedFloat32Array([
		base_yaw.rotation.y,
		shoulder_pitch.rotation.x,
		elbow_pitch.rotation.x,
		wrist_pitch.rotation.x,
		wrist_yaw.rotation.y,
		lamp_head_tilt.rotation.x,
	])
	_joint_velocity_rad_s = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	_joint_acceleration_rad_s2 = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


func _apply_servo_state_to_nodes() -> void:
	base_yaw.rotation.y = _joint_position_rad[JOINT_BASE_YAW]
	shoulder_pitch.rotation.x = _joint_position_rad[JOINT_SHOULDER_PITCH]
	elbow_pitch.rotation.x = _joint_position_rad[JOINT_ELBOW_PITCH]
	wrist_pitch.rotation.x = _joint_position_rad[JOINT_WRIST_PITCH]
	wrist_yaw.rotation.y = _joint_position_rad[JOINT_WRIST_YAW]
	lamp_head_tilt.rotation.x = _joint_position_rad[JOINT_HEAD_TILT]


func _shortest_angle_error(target_rad: float, current_rad: float) -> float:
	return wrapf(target_rad - current_rad, -PI, PI)


func _clamp_vector_length(value: Vector3, max_length: float) -> Vector3:
	var safe_max: float = maxf(0.0, max_length)
	var length_value: float = value.length()
	if length_value > safe_max and length_value > 0.0001:
		return value.normalized() * safe_max
	return value


func _move_vector_toward(current: Vector3, target: Vector3, max_delta: float) -> Vector3:
	var delta_vec: Vector3 = target - current
	var distance: float = delta_vec.length()
	if distance <= max_delta or distance <= 0.0001:
		return target
	return current + delta_vec.normalized() * max_delta


func _optional_norm_float(source: Dictionary, key: String, default_value: float) -> float:
	var value: Variant = source.get(key, null)
	var value_type: int = typeof(value)
	if value_type == TYPE_FLOAT or value_type == TYPE_INT:
		return clampf(float(value), 0.0, 1.0)
	return default_value


func _dictionary_value(source: Dictionary, key: String) -> Dictionary:
	var value: Variant = source.get(key, {})
	if typeof(value) == TYPE_DICTIONARY:
		return value as Dictionary
	return {}


func _make_material(color: Color, emissive: bool) -> StandardMaterial3D:
	var mat: StandardMaterial3D = StandardMaterial3D.new()
	mat.albedo_color = color
	if emissive:
		mat.emission_enabled = true
		mat.emission = color
		mat.emission_energy_multiplier = 0.6
	return mat


func _make_transparent_material(color: Color) -> StandardMaterial3D:
	var mat: StandardMaterial3D = StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = color
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return mat
