extends Node3D

const DEG: float = PI / 180.0
const MotionSkillLibrary = preload("res://scripts/lumos/MotionSkillLibrary.gd")
const LightSkillLibrary = preload("res://scripts/lumos/LightSkillLibrary.gd")

@export var face_follow_max_degrees: float = 82.0
@export var face_follow_vertical_max_degrees: float = 46.0
@export var face_follow_smooth_speed: float = 1.85
@export var motion_time_scale: float = 0.48
@export var smooth_speed_scale: float = 0.72
@export var camera_approach_offset: Vector3 = Vector3(0.24, 0.0, -0.26)
@export var root_shift_smooth_speed: float = 1.55
@export var distance_follow_enabled: bool = true
@export var desired_face_area_ratio: float = 0.070
@export var close_face_area_ratio: float = 0.150
@export var distance_follow_max_forward: float = 0.70
@export var distance_follow_max_back: float = 0.72

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


func _ready() -> void:
	_home_position = position
	_target_root_position = _home_position
	_bind_runtime_materials()
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


func apply_command(command: Dictionary) -> void:
	current_state = str(command.get("state", current_state))

	var behavior: Dictionary = _dictionary_value(command, "behavior")
	var next_motion: String = MotionSkillLibrary.normalize_motion(str(behavior.get("motion", current_motion)))
	if next_motion != current_motion:
		_motion_started_at = _time
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
	_update_light()


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

	var weight: float = clampf(delta * face_follow_smooth_speed, 0.0, 1.0)
	_smoothed_face_follow_x_deg = lerpf(_smoothed_face_follow_x_deg, desired_x_deg, weight)
	_smoothed_face_follow_y_deg = lerpf(_smoothed_face_follow_y_deg, desired_y_deg, weight)


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
	var weight: float = clampf(delta * root_shift_smooth_speed, 0.0, 1.0)
	position = position.lerp(_target_root_position, weight)


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

	_target["base_yaw"] = clampf(target_degrees[0] + follow_x_deg, -155.0, 155.0) * DEG
	_target["shoulder_pitch"] = clampf(target_degrees[1] - follow_y_deg * 0.28, -100.0, 100.0) * DEG
	_target["elbow_pitch"] = clampf(target_degrees[2], -150.0, 115.0) * DEG
	_target["wrist_pitch"] = clampf(target_degrees[3] - follow_y_deg * 0.34, -85.0, 85.0) * DEG
	_target["wrist_yaw"] = clampf(target_degrees[4] + follow_x_deg * 0.46, -95.0, 95.0) * DEG
	_target["head_tilt"] = clampf(target_degrees[5] + follow_y_deg * 0.88, -62.0, 46.0) * DEG


func _smooth_to_targets(delta: float) -> void:
	var speed: float = MotionSkillLibrary.smooth_speed_for(current_motion) * smooth_speed_scale
	var weight: float = clampf(delta * speed, 0.0, 1.0)
	base_yaw.rotation.y = lerp_angle(base_yaw.rotation.y, float(_target["base_yaw"]), weight)
	shoulder_pitch.rotation.x = lerp_angle(shoulder_pitch.rotation.x, float(_target["shoulder_pitch"]), weight)
	elbow_pitch.rotation.x = lerp_angle(elbow_pitch.rotation.x, float(_target["elbow_pitch"]), weight)
	wrist_pitch.rotation.x = lerp_angle(wrist_pitch.rotation.x, float(_target["wrist_pitch"]), weight)
	wrist_yaw.rotation.y = lerp_angle(wrist_yaw.rotation.y, float(_target["wrist_yaw"]), weight)
	lamp_head_tilt.rotation.x = lerp_angle(lamp_head_tilt.rotation.x, float(_target["head_tilt"]), weight)


func _update_light() -> void:
	var scaled_time: float = _time * motion_time_scale
	var light_values: PackedFloat32Array = LightSkillLibrary.values_for(current_light, scaled_time)
	if light_values.size() < 4:
		return
	var head_color: Color = Color(light_values[0], light_values[1], light_values[2])
	var energy: float = light_values[3]

	if spot_light != null:
		spot_light.light_energy = energy
		spot_light.light_color = head_color

	if head_material != null:
		head_material.albedo_color = head_color
		head_material.emission = head_color
		head_material.emission_energy_multiplier = energy * 0.55

	if cone_material != null:
		var alpha: float = clampf(0.075 + energy * 0.065, 0.07, 0.24)
		cone_material.albedo_color = Color(head_color.r, head_color.g, head_color.b, alpha)


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
