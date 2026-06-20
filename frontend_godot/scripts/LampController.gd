extends Node3D

const DEG: float = PI / 180.0
const MotionSkillLibrary = preload("res://scripts/lumos/MotionSkillLibrary.gd")
const LightSkillLibrary = preload("res://scripts/lumos/LightSkillLibrary.gd")

@export var face_follow_max_degrees: float = 46.0
@export var face_follow_smooth_speed: float = 2.10
@export var motion_time_scale: float = 0.58
@export var smooth_speed_scale: float = 0.84

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

var _time: float = 0.0
var _motion_started_at: float = 0.0
var _smoothed_face_follow_deg: float = 0.0
var _target: Dictionary = {
	"base_yaw": 0.0,
	"shoulder_pitch": -18.0 * DEG,
	"elbow_pitch": 36.0 * DEG,
	"wrist_pitch": -18.0 * DEG,
	"wrist_yaw": 0.0,
	"head_tilt": 0.0,
}


func _ready() -> void:
	_bind_runtime_materials()
	apply_command({
		"state": "idle",
		"engagement": {"status": "absent", "confidence": 0.0},
		"behavior": {"motion": "idle_breathe", "light": "dim_warm", "sound": null, "speech_text": null},
		"memory": {"last_detected_objects": []},
	})


func _bind_runtime_materials() -> void:
	# The geometry is editor-authored in LampRig.tscn. Runtime code only binds
	# mutable material instances for behavior-driven color/emission changes.
	head_material = _make_material(Color(1.0, 0.86, 0.48), true)
	cone_material = _make_transparent_material(Color(1.0, 0.82, 0.30, 0.18))

	head_mesh.material_override = head_material
	shade_mesh.material_override = head_material
	cone_mesh.material_override = cone_material
	spot_light.light_color = head_material.albedo_color


func apply_command(command: Dictionary) -> void:
	# This remains a bounded renderer. It maps backend command fields to known
	# skills and never performs perception, memory, or open-ended decision making.
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


func _process(delta: float) -> void:
	_time += delta
	_update_face_follow(delta)
	_update_motion_targets()
	_smooth_to_targets(delta)
	_update_light()


func _update_face_follow(delta: float) -> void:
	var desired_follow_deg: float = 0.0
	if face_x_norm >= 0.0 and face_x_norm <= 1.0 and not MotionSkillLibrary.blocks_face_follow(current_motion):
		# Camera-frame x=0 is user's left. Positive Godot yaw turns the lamp's
		# local -Z/front direction left from the viewer's perspective.
		desired_follow_deg = clampf((0.5 - face_x_norm) * face_follow_max_degrees * 2.0, -face_follow_max_degrees, face_follow_max_degrees)
	var weight: float = clampf(delta * face_follow_smooth_speed, 0.0, 1.0)
	_smoothed_face_follow_deg = lerpf(_smoothed_face_follow_deg, desired_follow_deg, weight)


func _update_motion_targets() -> void:
	var scaled_time: float = _time * motion_time_scale
	var motion_elapsed_s: float = maxf(0.0, _time - _motion_started_at)
	var target_degrees: PackedFloat32Array = MotionSkillLibrary.target_for(current_motion, scaled_time, motion_elapsed_s)
	_set_target_from_degrees(target_degrees)


func _set_target_from_degrees(target_degrees: PackedFloat32Array) -> void:
	if target_degrees.size() < MotionSkillLibrary.TARGET_SIZE:
		return
	var follow_weight: float = clampf(target_degrees[6], 0.0, 1.0)
	var follow_deg: float = _smoothed_face_follow_deg * follow_weight
	_target["base_yaw"] = clampf(target_degrees[0] + follow_deg, -135.0, 135.0) * DEG
	_target["shoulder_pitch"] = clampf(target_degrees[1], -95.0, 95.0) * DEG
	_target["elbow_pitch"] = clampf(target_degrees[2], -150.0, 115.0) * DEG
	_target["wrist_pitch"] = clampf(target_degrees[3], -80.0, 80.0) * DEG
	_target["wrist_yaw"] = clampf(target_degrees[4] + follow_deg * 0.42, -80.0, 80.0) * DEG
	_target["head_tilt"] = clampf(target_degrees[5], -55.0, 35.0) * DEG


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

	# Keep the physical SpotLight3D, emissive lamp head, shade, and visible cone synchronized.
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
