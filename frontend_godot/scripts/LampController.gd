extends Node3D

const DEG: float = PI / 180.0
const FACE_FOLLOW_MAX_DEG: float = 14.0
const FACE_FOLLOW_SMOOTH_SPEED: float = 4.0

var base_yaw: Node3D
var shoulder_pitch: Node3D
var elbow_pitch: Node3D
var wrist_pitch: Node3D
var wrist_yaw: Node3D
var lamp_head_tilt: Node3D
var spot_light: SpotLight3D
var head_material: StandardMaterial3D
var cone_material: StandardMaterial3D
var front_marker_material: StandardMaterial3D

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
	_build_lamp_rig()
	apply_command({
		"state": "idle",
		"engagement": {"status": "absent", "confidence": 0.0},
		"behavior": {"motion": "idle_breathe", "light": "dim_warm", "sound": null, "speech_text": null},
		"memory": {"last_detected_objects": []},
	})


func apply_command(command: Dictionary) -> void:
	# This is intentionally a bounded renderer. It reads backend command fields
	# and maps them to known animation targets; it does not infer engagement or
	# choose behavior policies on its own.
	current_state = str(command.get("state", current_state))

	var behavior: Dictionary = _dictionary_value(command, "behavior")
	current_motion = str(behavior.get("motion", current_motion))
	current_light = str(behavior.get("light", current_light))
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
	_update_light(delta)


func _build_lamp_rig() -> void:
	_clear_children()

	var base_material: StandardMaterial3D = _make_material(Color(0.34, 0.34, 0.36), false)
	var arm_material: StandardMaterial3D = _make_material(Color(0.72, 0.72, 0.76), false)
	head_material = _make_material(Color(1.0, 0.86, 0.48), true)
	cone_material = _make_transparent_material(Color(1.0, 0.82, 0.30, 0.18))
	front_marker_material = _make_material(Color(0.15, 0.45, 1.0), true)

	base_yaw = Node3D.new()
	base_yaw.name = "BaseYaw_DOF1"
	add_child(base_yaw)

	var base_mesh: MeshInstance3D = _cylinder_mesh("Base", 0.42, 0.18, base_material)
	base_mesh.position = Vector3(0.0, 0.09, 0.0)
	base_yaw.add_child(base_mesh)

	shoulder_pitch = Node3D.new()
	shoulder_pitch.name = "ShoulderPitch_DOF2"
	shoulder_pitch.position = Vector3(0.0, 0.24, 0.0)
	base_yaw.add_child(shoulder_pitch)

	var shoulder_joint: MeshInstance3D = _sphere_mesh("ShoulderJoint", 0.15, base_material)
	shoulder_pitch.add_child(shoulder_joint)

	var upper_arm: MeshInstance3D = _cylinder_mesh("UpperArm", 0.055, 1.15, arm_material)
	upper_arm.position = Vector3(0.0, 0.575, 0.0)
	shoulder_pitch.add_child(upper_arm)

	elbow_pitch = Node3D.new()
	elbow_pitch.name = "ElbowPitch_DOF3"
	elbow_pitch.position = Vector3(0.0, 1.15, 0.0)
	shoulder_pitch.add_child(elbow_pitch)

	var elbow_joint: MeshInstance3D = _sphere_mesh("ElbowJoint", 0.13, base_material)
	elbow_pitch.add_child(elbow_joint)

	var lower_arm: MeshInstance3D = _cylinder_mesh("LowerArm", 0.05, 0.90, arm_material)
	lower_arm.position = Vector3(0.0, 0.45, 0.0)
	elbow_pitch.add_child(lower_arm)

	wrist_pitch = Node3D.new()
	wrist_pitch.name = "WristPitch_DOF4"
	wrist_pitch.position = Vector3(0.0, 0.90, 0.0)
	elbow_pitch.add_child(wrist_pitch)

	var wrist_pitch_joint: MeshInstance3D = _sphere_mesh("WristPitchJoint", 0.105, base_material)
	wrist_pitch.add_child(wrist_pitch_joint)

	wrist_yaw = Node3D.new()
	wrist_yaw.name = "WristYaw_DOF5"
	wrist_pitch.add_child(wrist_yaw)

	lamp_head_tilt = Node3D.new()
	lamp_head_tilt.name = "LampHeadTilt_DOF6"
	lamp_head_tilt.position = Vector3(0.0, 0.16, -0.08)
	wrist_yaw.add_child(lamp_head_tilt)

	var head: MeshInstance3D = _sphere_mesh("LampHead", 0.22, head_material)
	head.scale = Vector3(1.15, 0.82, 1.0)
	lamp_head_tilt.add_child(head)

	var shade: MeshInstance3D = _cylinder_mesh("LampShade", 0.22, 0.18, head_material)
	shade.rotation_degrees = Vector3(90.0, 0.0, 0.0)
	shade.position = Vector3(0.0, -0.02, -0.16)
	lamp_head_tilt.add_child(shade)

	spot_light = SpotLight3D.new()
	spot_light.name = "LampSpotLight"
	spot_light.position = Vector3(0.0, -0.02, -0.25)
	spot_light.rotation_degrees = Vector3(-90.0, 0.0, 0.0)
	spot_light.spot_range = 4.0
	spot_light.spot_angle = 28.0
	spot_light.light_energy = 1.4
	lamp_head_tilt.add_child(spot_light)

	var cone: MeshInstance3D = _cone_mesh("VisibleLightCone", 0.24, 1.1, cone_material)
	cone.position = Vector3(0.0, -0.02, -0.62)
	cone.rotation_degrees = Vector3(90.0, 0.0, 0.0)
	lamp_head_tilt.add_child(cone)

	# Final-demo polish: this marker makes the lamp's local -Z/front side
	# unambiguous from the camera view without changing the 6-DOF animation axes.
	var front_lens_marker: MeshInstance3D = _cylinder_mesh("FrontLookMarker", 0.07, 0.018, front_marker_material)
	front_lens_marker.position = Vector3(0.0, -0.02, -0.275)
	front_lens_marker.rotation_degrees = Vector3(90.0, 0.0, 0.0)
	lamp_head_tilt.add_child(front_lens_marker)

	var look_tip: MeshInstance3D = _sphere_mesh("LookDirectionTip", 0.045, front_marker_material)
	look_tip.position = Vector3(0.0, -0.02, -0.43)
	lamp_head_tilt.add_child(look_tip)


func _clear_children() -> void:
	for child: Node in get_children():
		child.queue_free()


func _update_face_follow(delta: float) -> void:
	var desired_follow_deg: float = 0.0
	if face_x_norm >= 0.0 and face_x_norm <= 1.0 and current_motion != "sleep_rest" and current_motion != "sleep":
		# Camera-frame x=0 is user's left. Positive Godot yaw turns the lamp's
		# local -Z/front direction left from the viewer's perspective.
		desired_follow_deg = clampf((0.5 - face_x_norm) * FACE_FOLLOW_MAX_DEG * 2.0, -FACE_FOLLOW_MAX_DEG, FACE_FOLLOW_MAX_DEG)
	var weight: float = clampf(delta * FACE_FOLLOW_SMOOTH_SPEED, 0.0, 1.0)
	_smoothed_face_follow_deg = lerpf(_smoothed_face_follow_deg, desired_follow_deg, weight)


func _update_motion_targets() -> void:
	var t: float = _time
	var motion: String = current_motion

	if motion == "sleep_rest" or motion == "sleep":
		_set_target(0.0, -48.0, 78.0, -52.0, 0.0, -32.0, 0.0)
	elif motion == "idle" or motion == "idle_breathe":
		var breathe: float = sin(t * 0.85)
		var sway: float = sin(t * 0.48)
		_set_target(2.4 * sway, -18.0 + 1.4 * breathe, 37.0 + 1.2 * breathe, -19.0 + 1.8 * sin(t * 0.95), 1.6 * sway, 1.4 * sin(t * 0.72), 0.45)
	elif motion == "attentive_nod":
		var nod: float = maxf(0.0, sin(t * 3.9))
		_set_target(0.0, -13.0, 33.0, -24.0 - 3.6 * nod, 0.0, 3.0 + 4.0 * nod, 1.0)
	elif motion == "searching_glance":
		var scan: float = sin(t * 0.72)
		_set_target(21.0 * scan, -16.0, 39.0, -18.0, 16.0 * sin(t * 0.92), -5.0 + 2.5 * sin(t * 0.66), 0.30)
	elif motion == "curious_tilt":
		var anticipation: float = maxf(0.0, sin(t * 2.0))
		var tiny_bounce: float = 1.2 * sin(t * 3.0)
		_set_target(5.0 * sin(t * 0.78), -13.0 - tiny_bounce, 41.0 + tiny_bounce, -25.0 - 2.2 * anticipation, 13.0, -18.0 + 3.5 * sin(t * 1.75), 0.65)
	elif motion == "scanning":
		_set_target(38.0 * sin(t * 0.62), -11.0 + 3.5 * sin(t * 0.95), 42.0, -21.0, 24.0 * sin(t * 1.15), -8.0, 0.15)
	elif motion == "thinking" or motion == "recalling":
		_set_target(-2.0 + 2.6 * sin(t * 0.75), -14.5, 42.0, -30.0 + 2.0 * sin(t * 1.8), -7.0 + 3.0 * sin(t * 0.9), -17.0 + 2.2 * sin(t * 1.55), 0.40)
	else:
		_set_target(0.0, -18.0, 36.0, -18.0, 0.0, 0.0, 0.50)


func _set_target(base_deg: float, shoulder_deg: float, elbow_deg: float, wrist_pitch_deg: float, wrist_yaw_deg: float, head_tilt_deg: float, follow_weight: float) -> void:
	var follow_deg: float = _smoothed_face_follow_deg * clampf(follow_weight, 0.0, 1.0)
	_target["base_yaw"] = (base_deg + follow_deg) * DEG
	_target["shoulder_pitch"] = shoulder_deg * DEG
	_target["elbow_pitch"] = elbow_deg * DEG
	_target["wrist_pitch"] = wrist_pitch_deg * DEG
	_target["wrist_yaw"] = (wrist_yaw_deg + follow_deg * 0.32) * DEG
	_target["head_tilt"] = head_tilt_deg * DEG


func _smooth_to_targets(delta: float) -> void:
	var speed: float = 7.5
	if current_motion == "thinking" or current_motion == "recalling":
		speed = 10.5
	elif current_motion == "sleep_rest" or current_motion == "sleep":
		speed = 4.2
	var weight: float = clampf(delta * speed, 0.0, 1.0)
	base_yaw.rotation.y = lerp_angle(base_yaw.rotation.y, float(_target["base_yaw"]), weight)
	shoulder_pitch.rotation.x = lerp_angle(shoulder_pitch.rotation.x, float(_target["shoulder_pitch"]), weight)
	elbow_pitch.rotation.x = lerp_angle(elbow_pitch.rotation.x, float(_target["elbow_pitch"]), weight)
	wrist_pitch.rotation.x = lerp_angle(wrist_pitch.rotation.x, float(_target["wrist_pitch"]), weight)
	wrist_yaw.rotation.y = lerp_angle(wrist_yaw.rotation.y, float(_target["wrist_yaw"]), weight)
	lamp_head_tilt.rotation.x = lerp_angle(lamp_head_tilt.rotation.x, float(_target["head_tilt"]), weight)


func _update_light(_delta: float) -> void:
	var energy: float = 0.6
	var pulse: float = 0.0
	var head_color: Color = Color(1.0, 0.86, 0.48)
	var marker_color: Color = Color(0.15, 0.45, 1.0)

	if current_light == "sleep_red":
		pulse = 0.5 + 0.5 * sin(_time * 0.7)
		energy = 0.055 + pulse * 0.025
		head_color = Color(0.55, 0.035, 0.03)
		marker_color = head_color
	elif current_light == "dim_warm":
		pulse = 0.5 + 0.5 * sin(_time * 0.85)
		energy = 0.34 + pulse * 0.08
		head_color = Color(1.0, 0.86, 0.48)
	elif current_light == "steady_warm":
		energy = 1.16
		head_color = Color(1.0, 0.82, 0.40)
	elif current_light == "slow_pulse":
		pulse = 0.5 + 0.5 * sin(_time * 1.45)
		energy = 0.55 + pulse * 0.55
		head_color = Color(1.0, 0.72, 0.30)
	elif current_light == "soft_pulse":
		pulse = 0.5 + 0.5 * sin(_time * 2.15)
		energy = 0.82 + pulse * 0.55
		head_color = Color(1.0, 0.62, 0.24)
	elif current_light == "scan_sweep":
		pulse = 0.5 + 0.5 * sin(_time * 3.1)
		energy = 0.82 + pulse * 0.45
		head_color = Color(0.62, 0.80, 1.0)
		marker_color = Color(0.62, 0.80, 1.0)
	elif current_light == "focus_glow":
		pulse = 0.5 + 0.5 * sin(_time * 1.15)
		energy = 1.05 + pulse * 0.20
		head_color = Color(0.74, 0.76, 1.0)
		marker_color = Color(0.74, 0.76, 1.0)
	else:
		energy = 0.72

	# Frontend light audit/fix: previous versions changed the emissive lamp-head
	# material, but left the actual SpotLight3D color at its default. Keep the
	# physical light beam and visible cone synchronized with the lamp body color.
	if spot_light != null:
		spot_light.light_energy = energy
		spot_light.light_color = head_color

	if head_material != null:
		head_material.albedo_color = head_color
		head_material.emission = head_color
		head_material.emission_energy_multiplier = energy * 0.55

	if front_marker_material != null:
		front_marker_material.albedo_color = marker_color
		front_marker_material.emission = marker_color
		front_marker_material.emission_energy_multiplier = 0.42 + energy * 0.25

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


func _cylinder_mesh(node_name: String, radius: float, height: float, material: Material) -> MeshInstance3D:
	var instance: MeshInstance3D = MeshInstance3D.new()
	instance.name = node_name
	var mesh: CylinderMesh = CylinderMesh.new()
	mesh.top_radius = radius
	mesh.bottom_radius = radius
	mesh.height = height
	mesh.radial_segments = 32
	instance.mesh = mesh
	instance.material_override = material
	return instance


func _cone_mesh(node_name: String, radius: float, height: float, material: Material) -> MeshInstance3D:
	var instance: MeshInstance3D = MeshInstance3D.new()
	instance.name = node_name
	var mesh: CylinderMesh = CylinderMesh.new()
	mesh.top_radius = radius * 0.25
	mesh.bottom_radius = radius
	mesh.height = height
	mesh.radial_segments = 32
	instance.mesh = mesh
	instance.material_override = material
	return instance


func _sphere_mesh(node_name: String, radius: float, material: Material) -> MeshInstance3D:
	var instance: MeshInstance3D = MeshInstance3D.new()
	instance.name = node_name
	var mesh: SphereMesh = SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.radial_segments = 32
	mesh.rings = 16
	instance.mesh = mesh
	instance.material_override = material
	return instance


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
