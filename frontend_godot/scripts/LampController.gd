extends Node3D

const DEG := PI / 180.0

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

var current_state := "idle"
var current_motion := "idle_breathe"
var current_light := "dim_warm"
var current_sound := "none"
var current_speech_text := ""
var engagement_status := "absent"
var engagement_confidence := 0.0

var _time := 0.0
var _target := {
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

	var behavior: Dictionary = command.get("behavior", {})
	current_motion = str(behavior.get("motion", current_motion))
	current_light = str(behavior.get("light", current_light))
	current_sound = str(behavior.get("sound", "none"))
	var speech_value: Variant = behavior.get("speech_text", "")
	current_speech_text = "" if speech_value == null else str(speech_value)

	var engagement: Dictionary = command.get("engagement", {})
	engagement_status = str(engagement.get("status", engagement_status))
	engagement_confidence = float(engagement.get("confidence", engagement_confidence))


func _process(delta: float) -> void:
	_time += delta
	_update_motion_targets()
	_smooth_to_targets(delta)
	_update_light(delta)


func _build_lamp_rig() -> void:
	_clear_children()

	var base_material := _make_material(Color(0.34, 0.34, 0.36), false)
	var arm_material := _make_material(Color(0.72, 0.72, 0.76), false)
	head_material = _make_material(Color(1.0, 0.86, 0.48), true)
	cone_material = _make_transparent_material(Color(1.0, 0.82, 0.30, 0.18))
	front_marker_material = _make_material(Color(0.15, 0.45, 1.0), true)

	base_yaw = Node3D.new()
	base_yaw.name = "BaseYaw_DOF1"
	add_child(base_yaw)

	var base_mesh := _cylinder_mesh("Base", 0.42, 0.18, base_material)
	base_mesh.position = Vector3(0.0, 0.09, 0.0)
	base_yaw.add_child(base_mesh)

	shoulder_pitch = Node3D.new()
	shoulder_pitch.name = "ShoulderPitch_DOF2"
	shoulder_pitch.position = Vector3(0.0, 0.24, 0.0)
	base_yaw.add_child(shoulder_pitch)

	var shoulder_joint := _sphere_mesh("ShoulderJoint", 0.15, base_material)
	shoulder_pitch.add_child(shoulder_joint)

	var upper_arm := _cylinder_mesh("UpperArm", 0.055, 1.15, arm_material)
	upper_arm.position = Vector3(0.0, 0.575, 0.0)
	shoulder_pitch.add_child(upper_arm)

	elbow_pitch = Node3D.new()
	elbow_pitch.name = "ElbowPitch_DOF3"
	elbow_pitch.position = Vector3(0.0, 1.15, 0.0)
	shoulder_pitch.add_child(elbow_pitch)

	var elbow_joint := _sphere_mesh("ElbowJoint", 0.13, base_material)
	elbow_pitch.add_child(elbow_joint)

	var lower_arm := _cylinder_mesh("LowerArm", 0.05, 0.90, arm_material)
	lower_arm.position = Vector3(0.0, 0.45, 0.0)
	elbow_pitch.add_child(lower_arm)

	wrist_pitch = Node3D.new()
	wrist_pitch.name = "WristPitch_DOF4"
	wrist_pitch.position = Vector3(0.0, 0.90, 0.0)
	elbow_pitch.add_child(wrist_pitch)

	var wrist_pitch_joint := _sphere_mesh("WristPitchJoint", 0.105, base_material)
	wrist_pitch.add_child(wrist_pitch_joint)

	wrist_yaw = Node3D.new()
	wrist_yaw.name = "WristYaw_DOF5"
	wrist_pitch.add_child(wrist_yaw)

	lamp_head_tilt = Node3D.new()
	lamp_head_tilt.name = "LampHeadTilt_DOF6"
	lamp_head_tilt.position = Vector3(0.0, 0.16, -0.08)
	wrist_yaw.add_child(lamp_head_tilt)

	var head := _sphere_mesh("LampHead", 0.22, head_material)
	head.scale = Vector3(1.15, 0.82, 1.0)
	lamp_head_tilt.add_child(head)

	var shade := _cylinder_mesh("LampShade", 0.22, 0.18, head_material)
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
	for child in get_children():
		child.queue_free()


func _update_motion_targets() -> void:
	var t := _time
	var motion := current_motion

	if motion == "sleep_rest" or motion == "sleep":
		_set_target(0.0, -42.0, 70.0, -46.0, 0.0, -24.0)
	elif motion == "idle" or motion == "idle_breathe":
		_set_target(8.0 * sin(t * 0.65), -17.0 + 2.0 * sin(t * 1.0), 36.0, -18.0 + 2.5 * sin(t * 1.2), 0.0, 2.0 * sin(t * 1.1))
	elif motion == "attentive_nod":
		_set_target(0.0, -15.0, 34.0, -21.0 + 6.0 * sin(t * 4.2), 0.0, 7.0 * sin(t * 4.2))
	elif motion == "searching_glance":
		_set_target(32.0 * sin(t * 1.15), -14.0, 38.0, -17.0, 22.0 * sin(t * 1.8), -4.0)
	elif motion == "curious_tilt":
		_set_target(10.0 * sin(t * 1.3), -12.0, 41.0, -24.0, 14.0, -18.0 + 5.0 * sin(t * 2.4))
	elif motion == "scanning":
		_set_target(48.0 * sin(t * 0.85), -10.0 + 6.0 * sin(t * 1.1), 42.0, -20.0, 30.0 * sin(t * 1.7), -8.0)
	elif motion == "thinking" or motion == "recalling":
		_set_target(-8.0 + 4.0 * sin(t * 0.8), -13.0, 40.0, -26.0 + 3.0 * sin(t * 2.1), -10.0, -14.0 + 3.0 * sin(t * 1.9))
	else:
		_set_target(0.0, -18.0, 36.0, -18.0, 0.0, 0.0)


func _set_target(base_deg: float, shoulder_deg: float, elbow_deg: float, wrist_pitch_deg: float, wrist_yaw_deg: float, head_tilt_deg: float) -> void:
	_target["base_yaw"] = base_deg * DEG
	_target["shoulder_pitch"] = shoulder_deg * DEG
	_target["elbow_pitch"] = elbow_deg * DEG
	_target["wrist_pitch"] = wrist_pitch_deg * DEG
	_target["wrist_yaw"] = wrist_yaw_deg * DEG
	_target["head_tilt"] = head_tilt_deg * DEG


func _smooth_to_targets(delta: float) -> void:
	var weight: float = clampf(delta * 7.5, 0.0, 1.0)
	base_yaw.rotation.y = lerp_angle(base_yaw.rotation.y, _target["base_yaw"], weight)
	shoulder_pitch.rotation.x = lerp_angle(shoulder_pitch.rotation.x, _target["shoulder_pitch"], weight)
	elbow_pitch.rotation.x = lerp_angle(elbow_pitch.rotation.x, _target["elbow_pitch"], weight)
	wrist_pitch.rotation.x = lerp_angle(wrist_pitch.rotation.x, _target["wrist_pitch"], weight)
	wrist_yaw.rotation.y = lerp_angle(wrist_yaw.rotation.y, _target["wrist_yaw"], weight)
	lamp_head_tilt.rotation.x = lerp_angle(lamp_head_tilt.rotation.x, _target["head_tilt"], weight)


func _update_light(_delta: float) -> void:
	var energy: float = 0.6
	var pulse: float = 0.0
	var head_color: Color = Color(1.0, 0.86, 0.48)
	var marker_color: Color = Color(0.15, 0.45, 1.0)

	if current_light == "sleep_red":
		energy = 0.12
		head_color = Color(0.65, 0.04, 0.03)
		marker_color = Color(0.65, 0.04, 0.03)
	elif current_light == "dim_warm":
		energy = 0.55
	elif current_light == "steady_warm":
		energy = 1.25
	elif current_light == "slow_pulse":
		pulse = 0.5 + 0.5 * sin(_time * 2.0)
		energy = 0.75 + pulse * 0.55
	elif current_light == "soft_pulse":
		pulse = 0.5 + 0.5 * sin(_time * 3.2)
		energy = 0.95 + pulse * 0.75
	elif current_light == "scan_sweep":
		pulse = 0.5 + 0.5 * sin(_time * 5.0)
		energy = 1.0 + pulse * 0.45
	elif current_light == "focus_glow":
		pulse = 0.5 + 0.5 * sin(_time * 1.5)
		energy = 1.1 + pulse * 0.25
	else:
		energy = 0.8

	if spot_light != null:
		spot_light.light_energy = energy

	if head_material != null:
		head_material.albedo_color = head_color
		head_material.emission = head_color
		head_material.emission_energy_multiplier = energy * 0.55

	if front_marker_material != null:
		front_marker_material.albedo_color = marker_color
		front_marker_material.emission = marker_color
		front_marker_material.emission_energy_multiplier = 0.7 + energy * 0.25

	if cone_material != null:
		var alpha: float = clampf(0.10 + energy * 0.06, 0.10, 0.26)
		var c := cone_material.albedo_color
		c.a = alpha
		cone_material.albedo_color = c


func _cylinder_mesh(node_name: String, radius: float, height: float, material: Material) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.name = node_name
	var mesh := CylinderMesh.new()
	mesh.top_radius = radius
	mesh.bottom_radius = radius
	mesh.height = height
	mesh.radial_segments = 32
	instance.mesh = mesh
	instance.material_override = material
	return instance


func _cone_mesh(node_name: String, radius: float, height: float, material: Material) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.name = node_name
	var mesh := CylinderMesh.new()
	mesh.top_radius = radius * 0.25
	mesh.bottom_radius = radius
	mesh.height = height
	mesh.radial_segments = 32
	instance.mesh = mesh
	instance.material_override = material
	return instance


func _sphere_mesh(node_name: String, radius: float, material: Material) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.name = node_name
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.radial_segments = 32
	mesh.rings = 16
	instance.mesh = mesh
	instance.material_override = material
	return instance


func _make_material(color: Color, emissive: bool) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	if emissive:
		mat.emission_enabled = true
		mat.emission = color
		mat.emission_energy_multiplier = 0.6
	return mat


func _make_transparent_material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = color
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	return mat
