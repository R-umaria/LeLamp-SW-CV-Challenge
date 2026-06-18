extends Node3D

@export var response_visible_seconds: float = 8.0
@export var backend_stale_timeout_seconds: float = 60.0
@export var camera_position: Vector3 = Vector3(2.85, 2.05, -4.65)
@export var camera_target: Vector3 = Vector3(0.0, 1.22, 0.0)
@export var camera_fov_degrees: float = 48.0

@onready var udp_receiver: Node = $UdpCommandReceiver
@onready var lamp: Node3D = $LampRig
@onready var camera: Camera3D = $Camera3D
@onready var sun: DirectionalLight3D = $DirectionalLight3D
@onready var fill_light: OmniLight3D = $FillLight3D

var _debug_label: Label
var _response_panel: PanelContainer
var _response_label: Label

var _receiver_status: String = "starting"
var _last_command: Dictionary = {}
var _last_packet_local_time: String = "never"
var _last_packet_unix_time: float = 0.0
var _response_visible_until: float = 0.0
var _last_response_text: String = ""
var _sleeping_due_to_stale: bool = false


func _ready() -> void:
	_setup_camera_and_light()
	_build_scene_environment()
	_build_ui()

	udp_receiver.command_received.connect(_on_command_received)
	udp_receiver.receiver_status_changed.connect(_on_receiver_status_changed)
	udp_receiver.start()

	_last_packet_unix_time = Time.get_unix_time_from_system()
	_last_command = _default_command()
	lamp.apply_command(_last_command)
	_refresh_debug_ui()
	_update_response_panel()


func _process(_delta: float) -> void:
	_check_backend_stale_sleep()
	_refresh_debug_ui()
	_update_response_panel()


func _setup_camera_and_light() -> void:
	# The lamp head and visible cone point along the rig's local -Z axis.
	# Put the camera on that front side with a slight X offset so the final demo
	# shows the face, cone, head tilt, and arm joints at the same time.
	lamp.rotation_degrees = Vector3.ZERO
	camera.position = camera_position
	camera.look_at(camera_target, Vector3.UP)
	camera.fov = camera_fov_degrees
	camera.current = true

	# Key light from the viewer/front side; fill light keeps the back arm joints readable.
	sun.rotation_degrees = Vector3(-38.0, -32.0, 0.0)
	sun.light_energy = 1.45

	fill_light.position = Vector3(-2.3, 2.7, -2.4)
	fill_light.light_energy = 1.05
	fill_light.omni_range = 6.0


func _build_scene_environment() -> void:
	if has_node("DemoEnvironment"):
		return

	var root: Node3D = Node3D.new()
	root.name = "DemoEnvironment"
	add_child(root)

	var desk_material: StandardMaterial3D = _make_env_material(Color(0.43, 0.31, 0.21), 0.12)
	var wall_material: StandardMaterial3D = _make_env_material(Color(0.30, 0.33, 0.36), 0.05)
	var floor_material: StandardMaterial3D = _make_env_material(Color(0.18, 0.19, 0.20), 0.05)
	var trim_material: StandardMaterial3D = _make_env_material(Color(0.18, 0.20, 0.22), 0.08)
	var glass_material: StandardMaterial3D = _make_env_material(Color(0.28, 0.43, 0.62), 0.25)
	var accent_material: StandardMaterial3D = _make_env_material(Color(0.62, 0.50, 0.34), 0.18)

	root.add_child(_box_mesh("DeskSurface", Vector3(3.6, 0.08, 2.25), Vector3(0.0, -0.04, 0.05), desk_material))
	root.add_child(_box_mesh("DeskFrontLip", Vector3(3.7, 0.12, 0.06), Vector3(0.0, -0.03, -1.10), trim_material))
	root.add_child(_box_mesh("BackWall", Vector3(4.6, 2.4, 0.08), Vector3(0.0, 1.15, 1.18), wall_material))
	root.add_child(_box_mesh("FloorShadowPlane", Vector3(4.7, 0.05, 3.4), Vector3(0.0, -0.09, 0.25), floor_material))

	# Lightweight room cues: a small window and shelf behind the lamp keep the
	# scene readable without adding imported assets or physics cost.
	root.add_child(_box_mesh("WindowGlass", Vector3(0.88, 0.58, 0.025), Vector3(-1.18, 1.45, 1.125), glass_material))
	root.add_child(_box_mesh("WindowTop", Vector3(0.98, 0.045, 0.04), Vector3(-1.18, 1.765, 1.095), trim_material))
	root.add_child(_box_mesh("WindowBottom", Vector3(0.98, 0.045, 0.04), Vector3(-1.18, 1.135, 1.095), trim_material))
	root.add_child(_box_mesh("WindowLeft", Vector3(0.045, 0.64, 0.04), Vector3(-1.69, 1.45, 1.095), trim_material))
	root.add_child(_box_mesh("WindowRight", Vector3(0.045, 0.64, 0.04), Vector3(-0.67, 1.45, 1.095), trim_material))
	root.add_child(_box_mesh("BackShelf", Vector3(1.15, 0.07, 0.18), Vector3(1.05, 0.88, 1.02), accent_material))
	root.add_child(_box_mesh("SmallBookA", Vector3(0.10, 0.28, 0.16), Vector3(0.72, 1.04, 0.90), trim_material))
	root.add_child(_box_mesh("SmallBookB", Vector3(0.10, 0.22, 0.16), Vector3(0.86, 1.01, 0.90), glass_material))

	var world: WorldEnvironment = WorldEnvironment.new()
	world.name = "SoftAmbientWorld"
	var env: Environment = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.09, 0.10, 0.12)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.72, 0.67, 0.58)
	env.ambient_light_energy = 0.35
	world.environment = env
	add_child(world)


func _build_ui() -> void:
	var canvas: CanvasLayer = CanvasLayer.new()
	canvas.name = "LeLampCanvas"
	add_child(canvas)
	_build_debug_panel(canvas)
	_build_response_panel(canvas)


func _build_debug_panel(canvas: CanvasLayer) -> void:
	var panel: PanelContainer = PanelContainer.new()
	panel.name = "DebugPanel"
	panel.position = Vector2(12.0, 12.0)
	panel.custom_minimum_size = Vector2(440.0, 292.0)
	canvas.add_child(panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 10)
	margin.add_theme_constant_override("margin_top", 8)
	margin.add_theme_constant_override("margin_right", 10)
	margin.add_theme_constant_override("margin_bottom", 8)
	panel.add_child(margin)

	_debug_label = Label.new()
	_debug_label.name = "DebugLabel"
	_debug_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	margin.add_child(_debug_label)


func _build_response_panel(canvas: CanvasLayer) -> void:
	_response_panel = PanelContainer.new()
	_response_panel.name = "RecallResponsePanel"
	_response_panel.position = Vector2(450.0, 12.0)
	_response_panel.custom_minimum_size = Vector2(570.0, 165.0)
	_response_panel.visible = false
	canvas.add_child(_response_panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 16)
	margin.add_theme_constant_override("margin_top", 12)
	margin.add_theme_constant_override("margin_right", 16)
	margin.add_theme_constant_override("margin_bottom", 12)
	_response_panel.add_child(margin)

	var vbox: VBoxContainer = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	margin.add_child(vbox)

	var title: Label = Label.new()
	title.text = "LeLamp recall answer"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	vbox.add_child(title)

	_response_label = Label.new()
	_response_label.name = "RecallResponseLabel"
	_response_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_response_label.text = ""
	vbox.add_child(_response_label)


func _on_receiver_status_changed(message: String) -> void:
	_receiver_status = message
	_refresh_debug_ui()


func _on_command_received(command: Dictionary) -> void:
	_sleeping_due_to_stale = false
	_last_command = command
	_last_packet_local_time = Time.get_datetime_string_from_system(false, true)
	_last_packet_unix_time = Time.get_unix_time_from_system()
	lamp.apply_command(command)
	_maybe_show_recall_response(command)
	_refresh_debug_ui()


func _maybe_show_recall_response(command: Dictionary) -> void:
	var state_text: String = str(command.get("state", ""))
	var behavior: Dictionary = _dictionary_value(command, "behavior")
	var speech_value: Variant = behavior.get("speech_text", null)
	var speech_text: String = "" if speech_value == null else str(speech_value).strip_edges()
	if state_text == "recalling" and speech_text != "":
		_show_response_text(speech_text)


func _dictionary_value(source: Dictionary, key: String) -> Dictionary:
	var value: Variant = source.get(key, {})
	if typeof(value) == TYPE_DICTIONARY:
		return value as Dictionary
	return {}


func _show_response_text(text: String) -> void:
	_last_response_text = text
	_response_visible_until = Time.get_unix_time_from_system() + response_visible_seconds
	if _response_label != null:
		_response_label.text = text
	if _response_panel != null:
		_response_panel.visible = true


func _update_response_panel() -> void:
	if _response_panel == null:
		return
	var now: float = Time.get_unix_time_from_system()
	_response_panel.visible = _last_response_text != "" and now < _response_visible_until


func _check_backend_stale_sleep() -> void:
	if backend_stale_timeout_seconds <= 0.0:
		return
	if _sleeping_due_to_stale:
		return
	var now: float = Time.get_unix_time_from_system()
	var age_s: float = now - _last_packet_unix_time
	if age_s >= backend_stale_timeout_seconds:
		_enter_backend_stale_sleep(age_s)


func _enter_backend_stale_sleep(age_s: float) -> void:
	_sleeping_due_to_stale = true
	_last_command = _sleep_command(age_s)
	lamp.apply_command(_last_command)
	_refresh_debug_ui()


func _refresh_debug_ui() -> void:
	if _debug_label == null:
		return

	var engagement: Dictionary = _dictionary_value(_last_command, "engagement")
	var behavior: Dictionary = _dictionary_value(_last_command, "behavior")
	var speech_value: Variant = behavior.get("speech_text", "")
	var speech_text: String = "" if speech_value == null else str(speech_value)
	var now: float = Time.get_unix_time_from_system()
	var packet_age_s: float = maxf(0.0, now - _last_packet_unix_time)
	var connection_health: String = "backend stale / sleeping" if _sleeping_due_to_stale else "backend active/waiting"
	var face_x_text: String = _optional_debug_value(engagement.get("face_x_norm", null))
	var face_y_text: String = _optional_debug_value(engagement.get("face_y_norm", null))

	var lines: Array[String] = []
	lines.append("LeLamp Milestone 4.3.3 Frontend")
	lines.append("UDP commands: %s" % _receiver_status)
	lines.append("Text input: browser chat at http://127.0.0.1:8765")
	lines.append("Health: %s  age: %.1fs" % [connection_health, packet_age_s])
	lines.append("Backend stale sleep: %s" % ("yes" if _sleeping_due_to_stale else "no"))
	lines.append("View: front-three-quarter / local -Z")
	lines.append("State: %s" % str(_last_command.get("state", "unknown")))
	lines.append("Motion: %s" % str(behavior.get("motion", "none")))
	lines.append("Light: %s" % str(behavior.get("light", "none")))
	lines.append("Sound: %s" % str(behavior.get("sound", "none")))
	lines.append("Speech: %s" % speech_text)
	lines.append("Engagement: %s  confidence: %.2f" % [
		str(engagement.get("status", "unknown")),
		float(engagement.get("confidence", 0.0))
	])
	lines.append("Reason: %s" % str(engagement.get("reason", "none")))
	lines.append("Face follow hint: x=%s  y=%s" % [face_x_text, face_y_text])
	lines.append("Packet timestamp: %s" % str(_last_command.get("timestamp", "never")))
	lines.append("Local received: %s" % _last_packet_local_time)
	lines.append("Recall panel: %s" % ("visible" if _response_panel != null and _response_panel.visible else "hidden"))

	var debug_text: String = ""
	for line: String in lines:
		if debug_text != "":
			debug_text += "\n"
		debug_text += line
	_debug_label.text = debug_text


func _optional_debug_value(value: Variant) -> String:
	var value_type: int = typeof(value)
	if value_type == TYPE_FLOAT or value_type == TYPE_INT:
		return "%.2f" % float(value)
	return "n/a"


func _box_mesh(node_name: String, size: Vector3, position: Vector3, material: Material) -> MeshInstance3D:
	var instance: MeshInstance3D = MeshInstance3D.new()
	instance.name = node_name
	var mesh: BoxMesh = BoxMesh.new()
	mesh.size = size
	instance.mesh = mesh
	instance.position = position
	instance.material_override = material
	return instance


func _make_env_material(color: Color, roughness: float) -> StandardMaterial3D:
	var mat: StandardMaterial3D = StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = clampf(roughness, 0.0, 1.0)
	return mat


func _default_command() -> Dictionary:
	return {
		"timestamp": "not connected yet",
		"state": "idle",
		"engagement": {
			"status": "absent",
			"confidence": 0.0,
			"reason": "waiting_for_udp",
		},
		"behavior": {
			"motion": "idle_breathe",
			"light": "dim_warm",
			"sound": null,
			"speech_text": null,
		},
		"memory": {
			"last_detected_objects": [],
		},
	}


func _sleep_command(age_s: float) -> Dictionary:
	return {
		"timestamp": "backend stale for %.1fs" % age_s,
		"state": "sleep",
		"engagement": {
			"status": "absent",
			"confidence": 0.0,
			"reason": "backend_stale_local_sleep",
		},
		"behavior": {
			"motion": "sleep_rest",
			"light": "sleep_red",
			"sound": null,
			"speech_text": null,
		},
		"memory": {
			"last_detected_objects": [],
		},
	}
