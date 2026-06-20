extends Node3D

@export var response_visible_seconds: float = 8.0
@export var backend_stale_timeout_seconds: float = 60.0

# Keep this false for normal tinkering. When false, Godot uses the positions,
# rotations, scales, and light settings saved directly in Main.tscn.
@export var apply_demo_framing_on_start: bool = false
@export var camera_position: Vector3 = Vector3(2.20, 1.22, -2.10)
@export var camera_target: Vector3 = Vector3(-0.24, 0.98, 0.04)
@export var camera_fov_degrees: float = 42.0
@export var lamp_position: Vector3 = Vector3(-0.28, 0.0, 0.08)
@export var lamp_rotation_degrees: Vector3 = Vector3(0.0, -34.0, 0.0)
@export var lamp_scale: Vector3 = Vector3(0.82, 0.82, 0.82)

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
	_apply_optional_demo_framing()
	_build_ui()

	udp_receiver.command_received.connect(_on_command_received)
	udp_receiver.receiver_status_changed.connect(_on_receiver_status_changed)
	udp_receiver.start()

	_last_packet_unix_time = Time.get_unix_time_from_system()
	_last_command = _default_command()
	lamp.call("apply_command", _last_command)
	_refresh_debug_ui()
	_update_response_panel()


func _process(_delta: float) -> void:
	_check_backend_stale_sleep()
	_refresh_debug_ui()
	_update_response_panel()


func _apply_optional_demo_framing() -> void:
	if not apply_demo_framing_on_start:
		return

	lamp.position = lamp_position
	lamp.rotation_degrees = lamp_rotation_degrees
	lamp.scale = lamp_scale

	camera.position = camera_position
	camera.look_at(camera_target, Vector3.UP)
	camera.fov = camera_fov_degrees
	camera.current = true

	# These are optional defaults only. For visual tinkering, edit the scene nodes
	# directly in Main.tscn and leave apply_demo_framing_on_start disabled.
	sun.rotation_degrees = Vector3(-42.0, -18.0, 0.0)
	sun.light_energy = 1.18
	sun.shadow_enabled = true

	fill_light.position = Vector3(1.8, 1.9, -1.15)
	fill_light.light_energy = 0.52
	fill_light.omni_range = 5.0


func _build_ui() -> void:
	if has_node("LeLampCanvas"):
		return
	var canvas: CanvasLayer = CanvasLayer.new()
	canvas.name = "LeLampCanvas"
	add_child(canvas)
	_build_debug_panel(canvas)
	_build_response_panel(canvas)


func _build_debug_panel(canvas: CanvasLayer) -> void:
	var panel: PanelContainer = PanelContainer.new()
	panel.name = "DebugPanel"
	panel.position = Vector2(12.0, 12.0)
	panel.custom_minimum_size = Vector2(355.0, 224.0)
	canvas.add_child(panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 8)
	margin.add_theme_constant_override("margin_top", 7)
	margin.add_theme_constant_override("margin_right", 8)
	margin.add_theme_constant_override("margin_bottom", 7)
	panel.add_child(margin)

	_debug_label = Label.new()
	_debug_label.name = "DebugLabel"
	_debug_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_debug_label.add_theme_font_size_override("font_size", 12)
	margin.add_child(_debug_label)


func _build_response_panel(canvas: CanvasLayer) -> void:
	_response_panel = PanelContainer.new()
	_response_panel.name = "RecallResponsePanel"
	_response_panel.position = Vector2(390.0, 12.0)
	_response_panel.custom_minimum_size = Vector2(500.0, 132.0)
	_response_panel.visible = false
	canvas.add_child(_response_panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 14)
	margin.add_theme_constant_override("margin_top", 10)
	margin.add_theme_constant_override("margin_right", 14)
	margin.add_theme_constant_override("margin_bottom", 10)
	_response_panel.add_child(margin)

	var vbox: VBoxContainer = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 6)
	margin.add_child(vbox)

	var title: Label = Label.new()
	title.text = "LeLamp recall answer"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	title.add_theme_font_size_override("font_size", 14)
	vbox.add_child(title)

	_response_label = Label.new()
	_response_label.name = "RecallResponseLabel"
	_response_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_response_label.text = ""
	_response_label.add_theme_font_size_override("font_size", 14)
	vbox.add_child(_response_label)


func _on_receiver_status_changed(message: String) -> void:
	_receiver_status = message
	_refresh_debug_ui()


func _on_command_received(command: Dictionary) -> void:
	_sleeping_due_to_stale = false
	_last_command = command
	_last_packet_local_time = Time.get_datetime_string_from_system(false, true)
	_last_packet_unix_time = Time.get_unix_time_from_system()
	lamp.call("apply_command", command)
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
	lamp.call("apply_command", _last_command)
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
	lines.append("LeLamp Milestone 4.3.6 Editable Godot Scene")
	lines.append("UDP: %s" % _receiver_status)
	lines.append("Browser chat: http://127.0.0.1:8765")
	lines.append("Health: %s  age: %.1fs" % [connection_health, packet_age_s])
	lines.append("State: %s" % str(_last_command.get("state", "unknown")))
	lines.append("Motion: %s  Light: %s" % [str(behavior.get("motion", "none")), str(behavior.get("light", "none"))])
	lines.append("Engagement: %s  %.2f" % [str(engagement.get("status", "unknown")), float(engagement.get("confidence", 0.0))])
	lines.append("Reason: %s" % str(engagement.get("reason", "none")))
	lines.append("Face hint: x=%s  y=%s" % [face_x_text, face_y_text])
	lines.append("Recall panel: %s" % ("visible" if _response_panel != null and _response_panel.visible else "hidden"))
	if speech_text != "":
		lines.append("Speech: %s" % speech_text)

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
