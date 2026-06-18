extends Node3D

@onready var udp_receiver: Node = $UdpCommandReceiver
@onready var lamp: Node = $LampRig
@onready var camera: Camera3D = $Camera3D
@onready var sun: DirectionalLight3D = $DirectionalLight3D

var _debug_label: Label
var _receiver_status := "starting"
var _last_command: Dictionary = {}
var _last_packet_local_time := "never"


func _ready() -> void:
	_setup_camera_and_light()
	_build_debug_ui()

	udp_receiver.command_received.connect(_on_command_received)
	udp_receiver.receiver_status_changed.connect(_on_receiver_status_changed)
	udp_receiver.start()

	_last_command = _default_command()
	lamp.apply_command(_last_command)
	_refresh_debug_ui()


func _process(_delta: float) -> void:
	_refresh_debug_ui()


func _setup_camera_and_light() -> void:
	camera.position = Vector3(0.0, 2.0, 5.2)
	camera.look_at(Vector3(0.0, 1.35, 0.0), Vector3.UP)
	camera.fov = 45.0

	sun.rotation_degrees = Vector3(-45.0, 35.0, 0.0)
	sun.light_energy = 1.4


func _build_debug_ui() -> void:
	var canvas := CanvasLayer.new()
	canvas.name = "DebugCanvas"
	add_child(canvas)

	var panel := PanelContainer.new()
	panel.name = "DebugPanel"
	panel.position = Vector2(12, 12)
	panel.custom_minimum_size = Vector2(390, 190)
	canvas.add_child(panel)

	var margin := MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 10)
	margin.add_theme_constant_override("margin_top", 8)
	margin.add_theme_constant_override("margin_right", 10)
	margin.add_theme_constant_override("margin_bottom", 8)
	panel.add_child(margin)

	_debug_label = Label.new()
	_debug_label.name = "DebugLabel"
	_debug_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	margin.add_child(_debug_label)


func _on_receiver_status_changed(message: String) -> void:
	_receiver_status = message
	_refresh_debug_ui()


func _on_command_received(command: Dictionary) -> void:
	_last_command = command
	_last_packet_local_time = Time.get_datetime_string_from_system(false, true)
	lamp.apply_command(command)
	_refresh_debug_ui()


func _refresh_debug_ui() -> void:
	if _debug_label == null:
		return

	var engagement: Dictionary = _last_command.get("engagement", {})
	var behavior: Dictionary = _last_command.get("behavior", {})
	var speech_value: Variant = behavior.get("speech_text", "")
	var speech_text := "" if speech_value == null else str(speech_value)

	var lines: Array[String] = []
	lines.append("LeLamp Milestone 2 Frontend")
	lines.append("UDP: %s" % _receiver_status)
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
	lines.append("Packet timestamp: %s" % str(_last_command.get("timestamp", "never")))
	lines.append("Local received: %s" % _last_packet_local_time)

	var text := ""
	for line in lines:
		if text != "":
			text += "\n"
		text += line
	_debug_label.text = text


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
