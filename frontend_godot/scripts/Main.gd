extends Node3D

@export var response_visible_seconds: float = 8.0
@export var backend_stale_timeout_seconds: float = 60.0
@export var chat_host: String = "127.0.0.1"
@export var chat_port: int = 4243
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
var _chat_history: RichTextLabel
var _chat_input: LineEdit
var _send_button: Button
var _chat_udp: PacketPeerUDP = PacketPeerUDP.new()

var _receiver_status: String = "starting"
var _last_command: Dictionary = {}
var _last_packet_local_time: String = "never"
var _last_packet_unix_time: float = 0.0
var _response_visible_until: float = 0.0
var _last_response_text: String = ""
var _sleeping_due_to_stale: bool = false


func _ready() -> void:
	_setup_camera_and_light()
	_build_ui()

	udp_receiver.command_received.connect(_on_command_received)
	udp_receiver.receiver_status_changed.connect(_on_receiver_status_changed)
	udp_receiver.start()

	_last_packet_unix_time = Time.get_unix_time_from_system()
	_last_command = _default_command()
	lamp.apply_command(_last_command)
	_add_chat_history_line("lamp", "Live chat ready. Ask: Where did you last see my phone?")
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
	fill_light.light_energy = 0.9
	fill_light.omni_range = 6.0


func _build_ui() -> void:
	var canvas: CanvasLayer = CanvasLayer.new()
	canvas.name = "LeLampCanvas"
	add_child(canvas)
	_build_debug_panel(canvas)
	_build_response_panel(canvas)
	_build_chat_panel(canvas)


func _build_debug_panel(canvas: CanvasLayer) -> void:
	var panel: PanelContainer = PanelContainer.new()
	panel.name = "DebugPanel"
	panel.position = Vector2(12.0, 12.0)
	panel.custom_minimum_size = Vector2(420.0, 260.0)
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


func _build_chat_panel(canvas: CanvasLayer) -> void:
	var panel: PanelContainer = PanelContainer.new()
	panel.name = "LiveChatPanel"
	panel.position = Vector2(12.0, 300.0)
	panel.custom_minimum_size = Vector2(590.0, 245.0)
	canvas.add_child(panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 12)
	margin.add_theme_constant_override("margin_top", 10)
	margin.add_theme_constant_override("margin_right", 12)
	margin.add_theme_constant_override("margin_bottom", 10)
	panel.add_child(margin)

	var vbox: VBoxContainer = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	margin.add_child(vbox)

	var title: Label = Label.new()
	title.text = "Live grounded memory chat"
	vbox.add_child(title)

	_chat_history = RichTextLabel.new()
	_chat_history.name = "ChatHistory"
	_chat_history.custom_minimum_size = Vector2(560.0, 145.0)
	_chat_history.scroll_active = true
	_chat_history.fit_content = false
	vbox.add_child(_chat_history)

	var hbox: HBoxContainer = HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 8)
	vbox.add_child(hbox)

	_chat_input = LineEdit.new()
	_chat_input.name = "ChatInput"
	_chat_input.placeholder_text = "Ask: Where did you last see my phone?"
	_chat_input.custom_minimum_size = Vector2(450.0, 34.0)
	_chat_input.text_submitted.connect(_on_chat_submitted)
	hbox.add_child(_chat_input)

	_send_button = Button.new()
	_send_button.name = "SendChatButton"
	_send_button.text = "Send"
	_send_button.pressed.connect(_on_send_pressed)
	hbox.add_child(_send_button)


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


func _on_send_pressed() -> void:
	_send_chat_text(_chat_input.text)


func _on_chat_submitted(text: String) -> void:
	_send_chat_text(text)


func _send_chat_text(text: String) -> void:
	var query: String = text.strip_edges()
	if query == "":
		return

	var payload: Dictionary = {
		"type": "chat_query",
		"timestamp": Time.get_datetime_string_from_system(false, true),
		"text": query,
	}
	var encoded: PackedByteArray = JSON.stringify(payload).to_utf8_buffer()
	var err: int = _chat_udp.set_dest_address(chat_host, chat_port)
	if err != OK:
		_add_chat_history_line("system", "Could not address backend chat UDP: %s" % error_string(err))
		return
	err = _chat_udp.put_packet(encoded)
	if err != OK:
		_add_chat_history_line("system", "Could not send chat packet: %s" % error_string(err))
		return

	_add_chat_history_line("you", query)
	_chat_input.clear()


func _add_chat_history_line(speaker: String, text: String) -> void:
	if _chat_history == null:
		return
	var timestamp: String = Time.get_time_string_from_system()
	_chat_history.append_text("[%s] %s: %s\n" % [timestamp, speaker, text])
	_chat_history.scroll_to_line(_chat_history.get_line_count())


func _maybe_show_recall_response(command: Dictionary) -> void:
	var state_text: String = str(command.get("state", ""))
	var behavior: Dictionary = _dictionary_value(command, "behavior")
	var speech_value: Variant = behavior.get("speech_text", null)
	var speech_text: String = "" if speech_value == null else str(speech_value).strip_edges()
	if state_text == "recalling" and speech_text != "":
		_show_response_text(speech_text)
		_add_chat_history_line("lamp", speech_text)


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
	_add_chat_history_line("system", "Backend command stream is stale; lamp entered local sleep mode.")
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

	var lines: Array[String] = []
	lines.append("LeLamp Milestone 4.3 Frontend")
	lines.append("UDP commands: %s" % _receiver_status)
	lines.append("Chat target: udp://%s:%d" % [chat_host, chat_port])
	lines.append("Health: %s  age: %.1fs" % [connection_health, packet_age_s])
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
	lines.append("Packet timestamp: %s" % str(_last_command.get("timestamp", "never")))
	lines.append("Local received: %s" % _last_packet_local_time)
	lines.append("Recall panel: %s" % ("visible" if _response_panel != null and _response_panel.visible else "hidden"))

	var debug_text: String = ""
	for line: String in lines:
		if debug_text != "":
			debug_text += "\n"
		debug_text += line
	_debug_label.text = debug_text


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
