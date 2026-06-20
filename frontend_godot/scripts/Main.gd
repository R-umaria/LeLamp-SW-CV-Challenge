extends Node3D

@export var response_visible_seconds: float = 8.0
@export var backend_stale_timeout_seconds: float = 60.0
@export var camera_position: Vector3 = Vector3(2.55, 1.30, -2.35)
@export var camera_target: Vector3 = Vector3(-0.22, 0.96, 0.06)
@export var camera_fov_degrees: float = 44.0

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
	_setup_camera_lamp_and_light()
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


func _setup_camera_lamp_and_light() -> void:
	# Scene composition keeps the lamp on the tabletop in front of the window,
	# but scales it up so it reads clearly as the demo's main embodied agent.
	lamp.position = Vector3(-0.28, 0.0, 0.08)
	lamp.scale = Vector3(0.82, 0.82, 0.82)
	lamp.rotation_degrees = Vector3(0.0, -34.0, 0.0)

	camera.position = camera_position
	camera.look_at(camera_target, Vector3.UP)
	camera.fov = camera_fov_degrees
	camera.current = true

	# Soft daylight comes from the window side. The fill light keeps the enlarged
	# lamp readable in the closer camera framing without flattening the taller room.
	sun.rotation_degrees = Vector3(-42.0, -18.0, 0.0)
	sun.light_energy = 1.18
	sun.shadow_enabled = true

	fill_light.position = Vector3(1.8, 1.9, -1.15)
	fill_light.light_energy = 0.52
	fill_light.omni_range = 5.0


func _build_scene_environment() -> void:
	if has_node("RoomRoot"):
		return

	var root: Node3D = Node3D.new()
	root.name = "RoomRoot"
	add_child(root)
	move_child(root, 0)

	var wall_material: StandardMaterial3D = _make_env_material(Color(0.86, 0.85, 0.81), 0.62)
	var ceiling_material: StandardMaterial3D = _make_env_material(Color(0.78, 0.78, 0.75), 0.68)
	var floor_material: StandardMaterial3D = _make_env_material(Color(0.36, 0.34, 0.30), 0.70)
	var trim_material: StandardMaterial3D = _make_env_material(Color(0.48, 0.48, 0.45), 0.54)
	var dark_frame_material: StandardMaterial3D = _make_env_material(Color(0.16, 0.17, 0.16), 0.42)
	var sky_material: StandardMaterial3D = _make_env_material(Color(0.70, 0.84, 0.98), 0.95)
	var glass_material: StandardMaterial3D = _make_translucent_env_material(Color(0.74, 0.88, 1.0, 0.30), 0.18)
	var wood_material: StandardMaterial3D = _make_env_material(Color(0.66, 0.49, 0.32), 0.58)
	var wood_dark_material: StandardMaterial3D = _make_env_material(Color(0.50, 0.36, 0.23), 0.60)
	var wood_light_material: StandardMaterial3D = _make_env_material(Color(0.66, 0.49, 0.32), 0.58)
	var building_material: StandardMaterial3D = _make_env_material(Color(0.66, 0.69, 0.70), 0.75)
	var distant_building_material: StandardMaterial3D = _make_env_material(Color(0.78, 0.80, 0.80), 0.80)
	var tree_material: StandardMaterial3D = _make_env_material(Color(0.38, 0.58, 0.32), 0.82)
	var tree_light_material: StandardMaterial3D = _make_env_material(Color(0.50, 0.68, 0.40), 0.86)
	var handle_material: StandardMaterial3D = _make_env_material(Color(0.06, 0.055, 0.045), 0.42)

	_build_room_shell(root, wall_material, ceiling_material, floor_material, trim_material)
	_build_reference_desk(root, wood_material, wood_dark_material, wood_light_material, handle_material)
	_build_large_window(root, dark_frame_material, sky_material, glass_material)
	_build_outdoor_silhouette(root, building_material, distant_building_material, tree_material, tree_light_material)
	_add_daylight_fill(root)
	_add_world_environment()


func _build_room_shell(root: Node3D, wall_material: Material, ceiling_material: Material, floor_material: Material, trim_material: Material) -> void:
	# Taller and wider shell: the lamp no longer feels squeezed under a low ceiling.
	root.add_child(_box_mesh("BackWall", Vector3(8.6, 4.25, 0.10), Vector3(0.0, 1.52, 1.58), wall_material))
	root.add_child(_box_mesh("LeftWall", Vector3(0.10, 4.25, 4.65), Vector3(-4.25, 1.52, -0.42), wall_material))
	root.add_child(_box_mesh("Ceiling", Vector3(8.7, 0.08, 4.75), Vector3(0.0, 3.68, -0.42), ceiling_material))
	root.add_child(_box_mesh("Floor", Vector3(8.7, 0.08, 4.75), Vector3(0.0, -0.62, -0.42), floor_material))
	root.add_child(_box_mesh("BackBaseboard", Vector3(8.55, 0.08, 0.08), Vector3(0.0, -0.18, 1.50), trim_material))
	root.add_child(_box_mesh("LeftBaseboard", Vector3(0.09, 0.08, 4.65), Vector3(-4.19, -0.18, -0.42), trim_material))
	root.add_child(_box_mesh("LeftBackCornerTrim", Vector3(0.08, 4.15, 0.08), Vector3(-4.19, 1.52, 1.50), trim_material))


func _build_reference_desk(root: Node3D, wood_material: Material, wood_dark_material: Material, wood_light_material: Material, handle_material: Material) -> void:
	# Tabletop top surface is y=0.0 so the existing lamp rig can sit on it without
	# changing animation joint offsets. The desk is intentionally plain: no
	# procedural grain strips or texture overlays.
	root.add_child(_box_mesh("WideWoodTabletop", Vector3(6.55, 0.16, 2.75), Vector3(0.0, -0.08, -0.30), wood_material))
	root.add_child(_box_mesh("FrontLeftTableLeg", Vector3(0.22, 0.70, 0.22), Vector3(-2.95, -0.50, -1.48), wood_dark_material))
	root.add_child(_box_mesh("FrontRightTableLeg", Vector3(0.22, 0.70, 0.22), Vector3(2.95, -0.50, -1.48), wood_dark_material))
	root.add_child(_box_mesh("BackLeftTableLeg", Vector3(0.20, 0.64, 0.20), Vector3(-2.85, -0.48, 0.72), wood_dark_material))
	root.add_child(_box_mesh("BackRightTableLeg", Vector3(0.20, 0.64, 0.20), Vector3(2.85, -0.48, 0.72), wood_dark_material))
	root.add_child(_box_mesh("TableFrontThickEdge", Vector3(6.62, 0.20, 0.12), Vector3(0.0, -0.20, -1.70), wood_dark_material))
	root.add_child(_box_mesh("TableRightSideEdge", Vector3(0.14, 0.18, 2.76), Vector3(3.22, -0.19, -0.30), wood_dark_material))
	root.add_child(_box_mesh("LeftDeskSidePanel", Vector3(0.18, 0.86, 1.82), Vector3(-2.90, -0.58, -0.52), wood_dark_material))
	root.add_child(_box_mesh("RightDeskSidePanel", Vector3(0.18, 0.86, 1.70), Vector3(3.03, -0.58, -0.62), wood_dark_material))
	root.add_child(_box_mesh("RearSupportPanel", Vector3(5.80, 0.52, 0.12), Vector3(0.02, -0.58, 0.86), wood_dark_material))
	root.add_child(_box_mesh("RightDrawerBlock", Vector3(1.35, 0.56, 0.13), Vector3(1.70, -0.58, -1.50), wood_material))
	root.add_child(_box_mesh("RightDrawerTopLine", Vector3(1.32, 0.035, 0.145), Vector3(1.70, -0.31, -1.585), wood_light_material))
	root.add_child(_box_mesh("RightDrawerHandle", Vector3(0.38, 0.08, 0.04), Vector3(1.70, -0.58, -1.61), handle_material))


func _build_large_window(root: Node3D, frame_material: Material, sky_material: Material, glass_material: Material) -> void:
	# The window fills most of the back wall and stays behind the lamp, matching the
	# reference image scale ratio without importing any external texture assets.
	root.add_child(_box_mesh("WindowSkyPanel", Vector3(4.72, 2.26, 0.035), Vector3(0.17, 1.73, 1.235), sky_material))
	root.add_child(_box_mesh("WindowGlassOverlay", Vector3(4.68, 2.20, 0.020), Vector3(0.17, 1.73, 1.205), glass_material))
	root.add_child(_box_mesh("WindowTopFrame", Vector3(5.08, 0.12, 0.13), Vector3(0.17, 2.92, 1.185), frame_material))
	root.add_child(_box_mesh("WindowBottomFrame", Vector3(5.08, 0.12, 0.13), Vector3(0.17, 0.54, 1.185), frame_material))
	root.add_child(_box_mesh("WindowLeftFrame", Vector3(0.13, 2.48, 0.13), Vector3(-2.43, 1.73, 1.185), frame_material))
	root.add_child(_box_mesh("WindowRightFrame", Vector3(0.13, 2.48, 0.13), Vector3(2.77, 1.73, 1.185), frame_material))
	root.add_child(_box_mesh("WindowInnerTopShadow", Vector3(4.82, 0.045, 0.10), Vector3(0.17, 2.73, 1.145), frame_material))
	root.add_child(_box_mesh("WindowSill", Vector3(5.18, 0.10, 0.28), Vector3(0.17, 0.44, 1.06), frame_material))


func _build_outdoor_silhouette(root: Node3D, building_material: Material, distant_building_material: Material, tree_material: Material, tree_light_material: Material) -> void:
	# Stylized flat scenery placed just in front of the sky panel. It reads as city
	# and tree silhouettes through the large window while remaining cheap to render.
	root.add_child(_box_mesh("CityBlock_00", Vector3(0.30, 0.34, 0.045), Vector3(-2.02, 0.90, 1.165), distant_building_material))
	root.add_child(_box_mesh("CityBlock_01", Vector3(0.24, 0.56, 0.045), Vector3(-1.62, 1.00, 1.165), building_material))
	root.add_child(_box_mesh("CityBlock_02", Vector3(0.38, 0.28, 0.045), Vector3(-1.15, 0.86, 1.165), distant_building_material))
	root.add_child(_box_mesh("CityBlock_03", Vector3(0.26, 0.44, 0.045), Vector3(-0.57, 0.95, 1.165), building_material))
	root.add_child(_box_mesh("CityBlock_04", Vector3(0.42, 0.25, 0.045), Vector3(-0.10, 0.84, 1.165), distant_building_material))
	root.add_child(_box_mesh("CityBlock_05", Vector3(0.32, 0.48, 0.045), Vector3(0.43, 0.96, 1.165), building_material))
	root.add_child(_box_mesh("CityBlock_06", Vector3(0.24, 0.32, 0.045), Vector3(0.95, 0.88, 1.165), distant_building_material))
	root.add_child(_box_mesh("CityBlock_07", Vector3(0.36, 0.40, 0.045), Vector3(1.44, 0.93, 1.165), building_material))
	root.add_child(_box_mesh("CityBlock_08", Vector3(0.32, 0.27, 0.045), Vector3(1.94, 0.86, 1.165), distant_building_material))
	root.add_child(_box_mesh("CityBlock_09", Vector3(0.26, 0.38, 0.045), Vector3(2.35, 0.92, 1.165), building_material))
	root.add_child(_box_mesh("TreeBandBack", Vector3(4.45, 0.16, 0.045), Vector3(0.23, 0.69, 1.145), tree_light_material))
	root.add_child(_box_mesh("TreeBandFront", Vector3(4.55, 0.12, 0.050), Vector3(0.22, 0.61, 1.125), tree_material))
	_add_window_tree(root, "TreeBlob_00", -2.05, 0.73, 0.16, Vector3(1.10, 0.70, 0.25), tree_material)
	_add_window_tree(root, "TreeBlob_01", -1.64, 0.72, 0.15, Vector3(1.00, 0.66, 0.25), tree_light_material)
	_add_window_tree(root, "TreeBlob_02", -1.18, 0.74, 0.17, Vector3(1.20, 0.72, 0.25), tree_material)
	_add_window_tree(root, "TreeBlob_03", -0.66, 0.70, 0.14, Vector3(1.00, 0.68, 0.25), tree_light_material)
	_add_window_tree(root, "TreeBlob_04", -0.12, 0.73, 0.17, Vector3(1.22, 0.72, 0.25), tree_material)
	_add_window_tree(root, "TreeBlob_05", 0.42, 0.70, 0.14, Vector3(1.00, 0.65, 0.25), tree_light_material)
	_add_window_tree(root, "TreeBlob_06", 0.94, 0.73, 0.16, Vector3(1.18, 0.70, 0.25), tree_material)
	_add_window_tree(root, "TreeBlob_07", 1.46, 0.71, 0.15, Vector3(1.05, 0.66, 0.25), tree_light_material)
	_add_window_tree(root, "TreeBlob_08", 2.00, 0.73, 0.16, Vector3(1.15, 0.68, 0.25), tree_material)
	_add_window_tree(root, "TreeBlob_09", 2.42, 0.69, 0.13, Vector3(1.00, 0.65, 0.25), tree_light_material)


func _add_window_tree(root: Node3D, node_name: String, x_pos: float, y_pos: float, radius: float, scale_vec: Vector3, material: Material) -> void:
	var tree: MeshInstance3D = _low_poly_sphere_mesh(node_name, radius, Vector3(x_pos, y_pos, 1.105), scale_vec, material)
	root.add_child(tree)


func _add_daylight_fill(root: Node3D) -> void:
	var window_light: OmniLight3D = OmniLight3D.new()
	window_light.name = "WindowSoftDaylight"
	window_light.position = Vector3(-0.35, 2.16, 0.88)
	window_light.light_energy = 0.42
	window_light.omni_range = 4.2
	root.add_child(window_light)

	var tabletop_fill: OmniLight3D = OmniLight3D.new()
	tabletop_fill.name = "TabletopWarmBounce"
	tabletop_fill.position = Vector3(1.25, 0.62, -1.25)
	tabletop_fill.light_color = Color(1.0, 0.86, 0.68)
	tabletop_fill.light_energy = 0.18
	tabletop_fill.omni_range = 3.5
	root.add_child(tabletop_fill)


func _add_world_environment() -> void:
	if has_node("SoftAmbientWorld"):
		return
	var world: WorldEnvironment = WorldEnvironment.new()
	world.name = "SoftAmbientWorld"
	var env: Environment = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.78, 0.83, 0.88)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.82, 0.80, 0.74)
	env.ambient_light_energy = 0.44
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.tonemap_exposure = 1.0
	env.tonemap_white = 1.0
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
	lines.append("LeLamp Milestone 4.3.5 Frontend Polish")
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


func _box_mesh(node_name: String, size: Vector3, position: Vector3, material: Material) -> MeshInstance3D:
	var instance: MeshInstance3D = MeshInstance3D.new()
	instance.name = node_name
	var mesh: BoxMesh = BoxMesh.new()
	mesh.size = size
	instance.mesh = mesh
	instance.position = position
	instance.material_override = material
	return instance


func _low_poly_sphere_mesh(node_name: String, radius: float, position: Vector3, scale_vec: Vector3, material: Material) -> MeshInstance3D:
	var instance: MeshInstance3D = MeshInstance3D.new()
	instance.name = node_name
	var mesh: SphereMesh = SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.radial_segments = 8
	mesh.rings = 4
	instance.mesh = mesh
	instance.position = position
	instance.scale = scale_vec
	instance.material_override = material
	return instance


func _make_env_material(color: Color, roughness: float) -> StandardMaterial3D:
	var mat: StandardMaterial3D = StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = clampf(roughness, 0.0, 1.0)
	mat.metallic = 0.0
	return mat


func _make_translucent_env_material(color: Color, roughness: float) -> StandardMaterial3D:
	var mat: StandardMaterial3D = StandardMaterial3D.new()
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
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
