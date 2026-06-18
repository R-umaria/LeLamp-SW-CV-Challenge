extends Node

signal command_received(command: Dictionary)
signal receiver_status_changed(message: String)

@export var listen_address: String = "0.0.0.0"
@export var listen_port: int = 4242
@export var autostart: bool = true
@export var max_packets_per_frame: int = 8

var _udp := PacketPeerUDP.new()
var _is_listening := false
var last_packet_wall_time: float = 0.0
var last_packet_protocol_timestamp: String = "never"
var last_error: String = ""


func _ready() -> void:
    if autostart:
        start()


func start() -> void:
    if _is_listening:
        return

    var err := _udp.bind(listen_port, listen_address)
    if err != OK:
        last_error = "UDP bind failed: %s" % error_string(err)
        push_error(last_error)
        receiver_status_changed.emit(last_error)
        return

    _is_listening = true
    var message := "Listening on udp://%s:%d" % [listen_address, listen_port]
    print(message)
    receiver_status_changed.emit(message)


func stop() -> void:
    if not _is_listening:
        return
    _udp.close()
    _is_listening = false
    receiver_status_changed.emit("UDP receiver stopped")


func _exit_tree() -> void:
    stop()


func _process(_delta: float) -> void:
    if not _is_listening:
        return

    var packets_read := 0
    while _udp.get_available_packet_count() > 0 and packets_read < max_packets_per_frame:
        packets_read += 1
        var packet := _udp.get_packet()
        var text := packet.get_string_from_utf8()
        _handle_packet_text(text)


func _handle_packet_text(text: String) -> void:
    var parsed: Variant = JSON.parse_string(text)
    if typeof(parsed) != TYPE_DICTIONARY:
        last_error = "Ignored malformed JSON packet"
        push_warning(last_error + ": " + text.left(120))
        return

    var command: Dictionary = parsed
    last_packet_wall_time = Time.get_unix_time_from_system()
    last_packet_protocol_timestamp = str(command.get("timestamp", "missing"))
    command_received.emit(command)
