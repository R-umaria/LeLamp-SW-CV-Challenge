"""Lumos backend vertical slice with Milestone 4 grounded recall.

Run from the project root with:
    python -m backend.main --show-window

Milestone 4 preserves the stable engagement detector, object memory, isolated
logging, command JSON shape, and Godot UDP bridge. It adds optional interactive
text recall as a bounded side channel: the user can type a question, Python
retrieves SQLite memory, and Godot receives a controlled ``recalling`` command.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.behavior.behavior_policy import behavior_for_transition, behavior_with_gesture_override
from backend.behavior.command_protocol import build_behavior_command
from backend.behavior.recall_feedback import behavior_for_recall_result, recall_target_for_result
from backend.behavior.godot_udp_sender import GodotUdpSender
from backend.behavior.speaker_policy import SpeakerPolicyDecision, apply_speaker_policy
from backend.audio.audio_capture import AudioCaptureWorker
from backend.audio.gcc_phat import DirectionOfArrivalResult, estimate_direction_of_arrival
from backend.audio.voice_activity_detector import VoiceActivityDetector, VoiceActivityResult
from backend.behavior.state_machine import InteractionStateMachine, LampState
from backend.conversation.chat_udp_receiver import ChatUdpReceiver
from backend.conversation.recall_agent import RecallAgent
from backend.conversation.recall_worker import RecallWorker, RecallWorkItem, RecallWorkResult
from backend.conversation.web_chat_server import WebChatRequest, WebChatServer
from backend.evaluation.latency_logger import LatencyLogger
from backend.memory.scene_memory import SceneMemory
from backend.perception.camera import OpenCVCamera
from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.perception.engagement_detector import FaceEngagementDetector
from backend.perception.speaker_awareness_worker import SpeakerAwarenessWorker
from backend.perception.gesture_detector import HandGestureDetector, HandGestureResult
from backend.perception.object_detector import YoloObjectDetector
from backend.perception.temporal_smoother import EngagementSmoother
from backend.utils.config import (
    AudioConfig,
    CameraConfig,
    DirectionOfArrivalConfig,
    EngagementConfig,
    GodotUdpConfig,
    HandGestureConfig,
    MemoryConfig,
    ObjectDetectionConfig,
    SpeakerAwarenessConfig,
    RuntimeConfig,
    SmoothingConfig,
    StateMachineConfig,
)
from backend.utils.logging_utils import setup_logging
from backend.utils.preview_window import PreviewWindow, PreviewWindowConfig
from backend.utils.run_paths import create_run_paths, write_latest_pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lumos backend: engagement + object memory + grounded recall")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seek-after", type=float, default=5.0)
    parser.add_argument("--emit-interval", type=float, default=1.0)
    parser.add_argument("--godot-udp", action="store_true", help="Enable best-effort UDP command streaming to the Godot frontend.")
    parser.add_argument("--godot-host", type=str, default="127.0.0.1", help="Godot UDP host. Use 127.0.0.1 for local demo.")
    parser.add_argument("--godot-port", type=int, default=4242, help="Godot UDP listen port.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Root log directory. Each run writes under <log-dir>/runs/<run_id>/")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id. Defaults to timestamp YYYY-MM-DD_HH-MM-SS.")
    parser.add_argument("--no-latest", action="store_true", help="Do not mirror this run into logs/latest.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Esc/Ctrl-C")

    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--min-state-dwell", type=float, default=0.95)
    parser.add_argument("--exit-disengaged-frames", type=int, default=9)
    parser.add_argument("--exit-absent-frames", type=int, default=12)
    parser.add_argument("--engaged-recovery-frames", type=int, default=2)
    parser.add_argument("--clear-engaged-confidence", type=float, default=0.72)

    parser.add_argument("--center-tolerance-x", type=float, default=0.46)
    parser.add_argument("--center-tolerance-y", type=float, default=0.44)
    parser.add_argument("--min-face-area-ratio", type=float, default=0.018)
    parser.add_argument("--min-candidate-area-ratio", type=float, default=0.006)
    parser.add_argument("--cascade-min-neighbors", type=int, default=6)

    parser.add_argument("--enable-objects", action="store_true", help="Enable optional YOLO object detection and scene-memory writes.")
    parser.add_argument("--object-model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model path/name, e.g. yolov8n.pt")
    parser.add_argument("--object-interval", type=float, default=2.0, help="Seconds between object-detection passes.")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path.")
    parser.add_argument("--save-object-frames", action="store_true", help="Save annotated evidence frames when a memory record is written.")
    parser.add_argument("--object-confidence", type=float, default=0.35, help="YOLO confidence threshold for object detection.")
    parser.add_argument("--memory-dedupe-window", type=float, default=8.0, help="Seconds to suppress repeated same-object/same-location memory writes.")
    parser.add_argument("--object-frame-dir", type=str, default="data/object_frames", help="Directory for saved object evidence frames.")

    parser.add_argument("--enable-gestures", action="store_true", help="Enable MediaPipe hand gestures: index beckon moves Lumos closer; open palm moves it away.")
    parser.add_argument("--gesture-interval", type=float, default=0.10, help="Seconds between hand-gesture detection passes.")
    parser.add_argument("--gesture-confidence", type=float, default=0.64, help="Minimum gesture confidence required to override the normal motion skill.")
    parser.add_argument("--gesture-hold", type=float, default=1.15, help="Seconds to hold the last gesture command after a brief hand landmark dropout.")

    parser.add_argument("--enable-audio", action="store_true", help="Enable non-blocking microphone capture for speaker awareness.")
    parser.add_argument("--list-audio-devices", action="store_true", help="List sounddevice input devices and exit. Use this to choose --audio-device.")
    parser.add_argument("--enable-speaker-awareness", action="store_true", help="Enable active speaker awareness fusion and behavior overrides.")
    parser.add_argument("--audio-device", type=str, default=None, help="Optional sounddevice input device id/name. Defaults to system input.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000, help="Audio sample rate for VAD/DOA.")
    parser.add_argument("--audio-block-ms", type=int, default=30, help="Microphone block size in milliseconds.")
    parser.add_argument("--vad-energy-threshold", type=float, default=2.4, help="RMS multiplier over rolling noise floor for speech activity.")
    parser.add_argument("--enable-doa", action="store_true", help="Enable stereo GCC-PHAT direction of arrival when stereo input is available.")
    parser.add_argument("--mic-distance-m", type=float, default=0.08, help="Distance between left/right microphones in meters for DOA.")
    parser.add_argument("--speaker-fusion-interval", type=float, default=0.25, help="Seconds between active-speaker fusion passes. Runs in a worker thread by default.")
    parser.add_argument("--speaker-frame-width", type=int, default=640, help="Max frame width for speaker face tracking worker. Lower improves preview FPS.")
    parser.add_argument("--speaker-debug", action="store_true", help="Log detailed face-track and speaker-fusion diagnostics.")

    parser.add_argument("--interactive-recall", action="store_true", help="Debug only: allow terminal recall questions while the backend is running.")
    parser.add_argument("--enable-godot-chat", action="store_true", help="Legacy/non-primary: listen for live chat queries from Godot over local UDP.")
    parser.add_argument("--chat-host", type=str, default="127.0.0.1", help="Backend UDP host for Godot chat messages.")
    parser.add_argument("--chat-port", type=int, default=4243, help="Backend UDP port for Godot chat messages.")
    parser.add_argument("--enable-web-chat", action="store_true", help="Enable browser-based recall chat served by Python.")
    parser.add_argument("--web-chat-host", type=str, default="127.0.0.1", help="Browser chat HTTP host.")
    parser.add_argument("--web-chat-port", type=int, default=8765, help="Browser chat HTTP port.")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase retrieved memory answers.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL for --use-llm.")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:1.5b", help="Ollama model for recall phrasing, e.g. qwen2.5:1.5b.")
    parser.add_argument("--llm-timeout", type=float, default=90.0, help="Ollama full chat/read timeout in seconds for recall phrasing.")
    parser.add_argument("--llm-connect-timeout", type=float, default=5.0, help="Ollama connection/status timeout in seconds.")
    parser.add_argument("--llm-max-tokens", type=int, default=120, help="Maximum Ollama output tokens for recall phrasing.")
    parser.add_argument("--ollama-keep-alive", type=str, default="1h", help="Ollama keep_alive duration, e.g. 1h.")
    parser.add_argument("--ollama-timeout", type=float, default=None, help="Backward-compatible alias for --llm-timeout.")
    parser.add_argument("--llm-required", action="store_true", help="Fail fast at startup if Ollama/model is unavailable; live answers still fall back but log failures.")
    parser.add_argument("--recall-lookback-hours", type=float, default=24.0, help="Only answer object-location questions from memories within this many hours. Use 0 for all-time recall.")
    parser.add_argument("--recall-point-hold-seconds", type=float, default=5.0, help="Seconds to keep Lumos pointing/sad after a completed recall before returning to attentive tracking.")

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--show-window", action="store_true", default=True)
    window_group.add_argument("--no-window", action="store_false", dest="show_window")
    parser.add_argument("--preview-scale", type=float, default=0.50, help="Scale factor for the OpenCV preview window when --show-window is enabled. Default 0.50 gives about 320x240 for a 640x480 camera.")
    parser.add_argument("--preview-width", type=int, default=0, help="Optional exact preview width in pixels. If set without --preview-height, aspect ratio is preserved.")
    parser.add_argument("--preview-height", type=int, default=0, help="Optional exact preview height in pixels. If set without --preview-width, aspect ratio is preserved.")
    parser.add_argument("--preview-x", type=int, default=24, help="Initial screen x-position for the OpenCV preview window.")
    parser.add_argument("--preview-y", type=int, default=24, help="Initial screen y-position for the OpenCV preview window.")
    parser.add_argument("--preview-flip-horizontal", action="store_true", help="Flip only the displayed preview horizontally. This fixes a mirrored/reversed preview without changing perception, memory, or Godot face-follow coordinates.")
    return parser.parse_args()


def append_jsonl(paths: Path | list[Path] | tuple[Path, ...], payload: dict) -> None:
    if isinstance(paths, Path):
        output_paths = [paths]
    else:
        output_paths = list(paths)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")




def start_recall_input_thread(
    recall_queue: "queue.Queue[object]",
    stop_event: threading.Event,
    logger,
) -> threading.Thread:
    """Start a tiny stdin reader so camera/perception loop stays non-blocking."""

    def _worker() -> None:
        print("Interactive recall is for debugging only. Final demo input should use http://127.0.0.1:8765.", flush=True)
        print("Do not paste PowerShell commands into this prompt; command-looking lines will be ignored.", flush=True)
        print("Type a question such as: Where is the cup? Type :q, quit, or exit to stop.", flush=True)
        while not stop_event.is_set():
            try:
                line = input("recall> ")
            except EOFError:
                logger.info("Interactive recall input closed")
                break
            except Exception as exc:  # pragma: no cover - terminal/runtime guard
                logger.warning("Interactive recall input stopped: %s", exc)
                break

            query = line.strip()
            if not query:
                continue
            if query.lower() in {":q", "quit", "exit"}:
                logger.info("Interactive recall input stop requested")
                stop_event.set()
                break
            if looks_like_shell_command(query):
                logger.warning("Ignoring terminal recall input that looks like a shell command: %r", query)
                print("Ignored shell-looking line. Use the browser chat for demo questions.", flush=True)
                continue
            recall_queue.put(query)

    thread = threading.Thread(target=_worker, name="interactive-recall-input", daemon=True)
    thread.start()
    return thread


def looks_like_shell_command(text: str) -> bool:
    """Avoid treating pasted PowerShell/script lines as recall questions."""

    stripped = text.strip()
    lowered = stripped.lower()
    shell_prefixes = (
        "python ",
        "python3 ",
        "py ",
        "pip ",
        "ollama ",
        "curl ",
        "irm ",
        "invoke-webrequest",
        "invoke-restmethod",
        "cd ",
        "set-location",
        "mkdir ",
        "copy ",
        "move ",
        "del ",
        "dir",
        "ls",
    )
    if lowered.startswith(shell_prefixes):
        return True
    if lowered.startswith(("--", "$", ".\\", "./", "#")):
        return True
    if "`" in stripped and not stripped.endswith("?"):
        return True
    return False


def build_configs(
    args: argparse.Namespace,
) -> tuple[
    CameraConfig,
    EngagementConfig,
    SmoothingConfig,
    StateMachineConfig,
    RuntimeConfig,
    GodotUdpConfig,
    ObjectDetectionConfig,
    HandGestureConfig,
    MemoryConfig,
    AudioConfig,
    DirectionOfArrivalConfig,
    SpeakerAwarenessConfig,
]:
    camera_config = CameraConfig(index=args.camera_index, width=args.width, height=args.height)
    engagement_config = EngagementConfig(
        center_tolerance_x=args.center_tolerance_x,
        center_tolerance_y=args.center_tolerance_y,
        min_face_area_ratio=args.min_face_area_ratio,
        min_candidate_area_ratio=args.min_candidate_area_ratio,
        cascade_min_neighbors=args.cascade_min_neighbors,
    )
    smoothing_config = SmoothingConfig(
        window_size=args.smoothing_window,
        clear_engaged_confidence=args.clear_engaged_confidence,
    )
    state_config = StateMachineConfig(
        seek_attention_after_s=args.seek_after,
        min_state_dwell_s=args.min_state_dwell,
        exit_engaged_disengaged_frames=args.exit_disengaged_frames,
        exit_engaged_absent_frames=args.exit_absent_frames,
        engaged_recovery_frames=args.engaged_recovery_frames,
        clear_engaged_confidence=args.clear_engaged_confidence,
    )
    runtime_config = RuntimeConfig(
        command_emit_interval_s=args.emit_interval,
        log_dir=args.log_dir,
        show_window=args.show_window,
    )
    godot_udp_config = GodotUdpConfig(
        enabled=args.godot_udp,
        host=args.godot_host,
        port=args.godot_port,
    )
    object_config = ObjectDetectionConfig(
        enabled=args.enable_objects,
        model_path=args.object_model,
        interval_s=max(0.1, args.object_interval),
        confidence=args.object_confidence,
    )
    gesture_config = HandGestureConfig(
        enabled=args.enable_gestures,
        interval_s=max(0.04, args.gesture_interval),
        min_gesture_confidence=args.gesture_confidence,
        hold_s=max(0.0, args.gesture_hold),
    )
    memory_config = MemoryConfig(
        db_path=args.memory_db,
        dedupe_window_s=max(0.0, args.memory_dedupe_window),
        save_object_frames=args.save_object_frames,
        frame_dir=args.object_frame_dir,
    )
    audio_config = AudioConfig(
        enabled=args.enable_audio,
        device=args.audio_device,
        sample_rate=max(8000, int(args.audio_sample_rate)),
        block_ms=max(10, int(args.audio_block_ms)),
        request_stereo=bool(args.enable_doa),
        vad_energy_threshold=max(1.05, float(args.vad_energy_threshold)),
    )
    doa_config = DirectionOfArrivalConfig(
        enabled=args.enable_doa,
        mic_distance_m=max(0.0, float(args.mic_distance_m)),
    )
    speaker_config = SpeakerAwarenessConfig(
        enabled=args.enable_speaker_awareness,
        debug=args.speaker_debug,
        fusion_interval_s=max(0.05, float(args.speaker_fusion_interval)),
        worker_frame_width=max(160, int(args.speaker_frame_width)),
    )
    return (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        gesture_config,
        memory_config,
        audio_config,
        doa_config,
        speaker_config,
    )


def main() -> int:
    args = parse_args()
    if args.list_audio_devices:
        ok, message = AudioCaptureWorker.list_input_devices()
        print(message)
        return 0 if ok else 2
    (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        gesture_config,
        memory_config,
        audio_config,
        doa_config,
        speaker_config,
    ) = build_configs(args)

    run_paths = create_run_paths(
        log_root=runtime_config.log_dir,
        run_id=args.run_id,
        mirror_latest=not args.no_latest,
    )
    write_latest_pointer(run_paths.log_root, run_paths.run_dir)

    latest_latency_path = None if args.no_latest else run_paths.latest_latency_path
    latest_commands_path = None if args.no_latest else run_paths.latest_commands_path
    latest_runtime_log_paths = [] if args.no_latest else [run_paths.latest_runtime_log_path]

    logger = setup_logging(run_paths.run_dir, extra_runtime_log_paths=latest_runtime_log_paths)
    latency_paths = [run_paths.latency_path] + ([latest_latency_path] if latest_latency_path else [])
    latency_logger = LatencyLogger(latency_paths)
    command_paths = [run_paths.commands_path] + ([latest_commands_path] if latest_commands_path else [])
    commands_path = run_paths.commands_path
    speaker_event_paths = [run_paths.run_dir / "speaker_events.jsonl"]
    if not args.no_latest:
        speaker_event_paths.append(run_paths.latest_dir / "speaker_events.jsonl")
    godot_sender = GodotUdpSender(godot_udp_config, logger=logger)

    camera = OpenCVCamera(camera_config.index, camera_config.width, camera_config.height)
    detector = FaceEngagementDetector(engagement_config)
    smoother = EngagementSmoother(smoothing_config)
    fsm = InteractionStateMachine(state_config)
    object_detector = YoloObjectDetector(object_config, logger=logger)
    gesture_detector = HandGestureDetector(gesture_config, logger=logger)
    scene_memory = SceneMemory(memory_config, logger=logger)
    audio_capture = AudioCaptureWorker(audio_config, logger=logger)
    voice_detector = VoiceActivityDetector(audio_config)
    speaker_worker = SpeakerAwarenessWorker(
        engagement_config,
        speaker_config,
        logger=logger,
        event_paths=speaker_event_paths,
    )
    preview_window = PreviewWindow(
        PreviewWindowConfig(
            title="Lumos - CV Preview",
            scale=args.preview_scale,
            width=args.preview_width,
            height=args.preview_height,
            x=args.preview_x,
            y=args.preview_y,
            flip_horizontal=args.preview_flip_horizontal,
        ),
        logger=logger,
    )

    llm_timeout = float(args.ollama_timeout) if args.ollama_timeout is not None else float(args.llm_timeout)

    recall_agent = None
    recall_queue: queue.Queue[object] | None = None
    recall_result_queue: queue.Queue[RecallWorkResult] | None = None
    recall_worker: RecallWorker | None = None
    pending_recalls: dict[str, dict] = {}
    recall_stop_event = threading.Event()
    if args.interactive_recall or args.enable_godot_chat or args.enable_web_chat:
        recall_log_paths = [run_paths.run_dir / "recall.jsonl"]
        if not args.no_latest:
            recall_log_paths.append(run_paths.latest_dir / "recall.jsonl")
        recall_agent = RecallAgent(
            memory_db=args.memory_db,
            use_llm=args.use_llm,
            llm_required=args.llm_required,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            llm_timeout_s=llm_timeout,
            llm_connect_timeout_s=args.llm_connect_timeout,
            ollama_keep_alive=args.ollama_keep_alive,
            llm_max_tokens=args.llm_max_tokens,
            recall_lookback_hours=None if args.recall_lookback_hours <= 0 else args.recall_lookback_hours,
            log_paths=recall_log_paths,
            logger=logger,
        )
        recall_queue = queue.Queue()
        recall_result_queue = queue.Queue()
        recall_worker = RecallWorker(recall_agent, recall_result_queue, logger=logger)
        recall_worker.start()
        if args.interactive_recall:
            start_recall_input_thread(recall_queue, recall_stop_event, logger)

    web_chat_server = None
    if args.enable_web_chat:
        if recall_queue is None:
            recall_queue = queue.Queue()
        web_chat_server = WebChatServer(
            output_queue=recall_queue,
            host=args.web_chat_host,
            port=args.web_chat_port,
            request_timeout_s=max(30.0, llm_timeout + 15.0),
            logger=logger,
        )
        web_chat_server.start()

    chat_receiver = None
    if args.enable_godot_chat:
        if recall_queue is None:
            recall_queue = queue.Queue()
        chat_receiver = ChatUdpReceiver(
            output_queue=recall_queue,
            host=args.chat_host,
            port=args.chat_port,
            logger=logger,
        )
        chat_receiver.start()

    if args.use_llm and args.llm_required and recall_agent is not None:
        status = recall_agent.ollama_status
        if status is None or not (status.ok and status.model_available):
            logger.error(
                "--llm-required startup check failed url=%s model=%s error=%s",
                args.ollama_url,
                args.ollama_model,
                None if status is None else status.error,
            )
            return 1

    logger.info("Starting Milestone 4.3.3 backend with expressive Godot polish, optional face-follow hints, non-blocking browser chat, and homelab Ollama support")
    logger.info("Run id=%s", run_paths.run_id)
    logger.info("Run directory=%s", run_paths.run_dir)
    if not args.no_latest:
        logger.info("Latest mirror directory=%s", run_paths.latest_dir)
    logger.info("Camera index=%s size=%sx%s", camera_config.index, camera_config.width, camera_config.height)
    if runtime_config.show_window:
        logger.info(
            "Preview config scale=%.2f width=%s height=%s position=(%s,%s) flip_horizontal=%s",
            args.preview_scale,
            args.preview_width,
            args.preview_height,
            args.preview_x,
            args.preview_y,
            args.preview_flip_horizontal,
        )
    logger.info(
        "Stability config smoothing_window=%s min_dwell=%.2fs exit_disengaged_frames=%s exit_absent_frames=%s min_face_area=%.3f min_candidate_area=%.3f",
        smoothing_config.window_size,
        state_config.min_state_dwell_s,
        state_config.exit_engaged_disengaged_frames,
        state_config.exit_engaged_absent_frames,
        engagement_config.min_face_area_ratio,
        engagement_config.min_candidate_area_ratio,
    )
    logger.info(
        "Object config enabled=%s active=%s model=%s interval=%.2fs confidence=%.2f",
        object_config.enabled,
        object_detector.enabled,
        object_config.model_path,
        object_config.interval_s,
        object_config.confidence,
    )
    logger.info(
        "Gesture config enabled=%s active=%s interval=%.2fs confidence=%.2f hold=%.2fs",
        gesture_config.enabled,
        gesture_detector.enabled,
        gesture_config.interval_s,
        gesture_config.min_gesture_confidence,
        gesture_config.hold_s,
    )
    logger.info(
        "Memory config db=%s save_frames=%s dedupe_window=%.1fs",
        memory_config.db_path,
        memory_config.save_object_frames,
        memory_config.dedupe_window_s,
    )
    logger.info("Commands will be saved to %s", commands_path)
    logger.info(
        "Recall config interactive=%s godot_chat=%s web_chat=%s web_url=http://%s:%s chat_udp=%s:%s use_llm=%s llm_required=%s ollama_url=%s ollama_model=%s llm_timeout=%.1fs connect_timeout=%.1fs llm_max_tokens=%s keep_alive=%s",
        args.interactive_recall,
        args.enable_godot_chat,
        args.enable_web_chat,
        args.web_chat_host,
        args.web_chat_port,
        args.chat_host,
        args.chat_port,
        args.use_llm,
        args.llm_required,
        args.ollama_url,
        args.ollama_model,
        llm_timeout,
        args.llm_connect_timeout,
        args.llm_max_tokens,
        args.ollama_keep_alive,
    )
    logger.info(
        "Speaker awareness config audio_enabled=%s speaker_enabled=%s doa_enabled=%s device=%s sample_rate=%s block_ms=%s vad_threshold=%.2f mic_distance_m=%.3f fusion_interval=%.3f worker_frame_width=%s",
        audio_config.enabled,
        speaker_config.enabled,
        doa_config.enabled,
        audio_config.device or "default",
        audio_config.sample_rate,
        audio_config.block_ms,
        audio_config.vad_energy_threshold,
        doa_config.mic_distance_m,
        speaker_config.fusion_interval_s,
        speaker_config.worker_frame_width,
    )
    if speaker_config.enabled and not audio_config.enabled:
        reason_payload = {"event": "speaker_awareness_disabled_reason", "reason": "enable_speaker_awareness_without_enable_audio"}
        append_jsonl(speaker_event_paths, reason_payload)
        logger.warning("speaker_awareness_disabled_reason reason=%s", reason_payload["reason"])
    if godot_udp_config.enabled:
        logger.info("Commands will also be streamed to Godot via udp://%s:%s", godot_udp_config.host, godot_udp_config.port)

    frame_count = 0
    last_emit_at = 0.0
    last_object_detection_at = 0.0
    last_gesture_detection_at = 0.0
    last_audio_sequence = -1
    last_speaker_snapshot_sequence = -1
    last_speaker_submit_at = 0.0
    last_voice_log_at = 0.0
    last_voice_signature = None
    last_doa_log_at = 0.0
    last_doa_signature = None
    last_voice_result: VoiceActivityResult | None = None
    last_doa_result: DirectionOfArrivalResult | None = None
    last_speaker_result: ActiveSpeakerResult | None = None
    speaker_policy_decision = SpeakerPolicyDecision({}, False, "not_evaluated")
    gesture_result = HandGestureResult("unavailable", 0.0, "gesture_detection_disabled")
    fps_ema = 0.0
    last_godot_udp_send_ms = None
    last_detected_objects: list[dict] = []
    active_recall_feedback: dict | None = None
    display_detections = []

    try:
        audio_started = False
        if audio_config.enabled:
            audio_started = audio_capture.start()
        if speaker_config.enabled and audio_config.enabled and audio_started:
            speaker_worker.start()
        elif speaker_config.enabled and audio_config.enabled and not audio_started:
            append_jsonl(
                speaker_event_paths,
                {"event": "speaker_awareness_disabled_reason", "reason": audio_capture.disabled_reason or "audio_capture_unavailable"},
            )
        camera.open()
        while True:
            loop_start = time.perf_counter()

            camera_frame = camera.read()
            frame = camera_frame.frame

            t0 = time.perf_counter()
            raw_engagement = detector.detect(frame)
            engagement_ms = (time.perf_counter() - t0) * 1000.0

            object_detection_ms = ""
            gesture_detection_ms = ""
            audio_capture_ms = ""
            vad_ms = ""
            doa_ms = ""
            active_speaker_fusion_ms = ""
            speaker_policy_ms = ""
            memory_write_ms = ""
            memory_retrieval_ms = ""
            llm_response_ms = ""
            llm_attempted = ""
            llm_used = ""
            llm_error = ""
            llm_fallback_reason = ""
            recall_worker_ms = ""
            memory_write_count = 0
            memory_duplicate_skip_count = 0
            now = time.monotonic()

            if audio_capture.available:
                latest_audio = audio_capture.get_latest()
                if latest_audio is not None:
                    audio_capture_ms = round(float(latest_audio.capture_ms), 3)
                    if latest_audio.sequence != last_audio_sequence:
                        last_audio_sequence = latest_audio.sequence
                        t_audio = time.perf_counter()
                        last_voice_result = voice_detector.update(latest_audio)
                        vad_ms = round((time.perf_counter() - t_audio) * 1000.0, 3)
                        voice_signature = bool(last_voice_result.is_speech)
                        should_log_voice = (
                            voice_signature != last_voice_signature
                            or last_voice_result.is_speech
                            or (now - last_voice_log_at) >= 0.75
                        )
                        if should_log_voice:
                            append_jsonl(
                                speaker_event_paths,
                                {"event": "voice_activity", **last_voice_result.to_log_dict()},
                            )
                            logger.info(
                                "voice_activity active=%s confidence=%.3f rms=%.6f noise_floor=%.6f channels=%s latency_ms=%s",
                                last_voice_result.is_speech,
                                last_voice_result.confidence,
                                last_voice_result.rms,
                                last_voice_result.noise_floor,
                                last_voice_result.channels,
                                vad_ms,
                            )
                            last_voice_log_at = now
                            last_voice_signature = voice_signature
                        if doa_config.enabled:
                            t_doa = time.perf_counter()
                            last_doa_result = estimate_direction_of_arrival(
                                latest_audio.samples,
                                latest_audio.sample_rate,
                                doa_config.mic_distance_m,
                                timestamp=latest_audio.timestamp,
                                min_rms=doa_config.min_rms,
                                min_confidence=doa_config.min_confidence,
                            )
                            doa_ms = round((time.perf_counter() - t_doa) * 1000.0, 3)
                            doa_signature = (last_doa_result.available, last_doa_result.reason)
                            should_log_doa = (
                                doa_signature != last_doa_signature
                                or last_doa_result.available
                                or (now - last_doa_log_at) >= 0.75
                            )
                            if should_log_doa:
                                append_jsonl(
                                    speaker_event_paths,
                                    {"event": "doa_result", **last_doa_result.to_log_dict()},
                                )
                                logger.info(
                                    "doa_result available=%s azimuth=%s confidence=%.3f reason=%s latency_ms=%s",
                                    last_doa_result.available,
                                    last_doa_result.azimuth_deg,
                                    last_doa_result.confidence,
                                    last_doa_result.reason,
                                    doa_ms,
                                )
                                last_doa_log_at = now
                                last_doa_signature = doa_signature

            should_detect_objects = object_detector.enabled and (
                last_object_detection_at == 0.0 or (now - last_object_detection_at >= object_config.interval_s)
            )
            if should_detect_objects:
                t0 = time.perf_counter()
                display_detections = object_detector.detect(frame)
                object_detection_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                last_object_detection_at = now

                if display_detections:
                    logger.info(
                        "Detected objects count=%s objects=%s latency_ms=%.3f",
                        len(display_detections),
                        [d.to_log_dict() for d in display_detections],
                        object_detection_ms,
                    )
                else:
                    logger.info("Detected objects count=0 latency_ms=%.3f", object_detection_ms)

                memory_result = scene_memory.observe(
                    display_detections,
                    frame=frame,
                    frame_index=frame_count,
                    source="webcam",
                )
                last_detected_objects = memory_result.command_objects
                memory_write_ms = round(memory_result.memory_write_ms, 3)
                memory_write_count = len(memory_result.written_records)
                memory_duplicate_skip_count = memory_result.skipped_duplicates
                logger.info(
                    "Object memory update detections=%s writes=%s duplicates=%s memory_write_ms=%.3f",
                    len(display_detections),
                    memory_write_count,
                    memory_duplicate_skip_count,
                    memory_result.memory_write_ms,
                )

            should_detect_gestures = gesture_config.enabled and (
                last_gesture_detection_at == 0.0 or (now - last_gesture_detection_at >= gesture_config.interval_s)
            )
            if should_detect_gestures:
                t0 = time.perf_counter()
                gesture_result = gesture_detector.detect(frame, now=now)
                gesture_detection_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                last_gesture_detection_at = now
                if gesture_result.status in {"beckon", "palm_push"}:
                    logger.info(
                        "Detected gesture status=%s confidence=%.3f reason=%s latency_ms=%.3f",
                        gesture_result.status,
                        gesture_result.confidence,
                        gesture_result.reason,
                        gesture_detection_ms,
                    )

            if speaker_config.enabled and audio_config.enabled and speaker_worker.enabled:
                if last_speaker_submit_at == 0.0 or (now - last_speaker_submit_at) >= speaker_config.fusion_interval_s:
                    speaker_worker.submit(
                        frame_sequence=frame_count,
                        frame=frame,
                        engagement=raw_engagement,
                        voice=last_voice_result,
                        doa=last_doa_result,
                    )
                    last_speaker_submit_at = now
                speaker_snapshot = speaker_worker.snapshot()
                if speaker_snapshot.result is not None:
                    last_speaker_result = speaker_snapshot.result
                    if speaker_snapshot.frame_sequence != last_speaker_snapshot_sequence:
                        active_speaker_fusion_ms = speaker_snapshot.fusion_ms
                        last_speaker_snapshot_sequence = speaker_snapshot.frame_sequence

            t0 = time.perf_counter()
            smoothed_engagement = smoother.update(raw_engagement)
            smoothing_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            transition = fsm.update(smoothed_engagement)
            state_machine_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = behavior_for_transition(transition)
            gesture_payload = gesture_result.to_protocol_dict()
            behavior = behavior_with_gesture_override(behavior, gesture_payload)
            speaker_payload = None if last_speaker_result is None else last_speaker_result.to_protocol_dict()
            t_policy = time.perf_counter()
            speaker_policy_decision = apply_speaker_policy(behavior, last_speaker_result, transition.current_state, speaker_config)
            speaker_policy_ms = round((time.perf_counter() - t_policy) * 1000.0, 3)
            behavior = speaker_policy_decision.behavior
            if speaker_policy_decision.overridden:
                append_jsonl(
                    speaker_event_paths,
                    {
                        "event": "speaker_behavior_override",
                        "reason": speaker_policy_decision.reason,
                        "state": transition.current_state.value,
                        "behavior": behavior,
                        "speaker": speaker_payload or {},
                    },
                )
                logger.info(
                    "speaker_behavior_override reason=%s state=%s motion=%s light=%s",
                    speaker_policy_decision.reason,
                    transition.current_state.value,
                    behavior.get("motion"),
                    behavior.get("light"),
                )
            command = build_behavior_command(
                state=transition.current_state,
                engagement=smoothed_engagement,
                behavior=behavior,
                last_detected_objects=last_detected_objects,
                gesture=gesture_payload,
                speaker=speaker_payload,
            )
            command_ms = (time.perf_counter() - t0) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            instantaneous_fps = 1000.0 / max(total_ms, 1e-6)
            fps_ema = instantaneous_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * instantaneous_fps)

            now = time.monotonic()
            should_emit = transition.changed or (now - last_emit_at >= runtime_config.command_emit_interval_s)

            if transition.changed:
                logger.info(
                    "State transition %s -> %s | reason=%s | engagement=%s conf=%.2f",
                    transition.previous_state.value,
                    transition.current_state.value,
                    transition.reason,
                    smoothed_engagement.status,
                    smoothed_engagement.confidence,
                )

            if recall_worker is not None and recall_queue is not None and recall_result_queue is not None:
                while True:
                    try:
                        recall_item = recall_queue.get_nowait()
                    except queue.Empty:
                        break

                    web_request = recall_item if isinstance(recall_item, WebChatRequest) else None
                    recall_query = web_request.text if web_request is not None else str(recall_item)
                    source = "browser" if web_request is not None else "terminal_or_udp"
                    request_id = web_request.request_id if web_request is not None else RecallWorkItem(text=recall_query, source=source).request_id
                    work_item = RecallWorkItem(text=recall_query, request_id=request_id, source=source)
                    active_recall_feedback = None
                    pending_recalls[request_id] = {"source": source, "text": recall_query, "queued_at": time.monotonic()}
                    if web_request is not None and web_chat_server is not None:
                        web_chat_server.mark_started(request_id)
                    logger.info(
                        "recall_request_queued source=%s request_id=%s text=%r pending=%s",
                        source,
                        request_id,
                        recall_query,
                        len(pending_recalls),
                    )
                    thinking_command = build_behavior_command(
                        state=LampState.RECALLING,
                        engagement=smoothed_engagement,
                        behavior={
                            "motion": "thinking_slow",
                            "light": "focus_glow",
                            "sound": None,
                            "speech_text": "Thinking...",
                        },
                        last_detected_objects=last_detected_objects,
                        gesture=gesture_payload,
                        speaker=speaker_payload,
                    )
                    print(json.dumps(thinking_command, ensure_ascii=False), flush=True)
                    append_jsonl(command_paths, thinking_command)
                    last_godot_udp_send_ms = godot_sender.send(thinking_command)
                    last_emit_at = time.monotonic()
                    should_emit = False
                    recall_worker.submit(work_item)

                while True:
                    try:
                        work_result = recall_result_queue.get_nowait()
                    except queue.Empty:
                        break

                    pending_recalls.pop(work_result.request_id, None)
                    response_payload = work_result.response_payload()
                    recall_worker_ms = round(work_result.worker_ms, 3)
                    answer_text = response_payload.get("answer") or "The recall request failed."

                    if work_result.result is not None:
                        recall_result = work_result.result
                        memory_retrieval_ms = round(recall_result.memory_retrieval_ms, 3)
                        llm_response_ms = (
                            "" if recall_result.llm_response_ms is None else round(recall_result.llm_response_ms, 3)
                        )
                        llm_attempted = recall_result.llm_attempted
                        llm_used = recall_result.llm_used
                        llm_error = recall_result.llm_error or ""
                        llm_fallback_reason = recall_result.llm_fallback_reason or ""
                    else:
                        llm_attempted = False
                        llm_used = False
                        llm_error = work_result.error or "worker_error"
                        llm_fallback_reason = "worker_error"

                    recall_behavior = behavior_for_recall_result(work_result.result, str(answer_text))
                    recall_target = recall_target_for_result(work_result.result)
                    final_recall_command = build_behavior_command(
                        state=LampState.RECALLING,
                        engagement=smoothed_engagement,
                        behavior=recall_behavior,
                        last_detected_objects=last_detected_objects,
                        gesture=gesture_payload,
                        recall_target=recall_target,
                        speaker=speaker_payload,
                    )
                    print(json.dumps(final_recall_command, ensure_ascii=False), flush=True)
                    append_jsonl(command_paths, final_recall_command)
                    last_godot_udp_send_ms = godot_sender.send(final_recall_command)
                    last_emit_at = time.monotonic()
                    should_emit = False

                    hold_s = max(0.0, float(args.recall_point_hold_seconds))
                    if hold_s > 0.0:
                        hold_behavior = dict(recall_behavior)
                        # Only the first recall command carries the answer text.
                        # Repeated hold commands preserve the pose/light without
                        # re-triggering text display or speech every second.
                        hold_behavior["speech_text"] = None
                        active_recall_feedback = {
                            "expires_at": last_emit_at + hold_s,
                            "behavior": hold_behavior,
                            "recall_target": recall_target,
                            "request_id": work_result.request_id,
                        }
                        logger.info(
                            "recall_feedback_hold_started request_id=%s hold_s=%.2f found=%s location=%s",
                            work_result.request_id,
                            hold_s,
                            bool(recall_target.get("found", False)),
                            recall_target.get("location_label", "none"),
                        )

                    response_payload["godot_command_sent"] = bool(godot_udp_config.enabled)
                    response_payload["godot_udp_ms"] = last_godot_udp_send_ms
                    if work_result.source == "browser" and web_chat_server is not None:
                        web_chat_server.set_result(work_result.request_id, response_payload)

                    logger.info(
                        "Sent recall command to Godot state=recalling source=%s request_id=%s parsed_object=%s memory_id=%s recall_worker_ms=%s llm_attempted=%s llm_used=%s fallback_reason=%s",
                        work_result.source,
                        work_result.request_id,
                        None if work_result.result is None else work_result.result.parsed_object,
                        None if work_result.result is None or work_result.result.memory_record is None else work_result.result.memory_record.id,
                        recall_worker_ms,
                        llm_attempted,
                        llm_used,
                        llm_fallback_reason or "none",
                    )

            now = time.monotonic()
            if active_recall_feedback is not None and now >= float(active_recall_feedback["expires_at"]):
                logger.info(
                    "recall_feedback_hold_finished request_id=%s",
                    active_recall_feedback.get("request_id", "unknown"),
                )
                active_recall_feedback = None

            if should_emit:
                outgoing_command = command
                if pending_recalls:
                    outgoing_command = build_behavior_command(
                        state=LampState.RECALLING,
                        engagement=smoothed_engagement,
                        behavior={
                            "motion": "thinking_slow",
                            "light": "focus_glow",
                            "sound": None,
                            "speech_text": "Thinking...",
                        },
                        last_detected_objects=last_detected_objects,
                        gesture=gesture_payload,
                        speaker=speaker_payload,
                    )
                elif active_recall_feedback is not None:
                    outgoing_command = build_behavior_command(
                        state=LampState.RECALLING,
                        engagement=smoothed_engagement,
                        behavior=active_recall_feedback["behavior"],
                        last_detected_objects=last_detected_objects,
                        gesture=gesture_payload,
                        recall_target=active_recall_feedback["recall_target"],
                        speaker=speaker_payload,
                    )
                print(json.dumps(outgoing_command, ensure_ascii=False), flush=True)
                append_jsonl(command_paths, outgoing_command)
                last_godot_udp_send_ms = godot_sender.send(outgoing_command)
                last_emit_at = now

            latency_logger.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame_index": frame_count,
                    "capture_ms": round(camera_frame.capture_latency_ms, 3),
                    "engagement_detection_ms": round(engagement_ms, 3),
                    "object_detection_ms": object_detection_ms,
                    "gesture_detection_ms": gesture_detection_ms,
                    "audio_capture_ms": audio_capture_ms,
                    "vad_ms": vad_ms,
                    "doa_ms": doa_ms,
                    "active_speaker_fusion_ms": active_speaker_fusion_ms,
                    "speaker_policy_ms": speaker_policy_ms,
                    "memory_write_ms": memory_write_ms,
                    "memory_retrieval_ms": memory_retrieval_ms,
                    "llm_response_ms": llm_response_ms,
                    "llm_attempted": llm_attempted,
                    "llm_used": llm_used,
                    "llm_error": llm_error,
                    "llm_fallback_reason": llm_fallback_reason,
                    "recall_worker_ms": recall_worker_ms,
                    "pending_recall_count": len(pending_recalls),
                    "smoothing_ms": round(smoothing_ms, 3),
                    "state_machine_ms": round(state_machine_ms, 3),
                    "command_build_ms": round(command_ms, 3),
                    "godot_udp_send_ms": "" if last_godot_udp_send_ms is None else round(last_godot_udp_send_ms, 3),
                    "total_loop_ms": round(total_ms, 3),
                    "fps": round(fps_ema, 2),
                    "state": transition.current_state.value,
                    "state_elapsed_s": round(transition.state_elapsed_s, 3),
                    "raw_engagement_status": raw_engagement.status,
                    "raw_engagement_confidence": round(raw_engagement.confidence, 3),
                    "raw_engagement_reason": raw_engagement.reason,
                    "smoothed_engagement_status": smoothed_engagement.status,
                    "smoothed_engagement_confidence": round(smoothed_engagement.confidence, 3),
                    "smoothed_engagement_reason": smoothed_engagement.reason,
                    "face_bbox": smoothed_engagement.face_bbox or "",
                    "face_area_ratio": round(smoothed_engagement.face_area_ratio, 4),
                    "raw_face_count": raw_engagement.raw_face_count,
                    "candidate_count": raw_engagement.candidate_count,
                    "selected_face_score": round(smoothed_engagement.selected_face_score, 3),
                    "object_count": len(display_detections) if object_detector.enabled else 0,
                    "gesture_status": gesture_result.status,
                    "gesture_confidence": round(gesture_result.confidence, 3),
                    "gesture_reason": gesture_result.reason,
                    "speech_detected": "" if last_speaker_result is None else last_speaker_result.speech_detected,
                    "speaker_active_track_id": "" if last_speaker_result is None else (last_speaker_result.active_track_id or ""),
                    "speaker_active_track_location": "" if last_speaker_result is None else (last_speaker_result.active_track_location or ""),
                    "speaker_speaking_to_robot": "" if last_speaker_result is None else last_speaker_result.speaking_to_robot,
                    "speaker_confidence": "" if last_speaker_result is None else round(last_speaker_result.confidence, 3),
                    "speaker_reason": "" if last_speaker_result is None else last_speaker_result.reason,
                    "speaker_doa_azimuth_deg": "" if last_speaker_result is None or last_speaker_result.doa_azimuth_deg is None else round(last_speaker_result.doa_azimuth_deg, 2),
                    "speaker_policy_override": speaker_policy_decision.overridden,
                    "speaker_policy_reason": speaker_policy_decision.reason,
                    "memory_write_count": memory_write_count,
                    "memory_duplicate_skip_count": memory_duplicate_skip_count,
                    "consecutive_engaged": transition.consecutive_engaged,
                    "consecutive_disengaged": transition.consecutive_disengaged,
                    "consecutive_absent": transition.consecutive_absent,
                }
            )

            if runtime_config.show_window:
                key = preview_window.show(
                    frame,
                    raw_result=raw_engagement,
                    smoothed_result=smoothed_engagement,
                    state=transition.current_state.value,
                    state_elapsed_s=transition.state_elapsed_s,
                    fps=fps_ema,
                    engagement_config=engagement_config,
                    object_detections=display_detections,
                    object_overlay_enabled=object_detector.enabled,
                    gesture_result=gesture_result,
                    speaker_result=last_speaker_result,
                )
                if key in (27, ord("q")):
                    logger.info("Quit requested from preview window")
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                logger.info("Reached --max-frames=%s", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception("Backend stopped due to error: %s", exc)
        return 1
    finally:
        recall_stop_event.set()
        if chat_receiver is not None:
            chat_receiver.stop()
        if recall_worker is not None:
            recall_worker.stop()
        if web_chat_server is not None:
            web_chat_server.stop()
        gesture_detector.close()
        speaker_worker.stop()
        audio_capture.stop()
        godot_sender.close()
        camera.release()
        if runtime_config.show_window:
            preview_window.close()
            cv2.destroyAllWindows()
        logger.info("Stopped Lumos backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
