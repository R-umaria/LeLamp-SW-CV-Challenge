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
from backend.audio.speech_to_text import SpeechToTextWorker, SpeechTranscript, UtteranceSegmenter
from backend.audio.stt_intent_gate import STTIntentGate, parse_wake_words
from backend.audio.audio_output import AudioOutputWorker
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
    SpeechToTextConfig,
    AudioOutputConfig,
    RuntimeConfig,
    SmoothingConfig,
    StateMachineConfig,
)
from backend.utils.logging_utils import setup_logging
from backend.utils.preview_window import PreviewWindow, PreviewWindowConfig
from backend.utils.run_paths import create_run_paths, write_latest_pointer


def _bbox_overlap_ratio(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    """Return intersection area divided by the smaller box area for hand/object suppression."""
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    smaller_area = max(1, min(aw * ah, bw * bh))
    return float(intersection) / float(smaller_area)


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

    parser.add_argument("--enable-gestures", action="store_true", help="Enable MediaPipe hand gestures: beckon, palm push, thumbs up, pinch follow, and two-hand heart.")
    parser.add_argument("--gesture-interval", type=float, default=0.10, help="Seconds between hand-gesture detection passes.")
    parser.add_argument("--gesture-confidence", type=float, default=0.64, help="Minimum gesture confidence required to override the normal motion skill.")
    parser.add_argument("--gesture-hold", type=float, default=1.15, help="Seconds to hold the last gesture command after a brief hand landmark dropout.")
    parser.add_argument("--gesture-max-hands", type=int, default=2, help="Maximum hands to track. Use 2 for two-hand heart gesture support.")

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
    parser.add_argument("--speaker-policy-hold-s", type=float, default=1.15, help="Seconds to hold listening/sound-seek behavior after a high-confidence speaker decision.")
    parser.add_argument("--speaker-seek-min-confidence", type=float, default=0.30, help="Minimum confidence for sound-seeking when speech is heard but no visible speaker is identified.")
    parser.add_argument("--speaker-secondary-min-area", type=float, default=0.006, help="Minimum frame-area ratio for secondary speaker face candidates. Default matches min candidate area so non-primary speakers are not dropped.")
    parser.add_argument("--speaker-debug", action="store_true", help="Log detailed face-track and speaker-fusion diagnostics.")

    parser.add_argument("--enable-stt", action="store_true", help="Enable gated local speech-to-text. Accepted transcripts are routed to the existing grounded recall pipeline.")
    parser.add_argument("--stt-backend", type=str, default="faster_whisper", choices=["faster_whisper", "whisper"], help="Local STT backend. faster_whisper is recommended for CPU demo use.")
    parser.add_argument("--stt-model-size", type=str, default="tiny.en", help="Whisper model size/name for STT, e.g. tiny.en, base.en, small.en.")
    parser.add_argument("--stt-device", type=str, default="cpu", help="STT device, usually cpu or cuda.")
    parser.add_argument("--stt-compute-type", type=str, default="int8", help="faster-whisper compute type, e.g. int8, int8_float16, float16.")
    parser.add_argument("--stt-language", type=str, default="en", help="STT language hint. Use empty string for auto-detect.")
    parser.add_argument("--stt-min-utterance-s", type=float, default=0.55, help="Minimum captured voice utterance length before STT can finalize.")
    parser.add_argument("--stt-max-utterance-s", type=float, default=6.0, help="Maximum captured voice utterance length.")
    parser.add_argument("--stt-end-silence-s", type=float, default=0.85, help="Silence duration that finalizes a captured utterance.")
    parser.add_argument("--stt-cooldown-s", type=float, default=1.25, help="Cooldown after one transcript before a new utterance can start.")
    parser.add_argument("--stt-speaker-min-confidence", type=float, default=0.45, help="Minimum active-speaker confidence required before recording an utterance for STT.")
    parser.add_argument("--stt-min-words", type=int, default=2, help="Reject transcripts with fewer words than this.")
    parser.add_argument("--stt-require-wake-word", action="store_true", help="Only accept voice recall when a configured Lumos wake phrase is present.")
    parser.add_argument("--stt-wake-words", type=str, default="Lumos,hey Lumos", help="Comma-separated wake phrases for voice recall gating.")
    parser.add_argument("--stt-min-confidence", type=float, default=0.0, help="Minimum STT confidence for voice recall gating when the backend reports confidence.")
    parser.add_argument("--stt-cooldown-sec", type=float, default=None, help="Alias for --stt-cooldown-s.")

    parser.add_argument("--enable-audio-output", action="store_true", help="Enable optional non-blocking local sound cues for behavior events.")
    parser.add_argument("--enable-tts", action="store_true", help="Enable optional local pyttsx3 TTS for speech_text responses.")
    parser.add_argument("--tts-rate", type=int, default=175, help="pyttsx3 speech rate for --enable-tts.")
    parser.add_argument("--tts-volume", type=float, default=0.85, help="pyttsx3 volume in [0,1] for --enable-tts.")

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
    SpeechToTextConfig,
    AudioOutputConfig,
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
        max_num_hands=max(1, min(2, int(args.gesture_max_hands))),
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
        policy_hold_s=max(0.0, float(args.speaker_policy_hold_s)),
        policy_seek_min_confidence=max(0.0, min(1.0, float(args.speaker_seek_min_confidence))),
        secondary_face_min_area_ratio=max(0.0, float(args.speaker_secondary_min_area)),
    )
    stt_cooldown = args.stt_cooldown_sec if args.stt_cooldown_sec is not None else args.stt_cooldown_s
    stt_config = SpeechToTextConfig(
        enabled=bool(args.enable_stt),
        backend=str(args.stt_backend),
        model_size=str(args.stt_model_size),
        device=str(args.stt_device),
        compute_type=str(args.stt_compute_type),
        language=str(args.stt_language or ""),
        speaker_min_confidence=max(0.0, min(1.0, float(args.stt_speaker_min_confidence))),
        min_utterance_s=max(0.15, float(args.stt_min_utterance_s)),
        max_utterance_s=max(0.5, float(args.stt_max_utterance_s)),
        end_silence_s=max(0.15, float(args.stt_end_silence_s)),
        cooldown_s=max(0.0, float(stt_cooldown)),
        intent_cooldown_s=max(0.0, float(stt_cooldown)),
        min_words=max(1, int(args.stt_min_words)),
        require_wake_word=bool(args.stt_require_wake_word),
        wake_words=parse_wake_words(args.stt_wake_words),
        min_confidence=max(0.0, min(1.0, float(args.stt_min_confidence))),
    )
    audio_output_config = AudioOutputConfig(
        enabled=bool(args.enable_audio_output),
        tts_enabled=bool(args.enable_tts),
        tts_rate=int(args.tts_rate),
        tts_volume=max(0.0, min(1.0, float(args.tts_volume))),
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
        stt_config,
        audio_output_config,
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
        stt_config,
        audio_output_config,
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
    stt_event_paths = [run_paths.run_dir / "stt_events.jsonl"]
    if not args.no_latest:
        speaker_event_paths.append(run_paths.latest_dir / "speaker_events.jsonl")
        stt_event_paths.append(run_paths.latest_dir / "stt_events.jsonl")
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
    stt_result_queue: queue.Queue[SpeechTranscript] = queue.Queue()
    utterance_segmenter = UtteranceSegmenter(stt_config, logger=logger)
    stt_worker = SpeechToTextWorker(stt_config, stt_result_queue, logger=logger)
    stt_intent_gate = STTIntentGate(stt_config)
    audio_output = AudioOutputWorker(audio_output_config, logger=logger)
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
    if args.interactive_recall or args.enable_godot_chat or args.enable_web_chat or args.enable_stt:
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

    logger.info("Starting Lumos backend with engagement, object memory, active speaker awareness, and optional voice STT recall")
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
        "Gesture config enabled=%s active=%s interval=%.2fs confidence=%.2f hold=%.2fs max_hands=%s",
        gesture_config.enabled,
        gesture_detector.enabled,
        gesture_config.interval_s,
        gesture_config.min_gesture_confidence,
        gesture_config.hold_s,
        gesture_config.max_num_hands,
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
        "Speaker awareness config audio_enabled=%s speaker_enabled=%s doa_enabled=%s device=%s sample_rate=%s block_ms=%s vad_threshold=%.2f mic_distance_m=%.3f fusion_interval=%.3f worker_frame_width=%s hold_s=%.2f seek_min_conf=%.2f secondary_min_area=%.3f",
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
        speaker_config.policy_hold_s,
        speaker_config.policy_seek_min_confidence,
        speaker_config.secondary_face_min_area_ratio,
    )
    logger.info(
        "STT config enabled=%s backend=%s model=%s device=%s compute_type=%s gated_by_speaker=True min_utterance=%.2fs max_utterance=%.2fs end_silence=%.2fs speaker_min_conf=%.2f require_wake=%s wake_words=%s min_conf=%.2f cooldown=%.2f",
        stt_config.enabled,
        stt_config.backend,
        stt_config.model_size,
        stt_config.device,
        stt_config.compute_type,
        stt_config.min_utterance_s,
        stt_config.max_utterance_s,
        stt_config.end_silence_s,
        stt_config.speaker_min_confidence,
        stt_config.require_wake_word,
        stt_config.wake_words,
        stt_config.min_confidence,
        stt_config.intent_cooldown_s,
    )
    logger.info(
        "Audio output config cues_enabled=%s tts_enabled=%s tts_rate=%s tts_volume=%.2f",
        audio_output_config.enabled,
        audio_output_config.tts_enabled,
        audio_output_config.tts_rate,
        audio_output_config.tts_volume,
    )
    if speaker_config.enabled and not audio_config.enabled:
        reason_payload = {"event": "speaker_awareness_disabled_reason", "reason": "enable_speaker_awareness_without_enable_audio"}
        append_jsonl(speaker_event_paths, reason_payload)
        logger.warning("speaker_awareness_disabled_reason reason=%s", reason_payload["reason"])
    if stt_config.enabled and not audio_config.enabled:
        append_jsonl(stt_event_paths, {"event": "stt_disabled_reason", "reason": "enable_stt_without_enable_audio"})
        logger.warning("stt_disabled_reason reason=enable_stt_without_enable_audio")
    if stt_config.enabled and not speaker_config.enabled:
        append_jsonl(stt_event_paths, {"event": "stt_disabled_reason", "reason": "enable_stt_without_enable_speaker_awareness"})
        logger.warning("stt_disabled_reason reason=enable_stt_without_enable_speaker_awareness")
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
    speaker_hold_behavior: dict | None = None
    speaker_hold_reason = "not_held"
    speaker_hold_until = 0.0
    gesture_result = HandGestureResult("unavailable", 0.0, "gesture_detection_disabled")
    fps_ema = 0.0
    last_godot_udp_send_ms = None
    last_detected_objects: list[dict] = []
    active_recall_feedback: dict | None = None
    display_detections = []
    stt_transcribe_ms = ""
    stt_last_text = ""
    stt_last_status = ""

    try:
        audio_started = False
        audio_output.start()
        if audio_config.enabled:
            audio_started = audio_capture.start()
        if speaker_config.enabled and audio_config.enabled and audio_started:
            speaker_worker.start()
        if stt_config.enabled and audio_config.enabled and speaker_config.enabled and audio_started:
            stt_worker.start()
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

            new_voice_pairs = []
            if audio_capture.available:
                new_audio_chunks = audio_capture.get_chunks_since(last_audio_sequence)
                if new_audio_chunks:
                    audio_capture_ms = round(float(new_audio_chunks[-1].capture_ms), 3)
                for latest_audio in new_audio_chunks:
                    last_audio_sequence = latest_audio.sequence
                    t_audio = time.perf_counter()
                    last_voice_result = voice_detector.update(latest_audio)
                    vad_ms = round((time.perf_counter() - t_audio) * 1000.0, 3)
                    new_voice_pairs.append((latest_audio, last_voice_result))
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

            should_detect_gestures = gesture_config.enabled and (
                last_gesture_detection_at == 0.0 or (now - last_gesture_detection_at >= gesture_config.interval_s)
            )
            if should_detect_gestures:
                t0 = time.perf_counter()
                gesture_result = gesture_detector.detect(frame, now=now)
                gesture_detection_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                last_gesture_detection_at = now
                if gesture_result.status in {"beckon", "palm_push", "thumbs_up", "pinch_follow", "heart"}:
                    logger.info(
                        "Detected gesture status=%s confidence=%.3f reason=%s target=%s latency_ms=%.3f",
                        gesture_result.status,
                        gesture_result.confidence,
                        gesture_result.reason,
                        gesture_result.hand_center_norm,
                        gesture_detection_ms,
                    )

            should_detect_objects = object_detector.enabled and (
                last_object_detection_at == 0.0 or (now - last_object_detection_at >= object_config.interval_s)
            )
            if should_detect_objects:
                t0 = time.perf_counter()
                display_detections = object_detector.detect(frame)
                object_detection_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                if (
                    gesture_result.status in {"beckon", "palm_push", "thumbs_up", "pinch_follow", "heart"}
                    and gesture_result.hand_bbox is not None
                ):
                    before_hand_filter = len(display_detections)
                    display_detections = [
                        detection
                        for detection in display_detections
                        if _bbox_overlap_ratio(gesture_result.hand_bbox, detection.bbox) < 0.35
                    ]
                    removed_by_hand_filter = before_hand_filter - len(display_detections)
                    if removed_by_hand_filter:
                        logger.info(
                            "Suppressed object detections overlapping active hand gesture count=%s gesture=%s",
                            removed_by_hand_filter,
                            gesture_result.status,
                        )
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

            if stt_config.enabled and audio_config.enabled and speaker_config.enabled and stt_worker.enabled:
                allow_stt_capture = not pending_recalls and active_recall_feedback is None
                for audio_chunk, voice_result_for_chunk in new_voice_pairs:
                    utterance = utterance_segmenter.update(
                        audio_chunk,
                        voice_result_for_chunk,
                        last_speaker_result,
                        now=now,
                        allow_capture=allow_stt_capture,
                    )
                    if utterance is not None:
                        append_jsonl(stt_event_paths, {"event": "stt_utterance", **utterance.to_log_dict()})
                        stt_worker.submit(utterance)

            t0 = time.perf_counter()
            smoothed_engagement = smoother.update(raw_engagement)
            smoothing_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            transition = fsm.update(smoothed_engagement)
            state_machine_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = behavior_for_transition(transition)
            gesture_payload = gesture_result.to_protocol_dict()
            gesture_active = (
                gesture_result.status in {"beckon", "palm_push", "thumbs_up", "pinch_follow", "heart"}
                and gesture_result.confidence >= gesture_config.min_gesture_confidence
            )
            speaker_payload = None if last_speaker_result is None else last_speaker_result.to_protocol_dict()
            t_policy = time.perf_counter()
            if gesture_active and transition.current_state != LampState.RECALLING:
                # Deliberate hand commands should control the lamp immediately.
                # Active-speaker awareness resumes once the gesture drops out.
                behavior = behavior_with_gesture_override(behavior, gesture_payload)
                speaker_policy_decision = SpeakerPolicyDecision(behavior, False, "gesture_has_priority")
                speaker_hold_behavior = None
                speaker_hold_reason = "not_held"
            else:
                speaker_policy_decision = apply_speaker_policy(behavior, last_speaker_result, transition.current_state, speaker_config)
                behavior = speaker_policy_decision.behavior
            speaker_policy_ms = round((time.perf_counter() - t_policy) * 1000.0, 3)
            if speaker_policy_decision.overridden:
                speaker_hold_behavior = dict(behavior)
                speaker_hold_reason = speaker_policy_decision.reason
                speaker_hold_until = now + speaker_config.policy_hold_s
            elif speaker_hold_behavior is not None and now <= speaker_hold_until and transition.current_state != LampState.RECALLING:
                behavior = dict(speaker_hold_behavior)
                speaker_policy_decision = SpeakerPolicyDecision(behavior, True, f"speaker_policy_hold:{speaker_hold_reason}")
            else:
                speaker_hold_behavior = None
                speaker_hold_reason = "not_held"
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
                interruption=speaker_policy_decision.to_protocol_dict(),
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

            if stt_config.enabled:
                while True:
                    try:
                        transcript = stt_result_queue.get_nowait()
                    except queue.Empty:
                        break
                    stt_transcribe_ms = round(transcript.transcribe_ms, 3)
                    stt_last_text = transcript.text
                    stt_last_status = transcript.reason
                    append_jsonl(stt_event_paths, {"event": "stt_transcript", **transcript.to_log_dict()})
                    if transcript.accepted:
                        gate_decision = stt_intent_gate.evaluate(transcript.text, confidence=transcript.confidence)
                        append_jsonl(stt_event_paths, {"event": "stt_intent_gate", "request_id": transcript.request_id, **gate_decision.to_log_dict()})
                        logger.info(
                            "stt_intent_gate request_id=%s accepted=%s reason=%s directed=%s clear_memory_query=%s text=%r",
                            transcript.request_id,
                            gate_decision.accepted,
                            gate_decision.reason,
                            gate_decision.directed_to_lumos,
                            gate_decision.clear_memory_query,
                            transcript.text,
                        )
                    else:
                        gate_decision = None
                    if transcript.accepted and gate_decision is not None and gate_decision.accepted and recall_queue is not None:
                        recall_queue.put(
                            RecallWorkItem(
                                text=transcript.text,
                                request_id=transcript.request_id,
                                source="voice_stt",
                            )
                        )
                        logger.info(
                            "voice_recall_transcript_queued request_id=%s text=%r speaker=%s",
                            transcript.request_id,
                            transcript.text,
                            transcript.speaker_track_id,
                        )
                    elif transcript.accepted and gate_decision is not None and not gate_decision.accepted:
                        logger.info(
                            "voice_recall_transcript_rejected request_id=%s reason=%s text=%r",
                            transcript.request_id,
                            gate_decision.reason,
                            transcript.text,
                        )
                    elif transcript.accepted:
                        logger.warning(
                            "voice_recall_transcript_dropped request_id=%s reason=recall_queue_unavailable text=%r",
                            transcript.request_id,
                            transcript.text,
                        )

            if recall_worker is not None and recall_queue is not None and recall_result_queue is not None:
                while True:
                    try:
                        recall_item = recall_queue.get_nowait()
                    except queue.Empty:
                        break

                    web_request = recall_item if isinstance(recall_item, WebChatRequest) else None
                    if isinstance(recall_item, RecallWorkItem):
                        recall_query = recall_item.text
                        source = recall_item.source
                        request_id = recall_item.request_id
                        work_item = recall_item
                    else:
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
                        interruption=speaker_policy_decision.to_protocol_dict(),
                    )
                    print(json.dumps(thinking_command, ensure_ascii=False), flush=True)
                    append_jsonl(command_paths, thinking_command)
                    audio_output.submit_behavior(thinking_command.get("behavior", {}))
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
                        interruption=speaker_policy_decision.to_protocol_dict(),
                    )
                    print(json.dumps(final_recall_command, ensure_ascii=False), flush=True)
                    append_jsonl(command_paths, final_recall_command)
                    audio_output.submit_behavior(final_recall_command.get("behavior", {}))
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
                        interruption=speaker_policy_decision.to_protocol_dict(),
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
                        interruption=speaker_policy_decision.to_protocol_dict(),
                    )
                print(json.dumps(outgoing_command, ensure_ascii=False), flush=True)
                append_jsonl(command_paths, outgoing_command)
                audio_output.submit_behavior(outgoing_command.get("behavior", {}))
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
                    "stt_transcribe_ms": stt_transcribe_ms,
                    "stt_last_text": stt_last_text,
                    "stt_last_status": stt_last_status,
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
                    "head_pose_status": smoothed_engagement.head_pose_status,
                    "head_pose_confidence": round(smoothed_engagement.head_pose_confidence, 3),
                    "head_pose_yaw": "" if smoothed_engagement.yaw is None else round(smoothed_engagement.yaw, 2),
                    "head_pose_pitch": "" if smoothed_engagement.pitch is None else round(smoothed_engagement.pitch, 2),
                    "head_pose_roll": "" if smoothed_engagement.roll is None else round(smoothed_engagement.roll, 2),
                    "head_pose_fallback_used": smoothed_engagement.fallback_mode_used,
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
                    "speaker_do_not_interrupt": speaker_policy_decision.do_not_interrupt,
                    "speaker_quiet_listening": speaker_policy_decision.quiet_listening,
                    "speaker_safe_to_respond": speaker_policy_decision.safe_to_respond,
                    "speaker_suppressed_behavior": speaker_policy_decision.suppressed_behavior or "",
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
        detector.close()
        stt_worker.stop()
        speaker_worker.stop()
        audio_capture.stop()
        audio_output.stop()
        godot_sender.close()
        camera.release()
        if runtime_config.show_window:
            preview_window.close()
            cv2.destroyAllWindows()
        logger.info("Stopped Lumos backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
