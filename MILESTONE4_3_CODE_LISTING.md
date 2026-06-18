# Milestone 4.3 Complete Changed File Code

## `backend/main.py`

```python

"""LeLamp backend vertical slice with Milestone 4 grounded recall.

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

from backend.behavior.behavior_policy import behavior_for_state
from backend.behavior.command_protocol import build_behavior_command
from backend.behavior.godot_udp_sender import GodotUdpSender
from backend.behavior.state_machine import InteractionStateMachine, LampState
from backend.conversation.chat_udp_receiver import ChatUdpReceiver
from backend.conversation.recall_agent import RecallAgent
from backend.evaluation.latency_logger import LatencyLogger
from backend.memory.scene_memory import SceneMemory
from backend.perception.camera import OpenCVCamera
from backend.perception.engagement_detector import FaceEngagementDetector, draw_engagement_overlay
from backend.perception.object_detector import YoloObjectDetector, draw_object_overlay
from backend.perception.temporal_smoother import EngagementSmoother
from backend.utils.config import (
    CameraConfig,
    EngagementConfig,
    GodotUdpConfig,
    MemoryConfig,
    ObjectDetectionConfig,
    RuntimeConfig,
    SmoothingConfig,
    StateMachineConfig,
)
from backend.utils.logging_utils import setup_logging
from backend.utils.run_paths import create_run_paths, write_latest_pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeLamp Milestone 4 backend: engagement + object memory + grounded recall")
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

    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--min-state-dwell", type=float, default=0.75)
    parser.add_argument("--exit-disengaged-frames", type=int, default=5)
    parser.add_argument("--exit-absent-frames", type=int, default=8)
    parser.add_argument("--engaged-recovery-frames", type=int, default=2)
    parser.add_argument("--clear-engaged-confidence", type=float, default=0.78)

    parser.add_argument("--center-tolerance-x", type=float, default=0.24)
    parser.add_argument("--center-tolerance-y", type=float, default=0.30)
    parser.add_argument("--min-face-area-ratio", type=float, default=0.020)
    parser.add_argument("--min-candidate-area-ratio", type=float, default=0.010)
    parser.add_argument("--cascade-min-neighbors", type=int, default=6)

    parser.add_argument("--enable-objects", action="store_true", help="Enable optional YOLO object detection and scene-memory writes.")
    parser.add_argument("--object-model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model path/name, e.g. yolov8n.pt")
    parser.add_argument("--object-interval", type=float, default=2.0, help="Seconds between object-detection passes.")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path.")
    parser.add_argument("--save-object-frames", action="store_true", help="Save annotated evidence frames when a memory record is written.")
    parser.add_argument("--object-confidence", type=float, default=0.35, help="YOLO confidence threshold for object detection.")
    parser.add_argument("--memory-dedupe-window", type=float, default=8.0, help="Seconds to suppress repeated same-object/same-location memory writes.")
    parser.add_argument("--object-frame-dir", type=str, default="data/object_frames", help="Directory for saved object evidence frames.")

    parser.add_argument("--interactive-recall", action="store_true", help="Allow text recall questions while the backend is running.")
    parser.add_argument("--enable-godot-chat", action="store_true", help="Listen for live chat queries from Godot over local UDP.")
    parser.add_argument("--chat-host", type=str, default="127.0.0.1", help="Backend UDP host for Godot chat messages.")
    parser.add_argument("--chat-port", type=int, default=4243, help="Backend UDP port for Godot chat messages.")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase retrieved memory answers.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL for --use-llm.")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model for recall phrasing, e.g. llama3.2 or qwen2.5.")
    parser.add_argument("--ollama-timeout", type=float, default=8.0, help="Ollama chat timeout in seconds for recall phrasing.")
    parser.add_argument("--llm-required", action="store_true", help="Fail fast at startup if Ollama/model is unavailable; live answers still fall back but log failures.")

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--show-window", action="store_true", default=True)
    window_group.add_argument("--no-window", action="store_false", dest="show_window")
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
    recall_queue: "queue.Queue[str]",
    stop_event: threading.Event,
    logger,
) -> threading.Thread:
    """Start a tiny stdin reader so camera/perception loop stays non-blocking."""

    def _worker() -> None:
        print("Interactive recall enabled. Type a question such as: Where is the cup?", flush=True)
        print("Type :q, quit, or exit to stop accepting recall questions.", flush=True)
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
            recall_queue.put(query)

    thread = threading.Thread(target=_worker, name="interactive-recall-input", daemon=True)
    thread.start()
    return thread


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
    MemoryConfig,
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
    memory_config = MemoryConfig(
        db_path=args.memory_db,
        dedupe_window_s=max(0.0, args.memory_dedupe_window),
        save_object_frames=args.save_object_frames,
        frame_dir=args.object_frame_dir,
    )
    return (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        memory_config,
    )


def main() -> int:
    args = parse_args()
    (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        memory_config,
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
    godot_sender = GodotUdpSender(godot_udp_config, logger=logger)

    camera = OpenCVCamera(camera_config.index, camera_config.width, camera_config.height)
    detector = FaceEngagementDetector(engagement_config)
    smoother = EngagementSmoother(smoothing_config)
    fsm = InteractionStateMachine(state_config)
    object_detector = YoloObjectDetector(object_config, logger=logger)
    scene_memory = SceneMemory(memory_config, logger=logger)

    recall_agent = None
    recall_queue: queue.Queue[str] | None = None
    recall_stop_event = threading.Event()
    if args.interactive_recall or args.enable_godot_chat:
        recall_log_paths = [run_paths.run_dir / "recall.jsonl"]
        if not args.no_latest:
            recall_log_paths.append(run_paths.latest_dir / "recall.jsonl")
        recall_agent = RecallAgent(
            memory_db=args.memory_db,
            use_llm=args.use_llm,
            llm_required=args.llm_required,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            ollama_timeout_s=args.ollama_timeout,
            log_paths=recall_log_paths,
            logger=logger,
        )
        recall_queue = queue.Queue()
        if args.interactive_recall:
            start_recall_input_thread(recall_queue, recall_stop_event, logger)

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

    logger.info("Starting Milestone 4.3 backend with isolated run logging")
    logger.info("Run id=%s", run_paths.run_id)
    logger.info("Run directory=%s", run_paths.run_dir)
    if not args.no_latest:
        logger.info("Latest mirror directory=%s", run_paths.latest_dir)
    logger.info("Camera index=%s size=%sx%s", camera_config.index, camera_config.width, camera_config.height)
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
        "Memory config db=%s save_frames=%s dedupe_window=%.1fs",
        memory_config.db_path,
        memory_config.save_object_frames,
        memory_config.dedupe_window_s,
    )
    logger.info("Commands will be saved to %s", commands_path)
    logger.info(
        "Recall config interactive=%s godot_chat=%s chat_udp=%s:%s use_llm=%s llm_required=%s ollama_url=%s ollama_model=%s ollama_timeout=%.1fs",
        args.interactive_recall,
        args.enable_godot_chat,
        args.chat_host,
        args.chat_port,
        args.use_llm,
        args.llm_required,
        args.ollama_url,
        args.ollama_model,
        args.ollama_timeout,
    )
    if godot_udp_config.enabled:
        logger.info("Commands will also be streamed to Godot via udp://%s:%s", godot_udp_config.host, godot_udp_config.port)

    frame_count = 0
    last_emit_at = 0.0
    last_object_detection_at = 0.0
    fps_ema = 0.0
    last_godot_udp_send_ms = None
    last_detected_objects: list[dict] = []
    display_detections = []

    try:
        camera.open()
        while True:
            loop_start = time.perf_counter()

            camera_frame = camera.read()
            frame = camera_frame.frame

            t0 = time.perf_counter()
            raw_engagement = detector.detect(frame)
            engagement_ms = (time.perf_counter() - t0) * 1000.0

            object_detection_ms = ""
            memory_write_ms = ""
            memory_retrieval_ms = ""
            llm_response_ms = ""
            llm_attempted = ""
            llm_used = ""
            llm_error = ""
            memory_write_count = 0
            memory_duplicate_skip_count = 0
            now = time.monotonic()
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

            t0 = time.perf_counter()
            smoothed_engagement = smoother.update(raw_engagement)
            smoothing_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            transition = fsm.update(smoothed_engagement)
            state_machine_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = behavior_for_state(transition.current_state)
            command = build_behavior_command(
                state=transition.current_state,
                engagement=smoothed_engagement,
                behavior=behavior,
                last_detected_objects=last_detected_objects,
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

            if should_emit:
                print(json.dumps(command, ensure_ascii=False), flush=True)
                append_jsonl(command_paths, command)
                last_godot_udp_send_ms = godot_sender.send(command)
                last_emit_at = now

            if recall_agent is not None and recall_queue is not None:
                while True:
                    try:
                        recall_query = recall_queue.get_nowait()
                    except queue.Empty:
                        break

                    recall_result = recall_agent.answer(recall_query)
                    memory_retrieval_ms = round(recall_result.memory_retrieval_ms, 3)
                    llm_response_ms = (
                        "" if recall_result.llm_response_ms is None else round(recall_result.llm_response_ms, 3)
                    )
                    llm_attempted = recall_result.llm_attempted
                    llm_used = recall_result.llm_used
                    llm_error = recall_result.llm_error or ""
                    recall_command = build_behavior_command(
                        state=LampState.RECALLING,
                        engagement=smoothed_engagement,
                        behavior={
                            "motion": "thinking",
                            "light": "focus_glow",
                            "sound": None,
                            "speech_text": recall_result.answer,
                        },
                        last_detected_objects=last_detected_objects,
                    )
                    print(json.dumps(recall_command, ensure_ascii=False), flush=True)
                    append_jsonl(command_paths, recall_command)
                    last_godot_udp_send_ms = godot_sender.send(recall_command)
                    last_emit_at = time.monotonic()
                    logger.info(
                        "Sent recall command to Godot state=recalling parsed_object=%s memory_id=%s llm_attempted=%s llm_used=%s",
                        recall_result.parsed_object,
                        recall_result.memory_record.id if recall_result.memory_record else None,
                        recall_result.llm_attempted,
                        recall_result.llm_used,
                    )

            latency_logger.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame_index": frame_count,
                    "capture_ms": round(camera_frame.capture_latency_ms, 3),
                    "engagement_detection_ms": round(engagement_ms, 3),
                    "object_detection_ms": object_detection_ms,
                    "memory_write_ms": memory_write_ms,
                    "memory_retrieval_ms": memory_retrieval_ms,
                    "llm_response_ms": llm_response_ms,
                    "llm_attempted": llm_attempted,
                    "llm_used": llm_used,
                    "llm_error": llm_error,
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
                    "memory_write_count": memory_write_count,
                    "memory_duplicate_skip_count": memory_duplicate_skip_count,
                    "consecutive_engaged": transition.consecutive_engaged,
                    "consecutive_disengaged": transition.consecutive_disengaged,
                    "consecutive_absent": transition.consecutive_absent,
                }
            )

            if runtime_config.show_window:
                draw_engagement_overlay(
                    frame,
                    raw_result=raw_engagement,
                    smoothed_result=smoothed_engagement,
                    state=transition.current_state.value,
                    state_elapsed_s=transition.state_elapsed_s,
                    fps=fps_ema,
                    config=engagement_config,
                )
                if object_detector.enabled:
                    draw_object_overlay(frame, display_detections)
                cv2.imshow("LeLamp Milestone 4.3 - Engagement/Object Memory/Live Recall", frame)
                key = cv2.waitKey(1) & 0xFF
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
        godot_sender.close()
        camera.release()
        if runtime_config.show_window:
            cv2.destroyAllWindows()
        logger.info("Stopped Milestone 4 backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/conversation/__init__.py`

```python

"""Conversation, intent parsing, live chat, and grounded recall modules."""

from backend.conversation.intent_parser import ConversationIntentType, ParsedConversationIntent, parse_conversation_intent
from backend.conversation.query_parser import ParsedObjectQuery, parse_object_query

__all__ = [
    "ConversationIntentType",
    "ParsedConversationIntent",
    "ParsedObjectQuery",
    "parse_conversation_intent",
    "parse_object_query",
]

```

## `backend/conversation/chat_udp_receiver.py`

```python

"""UDP chat-query receiver for the Godot live chat panel.

Godot sends JSON packets to this backend-side listener:

    {
      "type": "chat_query",
      "timestamp": "2026-06-18T13:30:00",
      "text": "Where did you last see my phone?"
    }

The receiver does not answer queries itself. It only validates the packet shape
and enqueues the text into the same bounded recall path used by terminal input.
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatPacket:
    text: str
    timestamp: str | None
    sender: tuple[str, int]
    received_at: float


class ChatUdpReceiver:
    """Threaded local UDP receiver for Godot chat messages."""

    def __init__(
        self,
        output_queue: "queue.Queue[str]",
        host: str = "127.0.0.1",
        port: int = 4243,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_queue = output_queue
        self.host = host
        self.port = int(port)
        self.logger = logger or logging.getLogger("lelamp")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.received_count = 0
        self.invalid_count = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="godot-chat-udp-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.25)
        self._sock = sock
        try:
            sock.bind((self.host, self.port))
            self.logger.info("Godot chat UDP receiver listening on udp://%s:%s", self.host, self.port)
            while not self._stop_event.is_set():
                try:
                    packet, sender = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop_event.is_set():
                        self.logger.warning("Godot chat UDP socket stopped unexpectedly: %s", exc)
                    break
                self._handle_packet(packet, sender)
        except OSError as exc:
            self.logger.error("Failed to bind Godot chat UDP receiver on udp://%s:%s error=%s", self.host, self.port, exc)
        finally:
            try:
                sock.close()
            except OSError:
                pass
            self._sock = None
            self.logger.info(
                "Godot chat UDP receiver stopped received=%s invalid=%s",
                self.received_count,
                self.invalid_count,
            )

    def _handle_packet(self, packet: bytes, sender: tuple[str, int]) -> None:
        try:
            text = packet.decode("utf-8")
            payload: Any = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.invalid_count += 1
            self.logger.warning("Ignored malformed Godot chat UDP packet sender=%s error=%s", sender, exc)
            return

        parsed = parse_chat_packet(payload, sender)
        if parsed is None:
            self.invalid_count += 1
            self.logger.warning("Ignored invalid Godot chat packet sender=%s payload=%r", sender, payload)
            return

        self.received_count += 1
        self.output_queue.put(parsed.text)
        self.logger.info(
            "Queued Godot chat query sender=%s timestamp=%s text=%r total=%s",
            sender,
            parsed.timestamp or "missing",
            parsed.text,
            self.received_count,
        )


def parse_chat_packet(payload: Any, sender: tuple[str, int] = ("unknown", 0)) -> ChatPacket | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "chat_query":
        return None
    raw_text = payload.get("text")
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None
    timestamp_value = payload.get("timestamp")
    timestamp = None if timestamp_value is None else str(timestamp_value)
    return ChatPacket(text=text[:1000], timestamp=timestamp, sender=sender, received_at=time.time())

```

## `backend/conversation/intent_parser.py`

```python

"""Conversation intent parser for live LeLamp recall.

The parser is intentionally deterministic. It does not try to solve general NLU;
it only separates demo-critical grounded intents before object extraction:

- object_last_seen: retrieve one object from SQLite
- list_recent_objects: summarize recent unique objects from SQLite
- unsupported: avoid pretending an arbitrary utterance is an object query
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.conversation.query_parser import ParsedObjectQuery, clean_query_text, is_list_recent_objects_query, parse_object_query


class ConversationIntentType(str, Enum):
    OBJECT_LAST_SEEN = "object_last_seen"
    LIST_RECENT_OBJECTS = "list_recent_objects"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ParsedConversationIntent:
    original_query: str
    intent: ConversationIntentType
    object_query: ParsedObjectQuery
    confidence: float
    reason: str

    @property
    def normalized_label(self) -> str | None:
        return self.object_query.normalized_label

    @property
    def target_text(self) -> str | None:
        return self.object_query.target_text

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "target_text": self.target_text,
            "normalized_label": self.normalized_label,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "object_query": self.object_query.to_dict(),
        }


_OBJECT_RECALL_SIGNALS = [
    re.compile(r"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\b"),
    re.compile(r"\bhave\s+you\s+seen\b"),
    re.compile(r"\bdid\s+you\s+(?:see|notice)\s+if\s+i\s+(?:had|have|was\s+holding|was\s+using)\b"),
    re.compile(r"\bdid\s+you\s+see\b"),
    re.compile(r"\b(?:find|locate)\b"),
]

_UNSUPPORTED_CHITCHAT = [
    re.compile(r"\bhow\s+are\s+you\b"),
    re.compile(r"\bwhat\s+is\s+your\s+name\b"),
    re.compile(r"\btell\s+me\s+a\s+joke\b"),
    re.compile(r"\bwho\s+are\s+you\b"),
]


def parse_conversation_intent(query: str) -> ParsedConversationIntent:
    cleaned = clean_query_text(query)
    object_query = parse_object_query(query)

    if not cleaned:
        return ParsedConversationIntent(query, ConversationIntentType.UNSUPPORTED, object_query, 0.0, "empty_query")

    if is_list_recent_objects_query(cleaned):
        return ParsedConversationIntent(
            query,
            ConversationIntentType.LIST_RECENT_OBJECTS,
            object_query,
            0.96,
            "list_recent_objects_pattern",
        )

    if any(pattern.search(cleaned) is not None for pattern in _UNSUPPORTED_CHITCHAT):
        return ParsedConversationIntent(query, ConversationIntentType.UNSUPPORTED, _empty_object_query(query), 0.90, "unsupported_chitchat")

    if object_query.normalized_label and any(pattern.search(cleaned) is not None for pattern in _OBJECT_RECALL_SIGNALS):
        return ParsedConversationIntent(
            query,
            ConversationIntentType.OBJECT_LAST_SEEN,
            object_query,
            max(0.70, object_query.confidence),
            object_query.strategy,
        )

    # Compact object-only queries are useful in the terminal during debugging.
    if object_query.normalized_label and object_query.strategy == "known_alias_fallback":
        return ParsedConversationIntent(
            query,
            ConversationIntentType.OBJECT_LAST_SEEN,
            object_query,
            object_query.confidence,
            "known_alias_compact_query",
        )

    return ParsedConversationIntent(query, ConversationIntentType.UNSUPPORTED, object_query, 0.35, "no_supported_intent")


def _empty_object_query(query: str) -> ParsedObjectQuery:
    return ParsedObjectQuery(query, None, None, 0.0, "unsupported_intent")

```

## `backend/conversation/llm_client.py`

```python

"""Local Ollama client for structured, grounded conversation responses.

Milestone 4.3 intentionally uses Ollama's ``/api/chat`` endpoint instead of
raw ``/api/generate`` phrasing. The model is constrained with a JSON schema and
all callers still validate the returned facts against SQLite memory before the
answer is trusted.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OllamaStatus:
    """Connectivity and model-availability diagnostic for Ollama."""

    ok: bool
    base_url: str
    model: str
    model_available: bool
    latency_ms: float
    error: str | None = None
    available_models: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "base_url": self.base_url,
            "model": self.model,
            "model_available": self.model_available,
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "available_models": list(self.available_models),
        }


@dataclass(frozen=True)
class LLMResponse:
    """Result from a local LLM request.

    ``text`` is the raw assistant content. ``json_data`` is populated only when
    the raw content parsed as a JSON object. ``used_llm`` means Ollama returned a
    non-empty response; it does *not* mean the answer is trusted. Trust is decided
    later by the recall validator.
    """

    text: str
    attempted: bool
    used_llm: bool
    latency_ms: float
    error: str | None = None
    fallback_reason: str | None = None
    json_data: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "attempted": self.attempted,
            "used_llm": self.used_llm,
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "fallback_reason": self.fallback_reason,
            "json_data": self.json_data,
        }


class OllamaClient:
    """Small Ollama wrapper using only the Python standard library."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_s: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)

    def check_connectivity(self) -> OllamaStatus:
        """Return whether Ollama is reachable and whether the requested model exists."""

        start = time.perf_counter()
        request = urllib.request.Request(url=f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_s, 3.0)) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            model_names = _extract_model_names(parsed)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            model_available = _model_name_matches(self.model, model_names)
            return OllamaStatus(
                ok=True,
                base_url=self.base_url,
                model=self.model,
                model_available=model_available,
                latency_ms=elapsed_ms,
                error=None if model_available else f"model_not_found: {self.model}",
                available_models=tuple(model_names),
            )
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return OllamaStatus(False, self.base_url, self.model, False, elapsed_ms, f"ollama_connection_error: {exc}")
        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return OllamaStatus(False, self.base_url, self.model, False, elapsed_ms, f"ollama_timeout: {exc}")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return OllamaStatus(False, self.base_url, self.model, False, elapsed_ms, f"ollama_status_error: {exc}")

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        *,
        max_tokens: int = 180,
    ) -> LLMResponse:
        """Call Ollama ``/api/chat`` and request a JSON object matching ``schema``."""

        start = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "format": dict(schema),
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": int(max_tokens),
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            message = parsed.get("message", {})
            text = ""
            if isinstance(message, dict):
                text = str(message.get("content", "")).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not text:
                return LLMResponse(
                    text="",
                    attempted=True,
                    used_llm=False,
                    latency_ms=elapsed_ms,
                    error="ollama_returned_empty_chat_content",
                    fallback_reason="empty_response",
                )
            json_data = _parse_json_object(text)
            if json_data is None:
                return LLMResponse(
                    text=text,
                    attempted=True,
                    used_llm=True,
                    latency_ms=elapsed_ms,
                    error="ollama_returned_invalid_json",
                    fallback_reason="invalid_json",
                    json_data=None,
                )
            return LLMResponse(
                text=text,
                attempted=True,
                used_llm=True,
                latency_ms=elapsed_ms,
                error=None,
                fallback_reason=None,
                json_data=json_data,
            )
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            body_text = _read_error_body(exc)
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_http_error_{exc.code}: {body_text or exc.reason}",
                fallback_reason="http_error",
            )
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_connection_error: {exc}",
                fallback_reason="connection_error",
            )
        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_timeout: {exc}",
                fallback_reason="timeout",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_error: {exc}",
                fallback_reason="runtime_error",
            )

    def generate(self, prompt: str) -> LLMResponse:
        """Backward-compatible helper; prefer ``chat_json`` for recall."""

        schema = {
            "type": "object",
            "properties": {"spoken_answer": {"type": "string"}},
            "required": ["spoken_answer"],
            "additionalProperties": False,
        }
        return self.chat_json([{"role": "user", "content": prompt}], schema)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Some models wrap JSON in text despite schema mode. Recover only when a
        # single clear object exists; otherwise reject so validation can log why.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _extract_model_names(payload: dict[str, Any]) -> list[str]:
    models = payload.get("models", [])
    names: list[str] = []
    if isinstance(models, list):
        for model_info in models:
            if not isinstance(model_info, dict):
                continue
            name = str(model_info.get("name") or model_info.get("model") or "").strip()
            if name:
                names.append(name)
    return sorted(set(names))


def _model_name_matches(requested: str, available: list[str]) -> bool:
    requested = requested.strip()
    if not requested:
        return False
    for name in available:
        if name == requested:
            return True
        if name.startswith(requested + ":"):
            return True
        if requested.endswith(":latest") and name == requested.removesuffix(":latest"):
            return True
    return False


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip()[:300]
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Ollama connectivity for LeLamp grounded recall")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-model", type=str, default="llama3.2")
    parser.add_argument("--ollama-timeout", type=float, default=8.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OllamaClient(args.ollama_url, args.ollama_model, timeout_s=args.ollama_timeout)
    status = client.check_connectivity()
    if args.json:
        print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
    else:
        if status.ok and status.model_available:
            print(f"Ollama OK: {status.base_url} has model {status.model} ({status.latency_ms:.1f} ms)")
        elif status.ok:
            print(f"Ollama reachable, but model '{status.model}' was not found.")
            print("Available models: " + (", ".join(status.available_models) or "none"))
        else:
            print(f"Ollama unavailable: {status.error}")
    return 0 if status.ok and status.model_available else 1


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/conversation/query_parser.py`

```python

"""Deterministic object-query parser for grounded memory recall.

Milestone 4.1 keeps parsing deterministic and local. The parser extracts one
object target from short recall questions such as:
    - Where did you last see my phone?
    - Where is the cup?
    - Have you seen my mouse?
    - Did you see my spectacles that I left near the cup?

Important MVP rule:
    Nearby/context objects are not treated as the recall target when the user
    explicitly asks about a different possessive object. For example,
    "my spectacles ... near the cup" parses as ``glasses``, not ``cup``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Kept local to conversation code so the recall CLI does not import OpenCV,
# Ultralytics, or the object detector just to parse text.
OBJECT_ALIASES: dict[str, str] = {
    "phone": "phone",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "laptop": "laptop",
    "computer": "laptop",
    "notebook": "laptop",
    "monitor": "monitor",
    "screen": "monitor",
    "tv": "monitor",
    "television": "monitor",
    "cup": "cup",
    "mug": "cup",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "book": "book",
    "bottle": "bottle",
    "remote": "remote",
    "remote control": "remote",
    "clock": "clock",
    "chair": "chair",
    "backpack": "backpack",
    "bag": "bag",
    "handbag": "bag",
    "vase": "vase",
    "scissors": "scissors",
    "stapler": "stapler",
    "pen": "pen",
    "pencil": "pencil",
    "spectacles": "glasses",
    "glasses": "glasses",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "around",
    "at",
    "by",
    "chance",
    "did",
    "do",
    "for",
    "have",
    "having",
    "had",
    "has",
    "if",
    "i",
    "is",
    "it",
    "last",
    "locate",
    "location",
    "detect",
    "detected",
    "objects",
    "object",
    "me",
    "my",
    "near",
    "of",
    "on",
    "please",
    "remember",
    "see",
    "seen",
    "show",
    "that",
    "the",
    "there",
    "was",
    "were",
    "where",
    "you",
}

# Clause boundaries keep relation/context phrases from being interpreted as the
# target object. In "my spectacles that I left near the cup", the target phrase
# ends before "that" and the nearby cup is ignored for target selection.
BOUNDARY_WORDS = {
    "around",
    "at",
    "behind",
    "beside",
    "by",
    "from",
    "i",
    "in",
    "inside",
    "left",
    "near",
    "next",
    "on",
    "over",
    "right",
    "that",
    "under",
    "which",
    "with",
    "you",
}

_BOUNDARY_REGEX = "|".join(sorted(re.escape(word) for word in BOUNDARY_WORDS))
_TARGET_TEXT = rf"(?P<object>[a-z0-9][a-z0-9\s'-]*?)(?=\s+(?:{_BOUNDARY_REGEX})\b|\s*$)"

OBJECT_PHRASE_PATTERNS = [
    re.compile(rf"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bhave\s+you\s+seen\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bdid\s+you\s+see\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\b(?:find|locate)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\b(?:was|were)\s+(?:i\s+)?(?:having|using|holding)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
]

POSSESSIVE_TARGET_PATTERN = re.compile(rf"\bmy\s+{_TARGET_TEXT}")

LIST_OBJECTS_PATTERNS = [
    re.compile(r"\bwhat\s+(?:objects|things|items)\s+did\s+you\s+(?:detect|see|remember)\b"),
    re.compile(r"\bwhat\s+do\s+you\s+remember\s+seeing\b"),
    re.compile(r"\bwhat\s+have\s+you\s+(?:seen|detected)\b"),
    re.compile(r"\bshow\s+me\s+(?:recent|the)?\s*(?:objects|things|items)\b"),
]

HAD_OBJECT_PATTERNS = [
    re.compile(rf"\bdid\s+you\s+see\s+if\s+i\s+(?:had|have|was\s+holding|was\s+using)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bdid\s+you\s+notice\s+if\s+i\s+(?:had|have|was\s+holding|was\s+using)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
]


@dataclass(frozen=True)
class ParsedObjectQuery:
    original_query: str
    target_text: str | None
    normalized_label: str | None
    confidence: float
    strategy: str

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "target_text": self.target_text,
            "normalized_label": self.normalized_label,
            "confidence": round(float(self.confidence), 3),
            "strategy": self.strategy,
        }


def clean_query_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"[^a-z0-9\s'-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def is_list_recent_objects_query(text: str) -> bool:
    cleaned = clean_query_text(text)
    return any(pattern.search(cleaned) is not None for pattern in LIST_OBJECTS_PATTERNS)


def normalize_object_label(label: str) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
    if cleaned in OBJECT_ALIASES:
        return OBJECT_ALIASES[cleaned]
    # Conservative singular fallback for simple plurals: cups -> cup.
    if cleaned.endswith("s") and cleaned[:-1] in OBJECT_ALIASES:
        return OBJECT_ALIASES[cleaned[:-1]]
    return cleaned


def parse_object_query(query: str, aliases: dict[str, str] | None = None) -> ParsedObjectQuery:
    alias_map = aliases or OBJECT_ALIASES
    cleaned = clean_query_text(query)
    if not cleaned:
        return ParsedObjectQuery(query, None, None, 0.0, "empty_query")

    if is_list_recent_objects_query(cleaned):
        return ParsedObjectQuery(query, None, None, 0.0, "list_recent_objects_query")

    # Strong phrasing that previously failed as target="if":
    # "Did you see if I had a stapler?" means the object is stapler.
    for pattern in HAD_OBJECT_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = _normalize_with_aliases(target, alias_map)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.86,
                strategy="had_object_pattern",
            )

    # Strongest signal: explicit possessive target. This fixes questions like
    # "did you see my spectacles that I left near the cup?" by selecting
    # spectacles/glasses instead of the contextual cup.
    possessive_target = _extract_possessive_target(cleaned)
    if possessive_target:
        normalized = _normalize_with_aliases(possessive_target, alias_map)
        return ParsedObjectQuery(
            original_query=query,
            target_text=possessive_target,
            normalized_label=normalized or None,
            confidence=0.98,
            strategy="explicit_possessive",
        )

    # Next, parse a direct object from common recall question shapes. This can
    # return unsupported labels such as "stapler". The recall agent will then do
    # an exact memory lookup and answer that it does not remember seeing it.
    for pattern in OBJECT_PHRASE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = _normalize_with_aliases(target, alias_map)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.75,
                strategy="question_pattern",
            )

    # Known alias fallback for compact queries like "cup?" or "phone location".
    # Longest alias wins so "cell phone" wins before "phone".
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if _contains_phrase(cleaned, alias):
            return ParsedObjectQuery(
                original_query=query,
                target_text=alias,
                normalized_label=alias_map[alias],
                confidence=0.60,
                strategy="known_alias_fallback",
            )

    fallback = _last_content_token(cleaned.split())
    if fallback:
        normalized = _normalize_with_aliases(fallback, alias_map)
        return ParsedObjectQuery(
            original_query=query,
            target_text=fallback,
            normalized_label=normalized or None,
            confidence=0.35,
            strategy="fallback_last_token",
        )

    return ParsedObjectQuery(query, None, None, 0.0, "no_object_found")


def _extract_possessive_target(text: str) -> str | None:
    match = POSSESSIVE_TARGET_PATTERN.search(text)
    if not match:
        return None
    return _sanitize_object_phrase(match.group("object")) or None


def _normalize_with_aliases(label: str, alias_map: dict[str, str]) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
    if cleaned in alias_map:
        return alias_map[cleaned]
    if cleaned.endswith("s") and cleaned[:-1] in alias_map:
        return alias_map[cleaned[:-1]]
    return cleaned


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _sanitize_object_phrase(phrase: str) -> str:
    cleaned = clean_query_text(phrase)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    tokens = [token for token in cleaned.split() if token not in STOPWORDS]
    return " ".join(tokens).strip(" -'")


def _trim_after_boundary(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for token in tokens:
        stripped = token.strip(" -'")
        if stripped in BOUNDARY_WORDS:
            break
        kept.append(stripped)
    return " ".join(token for token in kept if token)


def _last_content_token(tokens: Iterable[str]) -> str | None:
    for token in reversed(list(tokens)):
        token = token.strip(" -'")
        if token and token not in STOPWORDS:
            return token
    return None

```

## `backend/conversation/recall_agent.py`

```python

"""Grounded live conversation agent for LeLamp Milestone 4.3.

The recall path is deliberately bounded:

    user text -> deterministic intent parser -> SQLite retrieval -> optional
    structured Ollama /api/chat phrasing -> strict fact validation.

The LLM never searches memory, never chooses locations, and never overrides
SQLite. If the model returns invalid JSON or contradicts retrieved memory, the
candidate is rejected and the deterministic grounded answer is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.conversation.intent_parser import ConversationIntentType, ParsedConversationIntent, parse_conversation_intent
from backend.conversation.llm_client import LLMResponse, OllamaClient, OllamaStatus
from backend.conversation.query_parser import ParsedObjectQuery
from backend.memory.memory_store import MemoryRecord, MemoryStore


RECALL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_type": {
            "type": "string",
            "enum": ["memory_answer", "no_memory", "recent_objects", "unsupported"],
        },
        "target_object": {"type": ["string", "null"]},
        "location_label": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "timestamp": {"type": ["string", "null"]},
        "spoken_answer": {"type": "string"},
    },
    "required": ["answer_type", "target_object", "location_label", "confidence", "timestamp", "spoken_answer"],
    "additionalProperties": False,
}

SYSTEM_MESSAGE = """You are the voice of a LeLamp-inspired robotic lamp.
You are only a wording layer. The backend has already retrieved the only facts you may use.
Return exactly one JSON object matching the provided schema.
Do not invent object locations. Do not mention frame paths, SQLite, JSON, IDs, logs, files, or implementation details.
Keep spoken_answer to one concise conversational sentence."""


@dataclass(frozen=True)
class RecallResult:
    user_query: str
    intent: ParsedConversationIntent
    parsed_object: str | None
    parsed: ParsedObjectQuery
    memory_record: MemoryRecord | None
    recent_records: tuple[MemoryRecord, ...]
    answer: str
    answer_type: str
    memory_retrieval_ms: float
    llm_response_ms: float | None
    llm_requested: bool
    llm_required: bool
    llm_attempted: bool
    llm_used: bool
    llm_error: str | None
    llm_fallback_reason: str | None
    llm_validation_reason: str | None
    llm_raw_response: str | None
    ollama_connection_ok: bool | None
    timestamp: str

    @property
    def used_llm(self) -> bool:
        """Backward-compatible alias for Milestone 4 callers."""
        return self.llm_used

    @property
    def llm_required_failed(self) -> bool:
        return self.llm_required and self.llm_requested and not self.llm_used and self.answer_type in {
            "memory_answer",
            "recent_objects",
        }

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "intent": self.intent.to_dict(),
            "answer_type": self.answer_type,
            "parsed_object": self.parsed_object,
            "parsed": self.parsed.to_dict(),
            "retrieved_memory_id": self.memory_record.id if self.memory_record else None,
            "memory_record": self.memory_record.to_dict() if self.memory_record else None,
            "recent_records": [record.to_dict() for record in self.recent_records],
            "memory_retrieval_ms": round(float(self.memory_retrieval_ms), 3),
            "llm_response_ms": None if self.llm_response_ms is None else round(float(self.llm_response_ms), 3),
            "llm_requested": self.llm_requested,
            "llm_required": self.llm_required,
            "llm_attempted": self.llm_attempted,
            "llm_used": self.llm_used,
            "used_llm": self.llm_used,
            "llm_required_failed": self.llm_required_failed,
            "llm_error": self.llm_error,
            "llm_fallback_reason": self.llm_fallback_reason,
            "llm_validation_reason": self.llm_validation_reason,
            "llm_raw_response": self.llm_raw_response,
            "ollama_connection_ok": self.ollama_connection_ok,
            "answer": self.answer,
        }


class RecallAgent:
    """Answers supported memory questions using SQLite-grounded facts."""

    def __init__(
        self,
        memory_db: str | Path = "data/scene_memory.sqlite",
        use_llm: bool = False,
        llm_required: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        ollama_timeout_s: float = 8.0,
        recent_object_limit: int = 8,
        log_paths: str | Path | Iterable[str | Path] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.memory_db = Path(memory_db)
        self.store = MemoryStore(self.memory_db)
        self.use_llm = bool(use_llm)
        self.llm_required = bool(llm_required)
        self.recent_object_limit = int(recent_object_limit)
        self.logger = logger or logging.getLogger("lelamp")
        self.log_paths = _coerce_paths(log_paths)
        self.ollama: OllamaClient | None = None
        self.ollama_status: OllamaStatus | None = None

        if self.use_llm:
            self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model, timeout_s=ollama_timeout_s)
            self.ollama_status = self.ollama.check_connectivity()
            if self.ollama_status.ok and self.ollama_status.model_available:
                self.logger.info(
                    "Ollama connectivity OK url=%s model=%s status_ms=%.3f",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    self.ollama_status.latency_ms,
                )
            elif self.ollama_status.ok:
                self.logger.warning(
                    "Ollama reachable but requested model is unavailable url=%s model=%s available_models=%s",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    list(self.ollama_status.available_models),
                )
            else:
                self.logger.warning(
                    "Ollama unavailable for recall url=%s model=%s error=%s",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    self.ollama_status.error,
                )

    def answer(self, user_query: str) -> RecallResult:
        intent = parse_conversation_intent(user_query)
        parsed = intent.object_query

        retrieval_start = time.perf_counter()
        memory_record: MemoryRecord | None = None
        recent_records: tuple[MemoryRecord, ...] = ()

        if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN and parsed.normalized_label:
            memory_record = self.store.find_latest_by_normalized_label(parsed.normalized_label)
        elif intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
            recent_records = tuple(self.store.recent_unique_by_normalized_label(limit=self.recent_object_limit))

        memory_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fallback_answer, answer_type = deterministic_conversation_answer(intent, memory_record, recent_records)
        final_answer = fallback_answer
        final_answer_type = answer_type

        llm_response: LLMResponse | None = None
        llm_attempted = False
        llm_used = False
        llm_error: str | None = None
        llm_fallback_reason: str | None = None
        llm_validation_reason: str | None = None
        llm_raw_response: str | None = None
        ollama_connection_ok: bool | None = None

        if self.use_llm:
            ollama_connection_ok = False if self.ollama_status is None else (
                self.ollama_status.ok and self.ollama_status.model_available
            )
            llm_response = self._try_llm_answer(user_query, intent, memory_record, recent_records)
            llm_attempted = llm_response.attempted
            llm_error = llm_response.error
            llm_fallback_reason = llm_response.fallback_reason
            llm_raw_response = llm_response.text if llm_response.text else None

            if llm_response.used_llm and llm_response.json_data is not None:
                is_valid, validation_reason = validate_structured_llm_response(
                    llm_response.json_data,
                    intent,
                    memory_record,
                    recent_records,
                )
                if is_valid:
                    final_answer = str(llm_response.json_data["spoken_answer"]).strip()
                    final_answer_type = str(llm_response.json_data["answer_type"])
                    llm_used = True
                    llm_error = None
                    llm_fallback_reason = None
                    llm_validation_reason = None
                else:
                    llm_validation_reason = validation_reason
                    llm_fallback_reason = validation_reason
                    llm_error = f"invalid_llm_response: {validation_reason}"
            elif llm_response.attempted and llm_response.fallback_reason:
                llm_validation_reason = llm_response.fallback_reason

            if self.use_llm and not llm_used:
                self.logger.warning(
                    "LLM candidate rejected query=%r intent=%s parsed_object=%s reason=%s error=%s attempted=%s raw=%r",
                    user_query,
                    intent.intent.value,
                    parsed.normalized_label,
                    llm_fallback_reason,
                    llm_error,
                    llm_attempted,
                    llm_raw_response,
                )

        result = RecallResult(
            user_query=user_query,
            intent=intent,
            parsed_object=parsed.normalized_label,
            parsed=parsed,
            memory_record=memory_record,
            recent_records=recent_records,
            answer=final_answer,
            answer_type=final_answer_type,
            memory_retrieval_ms=memory_retrieval_ms,
            llm_response_ms=None if llm_response is None else llm_response.latency_ms,
            llm_requested=self.use_llm,
            llm_required=self.llm_required,
            llm_attempted=llm_attempted,
            llm_used=llm_used,
            llm_error=llm_error,
            llm_fallback_reason=llm_fallback_reason,
            llm_validation_reason=llm_validation_reason,
            llm_raw_response=llm_raw_response,
            ollama_connection_ok=ollama_connection_ok,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )
        self._log_result(result)
        return result

    def _try_llm_answer(
        self,
        user_query: str,
        intent: ParsedConversationIntent,
        memory_record: MemoryRecord | None,
        recent_records: Sequence[MemoryRecord],
    ) -> LLMResponse:
        if intent.intent == ConversationIntentType.UNSUPPORTED:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, fallback_reason="unsupported_intent")
        if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN and memory_record is None:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, fallback_reason="no_memory_record")
        if intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS and not recent_records:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, fallback_reason="no_recent_objects")
        if self.ollama is None:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, error="ollama_client_not_initialized", fallback_reason="client_unavailable")
        if self.ollama_status is not None and not self.ollama_status.ok:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, error=self.ollama_status.error or "ollama_unavailable", fallback_reason="connectivity_check_failed")
        if self.ollama_status is not None and not self.ollama_status.model_available:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, error=self.ollama_status.error or f"model_not_found: {self.ollama.model}", fallback_reason="model_unavailable")

        messages = build_structured_messages(user_query, intent, memory_record, recent_records)
        return self.ollama.chat_json(messages, RECALL_RESPONSE_SCHEMA, max_tokens=180)

    def _log_result(self, result: RecallResult) -> None:
        payload = result.to_dict()
        self.logger.info(
            "Recall query=%r intent=%s parsed_object=%s answer_type=%s memory_id=%s recent_count=%s retrieval_ms=%.3f llm_attempted=%s llm_used=%s llm_ms=%s llm_error=%s fallback=%s answer=%r",
            result.user_query,
            result.intent.intent.value,
            result.parsed_object,
            result.answer_type,
            result.memory_record.id if result.memory_record else None,
            len(result.recent_records),
            result.memory_retrieval_ms,
            result.llm_attempted,
            result.llm_used,
            "" if result.llm_response_ms is None else f"{result.llm_response_ms:.3f}",
            result.llm_error,
            result.llm_fallback_reason,
            result.answer,
        )
        for path in self.log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_structured_messages(
    user_query: str,
    intent: ParsedConversationIntent,
    memory_record: MemoryRecord | None,
    recent_records: Sequence[MemoryRecord],
) -> list[dict[str, str]]:
    facts: dict[str, Any] = {
        "user_query": user_query,
        "intent": intent.intent.value,
        "parsed_object": intent.normalized_label,
        "schema": RECALL_RESPONSE_SCHEMA,
    }

    if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN and memory_record is not None:
        facts["retrieved_memory"] = _memory_record_for_llm(memory_record)
        facts["required_json_values"] = {
            "answer_type": "memory_answer",
            "target_object": memory_record.normalized_label,
            "location_label": memory_record.location_label,
            "confidence": round(float(memory_record.confidence), 3),
            "timestamp": memory_record.timestamp,
        }
        facts["fallback_answer"] = deterministic_object_answer(intent.object_query, memory_record)
    elif intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
        labels = [record.normalized_label for record in recent_records]
        facts["recent_objects"] = [_memory_record_for_llm(record) for record in recent_records]
        facts["allowed_labels"] = labels
        facts["required_json_values"] = {
            "answer_type": "recent_objects",
            "target_object": None,
            "location_label": None,
            "confidence": None,
            "timestamp": None,
        }
        facts["fallback_answer"] = deterministic_recent_objects_answer(recent_records)

    user_content = (
        "Use only these backend-provided memory facts. Return exactly one JSON object.\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def validate_structured_llm_response(
    candidate: Mapping[str, Any],
    intent: ParsedConversationIntent,
    memory_record: MemoryRecord | None,
    recent_records: Sequence[MemoryRecord],
) -> tuple[bool, str | None]:
    """Validate structured LLM output against retrieved SQLite facts."""

    required_keys = {"answer_type", "target_object", "location_label", "confidence", "timestamp", "spoken_answer"}
    missing = sorted(required_keys.difference(candidate.keys()))
    if missing:
        return False, "missing_schema_keys:" + ",".join(missing)

    answer_type = str(candidate.get("answer_type") or "").strip()
    spoken_answer = str(candidate.get("spoken_answer") or "").strip()
    if not spoken_answer:
        return False, "empty_spoken_answer"
    leak_reason = _validate_spoken_answer_text(spoken_answer, memory_record is not None)
    if leak_reason is not None:
        return False, leak_reason

    if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN:
        if memory_record is None:
            return False, "no_memory_record"
        if answer_type != "memory_answer":
            return False, "wrong_answer_type"
        expected_object = str(memory_record.normalized_label or intent.normalized_label or "").strip().lower()
        target_object = str(candidate.get("target_object") or "").strip().lower()
        if target_object != expected_object:
            return False, "target_object_mismatch"
        location_label = candidate.get("location_label")
        if not isinstance(location_label, str) or location_label != memory_record.location_label:
            return False, "location_label_mismatch"
        if memory_record.location_label.lower() not in spoken_answer.lower():
            return False, "spoken_answer_missing_location"
        if expected_object and expected_object not in spoken_answer.lower():
            return False, "spoken_answer_missing_target_object"
        if not _candidate_confidence_matches(candidate.get("confidence"), memory_record.confidence) and str(candidate.get("timestamp") or "") != memory_record.timestamp:
            return False, "missing_matching_confidence_or_timestamp"
        return True, None

    if intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
        if not recent_records:
            return False, "no_recent_objects"
        if answer_type != "recent_objects":
            return False, "wrong_answer_type"
        if candidate.get("target_object") is not None:
            return False, "target_object_should_be_null"
        if candidate.get("location_label") is not None:
            return False, "location_label_should_be_null"
        allowed_labels = [record.normalized_label.strip().lower() for record in recent_records if record.normalized_label.strip()]
        lower_answer = spoken_answer.lower()
        missing_labels = [label for label in allowed_labels if label not in lower_answer]
        if missing_labels:
            return False, "missing_recent_labels:" + ",".join(missing_labels)
        return True, None

    if answer_type != "unsupported":
        return False, "unsupported_intent_wrong_answer_type"
    return True, None


def deterministic_conversation_answer(
    intent: ParsedConversationIntent,
    memory_record: MemoryRecord | None,
    recent_records: Sequence[MemoryRecord],
) -> tuple[str, str]:
    if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN:
        if memory_record is None:
            return deterministic_no_memory_answer(intent.object_query), "no_memory"
        return deterministic_object_answer(intent.object_query, memory_record), "memory_answer"
    if intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
        if not recent_records:
            return "I do not remember seeing any objects yet.", "no_memory"
        return deterministic_recent_objects_answer(recent_records), "recent_objects"
    return (
        "I can answer grounded memory questions, like where I last saw your phone, or what objects I detected.",
        "unsupported",
    )


def deterministic_recall_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    """Backward-compatible object-only deterministic answer."""

    if memory_record is None:
        return deterministic_no_memory_answer(parsed)
    return deterministic_object_answer(parsed, memory_record)


def deterministic_object_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord) -> str:
    display_label = _display_object_label(parsed, memory_record)
    time_text = _friendly_time(memory_record.timestamp)
    return (
        f"I last saw your {display_label} on the {memory_record.location_label} "
        f"around {time_text}. My confidence was {memory_record.confidence:.2f}."
    )


def deterministic_no_memory_answer(parsed: ParsedObjectQuery) -> str:
    display_label = _display_object_label(parsed, None)
    if display_label == "that object":
        return "I can answer object-memory questions, like where I last saw your cup or phone."
    return f"I do not remember seeing your {display_label}."


def deterministic_recent_objects_answer(records: Sequence[MemoryRecord]) -> str:
    labels = []
    seen: set[str] = set()
    for record in records:
        label = record.normalized_label.strip().lower()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    if not labels:
        return "I do not remember seeing any objects yet."
    return f"I recently saw {_join_labels(labels)}."


def _join_labels(labels: Sequence[str]) -> str:
    if len(labels) == 1:
        return f"a {labels[0]}"
    if len(labels) == 2:
        return f"a {labels[0]} and {labels[1]}"
    prefixed = [f"a {labels[0]}"] + list(labels[1:])
    return ", ".join(prefixed[:-1]) + f", and {prefixed[-1]}"


def _memory_record_for_llm(memory_record: MemoryRecord) -> dict:
    return {
        "object_label": memory_record.object_label,
        "normalized_label": memory_record.normalized_label,
        "location_label": memory_record.location_label,
        "confidence": round(float(memory_record.confidence), 3),
        "timestamp": memory_record.timestamp,
        "source": memory_record.source,
    }


def _display_object_label(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    if memory_record is not None:
        return memory_record.normalized_label or memory_record.object_label
    if parsed.normalized_label:
        return parsed.normalized_label
    if parsed.target_text:
        return parsed.target_text
    return "that object"


def _friendly_time(timestamp_text: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp_text)
    except ValueError:
        return timestamp_text
    return parsed.strftime("%H:%M:%S on %Y-%m-%d")


def _candidate_confidence_matches(candidate_value: Any, expected_confidence: float) -> bool:
    try:
        candidate = float(candidate_value)
    except (TypeError, ValueError):
        return False
    return abs(candidate - float(expected_confidence)) <= 0.001


def _validate_spoken_answer_text(spoken_answer: str, memory_exists: bool) -> str | None:
    lowered = " ".join(spoken_answer.strip().split()).lower()
    implementation_leaks = (
        "frame_path",
        "frame path",
        ".jpg",
        ".png",
        "data\\",
        "data/",
        "json",
        "sqlite",
        "memory id",
        "database",
        "log",
        "file",
    )
    if any(marker in lowered for marker in implementation_leaks):
        return "implementation_detail_leak"

    contradictory_phrases = (
        "do not remember",
        "don't remember",
        "dont remember",
        "not remember",
        "no memory",
        "can't remember",
        "cannot remember",
        "do not recall",
        "don't recall",
        "have not seen",
        "haven't seen",
        "never seen",
    )
    if memory_exists and any(phrase in lowered for phrase in contradictory_phrases):
        return "contradicts_retrieved_memory"
    return None


def _validate_llm_grounding(
    answer: str,
    parsed: ParsedObjectQuery,
    memory_record: MemoryRecord | None,
) -> tuple[bool, str | None]:
    """Backward-compatible validator for old one-sentence tests.

    Milestone 4.3 uses ``validate_structured_llm_response``. This helper remains
    for existing guardrail tests and rejects the same unsafe cases.
    """

    if memory_record is None:
        return False, "no_memory_record"
    leak_reason = _validate_spoken_answer_text(answer, memory_exists=True)
    if leak_reason is not None:
        return False, leak_reason
    lowered = " ".join(str(answer or "").strip().split()).lower()
    if not lowered:
        return False, "empty_llm_answer"
    expected_location = str(memory_record.location_label or "").strip().lower()
    if expected_location and expected_location not in lowered:
        return False, "missing_stored_location"
    object_terms = {
        str(memory_record.normalized_label or "").strip().lower(),
        str(memory_record.object_label or "").strip().lower(),
        str(parsed.normalized_label or "").strip().lower(),
        str(parsed.target_text or "").strip().lower(),
    }
    object_terms = {term for term in object_terms if term}
    if object_terms and not any(term in lowered for term in object_terms):
        return False, "missing_target_object"
    expected_confidence_options = {f"{float(memory_record.confidence):.2f}", f"{float(memory_record.confidence):.3f}"}
    has_confidence = any(conf in lowered for conf in expected_confidence_options)
    has_timestamp = str(memory_record.timestamp or "").strip().lower() in lowered
    if not has_confidence and not has_timestamp:
        return False, "missing_confidence_or_timestamp"
    return True, None


def _coerce_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask LeLamp grounded questions about stored object memory")
    parser.add_argument("query", nargs="?", help="Question, e.g. 'Where did you last see my phone?'")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase retrieved memory answers")
    parser.add_argument("--llm-required", action="store_true", help="Return nonzero if an LLM-eligible answer falls back")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model, e.g. llama3.2 or qwen2.5")
    parser.add_argument("--ollama-timeout", type=float, default=8.0, help="Ollama chat timeout in seconds")
    parser.add_argument("--test-ollama", action="store_true", help="Check Ollama connectivity/model availability and exit")
    parser.add_argument("--json", action="store_true", help="Print machine-readable recall result JSON")
    parser.add_argument("--debug", action="store_true", help="Print debug metadata such as frame_path outside the spoken answer")
    parser.add_argument("--log-path", type=str, default="logs/recall.jsonl", help="JSONL recall log path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test_ollama:
        client = OllamaClient(args.ollama_url, args.ollama_model, timeout_s=args.ollama_timeout)
        status = client.check_connectivity()
        if args.json:
            print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
        else:
            if status.ok and status.model_available:
                print(f"Ollama OK: {status.base_url} has model {status.model} ({status.latency_ms:.1f} ms)")
            elif status.ok:
                print(f"Ollama reachable, but model '{status.model}' was not found.")
                print("Available models: " + (", ".join(status.available_models) or "none"))
            else:
                print(f"Ollama unavailable: {status.error}")
        return 0 if status.ok and status.model_available else 1

    if not args.query:
        print("Missing query. Example: python -m backend.conversation.recall_agent \"Where is the cup?\"")
        return 2

    console_level = logging.WARNING if args.json else logging.INFO
    logging.basicConfig(level=console_level, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = RecallAgent(
        memory_db=args.memory_db,
        use_llm=args.use_llm,
        llm_required=args.llm_required,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_timeout_s=args.ollama_timeout,
        log_paths=args.log_path,
    )
    result = agent.answer(args.query)

    if args.use_llm and not result.llm_used:
        warning = (
            "Warning: --use-llm was requested, but the deterministic fallback was used. "
            f"reason={result.llm_fallback_reason or 'unknown'} error={result.llm_error or 'none'}"
        )
        print(warning, file=sys.stderr)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(
            f"intent={result.intent.intent.value} "
            f"answer_type={result.answer_type} "
            f"parsed_object={result.parsed_object or 'none'} "
            f"memory_id={result.memory_record.id if result.memory_record else 'none'} "
            f"recent_count={len(result.recent_records)} "
            f"memory_retrieval_ms={result.memory_retrieval_ms:.3f} "
            f"llm_attempted={result.llm_attempted} "
            f"llm_used={result.llm_used} "
            f"llm_response_ms={'' if result.llm_response_ms is None else f'{result.llm_response_ms:.3f}'} "
            f"llm_error={result.llm_error or 'none'} "
            f"llm_fallback_reason={result.llm_fallback_reason or 'none'}"
        )
        if args.debug and result.memory_record is not None:
            print(f"frame_path={result.memory_record.frame_path or 'none'}")

    if result.llm_required_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/conversation/test_chat_udp_receiver.py`

```python

"""Milestone 4.3 Godot chat packet parser tests.

Run:
    python -m unittest backend.conversation.test_chat_udp_receiver -v
"""

from __future__ import annotations

import unittest

from backend.conversation.chat_udp_receiver import parse_chat_packet


class ChatUdpReceiverTests(unittest.TestCase):
    def test_accepts_valid_chat_query(self) -> None:
        packet = parse_chat_packet(
            {
                "type": "chat_query",
                "timestamp": "2026-06-18T13:30:00",
                "text": "Where did you last see my phone?",
            },
            ("127.0.0.1", 50000),
        )
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.text, "Where did you last see my phone?")
        self.assertEqual(packet.timestamp, "2026-06-18T13:30:00")

    def test_rejects_wrong_type(self) -> None:
        self.assertIsNone(parse_chat_packet({"type": "not_chat", "text": "hello"}))

    def test_rejects_empty_text(self) -> None:
        self.assertIsNone(parse_chat_packet({"type": "chat_query", "text": "   "}))


if __name__ == "__main__":
    unittest.main()

```

## `backend/conversation/test_intent_parser.py`

```python

"""Milestone 4.3 conversation intent tests.

Run:
    python -m unittest backend.conversation.test_intent_parser -v
"""

from __future__ import annotations

import unittest

from backend.conversation.intent_parser import ConversationIntentType, parse_conversation_intent


class IntentParserTests(unittest.TestCase):
    def assert_intent(self, query: str, expected_intent: ConversationIntentType, expected_object: str | None = None) -> None:
        parsed = parse_conversation_intent(query)
        self.assertEqual(parsed.intent, expected_intent, msg=parsed.to_dict())
        self.assertEqual(parsed.normalized_label, expected_object, msg=parsed.to_dict())

    def test_where_last_phone(self) -> None:
        self.assert_intent("Where did you last see my phone?", ConversationIntentType.OBJECT_LAST_SEEN, "phone")

    def test_where_is_cup(self) -> None:
        self.assert_intent("Where is my cup?", ConversationIntentType.OBJECT_LAST_SEEN, "cup")

    def test_have_you_seen_mouse(self) -> None:
        self.assert_intent("Have you seen my mouse?", ConversationIntentType.OBJECT_LAST_SEEN, "mouse")

    def test_did_you_see_if_i_had_stapler(self) -> None:
        self.assert_intent("Did you see if I had a stapler?", ConversationIntentType.OBJECT_LAST_SEEN, "stapler")

    def test_what_objects_detected(self) -> None:
        self.assert_intent("What objects did you detect?", ConversationIntentType.LIST_RECENT_OBJECTS, None)

    def test_what_do_you_remember_seeing(self) -> None:
        self.assert_intent("What do you remember seeing?", ConversationIntentType.LIST_RECENT_OBJECTS, None)

    def test_unsupported_chitchat(self) -> None:
        self.assert_intent("Tell me a joke", ConversationIntentType.UNSUPPORTED, None)


if __name__ == "__main__":
    unittest.main()

```

## `backend/conversation/test_query_parser.py`

```python

"""Milestone 4.3 parser regression tests.

Run:
    python -m unittest backend.conversation.test_query_parser -v
"""

from __future__ import annotations

import unittest

from backend.conversation.query_parser import parse_object_query


class QueryParserTests(unittest.TestCase):
    def assert_parses_to(self, query: str, expected: str | None) -> None:
        parsed = parse_object_query(query)
        self.assertEqual(parsed.normalized_label, expected, msg=parsed.to_dict())

    def test_phone(self) -> None:
        self.assert_parses_to("Where is my phone?", "phone")

    def test_cup(self) -> None:
        self.assert_parses_to("Where did you last see my cup?", "cup")

    def test_mouse(self) -> None:
        self.assert_parses_to("Have you seen my mouse?", "mouse")

    def test_spectacles_near_cup_prefers_possessive_target(self) -> None:
        self.assert_parses_to("By any chance, did you see my spectacles that I left near the cup?", "glasses")

    def test_stapler_unknown_object_still_parses_as_target(self) -> None:
        self.assert_parses_to("Where is the stapler?", "stapler")

    def test_did_you_see_if_i_had_stapler_does_not_parse_as_if(self) -> None:
        parsed = parse_object_query("Did you see if I had a stapler?")
        self.assertEqual(parsed.normalized_label, "stapler", msg=parsed.to_dict())
        self.assertNotEqual(parsed.normalized_label, "if")

    def test_list_query_does_not_parse_detected_as_object(self) -> None:
        parsed = parse_object_query("What objects did you detect?")
        self.assertIsNone(parsed.normalized_label, msg=parsed.to_dict())
        self.assertEqual(parsed.strategy, "list_recent_objects_query")


if __name__ == "__main__":
    unittest.main()

```

## `backend/conversation/test_recall_llm_guardrails.py`

```python

"""Milestone 4.3 structured LLM grounding tests.

Run:
    python -m unittest backend.conversation.test_recall_llm_guardrails -v
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.conversation.intent_parser import parse_conversation_intent
from backend.conversation.llm_client import LLMResponse, OllamaStatus
from backend.conversation.query_parser import parse_object_query
from backend.conversation.recall_agent import RecallAgent, validate_structured_llm_response, _validate_llm_grounding
from backend.memory.memory_store import MemoryRecord, MemoryStore


class FakeOllama:
    def __init__(self, payload: Mapping[str, Any] | str) -> None:
        self.payload = payload
        self.model = "fake-model"

    def chat_json(self, messages: Sequence[Mapping[str, str]], schema: Mapping[str, Any], *, max_tokens: int = 180) -> LLMResponse:
        if isinstance(self.payload, str):
            return LLMResponse(
                text=self.payload,
                attempted=True,
                used_llm=True,
                latency_ms=12.5,
                error="ollama_returned_invalid_json",
                fallback_reason="invalid_json",
                json_data=None,
            )
        return LLMResponse(
            text=json.dumps(self.payload),
            attempted=True,
            used_llm=True,
            latency_ms=12.5,
            error=None,
            fallback_reason=None,
            json_data=dict(self.payload),
        )


class LlmGroundingGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = parse_conversation_intent("Where did you last see my phone?")
        self.parsed = parse_object_query("Where did you last see my phone?")
        self.record = MemoryRecord(
            id="test-phone-record",
            object_label="cell phone",
            normalized_label="phone",
            location_label="center of view",
            bbox=(447, 487, 100, 96),
            confidence=0.733,
            timestamp="2026-06-18T13:22:33",
            source="webcam",
            frame_path="data\\object_frames\\frame_493_phone.jpg",
        )

    def valid_payload(self) -> dict[str, Any]:
        return {
            "answer_type": "memory_answer",
            "target_object": "phone",
            "location_label": "center of view",
            "confidence": 0.733,
            "timestamp": "2026-06-18T13:22:33",
            "spoken_answer": "I last saw your phone on the center of view around 2026-06-18T13:22:33.",
        }

    def assert_invalid_structured(self, payload: Mapping[str, Any], reason: str) -> None:
        ok, actual_reason = validate_structured_llm_response(payload, self.intent, self.record, ())
        self.assertFalse(ok, msg=payload)
        self.assertEqual(actual_reason, reason)

    def test_accepts_valid_structured_memory_answer(self) -> None:
        ok, reason = validate_structured_llm_response(self.valid_payload(), self.intent, self.record, ())
        self.assertTrue(ok, reason)

    def test_rejects_no_memory_answer_when_record_exists(self) -> None:
        payload = self.valid_payload()
        payload["answer_type"] = "no_memory"
        payload["target_object"] = None
        payload["location_label"] = None
        payload["confidence"] = None
        payload["timestamp"] = None
        payload["spoken_answer"] = "I do not remember seeing that."
        self.assert_invalid_structured(payload, "contradicts_retrieved_memory")

    def test_rejects_wrong_location_label(self) -> None:
        payload = self.valid_payload()
        payload["location_label"] = "right side of view"
        payload["spoken_answer"] = "I last saw your phone on the right side of view around 2026-06-18T13:22:33."
        self.assert_invalid_structured(payload, "location_label_mismatch")

    def test_rejects_frame_path_leak(self) -> None:
        payload = self.valid_payload()
        payload["spoken_answer"] = "I last saw your phone on the center of view. Frame path: data\\object_frames\\frame_493_phone.jpg"
        self.assert_invalid_structured(payload, "implementation_detail_leak")

    def test_rejects_missing_matching_confidence_and_timestamp(self) -> None:
        payload = self.valid_payload()
        payload["confidence"] = 0.111
        payload["timestamp"] = "2026-06-18T13:00:00"
        self.assert_invalid_structured(payload, "missing_matching_confidence_or_timestamp")

    def test_old_sentence_validator_reproduces_previous_bad_output(self) -> None:
        ok, reason = _validate_llm_grounding("I do not remember seeing that.", self.parsed, self.record)
        self.assertFalse(ok)
        self.assertEqual(reason, "contradicts_retrieved_memory")

    def make_agent_with_fake(self, db_path: Path, payload: Mapping[str, Any] | str) -> RecallAgent:
        agent = RecallAgent(memory_db=db_path, use_llm=False, logger=logging.getLogger("test"))
        agent.use_llm = True
        agent.ollama = FakeOllama(payload)  # type: ignore[assignment]
        agent.ollama_status = OllamaStatus(
            ok=True,
            base_url="http://localhost:11434",
            model="fake-model",
            model_available=True,
            latency_ms=1.0,
        )
        return agent

    def test_agent_rejects_structured_llm_that_contradicts_record(self) -> None:
        payload = self.valid_payload()
        payload["answer_type"] = "no_memory"
        payload["target_object"] = None
        payload["location_label"] = None
        payload["confidence"] = None
        payload["timestamp"] = None
        payload["spoken_answer"] = "I do not remember seeing that."
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)
            agent = self.make_agent_with_fake(db_path, payload)
            result = agent.answer("Where did you last see my phone?")

        self.assertTrue(result.llm_attempted)
        self.assertFalse(result.llm_used)
        self.assertEqual(result.llm_fallback_reason, "contradicts_retrieved_memory")
        self.assertIn("invalid_llm_response", result.llm_error or "")
        self.assertEqual(
            result.answer,
            "I last saw your phone on the center of view around 13:22:33 on 2026-06-18. My confidence was 0.73.",
        )

    def test_agent_uses_valid_structured_llm_answer(self) -> None:
        payload = self.valid_payload()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)
            agent = self.make_agent_with_fake(db_path, payload)
            result = agent.answer("Where did you last see my phone?")

        self.assertTrue(result.llm_attempted)
        self.assertTrue(result.llm_used)
        self.assertIsNone(result.llm_error)
        self.assertIsNone(result.llm_fallback_reason)
        self.assertEqual(result.answer, payload["spoken_answer"])

    def test_list_recent_objects_uses_sqlite_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)
            store.insert(MemoryRecord.create("cup", "cup", "left side of view", (1, 2, 3, 4), 0.82, timestamp="2026-06-18T13:25:00"))
            agent = RecallAgent(memory_db=db_path, use_llm=False, logger=logging.getLogger("test"))
            result = agent.answer("What objects did you detect?")

        self.assertEqual(result.intent.intent.value, "list_recent_objects")
        self.assertEqual(result.answer_type, "recent_objects")
        self.assertIn("cup", result.answer)
        self.assertIn("phone", result.answer)
        self.assertNotIn("detected", result.parsed_object or "")


if __name__ == "__main__":
    unittest.main()

```

## `backend/memory/memory_store.py`

```python

"""SQLite-backed object memory store for Milestone 3.

The schema is intentionally small and explainable. Records represent what the
lamp saw, where it approximately appeared in the camera frame, when it was seen,
and optional evidence through a saved frame path.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Iterable, Optional


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    object_label: str
    normalized_label: str
    location_label: str
    bbox: BBox
    confidence: float
    timestamp: str
    source: str = "webcam"
    frame_path: Optional[str] = None

    @classmethod
    def create(
        cls,
        object_label: str,
        normalized_label: str,
        location_label: str,
        bbox: BBox,
        confidence: float,
        source: str = "webcam",
        frame_path: str | None = None,
        timestamp: str | None = None,
    ) -> "MemoryRecord":
        return cls(
            id=str(uuid.uuid4()),
            object_label=object_label,
            normalized_label=normalized_label,
            location_label=location_label,
            bbox=tuple(int(v) for v in bbox),
            confidence=float(confidence),
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            source=source,
            frame_path=frame_path,
        )

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "normalized_label": self.normalized_label,
            "location_label": self.location_label,
            "bbox": json.dumps(list(self.bbox)),
            "confidence": float(self.confidence),
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "normalized_label": self.normalized_label,
            "location_label": self.location_label,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
        }


class MemoryStore:
    """SQLite store for object-memory records."""

    def __init__(self, db_path: str | Path = "data/scene_memory.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS object_memory (
                    id TEXT PRIMARY KEY,
                    object_label TEXT NOT NULL,
                    normalized_label TEXT NOT NULL,
                    location_label TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    frame_path TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_label_time ON object_memory(normalized_label, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_location_time ON object_memory(location_label, timestamp)"
            )

    def insert(self, record: MemoryRecord) -> None:
        row = record.to_row()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO object_memory (
                    id,
                    object_label,
                    normalized_label,
                    location_label,
                    bbox,
                    confidence,
                    timestamp,
                    source,
                    frame_path
                ) VALUES (
                    :id,
                    :object_label,
                    :normalized_label,
                    :location_label,
                    :bbox,
                    :confidence,
                    :timestamp,
                    :source,
                    :frame_path
                )
                """,
                row,
            )

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]


    def recent_unique_by_normalized_label(self, limit: int = 8, row_limit: int = 200) -> list[MemoryRecord]:
        """Return latest records for recent unique normalized labels.

        SQLite remains the source of truth. This method is used for questions like
        "What objects did you detect?" so the answer is grounded in stored rows,
        not in the object detector vocabulary or LLM guesses.
        """

        records = self.recent(limit=max(int(row_limit), int(limit)))
        seen: set[str] = set()
        unique: list[MemoryRecord] = []
        for record in records:
            label = record.normalized_label.strip().lower()
            if not label or label in seen:
                continue
            seen.add(label)
            unique.append(record)
            if len(unique) >= int(limit):
                break
        return unique

    def find(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        normalized_query = query.strip().lower()
        like = f"%{normalized_query}%"
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) LIKE ? OR lower(object_label) LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (like, like, int(limit)),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_latest_by_normalized_label(self, normalized_label: str) -> MemoryRecord | None:
        """Return the latest exact normalized-label match for grounded recall."""
        normalized = normalized_label.strip().lower()
        if not normalized:
            return None

        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_recent_duplicate(
        self,
        normalized_label: str,
        location_label: str,
        within_seconds: float,
        now: datetime | None = None,
    ) -> MemoryRecord | None:
        """Return a same-label/same-location record inside the dedupe window."""
        cutoff = (now or datetime.now()) - timedelta(seconds=float(within_seconds))
        cutoff_text = cutoff.isoformat(timespec="seconds")
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE normalized_label = ?
                  AND location_label = ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized_label, location_label, cutoff_text),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear(self) -> int:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM object_memory")
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def count(self) -> int:
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM object_memory").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        raw_bbox = row["bbox"]
        bbox_values = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
        bbox: BBox = tuple(int(v) for v in bbox_values)  # type: ignore[assignment]
        return MemoryRecord(
            id=str(row["id"]),
            object_label=str(row["object_label"]),
            normalized_label=str(row["normalized_label"]),
            location_label=str(row["location_label"]),
            bbox=bbox,
            confidence=float(row["confidence"]),
            timestamp=str(row["timestamp"]),
            source=str(row["source"]),
            frame_path=row["frame_path"],
        )


def records_to_table(records: Iterable[MemoryRecord]) -> str:
    """Render records as a readable fixed-width CLI table."""
    rows = list(records)
    if not rows:
        return "No memory records found."

    headers = ["timestamp", "object", "normalized", "location", "conf", "bbox", "frame"]
    data = []
    for record in rows:
        data.append(
            [
                record.timestamp,
                record.object_label,
                record.normalized_label,
                record.location_label,
                f"{record.confidence:.2f}",
                str(list(record.bbox)),
                record.frame_path or "",
            ]
        )

    widths = [len(header) for header in headers]
    for row in data:
        for idx, cell in enumerate(row):
            widths[idx] = min(max(widths[idx], len(cell)), 42)

    def fit(value: str, width: int) -> str:
        return value if len(value) <= width else value[: max(0, width - 1)] + "…"

    lines = []
    lines.append(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in data:
        lines.append(" | ".join(fit(cell, widths[idx]).ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)

```

## `frontend_godot/scripts/Main.gd`

```gd

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

```

## `frontend_godot/scripts/LampController.gd`

```gd

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

```

## `README_MILESTONE4_3_PATCH.md`

```markdown

# Milestone 4.3 Patch — Real Conversation + LLM Repair + Live Chat + Sleep Mode

This patch repairs the Milestone 4.2.1 recall path instead of adding another fallback-only layer.

## What changed

### Backend conversation path

- Replaced raw Ollama `/api/generate` phrasing with `/api/chat` structured output.
- Added `RECALL_RESPONSE_SCHEMA` validation with these fields:
  - `answer_type`
  - `target_object`
  - `location_label`
  - `confidence`
  - `timestamp`
  - `spoken_answer`
- Added strict validation so the LLM is accepted only when it matches retrieved SQLite memory.
- Added `--llm-required` to the recall CLI. If a memory answer is LLM-eligible and the LLM fails validation, the CLI returns nonzero.
- Added deterministic fallback, but fallback is logged as fallback and never counted as `llm_used=true`.

### Intent parsing

- Added `backend/conversation/intent_parser.py`.
- Supported intents:
  - `object_last_seen`
  - `list_recent_objects`
  - `unsupported`
- Fixed these regressions:
  - `Did you see if I had a stapler?` parses as `object_last_seen/stapler`, not `if`.
  - `What objects did you detect?` parses as `list_recent_objects`, not `detected`.
  - `What do you remember seeing?` parses as `list_recent_objects`.

### Live Godot chat

- Added `backend/conversation/chat_udp_receiver.py`.
- Backend listens for Godot chat JSON on UDP port `4243` by default:

```json
{
  "type": "chat_query",
  "timestamp": "2026-06-18T13:30:00",
  "text": "Where did you last see my phone?"
}
```

- Godot now has a live chat panel with:
  - chat history
  - text input
  - Send button
  - Enter-to-send
- Backend answers through the existing backend-to-Godot command JSON shape using:
  - `state=recalling`
  - `behavior.motion=thinking`
  - `behavior.light=focus_glow`
  - `behavior.speech_text=<grounded answer>`

### Godot stale-backend sleep mode

- Godot tracks the local time of the last backend command packet.
- If no backend command arrives for `60` seconds, Godot enters local sleep mode:
  - low resting pose
  - dim red light
  - no active searching/nodding
  - debug UI says `backend stale / sleeping`
- New backend command packets immediately exit sleep mode.
- This is local connection-health handling only; Godot still does not make engagement, memory, or recall decisions.

## Changed files

- `backend/main.py`
- `backend/conversation/__init__.py`
- `backend/conversation/chat_udp_receiver.py`
- `backend/conversation/intent_parser.py`
- `backend/conversation/llm_client.py`
- `backend/conversation/query_parser.py`
- `backend/conversation/recall_agent.py`
- `backend/conversation/test_chat_udp_receiver.py`
- `backend/conversation/test_intent_parser.py`
- `backend/conversation/test_query_parser.py`
- `backend/conversation/test_recall_llm_guardrails.py`
- `backend/memory/memory_store.py`
- `frontend_godot/scripts/Main.gd`
- `frontend_godot/scripts/LampController.gd`

## PowerShell validation commands

Run these from the project root.

### 1. Run parser/conversation tests

```powershell
python -m unittest `
  backend.conversation.test_query_parser `
  backend.conversation.test_intent_parser `
  backend.conversation.test_recall_llm_guardrails `
  backend.conversation.test_chat_udp_receiver `
  -v
```

Expected: all tests pass. This reproduces the previous bad LLM output and verifies it is rejected with `contradicts_retrieved_memory`.

### 2. Pull/check Ollama model

In one terminal, start Ollama if it is not already running:

```powershell
ollama serve
```

In another terminal:

```powershell
ollama pull llama3.2
python -m backend.conversation.llm_client --ollama-url http://localhost:11434 --ollama-model llama3.2 --json
```

Expected: `ok=true` and `model_available=true`.

### 3. Run recall with `--llm-required`

```powershell
python -m backend.conversation.recall_agent "Where did you last see my phone?" `
  --memory-db data/scene_memory.sqlite `
  --use-llm `
  --llm-required `
  --ollama-url http://localhost:11434 `
  --ollama-model llama3.2 `
  --json
```

Expected when a matching phone row exists and Ollama returns valid grounded JSON:

- `answer_type="memory_answer"`
- `parsed_object="phone"`
- `retrieved_memory_id` is not null
- `llm_attempted=true`
- `llm_used=true`
- `llm_error=null`

If the LLM says it does not remember while the SQLite record exists, the CLI exits nonzero and logs `contradicts_retrieved_memory`.

### 4. Run backend with Godot chat UDP receiver

```powershell
python -m backend.main `
  --show-window `
  --godot-udp `
  --enable-objects `
  --interactive-recall `
  --enable-godot-chat `
  --chat-port 4243 `
  --use-llm `
  --llm-required `
  --ollama-url http://localhost:11434 `
  --ollama-model llama3.2 `
  --memory-db data/scene_memory.sqlite `
  --object-model yolov8n.pt `
  --save-object-frames
```

Then run the Godot project and ask in the chat panel:

```text
Where did you last see my phone?
```

### 5. Optional manual UDP chat packet test

Use this if you want to test the backend receiver without typing in Godot:

```powershell
$payload = @{
  type = "chat_query"
  timestamp = (Get-Date).ToString("s")
  text = "What objects did you detect?"
} | ConvertTo-Json -Compress

$client = New-Object System.Net.Sockets.UdpClient
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
[void]$client.Send($bytes, $bytes.Length, "127.0.0.1", 4243)
$client.Close()
```

## Demo pass/fail criteria

PASS if Godot chat can ask `Where did you last see my phone?` while the backend is running and receives a valid LLM-grounded answer with `llm_used=true` in `logs/latest/recall.jsonl`.

PASS if `What objects did you detect?` lists recent unique objects retrieved from SQLite.

PASS if stopping the backend causes Godot to enter local sleep mode after 60 seconds and a new backend command wakes it immediately.

FAIL if normal object recall only works by stopping the backend and running CLI commands.

FAIL if the LLM says no memory when SQLite has a matching record and that answer is accepted.

FAIL if `stapler` parses as `if` or `What objects did you detect?` parses as `detected`.

```
