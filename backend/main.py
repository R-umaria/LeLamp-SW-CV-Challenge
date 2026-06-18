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
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase retrieved memory answers.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL for --use-llm.")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model for recall phrasing, e.g. llama3.2 or qwen2.5.")

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
    if args.interactive_recall:
        recall_log_paths = [run_paths.run_dir / "recall.jsonl"]
        if not args.no_latest:
            recall_log_paths.append(run_paths.latest_dir / "recall.jsonl")
        recall_agent = RecallAgent(
            memory_db=args.memory_db,
            use_llm=args.use_llm,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            log_paths=recall_log_paths,
            logger=logger,
        )
        recall_queue = queue.Queue()
        start_recall_input_thread(recall_queue, recall_stop_event, logger)

    logger.info("Starting Milestone 4 backend with isolated run logging")
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
        "Recall config interactive=%s use_llm=%s ollama_url=%s ollama_model=%s",
        args.interactive_recall,
        args.use_llm,
        args.ollama_url,
        args.ollama_model,
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
                        "Sent recall command to Godot state=recalling parsed_object=%s memory_id=%s",
                        recall_result.parsed_object,
                        recall_result.memory_record.id if recall_result.memory_record else None,
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
                cv2.imshow("LeLamp Milestone 4 - Engagement/Object Memory/Recall", frame)
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
        godot_sender.close()
        camera.release()
        if runtime_config.show_window:
            cv2.destroyAllWindows()
        logger.info("Stopped Milestone 4 backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
