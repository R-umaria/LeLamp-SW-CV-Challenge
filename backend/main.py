"""Milestone 1 backend vertical slice.

Run from the project root with:
    python -m backend.main --show-window

This opens the webcam, estimates engagement from face presence/position, updates a
finite state machine, emits protocol-shaped JSON commands, and logs latency.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.behavior.behavior_policy import behavior_for_state
from backend.behavior.command_protocol import build_behavior_command
from backend.behavior.state_machine import InteractionStateMachine
from backend.evaluation.latency_logger import LatencyLogger
from backend.perception.camera import OpenCVCamera
from backend.perception.engagement_detector import FaceEngagementDetector, draw_engagement_overlay
from backend.utils.config import CameraConfig, EngagementConfig, RuntimeConfig, StateMachineConfig
from backend.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeLamp Milestone 1 backend")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seek-after", type=float, default=5.0)
    parser.add_argument("--absent-grace", type=float, default=2.0)
    parser.add_argument("--emit-interval", type=float, default=1.0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Esc/Ctrl-C")

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--show-window", action="store_true", default=True)
    window_group.add_argument("--no-window", action="store_false", dest="show_window")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    runtime_config = RuntimeConfig(
        command_emit_interval_s=args.emit_interval,
        log_dir=args.log_dir,
        show_window=args.show_window,
    )
    camera_config = CameraConfig(index=args.camera_index, width=args.width, height=args.height)
    state_config = StateMachineConfig(
        seek_attention_after_s=args.seek_after,
        absent_grace_s=args.absent_grace,
    )

    logger = setup_logging(runtime_config.log_dir)
    latency_logger = LatencyLogger(Path(runtime_config.log_dir) / "latency.csv")
    commands_path = Path(runtime_config.log_dir) / "commands.jsonl"

    camera = OpenCVCamera(camera_config.index, camera_config.width, camera_config.height)
    detector = FaceEngagementDetector(EngagementConfig())
    fsm = InteractionStateMachine(state_config)

    logger.info("Starting Milestone 1 backend")
    logger.info("Camera index=%s size=%sx%s", camera_config.index, camera_config.width, camera_config.height)
    logger.info("Commands will be saved to %s", commands_path)

    frame_count = 0
    last_emit_at = 0.0

    try:
        camera.open()
        while True:
            loop_start = time.perf_counter()

            camera_frame = camera.read()
            frame = camera_frame.frame

            t0 = time.perf_counter()
            engagement = detector.detect(frame)
            engagement_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            transition = fsm.update(engagement)
            state_machine_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = behavior_for_state(transition.current_state)
            command = build_behavior_command(
                state=transition.current_state,
                engagement=engagement,
                behavior=behavior,
                last_detected_objects=[],
            )
            command_ms = (time.perf_counter() - t0) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            now = time.monotonic()
            should_emit = transition.changed or (now - last_emit_at >= runtime_config.command_emit_interval_s)

            if transition.changed:
                logger.info(
                    "State transition %s -> %s | reason=%s | engagement=%s conf=%.2f",
                    transition.previous_state.value,
                    transition.current_state.value,
                    transition.reason,
                    engagement.status,
                    engagement.confidence,
                )

            if should_emit:
                print(json.dumps(command, ensure_ascii=False), flush=True)
                append_jsonl(commands_path, command)
                last_emit_at = now

            latency_logger.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "capture_ms": round(camera_frame.capture_latency_ms, 3),
                    "engagement_detection_ms": round(engagement_ms, 3),
                    "state_machine_ms": round(state_machine_ms, 3),
                    "command_build_ms": round(command_ms, 3),
                    "total_loop_ms": round(total_ms, 3),
                    "state": transition.current_state.value,
                    "engagement_status": engagement.status,
                    "engagement_confidence": round(engagement.confidence, 3),
                    "engagement_reason": engagement.reason,
                }
            )

            if runtime_config.show_window:
                draw_engagement_overlay(frame, engagement, transition.current_state.value)
                cv2.imshow("LeLamp Milestone 1 - Engagement/FSM", frame)
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
        camera.release()
        if runtime_config.show_window:
            cv2.destroyAllWindows()
        logger.info("Stopped Milestone 1 backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
