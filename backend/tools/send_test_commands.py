"""Send fake LeLamp behavior commands to Godot without a webcam.

Run from the project root:
    python -m backend.tools.send_test_commands --count 24 --interval 1.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import datetime
from typing import Iterable

from backend.behavior.godot_udp_sender import GodotUdpSender
from backend.utils.config import GodotUdpConfig


TEST_SEQUENCE: list[dict] = [
    {
        "state": "idle",
        "engagement": {"status": "absent", "confidence": 0.20, "reason": "test_idle"},
        "behavior": {"motion": "idle_breathe", "light": "dim_warm", "sound": None, "speech_text": None},
    },
    {
        "state": "engaged",
        "engagement": {"status": "engaged", "confidence": 0.92, "reason": "test_face_centered", "face_x_norm": 0.50, "face_y_norm": 0.48},
        "behavior": {"motion": "attentive_nod", "light": "steady_warm", "sound": None, "speech_text": None},
    },
    {
        "state": "disengaged",
        "engagement": {"status": "disengaged", "confidence": 0.77, "reason": "test_head_turned", "face_x_norm": 0.18, "face_y_norm": 0.50},
        "behavior": {"motion": "searching_glance", "light": "slow_pulse", "sound": None, "speech_text": None},
    },
    {
        "state": "seeking_attention",
        "engagement": {"status": "disengaged", "confidence": 0.84, "reason": "test_sustained_disengagement", "face_x_norm": 0.82, "face_y_norm": 0.52},
        "behavior": {"motion": "curious_tilt", "light": "soft_pulse", "sound": "gentle_chime", "speech_text": None},
    },
    {
        "state": "scanning",
        "engagement": {"status": "engaged", "confidence": 0.70, "reason": "test_scene_scan", "face_x_norm": 0.36, "face_y_norm": 0.46},
        "behavior": {"motion": "scanning", "light": "scan_sweep", "sound": None, "speech_text": None},
    },
    {
        "state": "recalling",
        "engagement": {"status": "engaged", "confidence": 0.88, "reason": "test_memory_query", "face_x_norm": 0.50, "face_y_norm": 0.50},
        "behavior": {
            "motion": "thinking",
            "light": "focus_glow",
            "sound": None,
            "speech_text": "Let me check what I remember.",
        },
    },
    {
        "state": "sleep",
        "engagement": {"status": "absent", "confidence": 0.0, "reason": "test_backend_stale_sleep"},
        "behavior": {"motion": "sleep_rest", "light": "sleep_red", "sound": None, "speech_text": None},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send fake LeLamp commands to the Godot UDP frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Godot UDP host. Use 127.0.0.1 for local demo.")
    parser.add_argument("--port", type=int, default=4242, help="Godot UDP listen port.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between commands.")
    parser.add_argument("--count", type=int, default=0, help="Number of packets to send. 0 means loop until Ctrl-C.")
    parser.add_argument("--print-json", action="store_true", help="Print each command JSON to stdout.")
    return parser.parse_args()


def with_protocol_envelope(template: dict) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "state": template["state"],
        "engagement": dict(template["engagement"]),
        "behavior": dict(template["behavior"]),
        "memory": {
            "last_detected_objects": [],
        },
    }


def limited_cycle(sequence: Iterable[dict], count: int) -> Iterable[dict]:
    if count <= 0:
        yield from itertools.cycle(sequence)
    else:
        cycle = itertools.cycle(sequence)
        for _ in range(count):
            yield next(cycle)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("send_test_commands")

    sender = GodotUdpSender(
        GodotUdpConfig(enabled=True, host=args.host, port=args.port),
        logger=logger,
    )

    logger.info("Sending test commands to udp://%s:%s", args.host, args.port)
    logger.info("Press Ctrl-C to stop when --count is 0")

    sent = 0
    try:
        for template in limited_cycle(TEST_SEQUENCE, args.count):
            command = with_protocol_envelope(template)
            elapsed_ms = sender.send(command)
            sent += 1
            behavior = command["behavior"]
            logger.info(
                "sent #%s state=%s motion=%s light=%s udp_send_ms=%s",
                sent,
                command["state"],
                behavior.get("motion"),
                behavior.get("light"),
                "n/a" if elapsed_ms is None else f"{elapsed_ms:.3f}",
            )
            if args.print_json:
                print(json.dumps(command, ensure_ascii=False), flush=True)
            time.sleep(max(args.interval, 0.05))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        sender.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
