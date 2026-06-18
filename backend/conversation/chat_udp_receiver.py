"""Legacy UDP chat-query receiver for pre-4.3.2 Godot builds.

Older Godot builds could send JSON packets to this backend-side listener:

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
