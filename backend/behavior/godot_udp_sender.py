"""UDP command sender for the Godot embodiment frontend.

Milestone 2 keeps Python as the intelligence layer and Godot as the expressive
embodiment layer. This module only transports already-built protocol commands;
it does not choose behaviors or modify the command payload.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Mapping, Optional

from backend.utils.config import GodotUdpConfig


class GodotUdpSender:
    """Best-effort UDP sender for backend -> Godot command packets.

    UDP is intentionally connectionless. If Godot is not open or not listening,
    ``sendto`` usually succeeds and the packet is simply dropped by the OS. Any
    local socket/serialization failure is logged and suppressed so the webcam
    backend never crashes because the frontend is unavailable.
    """

    def __init__(self, config: GodotUdpConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._socket: Optional[socket.socket] = None
        self._send_count = 0
        self._last_error: Optional[str] = None

        if self.config.enabled:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
            self.logger.info(
                "Godot UDP sender enabled host=%s port=%s",
                self.config.host,
                self.config.port,
            )
        else:
            self.logger.info("Godot UDP sender disabled")

    @property
    def send_count(self) -> int:
        return self._send_count

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def send(self, command: Mapping) -> Optional[float]:
        """Send one protocol command.

        Returns:
            Elapsed send time in milliseconds when enabled, otherwise ``None``.
            Errors are logged and also return ``None``.
        """

        if not self.config.enabled:
            return None

        if self._socket is None:
            self._last_error = "socket_not_initialized"
            self.logger.warning("Godot UDP send skipped: socket not initialized")
            return None

        started = time.perf_counter()
        try:
            data = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(data) > self.config.max_packet_bytes:
                self._last_error = f"packet_too_large:{len(data)}"
                self.logger.warning(
                    "Godot UDP packet too large: %s bytes exceeds limit %s bytes",
                    len(data),
                    self.config.max_packet_bytes,
                )
                return None

            self._socket.sendto(data, (self.config.host, self.config.port))
            self._send_count += 1
            self._last_error = None
            return (time.perf_counter() - started) * 1000.0
        except (TypeError, ValueError) as exc:
            self._last_error = f"json_error:{exc}"
            self.logger.warning("Godot UDP serialization failed: %s", exc)
            return None
        except OSError as exc:
            self._last_error = f"socket_error:{exc}"
            self.logger.warning("Godot UDP send failed: %s", exc)
            return None

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
            self.logger.info("Godot UDP sender closed after %s packets", self._send_count)
