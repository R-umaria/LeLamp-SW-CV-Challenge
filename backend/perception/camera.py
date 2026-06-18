"""OpenCV webcam wrapper with basic error handling and capture timing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "OpenCV is required for webcam capture. Install with: pip install opencv-python"
    ) from exc


@dataclass
class CameraFrame:
    frame: object
    capture_latency_ms: float


class OpenCVCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.index}. "
                "Check camera permissions, index, and whether another app is using it."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> CameraFrame:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera has not been opened. Call open() before read().")

        t0 = time.perf_counter()
        ok, frame = self._cap.read()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from webcam.")

        return CameraFrame(frame=frame, capture_latency_ms=latency_ms)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
