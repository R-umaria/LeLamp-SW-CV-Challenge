"""OpenCV webcam wrapper with backend fallback and capture timing.

Milestone 3 note:
On Windows, OpenCV's default MSMF backend can open a camera but fail to read
frames with errors such as CAP_MSMF can't grab frame. This wrapper tries
DirectShow first on Windows, then MSMF, then OpenCV's default backend.
"""

from __future__ import annotations

import platform
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
    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        read_retries: int = 5,
        warmup_frames: int = 5,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.read_retries = max(1, read_retries)
        self.warmup_frames = max(0, warmup_frames)
        self._cap: Optional[cv2.VideoCapture] = None
        self.backend_name = "unopened"

    def open(self) -> None:
        errors: list[str] = []

        for backend_name, backend_id in self._candidate_backends():
            cap = self._open_with_backend(backend_id)
            if cap is None or not cap.isOpened():
                errors.append(f"{backend_name}: could not open")
                if cap is not None:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Not all backends support this property. Ignore failures.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if self._can_read_initial_frame(cap):
                self._cap = cap
                self.backend_name = backend_name
                return

            errors.append(f"{backend_name}: opened but could not read frame")
            cap.release()

        joined_errors = "; ".join(errors) if errors else "no backends attempted"
        raise RuntimeError(
            f"Could not open/read webcam index {self.index}. Tried: {joined_errors}. "
            "Check camera permissions, camera index, and whether another app is using it."
        )

    def _candidate_backends(self) -> list[tuple[str, Optional[int]]]:
        system = platform.system().lower()

        if system == "windows":
            return [
                ("dshow", cv2.CAP_DSHOW),
                ("msmf", cv2.CAP_MSMF),
                ("default", None),
            ]

        return [
            ("default", None),
        ]

    def _open_with_backend(self, backend_id: Optional[int]) -> cv2.VideoCapture:
        if backend_id is None:
            return cv2.VideoCapture(self.index)
        return cv2.VideoCapture(self.index, backend_id)

    def _can_read_initial_frame(self, cap: cv2.VideoCapture) -> bool:
        # Give the camera a short moment to start streaming.
        time.sleep(0.15)

        ok = False
        frame = None

        attempts = max(1, self.warmup_frames)
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
            time.sleep(0.05)

        return False

    def read(self) -> CameraFrame:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera has not been opened. Call open() before read().")

        start = time.perf_counter()
        last_error = "unknown"

        for _ in range(self.read_retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                latency_ms = (time.perf_counter() - start) * 1000.0
                return CameraFrame(frame=frame, capture_latency_ms=latency_ms)

            last_error = "OpenCV returned no frame"
            time.sleep(0.03)

        raise RuntimeError(
            f"Failed to read frame from webcam index {self.index} using backend "
            f"{self.backend_name}. Last error: {last_error}."
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None