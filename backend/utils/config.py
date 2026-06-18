"""Configuration defaults for Milestone 1.

Keep these values explicit and conservative so the engagement loop is easy to tune
while testing with a real webcam.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class EngagementConfig:
    # Face center must remain roughly near the camera/lamp center to count as engaged.
    center_tolerance_x: float = 0.22
    center_tolerance_y: float = 0.28
    min_face_area_ratio: float = 0.015
    cascade_scale_factor: float = 1.1
    cascade_min_neighbors: int = 5
    cascade_min_size: tuple[int, int] = (60, 60)


@dataclass(frozen=True)
class StateMachineConfig:
    # A centered face can transition immediately to engaged.
    # A non-centered face must persist before seeking attention.
    seek_attention_after_s: float = 5.0
    # If the face disappears briefly after engagement, treat it as disengagement first.
    absent_grace_s: float = 2.0


@dataclass(frozen=True)
class RuntimeConfig:
    command_emit_interval_s: float = 1.0
    log_dir: str = "logs"
    show_window: bool = True


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    engagement: EngagementConfig = EngagementConfig()
    state_machine: StateMachineConfig = StateMachineConfig()
    runtime: RuntimeConfig = RuntimeConfig()


DEFAULT_CONFIG = AppConfig()
