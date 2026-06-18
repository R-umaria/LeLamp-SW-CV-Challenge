"""Configuration defaults for the LeLamp backend.

Milestone 3 preserves the stable engagement/FSM/Godot path and adds optional
object detection plus SQLite scene memory. Object detection is disabled unless
explicitly enabled from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class EngagementConfig:
    # Engagement geometry. A face must be central and large enough to count as engaged.
    center_tolerance_x: float = 0.24
    center_tolerance_y: float = 0.30
    min_face_area_ratio: float = 0.020

    # Candidate filtering. This rejects many tiny false positives such as faces in
    # posters/photo frames while keeping a near-desk user detectable.
    min_candidate_area_ratio: float = 0.010
    max_candidate_area_ratio: float = 0.60

    # OpenCV Haar cascade settings. Higher min_neighbors reduces false positives
    # in low light at the cost of missing some weak faces.
    cascade_scale_factor: float = 1.08
    cascade_min_neighbors: int = 6
    cascade_min_size: tuple[int, int] = (64, 64)

    # Dark-room preprocessing.
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    blur_kernel_size: int = 3

    # Primary-face tracking. The detector scores candidates by continuity, size,
    # and center proximity instead of blindly picking the largest face every frame.
    primary_continuity_weight: float = 0.45
    primary_size_weight: float = 0.35
    primary_center_weight: float = 0.20
    max_primary_center_distance: float = 0.35
    max_primary_area_change_ratio: float = 2.75


@dataclass(frozen=True)
class SmoothingConfig:
    # Sliding window over raw frame-level predictions.
    window_size: int = 7
    engaged_vote_ratio: float = 0.55
    disengaged_vote_ratio: float = 0.60
    absent_vote_ratio: float = 0.75

    # Used by the FSM for fast but safe recovery when the user clearly returns.
    clear_engaged_confidence: float = 0.78


@dataclass(frozen=True)
class StateMachineConfig:
    # A user must remain disengaged before attention seeking begins.
    seek_attention_after_s: float = 5.0

    # State changes are suppressed until the current state has lasted this long,
    # except for clear engaged recovery.
    min_state_dwell_s: float = 0.75

    # Hysteresis: engaged -> disengaged/idle requires several consecutive smoothed
    # predictions rather than one noisy frame.
    exit_engaged_disengaged_frames: int = 5
    exit_engaged_absent_frames: int = 8

    # Recovery is deliberately quicker than disengagement so the lamp feels responsive.
    engaged_recovery_frames: int = 2
    clear_engaged_confidence: float = 0.78

    # No-face behavior after active states.
    absent_to_idle_frames: int = 12


@dataclass(frozen=True)
class RuntimeConfig:
    command_emit_interval_s: float = 1.0
    log_dir: str = "logs"
    show_window: bool = True


@dataclass(frozen=True)
class GodotUdpConfig:
    # Milestone 2 local embodiment bridge. Disabled by default so Milestone 1.5.1
    # behavior remains unchanged unless explicitly enabled.
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 4242
    max_packet_bytes: int = 8192


@dataclass(frozen=True)
class ObjectDetectionConfig:
    # Disabled by default to preserve the stable engagement-only run path.
    enabled: bool = False
    model_path: str = "yolov8n.pt"
    interval_s: float = 2.0
    confidence: float = 0.35
    max_objects_per_frame: int = 8
    allowed_labels: set[str] = field(
        default_factory=lambda: {
            "cell phone",
            "laptop",
            "keyboard",
            "mouse",
            "book",
            "cup",
            "bottle",
            "remote",
            "scissors",
            "clock",
            "vase",
            "backpack",
            "handbag",
            "chair",
            "tv",
            "monitor",
        }
    )


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str = "data/scene_memory.sqlite"
    dedupe_window_s: float = 8.0
    save_object_frames: bool = False
    frame_dir: str = "data/object_frames"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    engagement: EngagementConfig = EngagementConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    state_machine: StateMachineConfig = StateMachineConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    godot_udp: GodotUdpConfig = GodotUdpConfig()
    objects: ObjectDetectionConfig = ObjectDetectionConfig()
    memory: MemoryConfig = MemoryConfig()


DEFAULT_CONFIG = AppConfig()
