"""Configuration defaults for the Lumos backend.

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
    # Wide geometry keeps Lumos tracking a visible user across almost the full
    # webcam frame instead of treating normal up/down/side movement as disengagement.
    center_tolerance_x: float = 0.46
    center_tolerance_y: float = 0.44
    min_face_area_ratio: float = 0.018

    # Candidate filtering rejects tiny false positives while allowing a user who
    # is slightly farther from the camera to stay trackable.
    min_candidate_area_ratio: float = 0.006
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
    # Sliding window over raw frame-level predictions. Larger defaults reduce
    # attention-seeking flicker when a face is briefly near the detection edge.
    window_size: int = 9
    engaged_vote_ratio: float = 0.50
    disengaged_vote_ratio: float = 0.64
    absent_vote_ratio: float = 0.80

    # Used by the FSM for fast but safe recovery when the user clearly returns.
    clear_engaged_confidence: float = 0.72


@dataclass(frozen=True)
class StateMachineConfig:
    # A user must remain disengaged before attention seeking begins.
    seek_attention_after_s: float = 5.0

    # State changes are suppressed until the current state has lasted this long,
    # except for clear engaged recovery.
    min_state_dwell_s: float = 0.95

    # Hysteresis: engaged -> disengaged/idle requires several consecutive smoothed
    # predictions so edge-of-frame tracking does not trigger attention seeking.
    exit_engaged_disengaged_frames: int = 9
    exit_engaged_absent_frames: int = 12

    # Recovery is deliberately quicker than disengagement so the lamp feels responsive.
    engaged_recovery_frames: int = 2
    clear_engaged_confidence: float = 0.72

    # No-face behavior after active states.
    absent_to_idle_frames: int = 16


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
class HandGestureConfig:
    # Optional MediaPipe Hands path for deliberate gesture control.
    enabled: bool = False
    interval_s: float = 0.10
    max_num_hands: int = 1
    min_detection_confidence: float = 0.60
    min_tracking_confidence: float = 0.55
    min_gesture_confidence: float = 0.64
    min_hand_area_ratio: float = 0.018
    hold_s: float = 1.15

    # Landmark heuristics. Y coordinates are normalized image coordinates where
    # smaller values are higher in the frame.
    finger_extension_margin: float = 0.030
    finger_fold_margin: float = 0.005
    index_prominence_margin: float = 0.050

    # Temporal beckon detection: repeated index-tip movement within a short window.
    motion_window_s: float = 1.20
    beckon_motion_min_amplitude: float = 0.035
    beckon_min_direction_changes: int = 2


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str = "data/scene_memory.sqlite"
    dedupe_window_s: float = 8.0
    save_object_frames: bool = False
    frame_dir: str = "data/object_frames"


@dataclass(frozen=True)
class AudioConfig:
    # Optional microphone path for Milestone 4.11. Disabled by default so the
    # existing camera/FSM/Godot demo remains unchanged unless explicitly enabled.
    enabled: bool = False
    device: str | None = None
    sample_rate: int = 16000
    block_ms: int = 30
    request_stereo: bool = False

    # Rolling RMS VAD thresholds. ``vad_energy_threshold`` is a multiplier over
    # the learned noise floor, with ``vad_absolute_threshold`` as a safety floor.
    vad_energy_threshold: float = 2.4
    vad_absolute_threshold: float = 0.010
    vad_min_noise_floor: float = 0.003
    vad_noise_update_alpha: float = 0.08
    vad_noise_update_alpha_speech: float = 0.004
    vad_smoothing_blocks: int = 4
    vad_active_vote_ratio: float = 0.50


@dataclass(frozen=True)
class DirectionOfArrivalConfig:
    enabled: bool = False
    mic_distance_m: float = 0.08
    min_rms: float = 1e-4
    min_confidence: float = 0.12
    max_abs_azimuth_deg: float = 90.0


@dataclass(frozen=True)
class SpeakerAwarenessConfig:
    enabled: bool = False
    debug: bool = False
    fusion_interval_s: float = 0.25
    worker_frame_width: int = 640
    worker_log_interval_s: float = 1.0
    policy_min_confidence: float = 0.55
    policy_seek_min_confidence: float = 0.30
    policy_hold_s: float = 1.15
    mouth_motion_threshold: float = 0.18
    ambiguous_margin: float = 0.12
    face_track_ttl_s: float = 1.25
    max_face_match_distance_norm: float = 0.28
    min_face_iou: float = 0.15
    secondary_face_min_area_ratio: float = 0.018



@dataclass(frozen=True)
class SpeechToTextConfig:
    # Optional local STT path for Milestone 4.12. It is gated by active-speaker
    # awareness: Lumos records only short utterances from a person likely talking
    # to the lamp, then routes accepted transcripts into the existing recall path.
    enabled: bool = False
    backend: str = "faster_whisper"
    model_size: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1
    no_speech_threshold: float = 0.60

    # Utterance segmentation parameters. Defaults favor short demo questions such
    # as "did you see my phone" while avoiding long background transcription.
    speaker_min_confidence: float = 0.45
    min_utterance_s: float = 0.55
    max_utterance_s: float = 6.0
    end_silence_s: float = 0.85
    pre_roll_s: float = 0.45
    cooldown_s: float = 1.25
    max_queue_size: int = 2

    # Transcript filters. The first milestone is recall-focused, so reject tiny
    # Whisper fragments that are usually background noise or hallucination.
    min_chars: int = 5
    min_words: int = 2



@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    engagement: EngagementConfig = EngagementConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    state_machine: StateMachineConfig = StateMachineConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    godot_udp: GodotUdpConfig = GodotUdpConfig()
    objects: ObjectDetectionConfig = ObjectDetectionConfig()
    gestures: HandGestureConfig = HandGestureConfig()
    memory: MemoryConfig = MemoryConfig()
    audio: AudioConfig = AudioConfig()
    doa: DirectionOfArrivalConfig = DirectionOfArrivalConfig()
    speaker_awareness: SpeakerAwarenessConfig = SpeakerAwarenessConfig()
    speech_to_text: SpeechToTextConfig = SpeechToTextConfig()


DEFAULT_CONFIG = AppConfig()
