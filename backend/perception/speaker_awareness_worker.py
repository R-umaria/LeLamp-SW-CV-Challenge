"""Parallel active-speaker worker for Lumos.

Runs the expensive face-tracking / mouth-motion / audio-visual fusion path off
of the main camera/FSM/Godot loop. The worker consumes the newest frame only,
so slow speaker analysis can never backlog and make the preview stale.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from backend.audio.gcc_phat import DirectionOfArrivalResult
from backend.audio.voice_activity_detector import VoiceActivityResult
from backend.perception.active_speaker_detector import ActiveSpeakerDetector, ActiveSpeakerResult
from backend.perception.engagement_detector import EngagementResult
from backend.perception.face_tracker import FaceTrack, FaceTracker
from backend.utils.config import EngagementConfig, SpeakerAwarenessConfig


@dataclass(frozen=True)
class SpeakerWorkerSnapshot:
    result: ActiveSpeakerResult | None
    face_tracks: list[FaceTrack]
    fusion_ms: float | str
    frame_sequence: int
    updated_at_s: float
    worker_enabled: bool
    worker_reason: str


@dataclass(frozen=True)
class _WorkerInput:
    frame_sequence: int
    frame: object
    engagement: EngagementResult
    voice: VoiceActivityResult | None
    doa: DirectionOfArrivalResult | None
    timestamp_s: float


class SpeakerAwarenessWorker:
    """Latest-frame-only speaker fusion worker.

    This is intentionally a thread, not a process: OpenCV and MediaPipe release
    the GIL in their heavy native operations, and avoiding inter-process frame
    serialization matters more than theoretical Python-core parallelism here.
    """

    def __init__(
        self,
        engagement_config: EngagementConfig,
        speaker_config: SpeakerAwarenessConfig,
        logger: logging.Logger | None = None,
        event_paths: list[Path] | tuple[Path, ...] | None = None,
    ) -> None:
        self.engagement_config = engagement_config
        self.speaker_config = speaker_config
        self.logger = logger or logging.getLogger("lelamp")
        self.event_paths = list(event_paths or [])
        self.enabled = bool(speaker_config.enabled)
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._pending: Optional[_WorkerInput] = None
        self._latest_snapshot = SpeakerWorkerSnapshot(
            result=None,
            face_tracks=[],
            fusion_ms="",
            frame_sequence=-1,
            updated_at_s=0.0,
            worker_enabled=self.enabled,
            worker_reason="not_started" if self.enabled else "speaker_awareness_disabled",
        )
        self._thread: threading.Thread | None = None
        self._last_log_at_s = 0.0
        self._last_signature: tuple | None = None

    def start(self) -> bool:
        if not self.enabled:
            return False
        if self._thread is not None:
            return True
        self._thread = threading.Thread(target=self._run, name="speaker-awareness-worker", daemon=True)
        self._thread.start()
        self.logger.info(
            "speaker_worker_started interval_s=%.3f max_frame_width=%s debug=%s",
            self.speaker_config.fusion_interval_s,
            self.speaker_config.worker_frame_width,
            self.speaker_config.debug,
        )
        return True

    def submit(
        self,
        *,
        frame_sequence: int,
        frame,
        engagement: EngagementResult,
        voice: VoiceActivityResult | None,
        doa: DirectionOfArrivalResult | None,
    ) -> None:
        if not self.enabled:
            return
        now_s = time.time()
        with self._condition:
            self._pending = _WorkerInput(
                frame_sequence=int(frame_sequence),
                frame=frame.copy(),
                engagement=engagement,
                voice=voice,
                doa=doa,
                timestamp_s=now_s,
            )
            self._condition.notify()

    def snapshot(self) -> SpeakerWorkerSnapshot:
        with self._condition:
            return self._latest_snapshot

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _run(self) -> None:
        face_tracker: FaceTracker | None = None
        speaker_detector: ActiveSpeakerDetector | None = None
        try:
            face_tracker = FaceTracker(self.engagement_config, self.speaker_config, logger=self.logger)
            speaker_detector = ActiveSpeakerDetector(self.speaker_config)
            last_processed_at = 0.0
            while not self._stop_event.is_set():
                work_item = self._take_latest_work_item(last_processed_at)
                if work_item is None:
                    continue
                last_processed_at = time.monotonic()
                t0 = time.perf_counter()
                frame, engagement = self._prepare_frame_and_engagement(work_item.frame, work_item.engagement)
                face_tracks = face_tracker.update(frame, engagement_result=engagement, now=time.monotonic())
                result = speaker_detector.update(
                    face_tracks,
                    work_item.voice,
                    work_item.doa,
                    engagement,
                    now=work_item.timestamp_s,
                )
                fusion_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                snapshot = SpeakerWorkerSnapshot(
                    result=result,
                    face_tracks=face_tracks,
                    fusion_ms=fusion_ms,
                    frame_sequence=work_item.frame_sequence,
                    updated_at_s=time.time(),
                    worker_enabled=True,
                    worker_reason="ok",
                )
                with self._condition:
                    self._latest_snapshot = snapshot
                self._log_result(snapshot)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.logger.warning("speaker_worker_failed error=%s", exc)
            with self._condition:
                self._latest_snapshot = SpeakerWorkerSnapshot(
                    result=None,
                    face_tracks=[],
                    fusion_ms="",
                    frame_sequence=-1,
                    updated_at_s=time.time(),
                    worker_enabled=False,
                    worker_reason=f"worker_failed: {exc}",
                )
        finally:
            if face_tracker is not None:
                face_tracker.close()
            self.logger.info("speaker_worker_stopped")

    def _take_latest_work_item(self, last_processed_at: float) -> _WorkerInput | None:
        interval = max(0.02, float(self.speaker_config.fusion_interval_s))
        with self._condition:
            while not self._stop_event.is_set() and self._pending is None:
                self._condition.wait(timeout=0.10)
            if self._stop_event.is_set():
                return None
            wait_s = interval - (time.monotonic() - last_processed_at)
            if wait_s > 0:
                self._condition.wait(timeout=wait_s)
            item = self._pending
            self._pending = None
            return item

    def _prepare_frame_and_engagement(self, frame, engagement: EngagementResult) -> tuple[object, EngagementResult]:
        max_width = int(self.speaker_config.worker_frame_width or 0)
        height, width = frame.shape[:2]
        if max_width <= 0 or width <= max_width:
            return frame, engagement
        scale = float(max_width) / float(width)
        new_size = (max_width, max(1, int(round(height * scale))))
        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return resized, _scale_engagement_result(engagement, scale)

    def _log_result(self, snapshot: SpeakerWorkerSnapshot) -> None:
        result = snapshot.result
        if result is None:
            return
        payload = {
            "event": "speaker_result",
            **result.to_log_dict(),
            "face_tracks": [track.to_log_dict() for track in snapshot.face_tracks],
            "fusion_ms": snapshot.fusion_ms,
            "frame_sequence": snapshot.frame_sequence,
            "worker": True,
        }
        _append_jsonl(self.event_paths, payload)

        signature = (
            result.speech_detected,
            result.active_track_id,
            result.speaking_to_robot,
            result.reason,
        )
        now_s = time.monotonic()
        should_log = (
            self.speaker_config.debug
            or result.speech_detected
            or signature != self._last_signature
            or (now_s - self._last_log_at_s) >= float(self.speaker_config.worker_log_interval_s)
        )
        if not should_log:
            return
        self._last_signature = signature
        self._last_log_at_s = now_s
        self.logger.info(
            "speaker_result speech=%s track=%s to_robot=%s confidence=%.3f reason=%s fusion_ms=%s worker=true",
            result.speech_detected,
            result.active_track_id,
            result.speaking_to_robot,
            result.confidence,
            result.reason,
            snapshot.fusion_ms,
        )
        if self.speaker_config.debug:
            self.logger.info("speaker_debug face_tracks=%s", [track.to_log_dict() for track in snapshot.face_tracks])


def _scale_engagement_result(result: EngagementResult, scale: float) -> EngagementResult:
    if result.face_bbox is None:
        return result
    x, y, w, h = result.face_bbox
    scaled_bbox = (
        int(round(x * scale)),
        int(round(y * scale)),
        max(1, int(round(w * scale))),
        max(1, int(round(h * scale))),
    )
    return EngagementResult(
        status=result.status,
        confidence=result.confidence,
        reason=result.reason,
        face_bbox=scaled_bbox,
        face_center_norm=result.face_center_norm,
        face_area_ratio=result.face_area_ratio,
        raw_face_count=result.raw_face_count,
        candidate_count=result.candidate_count,
        selected_face_score=result.selected_face_score,
    )


def _append_jsonl(paths: list[Path] | tuple[Path, ...], payload: dict) -> None:
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Do not let diagnostic logging kill the perception loop.
            pass
