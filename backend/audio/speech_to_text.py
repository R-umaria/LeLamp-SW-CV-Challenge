"""Speech-to-text pipeline for Lumos voice recall.

This module is intentionally bounded: it transcribes short utterances that the
speaker-awareness pipeline has already classified as directed at Lumos. The STT
layer does not choose robot behavior directly; accepted transcripts are routed
into the existing grounded recall worker, exactly like browser-chat questions.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from backend.audio.audio_capture import AudioChunk
from backend.audio.voice_activity_detector import VoiceActivityResult
from backend.perception.active_speaker_detector import ActiveSpeakerResult
from backend.utils.config import SpeechToTextConfig


@dataclass(frozen=True)
class UtteranceAudio:
    """One short speech segment ready for transcription."""

    request_id: str
    start_timestamp: float
    end_timestamp: float
    samples: np.ndarray
    sample_rate: int
    channels: int
    speaker_track_id: str | None = None
    speaker_location: str | None = None
    speaker_confidence: float = 0.0

    @property
    def duration_s(self) -> float:
        if self.samples.size == 0 or self.sample_rate <= 0:
            return 0.0
        return float(self.samples.shape[0]) / float(self.sample_rate)

    def mono_float32(self, target_sample_rate: int = 16000) -> np.ndarray:
        """Return mono float32 audio, resampled with linear interpolation if needed."""
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 2:
            mono = samples.mean(axis=1)
        else:
            mono = samples.reshape(-1)
        mono = np.nan_to_num(mono, copy=False).astype(np.float32, copy=False)
        mono = np.clip(mono, -1.0, 1.0)
        if self.sample_rate <= 0 or int(self.sample_rate) == int(target_sample_rate) or mono.size == 0:
            return mono.astype(np.float32, copy=False)
        source_duration = mono.size / float(self.sample_rate)
        target_count = max(1, int(round(source_duration * float(target_sample_rate))))
        source_x = np.linspace(0.0, source_duration, num=mono.size, endpoint=False)
        target_x = np.linspace(0.0, source_duration, num=target_count, endpoint=False)
        return np.interp(target_x, source_x, mono).astype(np.float32)

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "start_timestamp": round(float(self.start_timestamp), 3),
            "end_timestamp": round(float(self.end_timestamp), 3),
            "duration_s": round(self.duration_s, 3),
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "speaker_track_id": self.speaker_track_id,
            "speaker_location": self.speaker_location,
            "speaker_confidence": round(float(self.speaker_confidence), 3),
        }


@dataclass(frozen=True)
class SpeechTranscript:
    """Completed STT result."""

    request_id: str
    text: str
    accepted: bool
    reason: str
    engine: str
    transcribe_ms: float
    utterance_duration_s: float
    language: str | None = None
    confidence: float | None = None
    speaker_track_id: str | None = None
    speaker_location: str | None = None
    created_at: float = field(default_factory=time.monotonic)

    def to_log_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "engine": self.engine,
            "transcribe_ms": round(float(self.transcribe_ms), 3),
            "utterance_duration_s": round(float(self.utterance_duration_s), 3),
            "language": self.language,
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
            "speaker_track_id": self.speaker_track_id,
            "speaker_location": self.speaker_location,
        }


class UtteranceSegmenter:
    """Turn VAD + active-speaker decisions into bounded utterance clips."""

    def __init__(self, config: SpeechToTextConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.enabled = bool(config.enabled)
        self._pre_roll: deque[AudioChunk] = deque(maxlen=max(1, int(round(config.pre_roll_s * 1000.0 / 30.0))))
        self._recording_chunks: list[AudioChunk] = []
        self._recording = False
        self._recording_start_ts = 0.0
        self._last_speech_monotonic = 0.0
        self._last_finish_monotonic = 0.0
        self._speaker_track_id: str | None = None
        self._speaker_location: str | None = None
        self._speaker_confidence = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    def reset(self) -> None:
        self._recording_chunks.clear()
        self._recording = False
        self._recording_start_ts = 0.0
        self._last_speech_monotonic = 0.0
        self._speaker_track_id = None
        self._speaker_location = None
        self._speaker_confidence = 0.0

    def update(
        self,
        chunk: AudioChunk,
        voice: VoiceActivityResult | None,
        speaker: ActiveSpeakerResult | None,
        *,
        now: float | None = None,
        allow_capture: bool = True,
    ) -> UtteranceAudio | None:
        if not self.enabled:
            return None
        now = time.monotonic() if now is None else float(now)
        self._pre_roll.append(chunk)

        voice_active = bool(voice is not None and voice.is_speech)
        speaker_to_lumos = bool(
            speaker is not None
            and speaker.speech_detected
            and speaker.speaking_to_robot is True
            and float(speaker.confidence) >= float(self.config.speaker_min_confidence)
        )
        eligible = bool(allow_capture and voice_active and speaker_to_lumos)

        if voice_active:
            self._last_speech_monotonic = now

        if not self._recording and eligible:
            if now - self._last_finish_monotonic < float(self.config.cooldown_s):
                return None
            self._recording = True
            self._recording_start_ts = float(chunk.timestamp)
            self._recording_chunks = list(self._pre_roll)
            self._speaker_track_id = speaker.active_track_id if speaker is not None else None
            self._speaker_location = speaker.active_track_location if speaker is not None else None
            self._speaker_confidence = float(speaker.confidence) if speaker is not None else 0.0
            self.logger.info(
                "stt_utterance_started speaker=%s location=%s confidence=%.3f",
                self._speaker_track_id,
                self._speaker_location,
                self._speaker_confidence,
            )
            return None

        if not self._recording:
            return None

        if not self._recording_chunks or self._recording_chunks[-1].sequence != chunk.sequence:
            self._recording_chunks.append(chunk)

        duration_s = self._chunks_duration_s(self._recording_chunks)
        silence_s = now - self._last_speech_monotonic if self._last_speech_monotonic > 0 else 0.0
        should_finish = False
        reason = ""
        if duration_s >= float(self.config.max_utterance_s):
            should_finish = True
            reason = "max_utterance_s"
        elif duration_s >= float(self.config.min_utterance_s) and silence_s >= float(self.config.end_silence_s):
            should_finish = True
            reason = "end_silence"

        if not should_finish:
            return None

        utterance = self._finish(chunk.timestamp, reason=reason, now=now)
        return utterance

    def flush(self, *, reason: str = "manual_flush", now: float | None = None) -> UtteranceAudio | None:
        if not self._recording or not self._recording_chunks:
            self.reset()
            return None
        now = time.monotonic() if now is None else float(now)
        return self._finish(self._recording_chunks[-1].timestamp, reason=reason, now=now)

    def _finish(self, end_timestamp: float, *, reason: str, now: float) -> UtteranceAudio | None:
        chunks = list(self._recording_chunks)
        speaker_track_id = self._speaker_track_id
        speaker_location = self._speaker_location
        speaker_confidence = self._speaker_confidence
        self._last_finish_monotonic = now
        self.reset()
        if not chunks:
            return None
        sample_rate = int(chunks[-1].sample_rate)
        channels = int(chunks[-1].channels)
        samples = np.concatenate([np.asarray(item.samples, dtype=np.float32) for item in chunks], axis=0)
        utterance = UtteranceAudio(
            request_id=uuid.uuid4().hex,
            start_timestamp=float(chunks[0].timestamp),
            end_timestamp=float(end_timestamp),
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            speaker_track_id=speaker_track_id,
            speaker_location=speaker_location,
            speaker_confidence=speaker_confidence,
        )
        self.logger.info(
            "stt_utterance_finished request_id=%s reason=%s duration_s=%.3f chunks=%s",
            utterance.request_id,
            reason,
            utterance.duration_s,
            len(chunks),
        )
        return utterance

    @staticmethod
    def _chunks_duration_s(chunks: Iterable[AudioChunk]) -> float:
        total_frames = 0
        sample_rate = 0
        for chunk in chunks:
            samples = np.asarray(chunk.samples)
            total_frames += int(samples.shape[0]) if samples.ndim >= 1 else 0
            sample_rate = int(chunk.sample_rate)
        if sample_rate <= 0:
            return 0.0
        return total_frames / float(sample_rate)


class LocalSpeechToTextEngine:
    """Small wrapper around optional local STT backends."""

    def __init__(self, config: SpeechToTextConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("lelamp")
        self.engine_name = str(config.backend)
        self._model = None
        self._load_error: str | None = None

    def transcribe(self, utterance: UtteranceAudio) -> SpeechTranscript:
        start = time.perf_counter()
        try:
            if self.config.backend == "faster_whisper":
                text, language, confidence = self._transcribe_faster_whisper(utterance)
                engine = "faster_whisper"
            elif self.config.backend == "whisper":
                text, language, confidence = self._transcribe_openai_whisper(utterance)
                engine = "whisper"
            else:
                raise RuntimeError(f"unsupported_stt_backend:{self.config.backend}")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            accepted, reason = self._accept_text(text)
            return SpeechTranscript(
                request_id=utterance.request_id,
                text=text,
                accepted=accepted,
                reason=reason,
                engine=engine,
                transcribe_ms=elapsed_ms,
                utterance_duration_s=utterance.duration_s,
                language=language,
                confidence=confidence,
                speaker_track_id=utterance.speaker_track_id,
                speaker_location=utterance.speaker_location,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return SpeechTranscript(
                request_id=utterance.request_id,
                text="",
                accepted=False,
                reason=f"stt_error:{exc}",
                engine=str(self.config.backend),
                transcribe_ms=elapsed_ms,
                utterance_duration_s=utterance.duration_s,
                speaker_track_id=utterance.speaker_track_id,
                speaker_location=utterance.speaker_location,
            )

    def _transcribe_faster_whisper(self, utterance: UtteranceAudio) -> tuple[str, str | None, float | None]:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install with: python -m pip install faster-whisper"
                ) from exc
            self.logger.info(
                "stt_model_loading backend=faster_whisper model=%s device=%s compute_type=%s",
                self.config.model_size,
                self.config.device,
                self.config.compute_type,
            )
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self.logger.info("stt_model_loaded backend=faster_whisper model=%s", self.config.model_size)

        audio = utterance.mono_float32(target_sample_rate=16000)
        segments, info = self._model.transcribe(
            audio,
            language=self.config.language or None,
            beam_size=max(1, int(self.config.beam_size)),
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=float(self.config.no_speech_threshold),
        )
        parts: list[str] = []
        confidences: list[float] = []
        for segment in segments:
            text = str(getattr(segment, "text", "") or "").strip()
            if text:
                parts.append(text)
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            if no_speech_prob is not None:
                confidences.append(max(0.0, min(1.0, 1.0 - float(no_speech_prob))))
        text = " ".join(parts).strip()
        language = getattr(info, "language", None)
        confidence = max(confidences) if confidences else None
        return text, language, confidence

    def _transcribe_openai_whisper(self, utterance: UtteranceAudio) -> tuple[str, str | None, float | None]:
        if self._model is None:
            try:
                import whisper  # type: ignore
            except Exception as exc:
                raise RuntimeError("openai-whisper is not installed. Install with: python -m pip install openai-whisper") from exc
            self.logger.info("stt_model_loading backend=whisper model=%s", self.config.model_size)
            self._model = whisper.load_model(self.config.model_size, device=self.config.device)
            self.logger.info("stt_model_loaded backend=whisper model=%s", self.config.model_size)

        audio = utterance.mono_float32(target_sample_rate=16000)
        result = self._model.transcribe(
            audio,
            language=self.config.language or None,
            fp16=False,
            condition_on_previous_text=False,
        )
        text = str(result.get("text", "") or "").strip()
        language = result.get("language")
        return text, language, None

    def _accept_text(self, text: str) -> tuple[bool, str]:
        cleaned = " ".join(str(text or "").strip().split())
        if not cleaned:
            return False, "empty_transcript"
        if len(cleaned) < int(self.config.min_chars):
            return False, "too_short"
        word_count = len([part for part in cleaned.split(" ") if part])
        if word_count < int(self.config.min_words):
            return False, "too_few_words"
        lowered = cleaned.lower()
        blocked = {
            "thank you",
            "thanks for watching",
            "subscribe",
            "bye",
            "you",
        }
        if lowered in blocked:
            return False, "common_hallucination_filter"
        return True, "accepted"


class SpeechToTextWorker:
    """Background STT worker so Whisper never blocks the webcam loop."""

    def __init__(
        self,
        config: SpeechToTextConfig,
        result_queue: "queue.Queue[SpeechTranscript]",
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.result_queue = result_queue
        self.logger = logger or logging.getLogger("lelamp")
        self.enabled = bool(config.enabled)
        self._input_queue: "queue.Queue[UtteranceAudio | None]" = queue.Queue(maxsize=max(1, int(config.max_queue_size)))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._engine = LocalSpeechToTextEngine(config, logger=self.logger)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="speech-to-text-worker", daemon=True)
        self._thread.start()
        self.logger.info(
            "stt_worker_started backend=%s model=%s device=%s compute_type=%s",
            self.config.backend,
            self.config.model_size,
            self.config.device,
            self.config.compute_type,
        )

    def submit(self, utterance: UtteranceAudio) -> bool:
        if not self.enabled:
            return False
        try:
            self._input_queue.put_nowait(utterance)
            return True
        except queue.Full:
            self.logger.warning("stt_utterance_dropped reason=stt_queue_full request_id=%s", utterance.request_id)
            return False

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        try:
            self._input_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self.logger.info("stt_worker_stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            item = self._input_queue.get()
            if item is None:
                break
            self.logger.info(
                "stt_transcribe_started request_id=%s duration_s=%.3f speaker=%s",
                item.request_id,
                item.duration_s,
                item.speaker_track_id,
            )
            transcript = self._engine.transcribe(item)
            self.logger.info(
                "stt_transcribe_finished request_id=%s accepted=%s reason=%s text=%r transcribe_ms=%.3f",
                transcript.request_id,
                transcript.accepted,
                transcript.reason,
                transcript.text,
                transcript.transcribe_ms,
            )
            self.result_queue.put(transcript)
