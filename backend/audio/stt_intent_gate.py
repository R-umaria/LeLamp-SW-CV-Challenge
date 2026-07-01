"""Wake-word and intent gate for noisy STT transcripts.

The browser chat remains the reliable recall path. Voice transcripts are accepted
only when they are directed at Lumos or are unambiguously object-memory queries.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

from backend.utils.config import SpeechToTextConfig


QUESTION_PATTERNS = (
    re.compile(r"\bwhere\s+(?:is|are|was|were|did|do|have|has)\b", re.I),
    re.compile(r"\bdid\s+you\s+(?:see|spot|notice|detect|find)\b", re.I),
    re.compile(r"\bwhere\s+did\s+you\s+last\s+see\b", re.I),
    re.compile(r"\blast\s+see\b", re.I),
    re.compile(r"\bremember\s+(?:seeing|where)\b", re.I),
)

OBJECT_HINTS = {
    "phone", "cell phone", "laptop", "keyboard", "mouse", "book", "cup", "bottle",
    "remote", "scissors", "clock", "vase", "backpack", "bag", "chair", "monitor", "tv",
}


@dataclass(frozen=True)
class STTIntentDecision:
    accepted: bool
    reason: str
    normalized_text: str
    directed_to_lumos: bool
    clear_memory_query: bool
    confidence: float | None = None

    def to_log_dict(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "normalized_text": self.normalized_text,
            "directed_to_lumos": bool(self.directed_to_lumos),
            "clear_memory_query": bool(self.clear_memory_query),
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
        }


class STTIntentGate:
    def __init__(self, config: SpeechToTextConfig) -> None:
        self.config = config
        self.wake_words = tuple(_normalize_phrase(w) for w in getattr(config, "wake_words", ()) if str(w).strip())
        self.require_wake_word = bool(getattr(config, "require_wake_word", False))
        self.min_confidence = float(getattr(config, "min_confidence", 0.0) or 0.0)
        self.cooldown_s = float(getattr(config, "intent_cooldown_s", getattr(config, "cooldown_s", 0.0)) or 0.0)
        self._last_accept_at = 0.0
        self._last_text = ""

    def evaluate(self, text: str, *, confidence: float | None = None, now: float | None = None) -> STTIntentDecision:
        now = time.monotonic() if now is None else float(now)
        normalized = _normalize_phrase(text)
        if not normalized:
            return STTIntentDecision(False, "empty_transcript", normalized, False, False, confidence)
        if confidence is not None and confidence < self.min_confidence:
            return STTIntentDecision(False, "confidence_below_threshold", normalized, False, False, confidence)
        directed = self._has_wake_word(normalized)
        clear_query = self._is_clear_memory_query(normalized)
        if self.require_wake_word and not directed:
            return STTIntentDecision(False, "missing_wake_word", normalized, directed, clear_query, confidence)
        if not directed and not clear_query:
            return STTIntentDecision(False, "not_lumos_directed_or_memory_query", normalized, directed, clear_query, confidence)
        if now - self._last_accept_at < self.cooldown_s and normalized == self._last_text:
            return STTIntentDecision(False, "duplicate_cooldown", normalized, directed, clear_query, confidence)
        if now - self._last_accept_at < min(self.cooldown_s, 1.5):
            return STTIntentDecision(False, "global_cooldown", normalized, directed, clear_query, confidence)
        self._last_accept_at = now
        self._last_text = normalized
        return STTIntentDecision(True, "accepted", normalized, directed, clear_query, confidence)

    def _has_wake_word(self, normalized: str) -> bool:
        if not self.wake_words:
            return False
        return any(normalized.startswith(w) or f" {w} " in f" {normalized} " for w in self.wake_words)

    def _is_clear_memory_query(self, normalized: str) -> bool:
        has_question_pattern = any(p.search(normalized) for p in QUESTION_PATTERNS)
        has_object_hint = any(re.search(rf"\b{re.escape(obj)}\b", normalized) for obj in OBJECT_HINTS)
        if has_question_pattern and has_object_hint:
            return True
        # Generic wording still accepted when phrased directly as a memory query.
        return bool(re.search(r"\bwhere\s+did\s+you\s+last\s+see\s+(?:my\s+|the\s+)?\w+", normalized))


def parse_wake_words(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = [str(p).strip() for p in raw]
    return tuple(_normalize_phrase(p) for p in parts if p)


def _normalize_phrase(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()
