"""Deterministic object-query parser for grounded memory recall.

The MVP deliberately avoids using an LLM for parsing. It extracts one target
object label from short text questions such as:
    - Where did you last see my phone?
    - Where is the cup?
    - Have you seen my mouse?

The output is a normalized label used for exact SQLite lookup. If the parser is
uncertain, it returns ``normalized_label=None`` rather than guessing a location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Keep aliases duplicated here instead of importing the detector module so the
# recall CLI can run without OpenCV/Ultralytics installed.
OBJECT_ALIASES: dict[str, str] = {
    "phone": "phone",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "laptop": "laptop",
    "computer": "laptop",
    "notebook": "laptop",
    "monitor": "monitor",
    "screen": "monitor",
    "tv": "monitor",
    "television": "monitor",
    "cup": "cup",
    "mug": "cup",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "book": "book",
    "bottle": "bottle",
    "remote": "remote",
    "remote control": "remote",
    "clock": "clock",
    "chair": "chair",
    "backpack": "backpack",
    "bag": "bag",
    "handbag": "bag",
    "vase": "vase",
    "scissors": "scissors",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "around",
    "at",
    "did",
    "do",
    "for",
    "have",
    "is",
    "it",
    "last",
    "locate",
    "location",
    "me",
    "my",
    "near",
    "of",
    "on",
    "please",
    "remember",
    "see",
    "seen",
    "show",
    "the",
    "there",
    "was",
    "were",
    "where",
    "you",
}

OBJECT_PHRASE_PATTERNS = [
    re.compile(r"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
    re.compile(r"\bhave\s+you\s+seen\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
    re.compile(r"\bdid\s+you\s+see\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
    re.compile(r"\b(?:find|locate)\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
]


@dataclass(frozen=True)
class ParsedObjectQuery:
    original_query: str
    target_text: str | None
    normalized_label: str | None
    confidence: float
    strategy: str

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "target_text": self.target_text,
            "normalized_label": self.normalized_label,
            "confidence": round(float(self.confidence), 3),
            "strategy": self.strategy,
        }


def clean_query_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"[^a-z0-9\s'-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def normalize_object_label(label: str) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if cleaned in OBJECT_ALIASES:
        return OBJECT_ALIASES[cleaned]
    # Conservative singular fallback for simple plurals: cups -> cup.
    if cleaned.endswith("s") and cleaned[:-1] in OBJECT_ALIASES:
        return OBJECT_ALIASES[cleaned[:-1]]
    return cleaned


def parse_object_query(query: str, aliases: dict[str, str] | None = None) -> ParsedObjectQuery:
    alias_map = aliases or OBJECT_ALIASES
    cleaned = clean_query_text(query)
    if not cleaned:
        return ParsedObjectQuery(query, None, None, 0.0, "empty_query")

    # Prefer exact known aliases, sorted by length so "cell phone" wins before
    # "phone" and "remote control" wins before "remote".
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if _contains_phrase(cleaned, alias):
            return ParsedObjectQuery(
                original_query=query,
                target_text=alias,
                normalized_label=alias_map[alias],
                confidence=0.95,
                strategy="known_alias",
            )

    for pattern in OBJECT_PHRASE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = normalize_object_label(target)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.65,
                strategy="question_pattern",
            )

    fallback = _last_content_token(cleaned.split())
    if fallback:
        normalized = normalize_object_label(fallback)
        return ParsedObjectQuery(
            original_query=query,
            target_text=fallback,
            normalized_label=normalized or None,
            confidence=0.35,
            strategy="fallback_last_token",
        )

    return ParsedObjectQuery(query, None, None, 0.0, "no_object_found")


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _sanitize_object_phrase(phrase: str) -> str:
    cleaned = clean_query_text(phrase)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    tokens = [token for token in cleaned.split() if token not in STOPWORDS]
    return " ".join(tokens).strip()


def _last_content_token(tokens: Iterable[str]) -> str | None:
    for token in reversed(list(tokens)):
        token = token.strip(" -'")
        if token and token not in STOPWORDS:
            return token
    return None
