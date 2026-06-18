"""Deterministic object-query parser for grounded memory recall.

Milestone 4.1 keeps parsing deterministic and local. The parser extracts one
object target from short recall questions such as:
    - Where did you last see my phone?
    - Where is the cup?
    - Have you seen my mouse?
    - Did you see my spectacles that I left near the cup?

Important MVP rule:
    Nearby/context objects are not treated as the recall target when the user
    explicitly asks about a different possessive object. For example,
    "my spectacles ... near the cup" parses as ``glasses``, not ``cup``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Kept local to conversation code so the recall CLI does not import OpenCV,
# Ultralytics, or the object detector just to parse text.
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
    "stapler": "stapler",
    "pen": "pen",
    "pencil": "pencil",
    "spectacles": "glasses",
    "glasses": "glasses",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "around",
    "at",
    "by",
    "chance",
    "did",
    "do",
    "for",
    "have",
    "having",
    "had",
    "has",
    "if",
    "so",
    "then",
    "i",
    "is",
    "it",
    "last",
    "locate",
    "location",
    "detect",
    "detected",
    "objects",
    "object",
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
    "that",
    "the",
    "there",
    "was",
    "were",
    "where",
    "you",
}

# Clause boundaries keep relation/context phrases from being interpreted as the
# target object. In "my spectacles that I left near the cup", the target phrase
# ends before "that" and the nearby cup is ignored for target selection.
BOUNDARY_WORDS = {
    "around",
    "at",
    "behind",
    "beside",
    "by",
    "from",
    "if",
    "i",
    "in",
    "inside",
    "left",
    "near",
    "next",
    "on",
    "over",
    "right",
    "that",
    "under",
    "where",
    "which",
    "with",
    "you",
}

_BOUNDARY_REGEX = "|".join(sorted(re.escape(word) for word in BOUNDARY_WORDS))
_TARGET_TEXT = rf"(?P<object>[a-z0-9][a-z0-9\s'-]*?)(?=\s+(?:{_BOUNDARY_REGEX})\b|\s*$)"

OBJECT_PHRASE_PATTERNS = [
    re.compile(rf"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bhave\s+you\s+seen\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bdid\s+you\s+see\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\b(?:find|locate)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\b(?:was|were)\s+(?:i\s+)?(?:having|using|holding)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
]

POSSESSIVE_TARGET_PATTERN = re.compile(rf"\bmy\s+{_TARGET_TEXT}")

LIST_OBJECTS_PATTERNS = [
    re.compile(r"\bwhat\s+(?:objects|things|items)\s+did\s+you\s+(?:detect|see|remember)\b"),
    re.compile(r"\bwhat\s+do\s+you\s+remember\s+seeing\b"),
    re.compile(r"\bwhat\s+have\s+you\s+(?:seen|detected)\b"),
    re.compile(r"\bshow\s+me\s+(?:recent|the)?\s*(?:objects|things|items)\b"),
]

HAD_OBJECT_PATTERNS = [
    re.compile(rf"\bdid\s+you\s+see\s+if\s+i\s+(?:had|have|was\s+holding|was\s+using)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bdid\s+you\s+notice\s+if\s+i\s+(?:had|have|was\s+holding|was\s+using)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
]

EXISTENCE_OBJECT_PATTERNS = [
    re.compile(rf"\bwas\s+there\s+(?:any\s+|a\s+|an\s+|the\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bwere\s+there\s+(?:any\s+|some\s+)?{_TARGET_TEXT}"),
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


def is_list_recent_objects_query(text: str) -> bool:
    cleaned = clean_query_text(text)
    return any(pattern.search(cleaned) is not None for pattern in LIST_OBJECTS_PATTERNS)


def normalize_object_label(label: str) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
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

    if is_list_recent_objects_query(cleaned):
        return ParsedObjectQuery(query, None, None, 0.0, "list_recent_objects_query")

    # Strong phrasing that previously failed as target="if":
    # "Did you see if I had a stapler?" means the object is stapler.
    for pattern in HAD_OBJECT_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = _normalize_with_aliases(target, alias_map)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.86,
                strategy="had_object_pattern",
            )

    # Existence phrasing for final-demo queries such as:
    # "Was there any pen in the view?"
    for pattern in EXISTENCE_OBJECT_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = _normalize_with_aliases(target, alias_map)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.82,
                strategy="existence_object_pattern",
            )

    # Strongest signal: explicit possessive target. This fixes questions like
    # "did you see my spectacles that I left near the cup?" by selecting
    # spectacles/glasses instead of the contextual cup.
    possessive_target = _extract_possessive_target(cleaned)
    if possessive_target:
        normalized = _normalize_with_aliases(possessive_target, alias_map)
        return ParsedObjectQuery(
            original_query=query,
            target_text=possessive_target,
            normalized_label=normalized or None,
            confidence=0.98,
            strategy="explicit_possessive",
        )

    # Next, parse a direct object from common recall question shapes. This can
    # return unsupported labels such as "stapler". The recall agent will then do
    # an exact memory lookup and answer that it does not remember seeing it.
    for pattern in OBJECT_PHRASE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = _normalize_with_aliases(target, alias_map)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.75,
                strategy="question_pattern",
            )

    # Known alias fallback for compact queries like "cup?" or "phone location".
    # Longest alias wins so "cell phone" wins before "phone".
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if _contains_phrase(cleaned, alias):
            return ParsedObjectQuery(
                original_query=query,
                target_text=alias,
                normalized_label=alias_map[alias],
                confidence=0.60,
                strategy="known_alias_fallback",
            )

    fallback = _last_content_token(cleaned.split())
    if fallback:
        normalized = _normalize_with_aliases(fallback, alias_map)
        return ParsedObjectQuery(
            original_query=query,
            target_text=fallback,
            normalized_label=normalized or None,
            confidence=0.35,
            strategy="fallback_last_token",
        )

    return ParsedObjectQuery(query, None, None, 0.0, "no_object_found")


def _extract_possessive_target(text: str) -> str | None:
    match = POSSESSIVE_TARGET_PATTERN.search(text)
    if not match:
        return None
    return _sanitize_object_phrase(match.group("object")) or None


def _normalize_with_aliases(label: str, alias_map: dict[str, str]) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
    if cleaned in alias_map:
        return alias_map[cleaned]
    if cleaned.endswith("s") and cleaned[:-1] in alias_map:
        return alias_map[cleaned[:-1]]
    return cleaned


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _sanitize_object_phrase(phrase: str) -> str:
    cleaned = clean_query_text(phrase)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    tokens = [token for token in cleaned.split() if token not in STOPWORDS]
    return " ".join(tokens).strip(" -'")


def _trim_after_boundary(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for token in tokens:
        stripped = token.strip(" -'")
        if stripped in BOUNDARY_WORDS:
            break
        kept.append(stripped)
    return " ".join(token for token in kept if token)


def _last_content_token(tokens: Iterable[str]) -> str | None:
    for token in reversed(list(tokens)):
        token = token.strip(" -'")
        if token and token not in STOPWORDS:
            return token
    return None
