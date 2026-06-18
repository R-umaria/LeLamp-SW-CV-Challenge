"""Conversation intent parser for live LeLamp recall.

The parser is intentionally deterministic. It does not try to solve general NLU;
it only separates demo-critical grounded intents before object extraction:

- object_last_seen: retrieve one object from SQLite
- list_recent_objects: summarize recent unique objects from SQLite
- unsupported: avoid pretending an arbitrary utterance is an object query
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.conversation.query_parser import ParsedObjectQuery, clean_query_text, is_list_recent_objects_query, parse_object_query


class ConversationIntentType(str, Enum):
    OBJECT_LAST_SEEN = "object_last_seen"
    LIST_RECENT_OBJECTS = "list_recent_objects"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ParsedConversationIntent:
    original_query: str
    intent: ConversationIntentType
    object_query: ParsedObjectQuery
    confidence: float
    reason: str

    @property
    def normalized_label(self) -> str | None:
        return self.object_query.normalized_label

    @property
    def target_text(self) -> str | None:
        return self.object_query.target_text

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "target_text": self.target_text,
            "normalized_label": self.normalized_label,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "object_query": self.object_query.to_dict(),
        }


_OBJECT_RECALL_SIGNALS = [
    re.compile(r"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\b"),
    re.compile(r"\bhave\s+you\s+seen\b"),
    re.compile(r"\bdid\s+you\s+(?:see|notice)\s+if\s+i\s+(?:had|have|was\s+holding|was\s+using)\b"),
    re.compile(r"\bdid\s+you\s+see\b"),
    re.compile(r"\b(?:find|locate)\b"),
]

_UNSUPPORTED_CHITCHAT = [
    re.compile(r"\bhow\s+are\s+you\b"),
    re.compile(r"\bwhat\s+is\s+your\s+name\b"),
    re.compile(r"\btell\s+me\s+a\s+joke\b"),
    re.compile(r"\bwho\s+are\s+you\b"),
]


def parse_conversation_intent(query: str) -> ParsedConversationIntent:
    cleaned = clean_query_text(query)
    object_query = parse_object_query(query)

    if not cleaned:
        return ParsedConversationIntent(query, ConversationIntentType.UNSUPPORTED, object_query, 0.0, "empty_query")

    if is_list_recent_objects_query(cleaned):
        return ParsedConversationIntent(
            query,
            ConversationIntentType.LIST_RECENT_OBJECTS,
            object_query,
            0.96,
            "list_recent_objects_pattern",
        )

    if any(pattern.search(cleaned) is not None for pattern in _UNSUPPORTED_CHITCHAT):
        return ParsedConversationIntent(query, ConversationIntentType.UNSUPPORTED, _empty_object_query(query), 0.90, "unsupported_chitchat")

    if object_query.normalized_label and any(pattern.search(cleaned) is not None for pattern in _OBJECT_RECALL_SIGNALS):
        return ParsedConversationIntent(
            query,
            ConversationIntentType.OBJECT_LAST_SEEN,
            object_query,
            max(0.70, object_query.confidence),
            object_query.strategy,
        )

    # Compact object-only queries are useful in the terminal during debugging.
    if object_query.normalized_label and object_query.strategy == "known_alias_fallback":
        return ParsedConversationIntent(
            query,
            ConversationIntentType.OBJECT_LAST_SEEN,
            object_query,
            object_query.confidence,
            "known_alias_compact_query",
        )

    return ParsedConversationIntent(query, ConversationIntentType.UNSUPPORTED, object_query, 0.35, "no_supported_intent")


def _empty_object_query(query: str) -> ParsedObjectQuery:
    return ParsedObjectQuery(query, None, None, 0.0, "unsupported_intent")
