"""Grounded live conversation agent for LeLamp Milestone 4.3.

The recall path is deliberately bounded:

    user text -> deterministic intent parser -> SQLite retrieval -> optional
    structured Ollama /api/chat phrasing -> strict fact validation.

The LLM never searches memory, never chooses locations, and never overrides
SQLite. If the model returns invalid JSON or contradicts retrieved memory, the
candidate is rejected and the deterministic grounded answer is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.conversation.intent_parser import ConversationIntentType, ParsedConversationIntent, parse_conversation_intent
from backend.conversation.llm_client import LLMResponse, OllamaClient, OllamaStatus
from backend.conversation.query_parser import ParsedObjectQuery
from backend.memory.memory_store import MemoryRecord, MemoryStore


RECALL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_type": {
            "type": "string",
            "enum": ["memory_answer", "no_memory", "recent_objects", "unsupported"],
        },
        "target_object": {"type": ["string", "null"]},
        "location_label": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "timestamp": {"type": ["string", "null"]},
        "spoken_answer": {"type": "string"},
    },
    "required": ["answer_type", "target_object", "location_label", "confidence", "timestamp", "spoken_answer"],
    "additionalProperties": False,
}

SYSTEM_MESSAGE = """You are the voice of a LeLamp-inspired robotic lamp.
You are only a wording layer. The backend has already retrieved the only facts you may use.
Return exactly one JSON object matching the provided schema.
Do not invent object locations. Do not mention frame paths, SQLite, JSON, IDs, logs, files, or implementation details.
Keep spoken_answer to one concise conversational sentence."""


@dataclass(frozen=True)
class RecallResult:
    user_query: str
    intent: ParsedConversationIntent
    parsed_object: str | None
    parsed: ParsedObjectQuery
    memory_record: MemoryRecord | None
    recent_records: tuple[MemoryRecord, ...]
    answer: str
    answer_type: str
    memory_retrieval_ms: float
    llm_response_ms: float | None
    llm_requested: bool
    llm_required: bool
    llm_attempted: bool
    llm_used: bool
    llm_error: str | None
    llm_fallback_reason: str | None
    llm_validation_reason: str | None
    llm_raw_response: str | None
    ollama_connection_ok: bool | None
    timestamp: str

    @property
    def used_llm(self) -> bool:
        """Backward-compatible alias for Milestone 4 callers."""
        return self.llm_used

    @property
    def llm_required_failed(self) -> bool:
        return self.llm_required and self.llm_requested and not self.llm_used and self.answer_type in {
            "memory_answer",
            "recent_objects",
        }

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "intent": self.intent.to_dict(),
            "answer_type": self.answer_type,
            "parsed_object": self.parsed_object,
            "parsed": self.parsed.to_dict(),
            "retrieved_memory_id": self.memory_record.id if self.memory_record else None,
            "memory_record": self.memory_record.to_dict() if self.memory_record else None,
            "recent_records": [record.to_dict() for record in self.recent_records],
            "memory_retrieval_ms": round(float(self.memory_retrieval_ms), 3),
            "llm_response_ms": None if self.llm_response_ms is None else round(float(self.llm_response_ms), 3),
            "llm_requested": self.llm_requested,
            "llm_required": self.llm_required,
            "llm_attempted": self.llm_attempted,
            "llm_used": self.llm_used,
            "used_llm": self.llm_used,
            "llm_required_failed": self.llm_required_failed,
            "llm_error": self.llm_error,
            "llm_fallback_reason": self.llm_fallback_reason,
            "llm_validation_reason": self.llm_validation_reason,
            "llm_raw_response": self.llm_raw_response,
            "ollama_connection_ok": self.ollama_connection_ok,
            "answer": self.answer,
        }


class RecallAgent:
    """Answers supported memory questions using SQLite-grounded facts."""

    def __init__(
        self,
        memory_db: str | Path = "data/scene_memory.sqlite",
        use_llm: bool = False,
        llm_required: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:1.5b",
        llm_timeout_s: float = 90.0,
        llm_connect_timeout_s: float = 5.0,
        ollama_keep_alive: str = "1h",
        llm_max_tokens: int = 120,
        recent_object_limit: int = 8,
        log_paths: str | Path | Iterable[str | Path] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.memory_db = Path(memory_db)
        self.store = MemoryStore(self.memory_db)
        self.use_llm = bool(use_llm)
        self.llm_required = bool(llm_required)
        self.recent_object_limit = int(recent_object_limit)
        self.llm_max_tokens = max(32, int(llm_max_tokens))
        self.logger = logger or logging.getLogger("lelamp")
        self.log_paths = _coerce_paths(log_paths)
        self.ollama: OllamaClient | None = None
        self.ollama_status: OllamaStatus | None = None

        if self.use_llm:
            self.ollama = OllamaClient(
                base_url=ollama_url,
                model=ollama_model,
                timeout_s=llm_timeout_s,
                connect_timeout_s=llm_connect_timeout_s,
                keep_alive=ollama_keep_alive,
            )
            self.ollama_status = self.ollama.check_connectivity()
            if self.ollama_status.ok and self.ollama_status.model_available:
                self.logger.info(
                    "Ollama connectivity OK url=%s model=%s status_ms=%.3f",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    self.ollama_status.latency_ms,
                )
            elif self.ollama_status.ok:
                self.logger.warning(
                    "Ollama reachable but requested model is unavailable url=%s model=%s available_models=%s",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    list(self.ollama_status.available_models),
                )
            else:
                self.logger.warning(
                    "Ollama unavailable for recall url=%s model=%s error=%s",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    self.ollama_status.error,
                )

    def answer(self, user_query: str) -> RecallResult:
        intent = parse_conversation_intent(user_query)
        parsed = intent.object_query

        retrieval_start = time.perf_counter()
        memory_record: MemoryRecord | None = None
        recent_records: tuple[MemoryRecord, ...] = ()

        if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN and parsed.normalized_label:
            memory_record = self.store.find_latest_by_normalized_label(parsed.normalized_label)
        elif intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
            recent_records = tuple(self.store.recent_unique_by_normalized_label(limit=self.recent_object_limit))

        memory_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fallback_answer, answer_type = deterministic_conversation_answer(intent, memory_record, recent_records)
        final_answer = fallback_answer
        final_answer_type = answer_type

        llm_response: LLMResponse | None = None
        llm_attempted = False
        llm_used = False
        llm_error: str | None = None
        llm_fallback_reason: str | None = None
        llm_validation_reason: str | None = None
        llm_raw_response: str | None = None
        ollama_connection_ok: bool | None = None

        if self.use_llm:
            ollama_connection_ok = False if self.ollama_status is None else (
                self.ollama_status.ok and self.ollama_status.model_available
            )
            llm_response = self._try_llm_answer(user_query, intent, memory_record, recent_records)
            llm_attempted = llm_response.attempted
            llm_error = llm_response.error
            llm_fallback_reason = llm_response.fallback_reason
            llm_raw_response = llm_response.text if llm_response.text else None

            if llm_response.used_llm and llm_response.json_data is not None:
                is_valid, validation_reason = validate_structured_llm_response(
                    llm_response.json_data,
                    intent,
                    memory_record,
                    recent_records,
                )
                if is_valid:
                    final_answer = str(llm_response.json_data["spoken_answer"]).strip()
                    final_answer_type = str(llm_response.json_data["answer_type"])
                    llm_used = True
                    llm_error = None
                    llm_fallback_reason = None
                    llm_validation_reason = None
                else:
                    llm_validation_reason = validation_reason
                    llm_fallback_reason = validation_reason
                    llm_error = f"invalid_llm_response: {validation_reason}"
            elif llm_response.attempted and llm_response.fallback_reason:
                llm_validation_reason = llm_response.fallback_reason

            if self.use_llm and not llm_used:
                self.logger.warning(
                    "LLM candidate rejected query=%r intent=%s parsed_object=%s reason=%s error=%s attempted=%s raw=%r",
                    user_query,
                    intent.intent.value,
                    parsed.normalized_label,
                    llm_fallback_reason,
                    llm_error,
                    llm_attempted,
                    llm_raw_response,
                )

        result = RecallResult(
            user_query=user_query,
            intent=intent,
            parsed_object=parsed.normalized_label,
            parsed=parsed,
            memory_record=memory_record,
            recent_records=recent_records,
            answer=final_answer,
            answer_type=final_answer_type,
            memory_retrieval_ms=memory_retrieval_ms,
            llm_response_ms=None if llm_response is None else llm_response.latency_ms,
            llm_requested=self.use_llm,
            llm_required=self.llm_required,
            llm_attempted=llm_attempted,
            llm_used=llm_used,
            llm_error=llm_error,
            llm_fallback_reason=llm_fallback_reason,
            llm_validation_reason=llm_validation_reason,
            llm_raw_response=llm_raw_response,
            ollama_connection_ok=ollama_connection_ok,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )
        self._log_result(result)
        return result

    def _try_llm_answer(
        self,
        user_query: str,
        intent: ParsedConversationIntent,
        memory_record: MemoryRecord | None,
        recent_records: Sequence[MemoryRecord],
    ) -> LLMResponse:
        if intent.intent == ConversationIntentType.UNSUPPORTED:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, fallback_reason="unsupported_intent")
        if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN and memory_record is None:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, fallback_reason="no_memory_record")
        if intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS and not recent_records:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, fallback_reason="no_recent_objects")
        if self.ollama is None:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, error="ollama_client_not_initialized", fallback_reason="client_unavailable")
        if self.ollama_status is not None and not self.ollama_status.ok:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, error=self.ollama_status.error or "ollama_unavailable", fallback_reason="connectivity_check_failed")
        if self.ollama_status is not None and not self.ollama_status.model_available:
            return LLMResponse("", attempted=False, used_llm=False, latency_ms=0.0, error=self.ollama_status.error or f"model_not_found: {self.ollama.model}", fallback_reason="model_unavailable")

        messages = build_structured_messages(user_query, intent, memory_record, recent_records)
        return self.ollama.chat_json(messages, RECALL_RESPONSE_SCHEMA, max_tokens=self.llm_max_tokens)

    def _log_result(self, result: RecallResult) -> None:
        payload = result.to_dict()
        self.logger.info(
            "Recall query=%r intent=%s parsed_object=%s answer_type=%s memory_id=%s recent_count=%s retrieval_ms=%.3f llm_attempted=%s llm_used=%s llm_ms=%s llm_error=%s fallback=%s answer=%r",
            result.user_query,
            result.intent.intent.value,
            result.parsed_object,
            result.answer_type,
            result.memory_record.id if result.memory_record else None,
            len(result.recent_records),
            result.memory_retrieval_ms,
            result.llm_attempted,
            result.llm_used,
            "" if result.llm_response_ms is None else f"{result.llm_response_ms:.3f}",
            result.llm_error,
            result.llm_fallback_reason,
            result.answer,
        )
        for path in self.log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_structured_messages(
    user_query: str,
    intent: ParsedConversationIntent,
    memory_record: MemoryRecord | None,
    recent_records: Sequence[MemoryRecord],
) -> list[dict[str, str]]:
    facts: dict[str, Any] = {
        "user_query": user_query,
        "intent": intent.intent.value,
        "parsed_object": intent.normalized_label,
    }

    if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN and memory_record is not None:
        facts["retrieved_memory"] = _memory_record_for_llm(memory_record)
        facts["required_json_values"] = {
            "answer_type": "memory_answer",
            "target_object": memory_record.normalized_label,
            "location_label": memory_record.location_label,
            "confidence": round(float(memory_record.confidence), 3),
            "timestamp": memory_record.timestamp,
        }
        facts["fallback_answer"] = deterministic_object_answer(intent.object_query, memory_record)
    elif intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
        labels = [record.normalized_label for record in recent_records]
        facts["recent_objects"] = [_memory_record_for_llm(record) for record in recent_records]
        facts["allowed_labels"] = labels
        facts["required_json_values"] = {
            "answer_type": "recent_objects",
            "target_object": None,
            "location_label": None,
            "confidence": None,
            "timestamp": None,
        }
        facts["fallback_answer"] = deterministic_recent_objects_answer(recent_records)

    user_content = (
        "Use only these backend-provided memory facts. Return one short JSON answer.\n"
        + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def validate_structured_llm_response(
    candidate: Mapping[str, Any],
    intent: ParsedConversationIntent,
    memory_record: MemoryRecord | None,
    recent_records: Sequence[MemoryRecord],
) -> tuple[bool, str | None]:
    """Validate structured LLM output against retrieved SQLite facts."""

    required_keys = {"answer_type", "target_object", "location_label", "confidence", "timestamp", "spoken_answer"}
    missing = sorted(required_keys.difference(candidate.keys()))
    if missing:
        return False, "missing_schema_keys:" + ",".join(missing)

    answer_type = str(candidate.get("answer_type") or "").strip()
    spoken_answer = str(candidate.get("spoken_answer") or "").strip()
    if not spoken_answer:
        return False, "empty_spoken_answer"
    leak_reason = _validate_spoken_answer_text(spoken_answer, memory_record is not None)
    if leak_reason is not None:
        return False, leak_reason

    if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN:
        if memory_record is None:
            return False, "no_memory_record"
        if answer_type != "memory_answer":
            return False, "wrong_answer_type"
        expected_object = str(memory_record.normalized_label or intent.normalized_label or "").strip().lower()
        target_object = str(candidate.get("target_object") or "").strip().lower()
        if target_object != expected_object:
            return False, "target_object_mismatch"
        location_label = candidate.get("location_label")
        if not isinstance(location_label, str) or location_label != memory_record.location_label:
            return False, "location_label_mismatch"
        if memory_record.location_label.lower() not in spoken_answer.lower():
            return False, "spoken_answer_missing_location"
        if expected_object and expected_object not in spoken_answer.lower():
            return False, "spoken_answer_missing_target_object"
        if not _candidate_confidence_matches(candidate.get("confidence"), memory_record.confidence) and str(candidate.get("timestamp") or "") != memory_record.timestamp:
            return False, "missing_matching_confidence_or_timestamp"
        return True, None

    if intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
        if not recent_records:
            return False, "no_recent_objects"
        if answer_type != "recent_objects":
            return False, "wrong_answer_type"
        if candidate.get("target_object") is not None:
            return False, "target_object_should_be_null"
        if candidate.get("location_label") is not None:
            return False, "location_label_should_be_null"
        allowed_labels = [record.normalized_label.strip().lower() for record in recent_records if record.normalized_label.strip()]
        lower_answer = spoken_answer.lower()
        missing_labels = [label for label in allowed_labels if label not in lower_answer]
        if missing_labels:
            return False, "missing_recent_labels:" + ",".join(missing_labels)
        return True, None

    if answer_type != "unsupported":
        return False, "unsupported_intent_wrong_answer_type"
    return True, None


def deterministic_conversation_answer(
    intent: ParsedConversationIntent,
    memory_record: MemoryRecord | None,
    recent_records: Sequence[MemoryRecord],
) -> tuple[str, str]:
    if intent.intent == ConversationIntentType.OBJECT_LAST_SEEN:
        if memory_record is None:
            return deterministic_no_memory_answer(intent.object_query), "no_memory"
        return deterministic_object_answer(intent.object_query, memory_record), "memory_answer"
    if intent.intent == ConversationIntentType.LIST_RECENT_OBJECTS:
        if not recent_records:
            return "I do not remember seeing any objects yet.", "no_memory"
        return deterministic_recent_objects_answer(recent_records), "recent_objects"
    return (
        "I can answer grounded memory questions, like where I last saw your phone, or what objects I detected.",
        "unsupported",
    )


def deterministic_recall_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    """Backward-compatible object-only deterministic answer."""

    if memory_record is None:
        return deterministic_no_memory_answer(parsed)
    return deterministic_object_answer(parsed, memory_record)


def deterministic_object_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord) -> str:
    display_label = _display_object_label(parsed, memory_record)
    time_text = _friendly_time(memory_record.timestamp)
    return (
        f"I last saw your {display_label} on the {memory_record.location_label} "
        f"around {time_text}. My confidence was {memory_record.confidence:.2f}."
    )


def deterministic_no_memory_answer(parsed: ParsedObjectQuery) -> str:
    display_label = _display_object_label(parsed, None)
    if display_label == "that object":
        return "I can answer object-memory questions, like where I last saw your cup or phone."
    return f"I do not remember seeing your {display_label}."


def deterministic_recent_objects_answer(records: Sequence[MemoryRecord]) -> str:
    labels = []
    seen: set[str] = set()
    for record in records:
        label = record.normalized_label.strip().lower()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    if not labels:
        return "I do not remember seeing any objects yet."
    return f"I recently saw {_join_labels(labels)}."


def _join_labels(labels: Sequence[str]) -> str:
    if len(labels) == 1:
        return f"a {labels[0]}"
    if len(labels) == 2:
        return f"a {labels[0]} and {labels[1]}"
    prefixed = [f"a {labels[0]}"] + list(labels[1:])
    return ", ".join(prefixed[:-1]) + f", and {prefixed[-1]}"


def _memory_record_for_llm(memory_record: MemoryRecord) -> dict:
    return {
        "object_label": memory_record.object_label,
        "normalized_label": memory_record.normalized_label,
        "location_label": memory_record.location_label,
        "confidence": round(float(memory_record.confidence), 3),
        "timestamp": memory_record.timestamp,
        "source": memory_record.source,
    }


def _display_object_label(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    if memory_record is not None:
        return memory_record.normalized_label or memory_record.object_label
    if parsed.normalized_label:
        return parsed.normalized_label
    if parsed.target_text:
        return parsed.target_text
    return "that object"


def _friendly_time(timestamp_text: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp_text)
    except ValueError:
        return timestamp_text
    return parsed.strftime("%H:%M:%S on %Y-%m-%d")


def _candidate_confidence_matches(candidate_value: Any, expected_confidence: float) -> bool:
    try:
        candidate = float(candidate_value)
    except (TypeError, ValueError):
        return False
    return abs(candidate - float(expected_confidence)) <= 0.001


def _validate_spoken_answer_text(spoken_answer: str, memory_exists: bool) -> str | None:
    lowered = " ".join(spoken_answer.strip().split()).lower()
    implementation_leaks = (
        "frame_path",
        "frame path",
        ".jpg",
        ".png",
        "data\\",
        "data/",
        "json",
        "sqlite",
        "memory id",
        "database",
        "log",
        "file",
    )
    if any(marker in lowered for marker in implementation_leaks):
        return "implementation_detail_leak"

    contradictory_phrases = (
        "do not remember",
        "don't remember",
        "dont remember",
        "not remember",
        "no memory",
        "can't remember",
        "cannot remember",
        "do not recall",
        "don't recall",
        "have not seen",
        "haven't seen",
        "never seen",
    )
    if memory_exists and any(phrase in lowered for phrase in contradictory_phrases):
        return "contradicts_retrieved_memory"
    return None


def _validate_llm_grounding(
    answer: str,
    parsed: ParsedObjectQuery,
    memory_record: MemoryRecord | None,
) -> tuple[bool, str | None]:
    """Backward-compatible validator for old one-sentence tests.

    Milestone 4.3 uses ``validate_structured_llm_response``. This helper remains
    for existing guardrail tests and rejects the same unsafe cases.
    """

    if memory_record is None:
        return False, "no_memory_record"
    leak_reason = _validate_spoken_answer_text(answer, memory_exists=True)
    if leak_reason is not None:
        return False, leak_reason
    lowered = " ".join(str(answer or "").strip().split()).lower()
    if not lowered:
        return False, "empty_llm_answer"
    expected_location = str(memory_record.location_label or "").strip().lower()
    if expected_location and expected_location not in lowered:
        return False, "missing_stored_location"
    object_terms = {
        str(memory_record.normalized_label or "").strip().lower(),
        str(memory_record.object_label or "").strip().lower(),
        str(parsed.normalized_label or "").strip().lower(),
        str(parsed.target_text or "").strip().lower(),
    }
    object_terms = {term for term in object_terms if term}
    if object_terms and not any(term in lowered for term in object_terms):
        return False, "missing_target_object"
    expected_confidence_options = {f"{float(memory_record.confidence):.2f}", f"{float(memory_record.confidence):.3f}"}
    has_confidence = any(conf in lowered for conf in expected_confidence_options)
    has_timestamp = str(memory_record.timestamp or "").strip().lower() in lowered
    if not has_confidence and not has_timestamp:
        return False, "missing_confidence_or_timestamp"
    return True, None


def _coerce_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask LeLamp grounded questions about stored object memory")
    parser.add_argument("query", nargs="?", help="Question, e.g. 'Where did you last see my phone?'")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase retrieved memory answers")
    parser.add_argument("--llm-required", action="store_true", help="Return nonzero if an LLM-eligible answer falls back")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:1.5b", help="Ollama model, e.g. qwen2.5:1.5b")
    parser.add_argument("--llm-timeout", type=float, default=90.0, help="Ollama full chat/read timeout in seconds")
    parser.add_argument("--llm-connect-timeout", type=float, default=5.0, help="Ollama connection/status timeout in seconds")
    parser.add_argument("--llm-max-tokens", type=int, default=120, help="Maximum Ollama output tokens for recall phrasing")
    parser.add_argument("--ollama-keep-alive", type=str, default="1h", help="Ollama keep_alive duration, e.g. 1h")
    parser.add_argument("--ollama-timeout", type=float, default=None, help="Backward-compatible alias for --llm-timeout")
    parser.add_argument("--test-ollama", action="store_true", help="Check Ollama connectivity/model availability and exit")
    parser.add_argument("--warm-ollama", action="store_true", help="Warm the selected Ollama model and exit")
    parser.add_argument("--json", action="store_true", help="Print machine-readable recall result JSON")
    parser.add_argument("--debug", action="store_true", help="Print debug metadata such as frame_path outside the spoken answer")
    parser.add_argument("--log-path", type=str, default="logs/recall.jsonl", help="JSONL recall log path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    llm_timeout = float(args.ollama_timeout) if args.ollama_timeout is not None else float(args.llm_timeout)

    if args.test_ollama or args.warm_ollama:
        client = OllamaClient(
            args.ollama_url,
            args.ollama_model,
            timeout_s=llm_timeout,
            connect_timeout_s=args.llm_connect_timeout,
            keep_alive=args.ollama_keep_alive,
        )
        status = client.check_connectivity()
        warm_response = None
        if status.ok and status.model_available and args.warm_ollama:
            warm_response = client.warm_up()
        payload = status.to_dict()
        if warm_response is not None:
            payload["warm_up"] = warm_response.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if status.ok and status.model_available:
                print(f"Ollama OK: {status.base_url} has model {status.model} ({status.latency_ms:.1f} ms)")
                if warm_response is not None:
                    print(
                        f"Warm-up attempted={warm_response.attempted} used_llm={warm_response.used_llm} "
                        f"latency_ms={warm_response.latency_ms:.1f} error={warm_response.error or 'none'}"
                    )
            elif status.ok:
                print(f"Ollama reachable, but model '{status.model}' was not found.")
                print("Available models: " + (", ".join(status.available_models) or "none"))
            else:
                print(f"Ollama unavailable: {status.error}")
        if not (status.ok and status.model_available):
            return 1
        if warm_response is not None and not warm_response.used_llm:
            return 1
        return 0

    if not args.query:
        print("Missing query. Example: python -m backend.conversation.recall_agent \"Where is the cup?\"")
        return 2

    console_level = logging.WARNING if args.json else logging.INFO
    logging.basicConfig(level=console_level, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = RecallAgent(
        memory_db=args.memory_db,
        use_llm=args.use_llm,
        llm_required=args.llm_required,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        llm_timeout_s=llm_timeout,
        llm_connect_timeout_s=args.llm_connect_timeout,
        ollama_keep_alive=args.ollama_keep_alive,
        llm_max_tokens=args.llm_max_tokens,
        log_paths=args.log_path,
    )
    result = agent.answer(args.query)

    if args.use_llm and not result.llm_used:
        warning = (
            "Warning: --use-llm was requested, but the deterministic fallback was used. "
            f"reason={result.llm_fallback_reason or 'unknown'} error={result.llm_error or 'none'}"
        )
        print(warning, file=sys.stderr)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(
            f"intent={result.intent.intent.value} "
            f"answer_type={result.answer_type} "
            f"parsed_object={result.parsed_object or 'none'} "
            f"memory_id={result.memory_record.id if result.memory_record else 'none'} "
            f"recent_count={len(result.recent_records)} "
            f"memory_retrieval_ms={result.memory_retrieval_ms:.3f} "
            f"llm_attempted={result.llm_attempted} "
            f"llm_used={result.llm_used} "
            f"llm_response_ms={'' if result.llm_response_ms is None else f'{result.llm_response_ms:.3f}'} "
            f"llm_error={result.llm_error or 'none'} "
            f"llm_fallback_reason={result.llm_fallback_reason or 'none'}"
        )
        if args.debug and result.memory_record is not None:
            print(f"frame_path={result.memory_record.frame_path or 'none'}")

    if result.llm_required_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
