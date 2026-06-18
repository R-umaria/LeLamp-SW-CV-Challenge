"""Grounded memory-recall agent for Milestone 4.1.

CLI examples:
    python -m backend.conversation.recall_agent "Where did you last see my phone?"
    python -m backend.conversation.recall_agent "Where is the cup?" --json
    python -m backend.conversation.recall_agent "Have you seen my mouse?" --use-llm
    python -m backend.conversation.recall_agent --test-ollama --ollama-model llama3.2

The recall path remains bounded:
    text query -> deterministic object parser -> exact SQLite lookup ->
    deterministic or strictly grounded LLM phrasing.
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
from typing import Iterable

from backend.conversation.llm_client import LLMResponse, OllamaClient, OllamaStatus
from backend.conversation.query_parser import ParsedObjectQuery, parse_object_query
from backend.memory.memory_store import MemoryRecord, MemoryStore


STRICT_GROUNDED_RECALL_PROMPT = """You are the voice of a LeLamp-inspired robotic lamp.

You are only a phrasing layer. A valid SQLite memory record has already been retrieved.
Use only that memory record and the grounded template sentence below.

Hard rules:
- Do not use outside knowledge.
- Do not infer, guess, or invent a location.
- Do not say where the object is now. Only say where it was last seen.
- Do not mention nearby objects or spatial relationships unless they are explicitly in the memory record.
- Do not mention frame paths, files, IDs, JSON, logs, or implementation details.
- Because a valid memory record is provided, do not say that you do not remember seeing the object.
- Keep the answer to one short conversational sentence.

Preferred answer:
{grounded_template_answer}

User question:
{user_query}

Parsed target object:
{parsed_object}

Memory record JSON:
{memory_record_json}

Answer with one sentence only:"""


@dataclass(frozen=True)
class RecallResult:
    user_query: str
    parsed_object: str | None
    parsed: ParsedObjectQuery
    memory_record: MemoryRecord | None
    answer: str
    memory_retrieval_ms: float
    llm_response_ms: float | None
    llm_requested: bool
    llm_attempted: bool
    llm_used: bool
    llm_error: str | None
    llm_fallback_reason: str | None
    ollama_connection_ok: bool | None
    timestamp: str

    @property
    def used_llm(self) -> bool:
        """Backward-compatible alias for Milestone 4 callers."""
        return self.llm_used

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "parsed_object": self.parsed_object,
            "parsed": self.parsed.to_dict(),
            "retrieved_memory_id": self.memory_record.id if self.memory_record else None,
            "memory_record": self.memory_record.to_dict() if self.memory_record else None,
            "memory_retrieval_ms": round(float(self.memory_retrieval_ms), 3),
            "llm_response_ms": None if self.llm_response_ms is None else round(float(self.llm_response_ms), 3),
            "llm_requested": self.llm_requested,
            "llm_attempted": self.llm_attempted,
            "llm_used": self.llm_used,
            "used_llm": self.llm_used,
            "llm_error": self.llm_error,
            "llm_fallback_reason": self.llm_fallback_reason,
            "ollama_connection_ok": self.ollama_connection_ok,
            "answer": self.answer,
        }


class RecallAgent:
    """Retrieves one grounded memory and phrases a response."""

    def __init__(
        self,
        memory_db: str | Path = "data/scene_memory.sqlite",
        use_llm: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        ollama_timeout_s: float = 8.0,
        log_paths: str | Path | Iterable[str | Path] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.memory_db = Path(memory_db)
        self.store = MemoryStore(self.memory_db)
        self.use_llm = bool(use_llm)
        self.logger = logger or logging.getLogger("lelamp")
        self.log_paths = _coerce_paths(log_paths)
        self.ollama: OllamaClient | None = None
        self.ollama_status: OllamaStatus | None = None

        if self.use_llm:
            self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model, timeout_s=ollama_timeout_s)
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
        parsed = parse_object_query(user_query)
        retrieval_start = time.perf_counter()
        memory_record = None
        if parsed.normalized_label:
            memory_record = self.store.find_latest_by_normalized_label(parsed.normalized_label)
        memory_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fallback_answer = deterministic_recall_answer(parsed, memory_record)
        final_answer = fallback_answer

        llm_response: LLMResponse | None = None
        llm_attempted = False
        llm_used = False
        llm_error: str | None = None
        llm_fallback_reason: str | None = None
        ollama_connection_ok: bool | None = None

        if self.use_llm:
            ollama_connection_ok = False if self.ollama_status is None else (
                self.ollama_status.ok and self.ollama_status.model_available
            )

            # Safer MVP rule: only ask the LLM to phrase an answer when a
            # concrete memory exists. Missing-memory answers stay deterministic.
            if memory_record is None:
                llm_fallback_reason = "no_memory_record"
            elif self.ollama is None:
                llm_error = "ollama_client_not_initialized"
                llm_fallback_reason = "client_unavailable"
            elif self.ollama_status is not None and not self.ollama_status.ok:
                llm_error = self.ollama_status.error or "ollama_unavailable"
                llm_fallback_reason = "connectivity_check_failed"
            elif self.ollama_status is not None and not self.ollama_status.model_available:
                llm_error = self.ollama_status.error or f"model_not_found: {self.ollama.model}"
                llm_fallback_reason = "model_unavailable"
            else:
                prompt = build_grounded_prompt(user_query, parsed, memory_record)
                llm_response = self.ollama.generate(prompt)
                llm_attempted = llm_response.attempted
                llm_used = llm_response.used_llm
                llm_error = llm_response.error
                llm_fallback_reason = llm_response.fallback_reason
                if llm_response.used_llm and llm_response.text:
                    candidate_answer = _clean_llm_text(llm_response.text)
                    is_grounded, validation_reason = _validate_llm_grounding(
                        candidate_answer, parsed, memory_record
                    )
                    if is_grounded:
                        final_answer = candidate_answer
                        llm_used = True
                    else:
                        final_answer = fallback_answer
                        llm_used = False
                        llm_fallback_reason = validation_reason
                        llm_error = f"invalid_llm_response: {validation_reason}"
                else:
                    final_answer = fallback_answer

            if self.use_llm and not llm_used:
                self.logger.warning(
                    "LLM fallback used query=%r parsed_object=%s reason=%s error=%s attempted=%s",
                    user_query,
                    parsed.normalized_label,
                    llm_fallback_reason,
                    llm_error,
                    llm_attempted,
                )

        result = RecallResult(
            user_query=user_query,
            parsed_object=parsed.normalized_label,
            parsed=parsed,
            memory_record=memory_record,
            answer=final_answer,
            memory_retrieval_ms=memory_retrieval_ms,
            llm_response_ms=None if llm_response is None else llm_response.latency_ms,
            llm_requested=self.use_llm,
            llm_attempted=llm_attempted,
            llm_used=llm_used,
            llm_error=llm_error,
            llm_fallback_reason=llm_fallback_reason,
            ollama_connection_ok=ollama_connection_ok,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )
        self._log_result(result)
        return result

    def _log_result(self, result: RecallResult) -> None:
        payload = result.to_dict()
        self.logger.info(
            "Recall query=%r parsed_object=%s memory_id=%s retrieval_ms=%.3f llm_attempted=%s llm_used=%s llm_ms=%s llm_error=%s fallback=%s answer=%r",
            result.user_query,
            result.parsed_object,
            result.memory_record.id if result.memory_record else None,
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


def build_grounded_prompt(user_query: str, parsed: ParsedObjectQuery, memory_record: MemoryRecord) -> str:
    return STRICT_GROUNDED_RECALL_PROMPT.format(
        user_query=user_query,
        parsed_object=parsed.normalized_label or "",
        grounded_template_answer=deterministic_recall_answer(parsed, memory_record),
        memory_record_json=json.dumps(_memory_record_for_prompt(memory_record), ensure_ascii=False, indent=2),
    )


def deterministic_recall_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    display_label = _display_object_label(parsed, memory_record)

    if not parsed.normalized_label:
        return "I am not sure which object you mean. Ask me about a specific object, like your cup or phone."

    if memory_record is None:
        return f"I do not remember seeing your {display_label}."

    time_text = _friendly_time(memory_record.timestamp)
    return (
        f"I last saw your {display_label} on the {memory_record.location_label} "
        f"around {time_text}. My confidence was {memory_record.confidence:.2f}."
    )


def _memory_record_for_prompt(memory_record: MemoryRecord) -> dict:
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


def _clean_llm_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    for prefix in ("Answer:", "Lamp:", "LeLamp:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _validate_llm_grounding(
    answer: str,
    parsed: ParsedObjectQuery,
    memory_record: MemoryRecord | None,
) -> tuple[bool, str | None]:
    """Accept an LLM answer only if it is consistent with retrieved memory.

    This is the critical safety valve for local LLM use. Connectivity and a
    non-empty generated response are not enough. If SQLite retrieval found a
    record, the displayed/spoken response must not contradict that record or
    omit the actual stored location. Invalid responses fall back to the
    deterministic grounded template.
    """

    if memory_record is None:
        return False, "no_memory_record"

    cleaned = " ".join(str(answer or "").strip().split())
    if not cleaned:
        return False, "empty_llm_answer"

    lowered = cleaned.lower()

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
    if any(phrase in lowered for phrase in contradictory_phrases):
        return False, "contradicts_retrieved_memory"

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
    )
    if any(marker in lowered for marker in implementation_leaks):
        return False, "implementation_detail_leak"

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

    expected_confidence_options = {
        f"{float(memory_record.confidence):.2f}",
        f"{float(memory_record.confidence):.3f}",
    }
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
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase a retrieved memory answer")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model, e.g. llama3.2 or qwen2.5")
    parser.add_argument("--ollama-timeout", type=float, default=8.0, help="Ollama generate timeout in seconds")
    parser.add_argument("--test-ollama", action="store_true", help="Check Ollama connectivity/model availability and exit")
    parser.add_argument("--json", action="store_true", help="Print machine-readable recall result JSON")
    parser.add_argument("--debug", action="store_true", help="Print debug metadata such as frame_path outside the spoken answer")
    parser.add_argument("--log-path", type=str, default="logs/recall.jsonl", help="JSONL recall log path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test_ollama:
        client = OllamaClient(args.ollama_url, args.ollama_model, timeout_s=args.ollama_timeout)
        status = client.check_connectivity()
        if args.json:
            print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
        else:
            if status.ok and status.model_available:
                print(f"Ollama OK: {status.base_url} has model {status.model} ({status.latency_ms:.1f} ms)")
            elif status.ok:
                print(f"Ollama reachable, but model '{status.model}' was not found.")
                print("Available models: " + (", ".join(status.available_models) or "none"))
            else:
                print(f"Ollama unavailable: {status.error}")
        return 0 if status.ok and status.model_available else 1

    if not args.query:
        print("Missing query. Example: python -m backend.conversation.recall_agent \"Where is the cup?\"")
        return 2

    console_level = logging.WARNING if args.json else logging.INFO
    logging.basicConfig(level=console_level, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = RecallAgent(
        memory_db=args.memory_db,
        use_llm=args.use_llm,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_timeout_s=args.ollama_timeout,
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
            f"parsed_object={result.parsed_object or 'none'} "
            f"memory_id={result.memory_record.id if result.memory_record else 'none'} "
            f"memory_retrieval_ms={result.memory_retrieval_ms:.3f} "
            f"llm_attempted={result.llm_attempted} "
            f"llm_used={result.llm_used} "
            f"llm_response_ms={'' if result.llm_response_ms is None else f'{result.llm_response_ms:.3f}'} "
            f"llm_error={result.llm_error or 'none'} "
            f"llm_fallback_reason={result.llm_fallback_reason or 'none'}"
        )
        if args.debug and result.memory_record is not None:
            print(f"frame_path={result.memory_record.frame_path or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
