"""Grounded memory-recall agent for Milestone 4.

CLI examples:
    python -m backend.conversation.recall_agent "Where did you last see my phone?"
    python -m backend.conversation.recall_agent "Where is the cup?" --json
    python -m backend.conversation.recall_agent "Have you seen my mouse?" --use-llm

The recall path is intentionally bounded:
    text query -> deterministic object parser -> exact SQLite lookup ->
    deterministic or strictly grounded LLM phrasing.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backend.conversation.llm_client import LLMResponse, OllamaClient
from backend.conversation.query_parser import ParsedObjectQuery, parse_object_query
from backend.memory.memory_store import MemoryRecord, MemoryStore


STRICT_GROUNDED_RECALL_PROMPT = """You are the voice of a LeLamp-inspired robotic lamp.

You must answer the user's object-location question using only the provided memory record.
Do not use outside knowledge.
Do not infer, guess, or invent a location.
Do not say where the object is now. Only say where it was last seen.
Do not mention objects that are not in the memory record.
If the memory record is missing or insufficient, say: "I do not remember seeing that."
Keep the answer to one short conversational sentence.

User question:
{user_query}

Parsed target object:
{parsed_object}

Memory record JSON:
{memory_record_json}

Answer:"""


@dataclass(frozen=True)
class RecallResult:
    user_query: str
    parsed_object: str | None
    parsed: ParsedObjectQuery
    memory_record: MemoryRecord | None
    answer: str
    memory_retrieval_ms: float
    llm_response_ms: float | None
    used_llm: bool
    llm_error: str | None
    timestamp: str

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
            "used_llm": self.used_llm,
            "llm_error": self.llm_error,
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
        log_paths: str | Path | Iterable[str | Path] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.memory_db = Path(memory_db)
        self.store = MemoryStore(self.memory_db)
        self.use_llm = bool(use_llm)
        self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model) if self.use_llm else None
        self.logger = logger or logging.getLogger("lelamp")
        self.log_paths = _coerce_paths(log_paths)

    def answer(self, user_query: str) -> RecallResult:
        parsed = parse_object_query(user_query)
        retrieval_start = time.perf_counter()
        memory_record = None
        if parsed.normalized_label:
            memory_record = self.store.find_latest_by_normalized_label(parsed.normalized_label)
        memory_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fallback_answer = deterministic_recall_answer(parsed, memory_record)
        llm_response: LLMResponse | None = None
        final_answer = fallback_answer

        # Safer MVP rule: only ask the LLM to phrase an answer when a concrete
        # memory exists. Missing-memory answers remain deterministic.
        if self.use_llm and self.ollama is not None and memory_record is not None:
            prompt = build_grounded_prompt(user_query, parsed, memory_record)
            llm_response = self.ollama.generate(prompt)
            if llm_response.used_llm and llm_response.text:
                final_answer = _clean_llm_text(llm_response.text)
            else:
                final_answer = fallback_answer

        result = RecallResult(
            user_query=user_query,
            parsed_object=parsed.normalized_label,
            parsed=parsed,
            memory_record=memory_record,
            answer=final_answer,
            memory_retrieval_ms=memory_retrieval_ms,
            llm_response_ms=None if llm_response is None else llm_response.latency_ms,
            used_llm=False if llm_response is None else llm_response.used_llm,
            llm_error=None if llm_response is None else llm_response.error,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )
        self._log_result(result)
        return result

    def _log_result(self, result: RecallResult) -> None:
        payload = result.to_dict()
        self.logger.info(
            "Recall query=%r parsed_object=%s memory_id=%s retrieval_ms=%.3f llm_ms=%s used_llm=%s answer=%r",
            result.user_query,
            result.parsed_object,
            result.memory_record.id if result.memory_record else None,
            result.memory_retrieval_ms,
            "" if result.llm_response_ms is None else f"{result.llm_response_ms:.3f}",
            result.used_llm,
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
        memory_record_json=json.dumps(memory_record.to_dict(), ensure_ascii=False, indent=2),
    )


def deterministic_recall_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    if not parsed.normalized_label:
        return "I am not sure which object you mean. Ask me about a specific object, like the cup or phone."

    if memory_record is None:
        return f"I do not remember seeing the {parsed.normalized_label}."

    label = memory_record.normalized_label or memory_record.object_label
    answer = (
        f"I last saw the {label} at the {memory_record.location_label} "
        f"at {memory_record.timestamp} with {memory_record.confidence:.2f} confidence."
    )
    if memory_record.frame_path:
        answer += f" I saved a reference frame at {memory_record.frame_path}."
    return answer


def _clean_llm_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    # Remove common leading labels if a local model includes them.
    for prefix in ("Answer:", "Lamp:", "LeLamp:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


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
    parser.add_argument("--json", action="store_true", help="Print machine-readable recall result JSON")
    parser.add_argument("--log-path", type=str, default="logs/recall.jsonl", help="JSONL recall log path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        log_paths=args.log_path,
    )
    result = agent.answer(args.query)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(
            f"parsed_object={result.parsed_object or 'none'} "
            f"memory_id={result.memory_record.id if result.memory_record else 'none'} "
            f"memory_retrieval_ms={result.memory_retrieval_ms:.3f} "
            f"llm_response_ms={'' if result.llm_response_ms is None else f'{result.llm_response_ms:.3f}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
