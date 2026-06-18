"""Run a three-question Ollama timing validation for the final demo path.

This command seeds a temporary scene-memory database, then asks three grounded
recall questions through the same RecallAgent used by the browser worker. It
fails if any LLM-eligible answer falls back, so --llm-required semantics remain
honest.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from backend.conversation.recall_agent import RecallAgent
from backend.memory.memory_store import MemoryRecord, MemoryStore


DEMO_QUERIES = (
    "Where did you last see the monitor?",
    "Was there any pen in the view?",
    "Where did you last see the cup?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate average Ollama recall time over three grounded questions")
    parser.add_argument("--ollama-url", type=str, default="http://10.0.0.70:11434")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--llm-timeout", type=float, default=90.0)
    parser.add_argument("--llm-connect-timeout", type=float, default=5.0)
    parser.add_argument("--llm-max-tokens", type=int, default=120)
    parser.add_argument("--ollama-keep-alive", type=str, default="1h")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def seed_memory(db_path: Path) -> None:
    store = MemoryStore(db_path)
    records = [
        MemoryRecord.create(
            object_label="tv",
            normalized_label="monitor",
            location_label="center of view",
            bbox=(220, 120, 180, 120),
            confidence=0.86,
            source="validation_seed",
            timestamp="2026-06-18T15:00:00",
        ),
        MemoryRecord.create(
            object_label="pen",
            normalized_label="pen",
            location_label="right side of desk",
            bbox=(400, 300, 45, 18),
            confidence=0.74,
            source="validation_seed",
            timestamp="2026-06-18T15:00:01",
        ),
        MemoryRecord.create(
            object_label="cup",
            normalized_label="cup",
            location_label="left side of desk",
            bbox=(90, 260, 70, 90),
            confidence=0.91,
            source="validation_seed",
            timestamp="2026-06-18T15:00:02",
        ),
    ]
    for record in records:
        store.insert(record)


def validate(args: argparse.Namespace, queries: Sequence[str] = DEMO_QUERIES) -> tuple[int, dict]:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    with tempfile.TemporaryDirectory(prefix="lelamp_llm_validation_") as tmpdir:
        db_path = Path(tmpdir) / "scene_memory.sqlite"
        seed_memory(db_path)
        agent = RecallAgent(
            memory_db=db_path,
            use_llm=True,
            llm_required=True,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            llm_timeout_s=args.llm_timeout,
            llm_connect_timeout_s=args.llm_connect_timeout,
            ollama_keep_alive=args.ollama_keep_alive,
            llm_max_tokens=args.llm_max_tokens,
        )
        rows = []
        for query in queries:
            result = agent.answer(query)
            rows.append(
                {
                    "query": query,
                    "answer": result.answer,
                    "parsed_object": result.parsed_object,
                    "answer_type": result.answer_type,
                    "llm_attempted": result.llm_attempted,
                    "llm_used": result.llm_used,
                    "llm_response_ms": result.llm_response_ms,
                    "llm_error": result.llm_error,
                    "llm_fallback_reason": result.llm_fallback_reason,
                }
            )

    llm_times = [float(row["llm_response_ms"]) for row in rows if row["llm_used"] and row["llm_response_ms"] is not None]
    passed = len(llm_times) == len(rows)
    payload = {
        "ok": passed,
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "llm_max_tokens": args.llm_max_tokens,
        "question_count": len(rows),
        "llm_used_count": len(llm_times),
        "avg_llm_response_ms": None if not llm_times else round(statistics.mean(llm_times), 3),
        "min_llm_response_ms": None if not llm_times else round(min(llm_times), 3),
        "max_llm_response_ms": None if not llm_times else round(max(llm_times), 3),
        "results": rows,
    }
    return (0 if passed else 1), payload


def main() -> int:
    args = parse_args()
    code, payload = validate(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"LLM timing validation ok={payload['ok']} model={payload['ollama_model']} url={payload['ollama_url']}")
        print(
            "questions={question_count} llm_used={llm_used_count} avg_ms={avg_llm_response_ms} min_ms={min_llm_response_ms} max_ms={max_llm_response_ms}".format(
                **payload
            )
        )
        for index, row in enumerate(payload["results"], start=1):
            print(
                f"{index}. llm_used={row['llm_used']} llm_ms={row['llm_response_ms']} "
                f"fallback={row['llm_fallback_reason'] or 'none'} object={row['parsed_object']} query={row['query']!r}"
            )
            print(f"   answer={row['answer']}")
        if code != 0:
            print("FAIL: at least one LLM-eligible recall question used deterministic fallback.", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
