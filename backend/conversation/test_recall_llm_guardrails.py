"""Milestone 4.2.1 LLM grounding guardrail tests.

Run:
    python -m unittest backend.conversation.test_recall_llm_guardrails -v
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from backend.conversation.llm_client import LLMResponse, OllamaStatus
from backend.conversation.query_parser import parse_object_query
from backend.conversation.recall_agent import RecallAgent, _validate_llm_grounding
from backend.memory.memory_store import MemoryRecord, MemoryStore


class FakeOllama:
    def __init__(self, text: str) -> None:
        self.text = text
        self.model = "fake-model"

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            attempted=True,
            used_llm=True,
            latency_ms=12.5,
            error=None,
            fallback_reason=None,
        )


class LlmGroundingGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parsed = parse_object_query("Where did you last see my phone?")
        self.record = MemoryRecord(
            id="test-phone-record",
            object_label="cell phone",
            normalized_label="phone",
            location_label="center of view",
            bbox=(447, 487, 100, 96),
            confidence=0.733,
            timestamp="2026-06-18T13:22:33",
            source="webcam",
            frame_path="data\\object_frames\\frame_493_phone.jpg",
        )

    def assert_invalid(self, answer: str, reason: str) -> None:
        ok, actual_reason = _validate_llm_grounding(answer, self.parsed, self.record)
        self.assertFalse(ok, msg=answer)
        self.assertEqual(actual_reason, reason)

    def test_accepts_grounded_answer(self) -> None:
        answer = "I last saw your phone on the center of view around 2026-06-18T13:22:33. My confidence was 0.733."
        ok, reason = _validate_llm_grounding(answer, self.parsed, self.record)
        self.assertTrue(ok, reason)

    def test_rejects_no_memory_answer_when_record_exists(self) -> None:
        self.assert_invalid("I do not remember seeing that.", "contradicts_retrieved_memory")

    def test_rejects_wrong_location(self) -> None:
        self.assert_invalid(
            "I last saw your phone on the right side of view around 2026-06-18T13:22:33. My confidence was 0.733.",
            "missing_stored_location",
        )

    def test_rejects_frame_path_leak(self) -> None:
        self.assert_invalid(
            "I last saw your phone on the center of view around 2026-06-18T13:22:33. My confidence was 0.733. Frame path: data\\object_frames\\frame_493_phone.jpg",
            "implementation_detail_leak",
        )

    def test_rejects_missing_confidence_and_timestamp(self) -> None:
        self.assert_invalid(
            "I last saw your phone on the center of view.",
            "missing_confidence_or_timestamp",
        )

    def test_agent_falls_back_when_llm_contradicts_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)

            agent = RecallAgent(memory_db=db_path, use_llm=False, logger=logging.getLogger("test"))
            agent.use_llm = True
            agent.ollama = FakeOllama("I do not remember seeing that.")  # type: ignore[assignment]
            agent.ollama_status = OllamaStatus(
                ok=True,
                base_url="http://localhost:11434",
                model="fake-model",
                model_available=True,
                latency_ms=1.0,
            )

            result = agent.answer("Where did you last see my phone?")

        self.assertTrue(result.llm_attempted)
        self.assertFalse(result.llm_used)
        self.assertEqual(result.llm_fallback_reason, "contradicts_retrieved_memory")
        self.assertIn("invalid_llm_response", result.llm_error or "")
        self.assertEqual(
            result.answer,
            "I last saw your phone on the center of view around 13:22:33 on 2026-06-18. My confidence was 0.73.",
        )

    def test_agent_uses_valid_grounded_llm_answer(self) -> None:
        valid_answer = "I last saw your phone on the center of view around 2026-06-18T13:22:33. My confidence was 0.733."
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)

            agent = RecallAgent(memory_db=db_path, use_llm=False, logger=logging.getLogger("test"))
            agent.use_llm = True
            agent.ollama = FakeOllama(valid_answer)  # type: ignore[assignment]
            agent.ollama_status = OllamaStatus(
                ok=True,
                base_url="http://localhost:11434",
                model="fake-model",
                model_available=True,
                latency_ms=1.0,
            )

            result = agent.answer("Where did you last see my phone?")

        self.assertTrue(result.llm_attempted)
        self.assertTrue(result.llm_used)
        self.assertIsNone(result.llm_error)
        self.assertIsNone(result.llm_fallback_reason)
        self.assertEqual(result.answer, valid_answer)


if __name__ == "__main__":
    unittest.main()
