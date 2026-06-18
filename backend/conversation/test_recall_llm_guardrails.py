"""Milestone 4.3 structured LLM grounding tests.

Run:
    python -m unittest backend.conversation.test_recall_llm_guardrails -v
"""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.conversation.intent_parser import parse_conversation_intent
from backend.conversation.llm_client import LLMResponse, OllamaStatus
from backend.conversation.query_parser import parse_object_query
from backend.conversation.recall_agent import RecallAgent, validate_structured_llm_response, _validate_llm_grounding
from backend.memory.memory_store import MemoryRecord, MemoryStore


class FakeOllama:
    def __init__(self, payload: Mapping[str, Any] | str) -> None:
        self.payload = payload
        self.model = "fake-model"

    def chat_json(self, messages: Sequence[Mapping[str, str]], schema: Mapping[str, Any], *, max_tokens: int = 180) -> LLMResponse:
        if isinstance(self.payload, str):
            return LLMResponse(
                text=self.payload,
                attempted=True,
                used_llm=True,
                latency_ms=12.5,
                error="ollama_returned_invalid_json",
                fallback_reason="invalid_json",
                json_data=None,
            )
        return LLMResponse(
            text=json.dumps(self.payload),
            attempted=True,
            used_llm=True,
            latency_ms=12.5,
            error=None,
            fallback_reason=None,
            json_data=dict(self.payload),
        )


class LlmGroundingGuardrailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = parse_conversation_intent("Where did you last see my phone?")
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

    def valid_payload(self) -> dict[str, Any]:
        return {
            "answer_type": "memory_answer",
            "target_object": "phone",
            "location_label": "center of view",
            "confidence": 0.733,
            "timestamp": "2026-06-18T13:22:33",
            "spoken_answer": "I last saw your phone on the center of view around 2026-06-18T13:22:33.",
        }

    def assert_invalid_structured(self, payload: Mapping[str, Any], reason: str) -> None:
        ok, actual_reason = validate_structured_llm_response(payload, self.intent, self.record, ())
        self.assertFalse(ok, msg=payload)
        self.assertEqual(actual_reason, reason)

    def test_accepts_valid_structured_memory_answer(self) -> None:
        ok, reason = validate_structured_llm_response(self.valid_payload(), self.intent, self.record, ())
        self.assertTrue(ok, reason)

    def test_rejects_no_memory_answer_when_record_exists(self) -> None:
        payload = self.valid_payload()
        payload["answer_type"] = "no_memory"
        payload["target_object"] = None
        payload["location_label"] = None
        payload["confidence"] = None
        payload["timestamp"] = None
        payload["spoken_answer"] = "I do not remember seeing that."
        self.assert_invalid_structured(payload, "contradicts_retrieved_memory")

    def test_rejects_wrong_location_label(self) -> None:
        payload = self.valid_payload()
        payload["location_label"] = "right side of view"
        payload["spoken_answer"] = "I last saw your phone on the right side of view around 2026-06-18T13:22:33."
        self.assert_invalid_structured(payload, "location_label_mismatch")

    def test_rejects_frame_path_leak(self) -> None:
        payload = self.valid_payload()
        payload["spoken_answer"] = "I last saw your phone on the center of view. Frame path: data\\object_frames\\frame_493_phone.jpg"
        self.assert_invalid_structured(payload, "implementation_detail_leak")

    def test_rejects_missing_matching_confidence_and_timestamp(self) -> None:
        payload = self.valid_payload()
        payload["confidence"] = 0.111
        payload["timestamp"] = "2026-06-18T13:00:00"
        self.assert_invalid_structured(payload, "missing_matching_confidence_or_timestamp")

    def test_old_sentence_validator_reproduces_previous_bad_output(self) -> None:
        ok, reason = _validate_llm_grounding("I do not remember seeing that.", self.parsed, self.record)
        self.assertFalse(ok)
        self.assertEqual(reason, "contradicts_retrieved_memory")

    def make_agent_with_fake(self, db_path: Path, payload: Mapping[str, Any] | str) -> RecallAgent:
        agent = RecallAgent(memory_db=db_path, use_llm=False, logger=logging.getLogger("test"))
        agent.use_llm = True
        agent.ollama = FakeOllama(payload)  # type: ignore[assignment]
        agent.ollama_status = OllamaStatus(
            ok=True,
            base_url="http://localhost:11434",
            model="fake-model",
            model_available=True,
            latency_ms=1.0,
        )
        return agent

    def test_agent_rejects_structured_llm_that_contradicts_record(self) -> None:
        payload = self.valid_payload()
        payload["answer_type"] = "no_memory"
        payload["target_object"] = None
        payload["location_label"] = None
        payload["confidence"] = None
        payload["timestamp"] = None
        payload["spoken_answer"] = "I do not remember seeing that."
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)
            agent = self.make_agent_with_fake(db_path, payload)
            result = agent.answer("Where did you last see my phone?")

        self.assertTrue(result.llm_attempted)
        self.assertFalse(result.llm_used)
        self.assertEqual(result.llm_fallback_reason, "contradicts_retrieved_memory")
        self.assertIn("invalid_llm_response", result.llm_error or "")
        self.assertEqual(
            result.answer,
            "I last saw your phone on the center of view around 13:22:33 on 2026-06-18. My confidence was 0.73.",
        )

    def test_agent_uses_valid_structured_llm_answer(self) -> None:
        payload = self.valid_payload()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)
            agent = self.make_agent_with_fake(db_path, payload)
            result = agent.answer("Where did you last see my phone?")

        self.assertTrue(result.llm_attempted)
        self.assertTrue(result.llm_used)
        self.assertIsNone(result.llm_error)
        self.assertIsNone(result.llm_fallback_reason)
        self.assertEqual(result.answer, payload["spoken_answer"])

    def test_list_recent_objects_uses_sqlite_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite"
            store = MemoryStore(db_path)
            store.insert(self.record)
            store.insert(MemoryRecord.create("cup", "cup", "left side of view", (1, 2, 3, 4), 0.82, timestamp="2026-06-18T13:25:00"))
            agent = RecallAgent(memory_db=db_path, use_llm=False, logger=logging.getLogger("test"))
            result = agent.answer("What objects did you detect?")

        self.assertEqual(result.intent.intent.value, "list_recent_objects")
        self.assertEqual(result.answer_type, "recent_objects")
        self.assertIn("cup", result.answer)
        self.assertIn("phone", result.answer)
        self.assertNotIn("detected", result.parsed_object or "")


if __name__ == "__main__":
    unittest.main()
