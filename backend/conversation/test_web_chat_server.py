"""Milestone 4.3.2 non-blocking browser chat server tests.

Run:
    python -m unittest backend.conversation.test_web_chat_server -v
"""

from __future__ import annotations

import json
import queue
import unittest
import urllib.request

from backend.conversation.web_chat_server import WebChatRequest, WebChatServer


class WebChatServerTests(unittest.TestCase):
    def test_post_chat_returns_request_id_without_waiting_for_answer(self) -> None:
        output_queue: queue.Queue[object] = queue.Queue()
        server = WebChatServer(output_queue, host="127.0.0.1", port=0, request_timeout_s=2.0)
        server.start()
        try:
            body = json.dumps({"text": "Where did you last see my phone?"}).encode("utf-8")
            request = urllib.request.Request(
                url=f"http://127.0.0.1:{server.port}/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["status"], "queued")
            self.assertIn("request_id", payload)

            item = output_queue.get(timeout=1.0)
            self.assertIsInstance(item, WebChatRequest)
            self.assertEqual(item.text, "Where did you last see my phone?")
            self.assertEqual(item.request_id, payload["request_id"])
        finally:
            server.stop()

    def test_get_result_polling_lifecycle(self) -> None:
        output_queue: queue.Queue[object] = queue.Queue()
        server = WebChatServer(output_queue, host="127.0.0.1", port=0, request_timeout_s=2.0)
        server.start()
        try:
            body = json.dumps({"text": "Where did you last see my phone?"}).encode("utf-8")
            request = urllib.request.Request(
                url=f"http://127.0.0.1:{server.port}/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=3.0) as response:
                queued = json.loads(response.read().decode("utf-8"))
            request_id = queued["request_id"]

            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/chat/result?request_id={request_id}", timeout=3.0) as response:
                pending = json.loads(response.read().decode("utf-8"))
            self.assertEqual(pending["status"], "queued")
            self.assertIsNone(pending["answer"])

            server.set_result(
                request_id,
                {
                    "ok": True,
                    "answer": "I last saw your phone on the center of view.",
                    "answer_type": "memory_answer",
                    "llm_used": True,
                    "llm_attempted": True,
                    "llm_error": None,
                    "llm_fallback_reason": None,
                    "memory_record": {"normalized_label": "phone", "location_label": "center of view"},
                },
            )
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/chat/result?request_id={request_id}", timeout=3.0) as response:
                done = json.loads(response.read().decode("utf-8"))
            self.assertEqual(done["status"], "done")
            self.assertTrue(done["llm_used"])
            self.assertEqual(done["answer_type"], "memory_answer")
        finally:
            server.stop()

    def test_get_health(self) -> None:
        output_queue: queue.Queue[object] = queue.Queue()
        server = WebChatServer(output_queue, host="127.0.0.1", port=0, request_timeout_s=2.0)
        server.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/health", timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["status"], "backend connected")
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
