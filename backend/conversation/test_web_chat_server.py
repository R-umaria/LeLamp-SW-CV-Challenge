"""Milestone 4.3.1 browser chat server tests.

Run:
    python -m unittest backend.conversation.test_web_chat_server -v
"""

from __future__ import annotations

import json
import queue
import threading
import unittest
import urllib.request

from backend.conversation.web_chat_server import WebChatRequest, WebChatServer


class WebChatServerTests(unittest.TestCase):
    def test_post_chat_round_trip(self) -> None:
        output_queue: queue.Queue[object] = queue.Queue()
        server = WebChatServer(output_queue, host="127.0.0.1", port=0, request_timeout_s=2.0)
        server.start()
        try:
            def worker() -> None:
                item = output_queue.get(timeout=2.0)
                self.assertIsInstance(item, WebChatRequest)
                request = item  # type: ignore[assignment]
                self.assertEqual(request.text, "Where did you last see my phone?")
                request.set_result(
                    {
                        "ok": True,
                        "answer": "I last saw your phone on the center of view.",
                        "answer_type": "memory_answer",
                        "llm_used": True,
                        "llm_attempted": True,
                        "llm_error": None,
                        "llm_fallback_reason": None,
                        "memory_record": {"normalized_label": "phone", "location_label": "center of view"},
                    }
                )

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
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
            self.assertTrue(payload["llm_used"])
            self.assertEqual(payload["answer_type"], "memory_answer")
            thread.join(timeout=1.0)
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
