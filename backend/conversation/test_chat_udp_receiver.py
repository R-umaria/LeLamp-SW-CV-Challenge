"""Milestone 4.3 Godot chat packet parser tests.

Run:
    python -m unittest backend.conversation.test_chat_udp_receiver -v
"""

from __future__ import annotations

import unittest

from backend.conversation.chat_udp_receiver import parse_chat_packet


class ChatUdpReceiverTests(unittest.TestCase):
    def test_accepts_valid_chat_query(self) -> None:
        packet = parse_chat_packet(
            {
                "type": "chat_query",
                "timestamp": "2026-06-18T13:30:00",
                "text": "Where did you last see my phone?",
            },
            ("127.0.0.1", 50000),
        )
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.text, "Where did you last see my phone?")
        self.assertEqual(packet.timestamp, "2026-06-18T13:30:00")

    def test_rejects_wrong_type(self) -> None:
        self.assertIsNone(parse_chat_packet({"type": "not_chat", "text": "hello"}))

    def test_rejects_empty_text(self) -> None:
        self.assertIsNone(parse_chat_packet({"type": "chat_query", "text": "   "}))


if __name__ == "__main__":
    unittest.main()
