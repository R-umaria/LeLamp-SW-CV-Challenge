"""Milestone 4.3 conversation intent tests.

Run:
    python -m unittest backend.conversation.test_intent_parser -v
"""

from __future__ import annotations

import unittest

from backend.conversation.intent_parser import ConversationIntentType, parse_conversation_intent


class IntentParserTests(unittest.TestCase):
    def assert_intent(self, query: str, expected_intent: ConversationIntentType, expected_object: str | None = None) -> None:
        parsed = parse_conversation_intent(query)
        self.assertEqual(parsed.intent, expected_intent, msg=parsed.to_dict())
        self.assertEqual(parsed.normalized_label, expected_object, msg=parsed.to_dict())

    def test_where_last_phone(self) -> None:
        self.assert_intent("Where did you last see my phone?", ConversationIntentType.OBJECT_LAST_SEEN, "phone")

    def test_where_is_cup(self) -> None:
        self.assert_intent("Where is my cup?", ConversationIntentType.OBJECT_LAST_SEEN, "cup")

    def test_have_you_seen_mouse(self) -> None:
        self.assert_intent("Have you seen my mouse?", ConversationIntentType.OBJECT_LAST_SEEN, "mouse")

    def test_did_you_see_if_i_had_stapler(self) -> None:
        self.assert_intent("Did you see if I had a stapler?", ConversationIntentType.OBJECT_LAST_SEEN, "stapler")

    def test_what_objects_detected(self) -> None:
        self.assert_intent("What objects did you detect?", ConversationIntentType.LIST_RECENT_OBJECTS, None)

    def test_what_do_you_remember_seeing(self) -> None:
        self.assert_intent("What do you remember seeing?", ConversationIntentType.LIST_RECENT_OBJECTS, None)

    def test_where_last_bottle(self) -> None:
        self.assert_intent("Where did you last see my bottle?", ConversationIntentType.OBJECT_LAST_SEEN, "bottle")

    def test_was_there_any_pen_in_view(self) -> None:
        self.assert_intent("Was there any pen in the view?", ConversationIntentType.OBJECT_LAST_SEEN, "pen")

    def test_unsupported_chitchat(self) -> None:
        self.assert_intent("Tell me a joke", ConversationIntentType.UNSUPPORTED, None)


if __name__ == "__main__":
    unittest.main()
