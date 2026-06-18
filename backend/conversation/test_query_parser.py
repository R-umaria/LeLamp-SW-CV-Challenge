"""Milestone 4.3 parser regression tests.

Run:
    python -m unittest backend.conversation.test_query_parser -v
"""

from __future__ import annotations

import unittest

from backend.conversation.query_parser import parse_object_query


class QueryParserTests(unittest.TestCase):
    def assert_parses_to(self, query: str, expected: str | None) -> None:
        parsed = parse_object_query(query)
        self.assertEqual(parsed.normalized_label, expected, msg=parsed.to_dict())

    def test_phone(self) -> None:
        self.assert_parses_to("Where is my phone?", "phone")

    def test_cup(self) -> None:
        self.assert_parses_to("Where did you last see my cup?", "cup")

    def test_mouse(self) -> None:
        self.assert_parses_to("Have you seen my mouse?", "mouse")

    def test_spectacles_near_cup_prefers_possessive_target(self) -> None:
        self.assert_parses_to("By any chance, did you see my spectacles that I left near the cup?", "glasses")

    def test_stapler_unknown_object_still_parses_as_target(self) -> None:
        self.assert_parses_to("Where is the stapler?", "stapler")

    def test_did_you_see_if_i_had_stapler_does_not_parse_as_if(self) -> None:
        parsed = parse_object_query("Did you see if I had a stapler?")
        self.assertEqual(parsed.normalized_label, "stapler", msg=parsed.to_dict())
        self.assertNotEqual(parsed.normalized_label, "if")

    def test_list_query_does_not_parse_detected_as_object(self) -> None:
        parsed = parse_object_query("What objects did you detect?")
        self.assertIsNone(parsed.normalized_label, msg=parsed.to_dict())
        self.assertEqual(parsed.strategy, "list_recent_objects_query")

    def test_where_last_bottle(self) -> None:
        self.assert_parses_to("Where did you last see my bottle?", "bottle")

    def test_was_there_any_pen_in_view(self) -> None:
        parsed = parse_object_query("Was there any pen in the view?")
        self.assertEqual(parsed.normalized_label, "pen", msg=parsed.to_dict())
        self.assertEqual(parsed.strategy, "existence_object_pattern")

    def test_was_there_any_tv_followup_maps_to_monitor(self) -> None:
        parsed = parse_object_query("was there any tv? If so, where was it?")
        self.assertEqual(parsed.normalized_label, "monitor", msg=parsed.to_dict())
        self.assertEqual(parsed.target_text, "tv", msg=parsed.to_dict())

    def test_was_there_any_tv_maps_to_monitor(self) -> None:
        parsed = parse_object_query("Was there any TV?")
        self.assertEqual(parsed.normalized_label, "monitor", msg=parsed.to_dict())
        self.assertEqual(parsed.target_text, "tv", msg=parsed.to_dict())


if __name__ == "__main__":
    unittest.main()
