"""Milestone 4.1 parser regression tests.

Run:
    python -m unittest backend.conversation.test_query_parser -v
"""

from __future__ import annotations

import unittest

from backend.conversation.query_parser import parse_object_query


class QueryParserTests(unittest.TestCase):
    def assert_parses_to(self, query: str, expected: str) -> None:
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


if __name__ == "__main__":
    unittest.main()
