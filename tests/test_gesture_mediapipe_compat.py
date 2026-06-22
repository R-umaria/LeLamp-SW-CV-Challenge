from __future__ import annotations

import importlib
import sys
import types


def test_gesture_detector_uses_internal_mediapipe_solutions_fallback(monkeypatch):
    """Windows MediaPipe wheels may not expose mp.solutions on the root module."""

    module_name = "backend.perception.gesture_detector"
    sys.modules.pop(module_name, None)

    fake_mediapipe = types.ModuleType("mediapipe")
    fake_python = types.ModuleType("mediapipe.python")
    fake_solutions = types.ModuleType("mediapipe.python.solutions")
    fake_hands = types.ModuleType("mediapipe.python.solutions.hands")

    class FakeHands:
        pass

    fake_hands.Hands = FakeHands

    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
    monkeypatch.setitem(sys.modules, "mediapipe.python", fake_python)
    monkeypatch.setitem(sys.modules, "mediapipe.python.solutions", fake_solutions)
    monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.hands", fake_hands)

    try:
        gesture_detector = importlib.import_module(module_name)
        assert gesture_detector.mp_hands is fake_hands
        assert gesture_detector._MEDIAPIPE_HANDS_IMPORT_ERROR is None
    finally:
        sys.modules.pop(module_name, None)
