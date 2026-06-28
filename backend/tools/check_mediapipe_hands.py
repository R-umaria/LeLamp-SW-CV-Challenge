"""Diagnose whether the current Python environment can load MediaPipe Hands.

Run from the Lumos project root with the same activated virtual environment:

    python -m backend.tools.check_mediapipe_hands
"""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    try:
        import mediapipe as mp  # type: ignore
    except Exception as exc:
        print(f"FAIL: import mediapipe failed: {exc!r}")
        return 1

    print(f"mediapipe module: {getattr(mp, '__file__', '<namespace/no __file__>')}")
    print(f"mediapipe version: {getattr(mp, '__version__', '<unknown>')}")

    ok = False
    checks: list[tuple[str, bool, str]] = []

    try:
        hands = mp.solutions.hands  # type: ignore[attr-defined]
        checks.append(("mp.solutions.hands", hasattr(hands, "Hands"), "ok"))
        ok = ok or hasattr(hands, "Hands")
    except Exception as exc:
        checks.append(("mp.solutions.hands", False, repr(exc)))

    for module_name in ("mediapipe.python.solutions.hands", "mediapipe.solutions.hands"):
        try:
            hands = importlib.import_module(module_name)
            checks.append((module_name, hasattr(hands, "Hands"), "ok"))
            ok = ok or hasattr(hands, "Hands")
        except Exception as exc:
            checks.append((module_name, False, repr(exc)))

    print("\nMediaPipe Hands import checks:")
    for name, success, detail in checks:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {name} ({detail})")

    if not ok:
        print("\nMediaPipe is installed, but MediaPipe Hands could not be loaded.")
        print("Recommended repair in this venv:")
        print("  python -m pip uninstall -y mediapipe")
        print("  python -m pip install --no-cache-dir mediapipe==0.10.14")
        return 2

    print("\nPASS: MediaPipe Hands is available for Lumos gesture detection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
