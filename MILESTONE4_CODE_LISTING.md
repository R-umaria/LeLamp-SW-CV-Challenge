# Milestone 4 Grounded Recall - Complete Code Listing

---

## .gitignore

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582; used by e.g. python-pdm
__pypackages__/

# env stuff
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# vscode
.vscode/

# JetBrains IDEs
.idea/
*.iml

# LeLamp runtime artifacts
logs/
test_logs/
data/*.sqlite
data/*.sqlite-*
data/object_frames/
data/captures/
captures/
*.pt
*.onnx

# Godot generated/editor cache
frontend_godot/.godot/
frontend_godot/.import/
frontend_godot/imported/

# OS noise
.DS_Store
Thumbs.db

```

---

## backend/conversation/__init__.py

```python
"""Conversation and grounded recall modules for Milestone 4."""

from backend.conversation.query_parser import ParsedObjectQuery, parse_object_query

__all__ = ["ParsedObjectQuery", "parse_object_query"]

```

---

## backend/conversation/query_parser.py

```python
"""Deterministic object-query parser for grounded memory recall.

The MVP deliberately avoids using an LLM for parsing. It extracts one target
object label from short text questions such as:
    - Where did you last see my phone?
    - Where is the cup?
    - Have you seen my mouse?

The output is a normalized label used for exact SQLite lookup. If the parser is
uncertain, it returns ``normalized_label=None`` rather than guessing a location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Keep aliases duplicated here instead of importing the detector module so the
# recall CLI can run without OpenCV/Ultralytics installed.
OBJECT_ALIASES: dict[str, str] = {
    "phone": "phone",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "laptop": "laptop",
    "computer": "laptop",
    "notebook": "laptop",
    "monitor": "monitor",
    "screen": "monitor",
    "tv": "monitor",
    "television": "monitor",
    "cup": "cup",
    "mug": "cup",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "book": "book",
    "bottle": "bottle",
    "remote": "remote",
    "remote control": "remote",
    "clock": "clock",
    "chair": "chair",
    "backpack": "backpack",
    "bag": "bag",
    "handbag": "bag",
    "vase": "vase",
    "scissors": "scissors",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "around",
    "at",
    "did",
    "do",
    "for",
    "have",
    "is",
    "it",
    "last",
    "locate",
    "location",
    "me",
    "my",
    "near",
    "of",
    "on",
    "please",
    "remember",
    "see",
    "seen",
    "show",
    "the",
    "there",
    "was",
    "were",
    "where",
    "you",
}

OBJECT_PHRASE_PATTERNS = [
    re.compile(r"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
    re.compile(r"\bhave\s+you\s+seen\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
    re.compile(r"\bdid\s+you\s+see\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
    re.compile(r"\b(?:find|locate)\s+(?:my\s+|the\s+|a\s+|an\s+)?(?P<object>[a-z0-9][a-z0-9\s-]*?)\s*\??$"),
]


@dataclass(frozen=True)
class ParsedObjectQuery:
    original_query: str
    target_text: str | None
    normalized_label: str | None
    confidence: float
    strategy: str

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "target_text": self.target_text,
            "normalized_label": self.normalized_label,
            "confidence": round(float(self.confidence), 3),
            "strategy": self.strategy,
        }


def clean_query_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"[^a-z0-9\s'-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def normalize_object_label(label: str) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if cleaned in OBJECT_ALIASES:
        return OBJECT_ALIASES[cleaned]
    # Conservative singular fallback for simple plurals: cups -> cup.
    if cleaned.endswith("s") and cleaned[:-1] in OBJECT_ALIASES:
        return OBJECT_ALIASES[cleaned[:-1]]
    return cleaned


def parse_object_query(query: str, aliases: dict[str, str] | None = None) -> ParsedObjectQuery:
    alias_map = aliases or OBJECT_ALIASES
    cleaned = clean_query_text(query)
    if not cleaned:
        return ParsedObjectQuery(query, None, None, 0.0, "empty_query")

    # Prefer exact known aliases, sorted by length so "cell phone" wins before
    # "phone" and "remote control" wins before "remote".
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if _contains_phrase(cleaned, alias):
            return ParsedObjectQuery(
                original_query=query,
                target_text=alias,
                normalized_label=alias_map[alias],
                confidence=0.95,
                strategy="known_alias",
            )

    for pattern in OBJECT_PHRASE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = normalize_object_label(target)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.65,
                strategy="question_pattern",
            )

    fallback = _last_content_token(cleaned.split())
    if fallback:
        normalized = normalize_object_label(fallback)
        return ParsedObjectQuery(
            original_query=query,
            target_text=fallback,
            normalized_label=normalized or None,
            confidence=0.35,
            strategy="fallback_last_token",
        )

    return ParsedObjectQuery(query, None, None, 0.0, "no_object_found")


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _sanitize_object_phrase(phrase: str) -> str:
    cleaned = clean_query_text(phrase)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    tokens = [token for token in cleaned.split() if token not in STOPWORDS]
    return " ".join(tokens).strip()


def _last_content_token(tokens: Iterable[str]) -> str | None:
    for token in reversed(list(tokens)):
        token = token.strip(" -'")
        if token and token not in STOPWORDS:
            return token
    return None

```

---

## backend/conversation/llm_client.py

```python
"""Optional local LLM client for grounded recall phrasing.

This module uses only the Python standard library so the recall CLI still works
without extra client packages. It targets Ollama's ``/api/generate`` endpoint and
fails closed: if Ollama is unavailable, callers receive an error and can use the
deterministic template response.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    used_llm: bool
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "used_llm": self.used_llm,
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
        }


class OllamaClient:
    """Small Ollama wrapper with timeout and deterministic settings."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_s: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)

    def generate(self, prompt: str) -> LLMResponse:
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 120,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            text = str(parsed.get("response", "")).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not text:
                return LLMResponse("", False, elapsed_ms, "ollama_returned_empty_response")
            return LLMResponse(text, True, elapsed_ms, None)
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse("", False, elapsed_ms, f"ollama_connection_error: {exc}")
        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse("", False, elapsed_ms, f"ollama_timeout: {exc}")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse("", False, elapsed_ms, f"ollama_error: {exc}")

```

---

## backend/conversation/recall_agent.py

```python
"""Grounded memory-recall agent for Milestone 4.

CLI examples:
    python -m backend.conversation.recall_agent "Where did you last see my phone?"
    python -m backend.conversation.recall_agent "Where is the cup?" --json
    python -m backend.conversation.recall_agent "Have you seen my mouse?" --use-llm

The recall path is intentionally bounded:
    text query -> deterministic object parser -> exact SQLite lookup ->
    deterministic or strictly grounded LLM phrasing.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backend.conversation.llm_client import LLMResponse, OllamaClient
from backend.conversation.query_parser import ParsedObjectQuery, parse_object_query
from backend.memory.memory_store import MemoryRecord, MemoryStore


STRICT_GROUNDED_RECALL_PROMPT = """You are the voice of a LeLamp-inspired robotic lamp.

You must answer the user's object-location question using only the provided memory record.
Do not use outside knowledge.
Do not infer, guess, or invent a location.
Do not say where the object is now. Only say where it was last seen.
Do not mention objects that are not in the memory record.
If the memory record is missing or insufficient, say: "I do not remember seeing that."
Keep the answer to one short conversational sentence.

User question:
{user_query}

Parsed target object:
{parsed_object}

Memory record JSON:
{memory_record_json}

Answer:"""


@dataclass(frozen=True)
class RecallResult:
    user_query: str
    parsed_object: str | None
    parsed: ParsedObjectQuery
    memory_record: MemoryRecord | None
    answer: str
    memory_retrieval_ms: float
    llm_response_ms: float | None
    used_llm: bool
    llm_error: str | None
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "parsed_object": self.parsed_object,
            "parsed": self.parsed.to_dict(),
            "retrieved_memory_id": self.memory_record.id if self.memory_record else None,
            "memory_record": self.memory_record.to_dict() if self.memory_record else None,
            "memory_retrieval_ms": round(float(self.memory_retrieval_ms), 3),
            "llm_response_ms": None if self.llm_response_ms is None else round(float(self.llm_response_ms), 3),
            "used_llm": self.used_llm,
            "llm_error": self.llm_error,
            "answer": self.answer,
        }


class RecallAgent:
    """Retrieves one grounded memory and phrases a response."""

    def __init__(
        self,
        memory_db: str | Path = "data/scene_memory.sqlite",
        use_llm: bool = False,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2",
        log_paths: str | Path | Iterable[str | Path] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.memory_db = Path(memory_db)
        self.store = MemoryStore(self.memory_db)
        self.use_llm = bool(use_llm)
        self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model) if self.use_llm else None
        self.logger = logger or logging.getLogger("lelamp")
        self.log_paths = _coerce_paths(log_paths)

    def answer(self, user_query: str) -> RecallResult:
        parsed = parse_object_query(user_query)
        retrieval_start = time.perf_counter()
        memory_record = None
        if parsed.normalized_label:
            memory_record = self.store.find_latest_by_normalized_label(parsed.normalized_label)
        memory_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fallback_answer = deterministic_recall_answer(parsed, memory_record)
        llm_response: LLMResponse | None = None
        final_answer = fallback_answer

        # Safer MVP rule: only ask the LLM to phrase an answer when a concrete
        # memory exists. Missing-memory answers remain deterministic.
        if self.use_llm and self.ollama is not None and memory_record is not None:
            prompt = build_grounded_prompt(user_query, parsed, memory_record)
            llm_response = self.ollama.generate(prompt)
            if llm_response.used_llm and llm_response.text:
                final_answer = _clean_llm_text(llm_response.text)
            else:
                final_answer = fallback_answer

        result = RecallResult(
            user_query=user_query,
            parsed_object=parsed.normalized_label,
            parsed=parsed,
            memory_record=memory_record,
            answer=final_answer,
            memory_retrieval_ms=memory_retrieval_ms,
            llm_response_ms=None if llm_response is None else llm_response.latency_ms,
            used_llm=False if llm_response is None else llm_response.used_llm,
            llm_error=None if llm_response is None else llm_response.error,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )
        self._log_result(result)
        return result

    def _log_result(self, result: RecallResult) -> None:
        payload = result.to_dict()
        self.logger.info(
            "Recall query=%r parsed_object=%s memory_id=%s retrieval_ms=%.3f llm_ms=%s used_llm=%s answer=%r",
            result.user_query,
            result.parsed_object,
            result.memory_record.id if result.memory_record else None,
            result.memory_retrieval_ms,
            "" if result.llm_response_ms is None else f"{result.llm_response_ms:.3f}",
            result.used_llm,
            result.answer,
        )
        for path in self.log_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_grounded_prompt(user_query: str, parsed: ParsedObjectQuery, memory_record: MemoryRecord) -> str:
    return STRICT_GROUNDED_RECALL_PROMPT.format(
        user_query=user_query,
        parsed_object=parsed.normalized_label or "",
        memory_record_json=json.dumps(memory_record.to_dict(), ensure_ascii=False, indent=2),
    )


def deterministic_recall_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    if not parsed.normalized_label:
        return "I am not sure which object you mean. Ask me about a specific object, like the cup or phone."

    if memory_record is None:
        return f"I do not remember seeing the {parsed.normalized_label}."

    label = memory_record.normalized_label or memory_record.object_label
    answer = (
        f"I last saw the {label} at the {memory_record.location_label} "
        f"at {memory_record.timestamp} with {memory_record.confidence:.2f} confidence."
    )
    if memory_record.frame_path:
        answer += f" I saved a reference frame at {memory_record.frame_path}."
    return answer


def _clean_llm_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    # Remove common leading labels if a local model includes them.
    for prefix in ("Answer:", "Lamp:", "LeLamp:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _coerce_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask LeLamp grounded questions about stored object memory")
    parser.add_argument("query", nargs="?", help="Question, e.g. 'Where did you last see my phone?'")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase a retrieved memory answer")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model, e.g. llama3.2 or qwen2.5")
    parser.add_argument("--json", action="store_true", help="Print machine-readable recall result JSON")
    parser.add_argument("--log-path", type=str, default="logs/recall.jsonl", help="JSONL recall log path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.query:
        print("Missing query. Example: python -m backend.conversation.recall_agent \"Where is the cup?\"")
        return 2

    console_level = logging.WARNING if args.json else logging.INFO
    logging.basicConfig(level=console_level, format="%(asctime)s | %(levelname)s | %(message)s")
    agent = RecallAgent(
        memory_db=args.memory_db,
        use_llm=args.use_llm,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        log_paths=args.log_path,
    )
    result = agent.answer(args.query)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(
            f"parsed_object={result.parsed_object or 'none'} "
            f"memory_id={result.memory_record.id if result.memory_record else 'none'} "
            f"memory_retrieval_ms={result.memory_retrieval_ms:.3f} "
            f"llm_response_ms={'' if result.llm_response_ms is None else f'{result.llm_response_ms:.3f}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## backend/main.py

```python
"""LeLamp backend vertical slice with Milestone 4 grounded recall.

Run from the project root with:
    python -m backend.main --show-window

Milestone 4 preserves the stable engagement detector, object memory, isolated
logging, command JSON shape, and Godot UDP bridge. It adds optional interactive
text recall as a bounded side channel: the user can type a question, Python
retrieves SQLite memory, and Godot receives a controlled ``recalling`` command.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV is required. Install with: pip install opencv-python") from exc

from backend.behavior.behavior_policy import behavior_for_state
from backend.behavior.command_protocol import build_behavior_command
from backend.behavior.godot_udp_sender import GodotUdpSender
from backend.behavior.state_machine import InteractionStateMachine, LampState
from backend.conversation.recall_agent import RecallAgent
from backend.evaluation.latency_logger import LatencyLogger
from backend.memory.scene_memory import SceneMemory
from backend.perception.camera import OpenCVCamera
from backend.perception.engagement_detector import FaceEngagementDetector, draw_engagement_overlay
from backend.perception.object_detector import YoloObjectDetector, draw_object_overlay
from backend.perception.temporal_smoother import EngagementSmoother
from backend.utils.config import (
    CameraConfig,
    EngagementConfig,
    GodotUdpConfig,
    MemoryConfig,
    ObjectDetectionConfig,
    RuntimeConfig,
    SmoothingConfig,
    StateMachineConfig,
)
from backend.utils.logging_utils import setup_logging
from backend.utils.run_paths import create_run_paths, write_latest_pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeLamp Milestone 4 backend: engagement + object memory + grounded recall")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seek-after", type=float, default=5.0)
    parser.add_argument("--emit-interval", type=float, default=1.0)
    parser.add_argument("--godot-udp", action="store_true", help="Enable best-effort UDP command streaming to the Godot frontend.")
    parser.add_argument("--godot-host", type=str, default="127.0.0.1", help="Godot UDP host. Use 127.0.0.1 for local demo.")
    parser.add_argument("--godot-port", type=int, default=4242, help="Godot UDP listen port.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Root log directory. Each run writes under <log-dir>/runs/<run_id>/")
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id. Defaults to timestamp YYYY-MM-DD_HH-MM-SS.")
    parser.add_argument("--no-latest", action="store_true", help="Do not mirror this run into logs/latest.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Esc/Ctrl-C")

    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--min-state-dwell", type=float, default=0.75)
    parser.add_argument("--exit-disengaged-frames", type=int, default=5)
    parser.add_argument("--exit-absent-frames", type=int, default=8)
    parser.add_argument("--engaged-recovery-frames", type=int, default=2)
    parser.add_argument("--clear-engaged-confidence", type=float, default=0.78)

    parser.add_argument("--center-tolerance-x", type=float, default=0.24)
    parser.add_argument("--center-tolerance-y", type=float, default=0.30)
    parser.add_argument("--min-face-area-ratio", type=float, default=0.020)
    parser.add_argument("--min-candidate-area-ratio", type=float, default=0.010)
    parser.add_argument("--cascade-min-neighbors", type=int, default=6)

    parser.add_argument("--enable-objects", action="store_true", help="Enable optional YOLO object detection and scene-memory writes.")
    parser.add_argument("--object-model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model path/name, e.g. yolov8n.pt")
    parser.add_argument("--object-interval", type=float, default=2.0, help="Seconds between object-detection passes.")
    parser.add_argument("--memory-db", type=str, default="data/scene_memory.sqlite", help="SQLite scene-memory database path.")
    parser.add_argument("--save-object-frames", action="store_true", help="Save annotated evidence frames when a memory record is written.")
    parser.add_argument("--object-confidence", type=float, default=0.35, help="YOLO confidence threshold for object detection.")
    parser.add_argument("--memory-dedupe-window", type=float, default=8.0, help="Seconds to suppress repeated same-object/same-location memory writes.")
    parser.add_argument("--object-frame-dir", type=str, default="data/object_frames", help="Directory for saved object evidence frames.")

    parser.add_argument("--interactive-recall", action="store_true", help="Allow text recall questions while the backend is running.")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama/local LLM only to phrase retrieved memory answers.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL for --use-llm.")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model for recall phrasing, e.g. llama3.2 or qwen2.5.")

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--show-window", action="store_true", default=True)
    window_group.add_argument("--no-window", action="store_false", dest="show_window")
    return parser.parse_args()


def append_jsonl(paths: Path | list[Path] | tuple[Path, ...], payload: dict) -> None:
    if isinstance(paths, Path):
        output_paths = [paths]
    else:
        output_paths = list(paths)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")




def start_recall_input_thread(
    recall_queue: "queue.Queue[str]",
    stop_event: threading.Event,
    logger,
) -> threading.Thread:
    """Start a tiny stdin reader so camera/perception loop stays non-blocking."""

    def _worker() -> None:
        print("Interactive recall enabled. Type a question such as: Where is the cup?", flush=True)
        print("Type :q, quit, or exit to stop accepting recall questions.", flush=True)
        while not stop_event.is_set():
            try:
                line = input("recall> ")
            except EOFError:
                logger.info("Interactive recall input closed")
                break
            except Exception as exc:  # pragma: no cover - terminal/runtime guard
                logger.warning("Interactive recall input stopped: %s", exc)
                break

            query = line.strip()
            if not query:
                continue
            if query.lower() in {":q", "quit", "exit"}:
                logger.info("Interactive recall input stop requested")
                stop_event.set()
                break
            recall_queue.put(query)

    thread = threading.Thread(target=_worker, name="interactive-recall-input", daemon=True)
    thread.start()
    return thread


def build_configs(
    args: argparse.Namespace,
) -> tuple[
    CameraConfig,
    EngagementConfig,
    SmoothingConfig,
    StateMachineConfig,
    RuntimeConfig,
    GodotUdpConfig,
    ObjectDetectionConfig,
    MemoryConfig,
]:
    camera_config = CameraConfig(index=args.camera_index, width=args.width, height=args.height)
    engagement_config = EngagementConfig(
        center_tolerance_x=args.center_tolerance_x,
        center_tolerance_y=args.center_tolerance_y,
        min_face_area_ratio=args.min_face_area_ratio,
        min_candidate_area_ratio=args.min_candidate_area_ratio,
        cascade_min_neighbors=args.cascade_min_neighbors,
    )
    smoothing_config = SmoothingConfig(
        window_size=args.smoothing_window,
        clear_engaged_confidence=args.clear_engaged_confidence,
    )
    state_config = StateMachineConfig(
        seek_attention_after_s=args.seek_after,
        min_state_dwell_s=args.min_state_dwell,
        exit_engaged_disengaged_frames=args.exit_disengaged_frames,
        exit_engaged_absent_frames=args.exit_absent_frames,
        engaged_recovery_frames=args.engaged_recovery_frames,
        clear_engaged_confidence=args.clear_engaged_confidence,
    )
    runtime_config = RuntimeConfig(
        command_emit_interval_s=args.emit_interval,
        log_dir=args.log_dir,
        show_window=args.show_window,
    )
    godot_udp_config = GodotUdpConfig(
        enabled=args.godot_udp,
        host=args.godot_host,
        port=args.godot_port,
    )
    object_config = ObjectDetectionConfig(
        enabled=args.enable_objects,
        model_path=args.object_model,
        interval_s=max(0.1, args.object_interval),
        confidence=args.object_confidence,
    )
    memory_config = MemoryConfig(
        db_path=args.memory_db,
        dedupe_window_s=max(0.0, args.memory_dedupe_window),
        save_object_frames=args.save_object_frames,
        frame_dir=args.object_frame_dir,
    )
    return (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        memory_config,
    )


def main() -> int:
    args = parse_args()
    (
        camera_config,
        engagement_config,
        smoothing_config,
        state_config,
        runtime_config,
        godot_udp_config,
        object_config,
        memory_config,
    ) = build_configs(args)

    run_paths = create_run_paths(
        log_root=runtime_config.log_dir,
        run_id=args.run_id,
        mirror_latest=not args.no_latest,
    )
    write_latest_pointer(run_paths.log_root, run_paths.run_dir)

    latest_latency_path = None if args.no_latest else run_paths.latest_latency_path
    latest_commands_path = None if args.no_latest else run_paths.latest_commands_path
    latest_runtime_log_paths = [] if args.no_latest else [run_paths.latest_runtime_log_path]

    logger = setup_logging(run_paths.run_dir, extra_runtime_log_paths=latest_runtime_log_paths)
    latency_paths = [run_paths.latency_path] + ([latest_latency_path] if latest_latency_path else [])
    latency_logger = LatencyLogger(latency_paths)
    command_paths = [run_paths.commands_path] + ([latest_commands_path] if latest_commands_path else [])
    commands_path = run_paths.commands_path
    godot_sender = GodotUdpSender(godot_udp_config, logger=logger)

    camera = OpenCVCamera(camera_config.index, camera_config.width, camera_config.height)
    detector = FaceEngagementDetector(engagement_config)
    smoother = EngagementSmoother(smoothing_config)
    fsm = InteractionStateMachine(state_config)
    object_detector = YoloObjectDetector(object_config, logger=logger)
    scene_memory = SceneMemory(memory_config, logger=logger)

    recall_agent = None
    recall_queue: queue.Queue[str] | None = None
    recall_stop_event = threading.Event()
    if args.interactive_recall:
        recall_log_paths = [run_paths.run_dir / "recall.jsonl"]
        if not args.no_latest:
            recall_log_paths.append(run_paths.latest_dir / "recall.jsonl")
        recall_agent = RecallAgent(
            memory_db=args.memory_db,
            use_llm=args.use_llm,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            log_paths=recall_log_paths,
            logger=logger,
        )
        recall_queue = queue.Queue()
        start_recall_input_thread(recall_queue, recall_stop_event, logger)

    logger.info("Starting Milestone 4 backend with isolated run logging")
    logger.info("Run id=%s", run_paths.run_id)
    logger.info("Run directory=%s", run_paths.run_dir)
    if not args.no_latest:
        logger.info("Latest mirror directory=%s", run_paths.latest_dir)
    logger.info("Camera index=%s size=%sx%s", camera_config.index, camera_config.width, camera_config.height)
    logger.info(
        "Stability config smoothing_window=%s min_dwell=%.2fs exit_disengaged_frames=%s exit_absent_frames=%s min_face_area=%.3f min_candidate_area=%.3f",
        smoothing_config.window_size,
        state_config.min_state_dwell_s,
        state_config.exit_engaged_disengaged_frames,
        state_config.exit_engaged_absent_frames,
        engagement_config.min_face_area_ratio,
        engagement_config.min_candidate_area_ratio,
    )
    logger.info(
        "Object config enabled=%s active=%s model=%s interval=%.2fs confidence=%.2f",
        object_config.enabled,
        object_detector.enabled,
        object_config.model_path,
        object_config.interval_s,
        object_config.confidence,
    )
    logger.info(
        "Memory config db=%s save_frames=%s dedupe_window=%.1fs",
        memory_config.db_path,
        memory_config.save_object_frames,
        memory_config.dedupe_window_s,
    )
    logger.info("Commands will be saved to %s", commands_path)
    logger.info(
        "Recall config interactive=%s use_llm=%s ollama_url=%s ollama_model=%s",
        args.interactive_recall,
        args.use_llm,
        args.ollama_url,
        args.ollama_model,
    )
    if godot_udp_config.enabled:
        logger.info("Commands will also be streamed to Godot via udp://%s:%s", godot_udp_config.host, godot_udp_config.port)

    frame_count = 0
    last_emit_at = 0.0
    last_object_detection_at = 0.0
    fps_ema = 0.0
    last_godot_udp_send_ms = None
    last_detected_objects: list[dict] = []
    display_detections = []

    try:
        camera.open()
        while True:
            loop_start = time.perf_counter()

            camera_frame = camera.read()
            frame = camera_frame.frame

            t0 = time.perf_counter()
            raw_engagement = detector.detect(frame)
            engagement_ms = (time.perf_counter() - t0) * 1000.0

            object_detection_ms = ""
            memory_write_ms = ""
            memory_retrieval_ms = ""
            llm_response_ms = ""
            memory_write_count = 0
            memory_duplicate_skip_count = 0
            now = time.monotonic()
            should_detect_objects = object_detector.enabled and (
                last_object_detection_at == 0.0 or (now - last_object_detection_at >= object_config.interval_s)
            )
            if should_detect_objects:
                t0 = time.perf_counter()
                display_detections = object_detector.detect(frame)
                object_detection_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                last_object_detection_at = now

                if display_detections:
                    logger.info(
                        "Detected objects count=%s objects=%s latency_ms=%.3f",
                        len(display_detections),
                        [d.to_log_dict() for d in display_detections],
                        object_detection_ms,
                    )
                else:
                    logger.info("Detected objects count=0 latency_ms=%.3f", object_detection_ms)

                memory_result = scene_memory.observe(
                    display_detections,
                    frame=frame,
                    frame_index=frame_count,
                    source="webcam",
                )
                last_detected_objects = memory_result.command_objects
                memory_write_ms = round(memory_result.memory_write_ms, 3)
                memory_write_count = len(memory_result.written_records)
                memory_duplicate_skip_count = memory_result.skipped_duplicates
                logger.info(
                    "Object memory update detections=%s writes=%s duplicates=%s memory_write_ms=%.3f",
                    len(display_detections),
                    memory_write_count,
                    memory_duplicate_skip_count,
                    memory_result.memory_write_ms,
                )

            t0 = time.perf_counter()
            smoothed_engagement = smoother.update(raw_engagement)
            smoothing_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            transition = fsm.update(smoothed_engagement)
            state_machine_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = behavior_for_state(transition.current_state)
            command = build_behavior_command(
                state=transition.current_state,
                engagement=smoothed_engagement,
                behavior=behavior,
                last_detected_objects=last_detected_objects,
            )
            command_ms = (time.perf_counter() - t0) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            instantaneous_fps = 1000.0 / max(total_ms, 1e-6)
            fps_ema = instantaneous_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * instantaneous_fps)

            now = time.monotonic()
            should_emit = transition.changed or (now - last_emit_at >= runtime_config.command_emit_interval_s)

            if transition.changed:
                logger.info(
                    "State transition %s -> %s | reason=%s | engagement=%s conf=%.2f",
                    transition.previous_state.value,
                    transition.current_state.value,
                    transition.reason,
                    smoothed_engagement.status,
                    smoothed_engagement.confidence,
                )

            if should_emit:
                print(json.dumps(command, ensure_ascii=False), flush=True)
                append_jsonl(command_paths, command)
                last_godot_udp_send_ms = godot_sender.send(command)
                last_emit_at = now

            if recall_agent is not None and recall_queue is not None:
                while True:
                    try:
                        recall_query = recall_queue.get_nowait()
                    except queue.Empty:
                        break

                    recall_result = recall_agent.answer(recall_query)
                    memory_retrieval_ms = round(recall_result.memory_retrieval_ms, 3)
                    llm_response_ms = (
                        "" if recall_result.llm_response_ms is None else round(recall_result.llm_response_ms, 3)
                    )
                    recall_command = build_behavior_command(
                        state=LampState.RECALLING,
                        engagement=smoothed_engagement,
                        behavior={
                            "motion": "thinking",
                            "light": "focus_glow",
                            "sound": None,
                            "speech_text": recall_result.answer,
                        },
                        last_detected_objects=last_detected_objects,
                    )
                    print(json.dumps(recall_command, ensure_ascii=False), flush=True)
                    append_jsonl(command_paths, recall_command)
                    last_godot_udp_send_ms = godot_sender.send(recall_command)
                    last_emit_at = time.monotonic()
                    logger.info(
                        "Sent recall command to Godot state=recalling parsed_object=%s memory_id=%s",
                        recall_result.parsed_object,
                        recall_result.memory_record.id if recall_result.memory_record else None,
                    )

            latency_logger.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "frame_index": frame_count,
                    "capture_ms": round(camera_frame.capture_latency_ms, 3),
                    "engagement_detection_ms": round(engagement_ms, 3),
                    "object_detection_ms": object_detection_ms,
                    "memory_write_ms": memory_write_ms,
                    "memory_retrieval_ms": memory_retrieval_ms,
                    "llm_response_ms": llm_response_ms,
                    "smoothing_ms": round(smoothing_ms, 3),
                    "state_machine_ms": round(state_machine_ms, 3),
                    "command_build_ms": round(command_ms, 3),
                    "godot_udp_send_ms": "" if last_godot_udp_send_ms is None else round(last_godot_udp_send_ms, 3),
                    "total_loop_ms": round(total_ms, 3),
                    "fps": round(fps_ema, 2),
                    "state": transition.current_state.value,
                    "state_elapsed_s": round(transition.state_elapsed_s, 3),
                    "raw_engagement_status": raw_engagement.status,
                    "raw_engagement_confidence": round(raw_engagement.confidence, 3),
                    "raw_engagement_reason": raw_engagement.reason,
                    "smoothed_engagement_status": smoothed_engagement.status,
                    "smoothed_engagement_confidence": round(smoothed_engagement.confidence, 3),
                    "smoothed_engagement_reason": smoothed_engagement.reason,
                    "face_bbox": smoothed_engagement.face_bbox or "",
                    "face_area_ratio": round(smoothed_engagement.face_area_ratio, 4),
                    "raw_face_count": raw_engagement.raw_face_count,
                    "candidate_count": raw_engagement.candidate_count,
                    "selected_face_score": round(smoothed_engagement.selected_face_score, 3),
                    "object_count": len(display_detections) if object_detector.enabled else 0,
                    "memory_write_count": memory_write_count,
                    "memory_duplicate_skip_count": memory_duplicate_skip_count,
                    "consecutive_engaged": transition.consecutive_engaged,
                    "consecutive_disengaged": transition.consecutive_disengaged,
                    "consecutive_absent": transition.consecutive_absent,
                }
            )

            if runtime_config.show_window:
                draw_engagement_overlay(
                    frame,
                    raw_result=raw_engagement,
                    smoothed_result=smoothed_engagement,
                    state=transition.current_state.value,
                    state_elapsed_s=transition.state_elapsed_s,
                    fps=fps_ema,
                    config=engagement_config,
                )
                if object_detector.enabled:
                    draw_object_overlay(frame, display_detections)
                cv2.imshow("LeLamp Milestone 4 - Engagement/Object Memory/Recall", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("Quit requested from preview window")
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                logger.info("Reached --max-frames=%s", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.exception("Backend stopped due to error: %s", exc)
        return 1
    finally:
        recall_stop_event.set()
        godot_sender.close()
        camera.release()
        if runtime_config.show_window:
            cv2.destroyAllWindows()
        logger.info("Stopped Milestone 4 backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

---

## backend/memory/memory_store.py

```python
"""SQLite-backed object memory store for Milestone 3.

The schema is intentionally small and explainable. Records represent what the
lamp saw, where it approximately appeared in the camera frame, when it was seen,
and optional evidence through a saved frame path.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    object_label: str
    normalized_label: str
    location_label: str
    bbox: BBox
    confidence: float
    timestamp: str
    source: str = "webcam"
    frame_path: Optional[str] = None

    @classmethod
    def create(
        cls,
        object_label: str,
        normalized_label: str,
        location_label: str,
        bbox: BBox,
        confidence: float,
        source: str = "webcam",
        frame_path: str | None = None,
        timestamp: str | None = None,
    ) -> "MemoryRecord":
        return cls(
            id=str(uuid.uuid4()),
            object_label=object_label,
            normalized_label=normalized_label,
            location_label=location_label,
            bbox=tuple(int(v) for v in bbox),
            confidence=float(confidence),
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            source=source,
            frame_path=frame_path,
        )

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "normalized_label": self.normalized_label,
            "location_label": self.location_label,
            "bbox": json.dumps(list(self.bbox)),
            "confidence": float(self.confidence),
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object_label": self.object_label,
            "normalized_label": self.normalized_label,
            "location_label": self.location_label,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 3),
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
        }


class MemoryStore:
    """SQLite store for object-memory records."""

    def __init__(self, db_path: str | Path = "data/scene_memory.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS object_memory (
                    id TEXT PRIMARY KEY,
                    object_label TEXT NOT NULL,
                    normalized_label TEXT NOT NULL,
                    location_label TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    frame_path TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_label_time ON object_memory(normalized_label, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_object_memory_location_time ON object_memory(location_label, timestamp)"
            )

    def insert(self, record: MemoryRecord) -> None:
        row = record.to_row()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO object_memory (
                    id,
                    object_label,
                    normalized_label,
                    location_label,
                    bbox,
                    confidence,
                    timestamp,
                    source,
                    frame_path
                ) VALUES (
                    :id,
                    :object_label,
                    :normalized_label,
                    :location_label,
                    :bbox,
                    :confidence,
                    :timestamp,
                    :source,
                    :frame_path
                )
                """,
                row,
            )

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find(self, query: str, limit: int = 20) -> list[MemoryRecord]:
        normalized_query = query.strip().lower()
        like = f"%{normalized_query}%"
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) LIKE ? OR lower(object_label) LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (like, like, int(limit)),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_latest_by_normalized_label(self, normalized_label: str) -> MemoryRecord | None:
        """Return the latest exact normalized-label match for grounded recall."""
        normalized = normalized_label.strip().lower()
        if not normalized:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE lower(normalized_label) = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_recent_duplicate(
        self,
        normalized_label: str,
        location_label: str,
        within_seconds: float,
        now: datetime | None = None,
    ) -> MemoryRecord | None:
        """Return a same-label/same-location record inside the dedupe window."""
        cutoff = (now or datetime.now()) - timedelta(seconds=float(within_seconds))
        cutoff_text = cutoff.isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM object_memory
                WHERE normalized_label = ?
                  AND location_label = ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (normalized_label, location_label, cutoff_text),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM object_memory")
            return int(cursor.rowcount if cursor.rowcount is not None else 0)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM object_memory").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        raw_bbox = row["bbox"]
        bbox_values = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
        bbox: BBox = tuple(int(v) for v in bbox_values)  # type: ignore[assignment]
        return MemoryRecord(
            id=str(row["id"]),
            object_label=str(row["object_label"]),
            normalized_label=str(row["normalized_label"]),
            location_label=str(row["location_label"]),
            bbox=bbox,
            confidence=float(row["confidence"]),
            timestamp=str(row["timestamp"]),
            source=str(row["source"]),
            frame_path=row["frame_path"],
        )


def records_to_table(records: Iterable[MemoryRecord]) -> str:
    """Render records as a readable fixed-width CLI table."""
    rows = list(records)
    if not rows:
        return "No memory records found."

    headers = ["timestamp", "object", "normalized", "location", "conf", "bbox", "frame"]
    data = []
    for record in rows:
        data.append(
            [
                record.timestamp,
                record.object_label,
                record.normalized_label,
                record.location_label,
                f"{record.confidence:.2f}",
                str(list(record.bbox)),
                record.frame_path or "",
            ]
        )

    widths = [len(header) for header in headers]
    for row in data:
        for idx, cell in enumerate(row):
            widths[idx] = min(max(widths[idx], len(cell)), 42)

    def fit(value: str, width: int) -> str:
        return value if len(value) <= width else value[: max(0, width - 1)] + "…"

    lines = []
    lines.append(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in data:
        lines.append(" | ".join(fit(cell, widths[idx]).ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)

```

---

## backend/behavior/state_machine.py

```python
"""Finite state machine for lamp interaction state.

Milestone 1.5 adds hysteresis and dwell-time gating. The FSM consumes the
smoothed engagement signal, not the raw per-frame detector output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.perception.engagement_detector import EngagementResult
from backend.utils.config import StateMachineConfig


class LampState(str, Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"
    SEEKING_ATTENTION = "seeking_attention"
    RECALLING = "recalling"


@dataclass(frozen=True)
class StateTransition:
    previous_state: LampState
    current_state: LampState
    changed: bool
    reason: str
    state_elapsed_s: float
    consecutive_engaged: int
    consecutive_disengaged: int
    consecutive_absent: int


class InteractionStateMachine:
    def __init__(self, config: StateMachineConfig) -> None:
        self.config = config
        self.state = LampState.IDLE
        self._state_started_at = time.monotonic()
        self._disengaged_started_at: Optional[float] = None
        self._consecutive_engaged = 0
        self._consecutive_disengaged = 0
        self._consecutive_absent = 0

    def update(self, engagement: EngagementResult, now: Optional[float] = None) -> StateTransition:
        now = now if now is not None else time.monotonic()
        previous = self.state
        target = self.state
        reason = engagement.reason

        self._update_consecutive_counts(engagement.status)
        dwell_s = now - self._state_started_at
        clear_engaged_recovery = self._is_clear_engaged_recovery(engagement)

        if engagement.status == "engaged":
            self._disengaged_started_at = None
            if clear_engaged_recovery:
                target = LampState.ENGAGED
                reason = "clear_engaged_recovery"
            else:
                reason = "holding_until_engaged_stable"

        elif engagement.status == "disengaged":
            if self._disengaged_started_at is None:
                self._disengaged_started_at = now

            if self.state == LampState.ENGAGED:
                if self._consecutive_disengaged >= self.config.exit_engaged_disengaged_frames:
                    target = LampState.DISENGAGED
                    reason = f"stable_disengaged_{self._consecutive_disengaged}_frames"
                else:
                    reason = f"hold_engaged_disengaged_count_{self._consecutive_disengaged}"
            elif self.state in (LampState.IDLE, LampState.DISENGAGED):
                target = LampState.DISENGAGED
                reason = engagement.reason
            elif self.state == LampState.SEEKING_ATTENTION:
                target = LampState.SEEKING_ATTENTION
                reason = "continue_seeking_attention"

            disengaged_elapsed = now - self._disengaged_started_at
            if target == LampState.DISENGAGED and disengaged_elapsed >= self.config.seek_attention_after_s:
                target = LampState.SEEKING_ATTENTION
                reason = f"stable_disengaged_for_{disengaged_elapsed:.1f}s"

        elif engagement.status == "absent":
            if self.state == LampState.ENGAGED:
                if self._consecutive_absent >= self.config.exit_engaged_absent_frames:
                    target = LampState.DISENGAGED
                    reason = f"stable_absent_{self._consecutive_absent}_frames_after_engaged"
                else:
                    reason = f"hold_engaged_absent_count_{self._consecutive_absent}"
            elif self.state in (LampState.DISENGAGED, LampState.SEEKING_ATTENTION):
                if self._consecutive_absent >= self.config.absent_to_idle_frames:
                    self._disengaged_started_at = None
                    target = LampState.IDLE
                    reason = f"stable_absent_{self._consecutive_absent}_frames_idle"
                else:
                    target = self.state
                    reason = f"hold_active_absent_count_{self._consecutive_absent}"
            else:
                target = LampState.IDLE
                reason = "no_face_idle"

        if target != previous and not clear_engaged_recovery and dwell_s < self.config.min_state_dwell_s:
            target = previous
            reason = f"min_dwell_hold_{dwell_s:.2f}s"

        changed = target != previous
        if changed:
            self.state = target
            self._state_started_at = now
            dwell_s = 0.0
        else:
            dwell_s = now - self._state_started_at

        return StateTransition(
            previous_state=previous,
            current_state=self.state,
            changed=changed,
            reason=reason,
            state_elapsed_s=dwell_s,
            consecutive_engaged=self._consecutive_engaged,
            consecutive_disengaged=self._consecutive_disengaged,
            consecutive_absent=self._consecutive_absent,
        )

    def _update_consecutive_counts(self, status: str) -> None:
        if status == "engaged":
            self._consecutive_engaged += 1
            self._consecutive_disengaged = 0
            self._consecutive_absent = 0
        elif status == "disengaged":
            self._consecutive_engaged = 0
            self._consecutive_disengaged += 1
            self._consecutive_absent = 0
        elif status == "absent":
            self._consecutive_engaged = 0
            self._consecutive_disengaged = 0
            self._consecutive_absent += 1
        else:
            self._consecutive_engaged = 0
            self._consecutive_disengaged = 0
            self._consecutive_absent = 0

    def _is_clear_engaged_recovery(self, engagement: EngagementResult) -> bool:
        return (
            engagement.status == "engaged"
            and engagement.confidence >= self.config.clear_engaged_confidence
            and self._consecutive_engaged >= self.config.engaged_recovery_frames
        )

```

---

## backend/behavior/behavior_policy.py

```python
"""Bounded mapping from interaction state to expressive behavior command fields."""

from __future__ import annotations

from backend.behavior.state_machine import LampState


BEHAVIOR_BY_STATE: dict[LampState, dict] = {
    LampState.IDLE: {
        "motion": "idle_breathe",
        "light": "dim_warm",
        "sound": None,
        "speech_text": None,
    },
    LampState.ENGAGED: {
        "motion": "attentive_nod",
        "light": "steady_warm",
        "sound": None,
        "speech_text": None,
    },
    LampState.DISENGAGED: {
        "motion": "searching_glance",
        "light": "slow_pulse",
        "sound": None,
        "speech_text": None,
    },
    LampState.SEEKING_ATTENTION: {
        "motion": "curious_tilt",
        "light": "soft_pulse",
        "sound": "gentle_chime",
        "speech_text": None,
    },
    LampState.RECALLING: {
        "motion": "thinking",
        "light": "focus_glow",
        "sound": None,
        "speech_text": None,
    },
}


def behavior_for_state(state: LampState) -> dict:
    # Return a copy so callers can safely annotate without mutating the policy table.
    return dict(BEHAVIOR_BY_STATE[state])

```

---

## backend/behavior/command_protocol.py

```python
"""JSON command builder matching the backend/frontend protocol shape."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Optional

from backend.behavior.state_machine import LampState
from backend.perception.engagement_detector import EngagementResult


def build_behavior_command(
    state: LampState,
    engagement: EngagementResult,
    behavior: Mapping,
    last_detected_objects: Optional[Iterable[Mapping]] = None,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "state": state.value,
        "engagement": engagement.to_protocol_dict(),
        "behavior": {
            "motion": behavior.get("motion"),
            "light": behavior.get("light"),
            "sound": behavior.get("sound"),
            "speech_text": behavior.get("speech_text"),
        },
        "memory": {
            # This field is stable across Milestones 3-4 and is consumed by the Godot overlay.
            "last_detected_objects": list(last_detected_objects or []),
        },
    }

```

---

## backend/evaluation/latency_logger.py

```python
"""CSV latency logger for engagement, memory, and recall evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


class LatencyLogger:
    FIELDNAMES = [
        "timestamp",
        "frame_index",
        "capture_ms",
        "engagement_detection_ms",
        "object_detection_ms",
        "memory_write_ms",
        "memory_retrieval_ms",
        "llm_response_ms",
        "smoothing_ms",
        "state_machine_ms",
        "command_build_ms",
        "godot_udp_send_ms",
        "total_loop_ms",
        "fps",
        "state",
        "state_elapsed_s",
        "raw_engagement_status",
        "raw_engagement_confidence",
        "raw_engagement_reason",
        "smoothed_engagement_status",
        "smoothed_engagement_confidence",
        "smoothed_engagement_reason",
        "face_bbox",
        "face_area_ratio",
        "raw_face_count",
        "candidate_count",
        "selected_face_score",
        "object_count",
        "memory_write_count",
        "memory_duplicate_skip_count",
        "consecutive_engaged",
        "consecutive_disengaged",
        "consecutive_absent",
    ]

    def __init__(self, paths: str | Path | Iterable[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            self.paths = [Path(paths)]
        else:
            self.paths = [Path(path) for path in paths]

        if not self.paths:
            raise ValueError("LatencyLogger requires at least one output path")

        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append(self, row: Mapping) -> None:
        clean_row = {field: row.get(field, "") for field in self.FIELDNAMES}
        for path in self.paths:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(clean_row)

```

---

## docs/milestone4_grounded_recall.md

```markdown
# Milestone 4: Grounded Memory Recall

## Goal

Milestone 4 adds text-based grounded memory recall on top of the working Milestone 3 object-memory backend.

The recall path is intentionally bounded:

```text
User question
  -> deterministic object parser
  -> exact SQLite lookup by normalized_label
  -> deterministic answer, or optional Ollama phrasing from the retrieved record only
  -> optional Godot command with state=recalling
```

Godot remains the embodiment layer. Python owns parsing, retrieval, answer grounding, logging, and command generation.

## Files created

```text
backend/conversation/__init__.py
backend/conversation/query_parser.py
backend/conversation/llm_client.py
backend/conversation/recall_agent.py
docs/milestone4_grounded_recall.md
```

## Files modified

```text
.gitignore
backend/main.py
backend/behavior/behavior_policy.py
backend/behavior/command_protocol.py
backend/behavior/state_machine.py
backend/evaluation/latency_logger.py
backend/memory/memory_store.py
```

## Strict grounded LLM prompt

```text
You are the voice of a LeLamp-inspired robotic lamp.

You must answer the user's object-location question using only the provided memory record.
Do not use outside knowledge.
Do not infer, guess, or invent a location.
Do not say where the object is now. Only say where it was last seen.
Do not mention objects that are not in the memory record.
If the memory record is missing or insufficient, say: "I do not remember seeing that."
Keep the answer to one short conversational sentence.

User question:
{user_query}

Parsed target object:
{parsed_object}

Memory record JSON:
{memory_record_json}

Answer:
```

The MVP only calls the LLM when a concrete SQLite memory record exists. Missing-memory responses remain deterministic.

## PowerShell test commands

Run these from the project root.

### 1. Confirm recent object memories exist

```powershell
python -m backend.memory.scene_memory --db data/scene_memory.sqlite --recent --limit 10
```

### 2. Deterministic recall without LLM

```powershell
python -m backend.conversation.recall_agent "Where did you last see my phone?" --memory-db data/scene_memory.sqlite
python -m backend.conversation.recall_agent "Where is the cup?" --memory-db data/scene_memory.sqlite
python -m backend.conversation.recall_agent "Have you seen my mouse?" --memory-db data/scene_memory.sqlite
```

### 3. Machine-readable recall

```powershell
python -m backend.conversation.recall_agent "Where is the cup?" --memory-db data/scene_memory.sqlite --json
```

### 4. Missing-memory grounding test

```powershell
python -m backend.conversation.recall_agent "Where is the stapler?" --memory-db data/scene_memory.sqlite --json
```

Expected behavior: `retrieved_memory_id` is `null`, and the answer says it does not remember seeing the stapler.

### 5. Optional Ollama/local LLM phrasing

Start Ollama separately, make sure the model exists locally, then run:

```powershell
python -m backend.conversation.recall_agent "Where is the cup?" --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model llama3.2
```

Alternative model:

```powershell
python -m backend.conversation.recall_agent "Where is the cup?" --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model qwen2.5
```

If Ollama is unavailable or returns an empty response, the agent falls back to the deterministic template.

### 6. Backend interactive recall with Godot

Start Godot first, then run:

```powershell
python -m backend.main --godot-udp --enable-objects --interactive-recall --memory-db data/scene_memory.sqlite --object-model yolov8n.pt --save-object-frames
```

Then type questions into the backend terminal:

```text
Where is the cup?
Have you seen my mouse?
Where did you last see my phone?
```

For each answer, the backend sends a command shaped like:

```json
{
  "state": "recalling",
  "behavior": {
    "motion": "thinking",
    "light": "focus_glow",
    "sound": null,
    "speech_text": "I last saw the cup at the center of view at ..."
  }
}
```

## Example outputs

### Found object

```text
I last saw the phone at the right side of view at 2026-06-18T12:01:18 with 0.68 confidence. I saved a reference frame at data\object_frames\frame_230_phone_1781798478256.jpg.
parsed_object=phone memory_id=328db801-ebd0-4049-ae8f-426179b7d5a8 memory_retrieval_ms=1.202 llm_response_ms=
```

### Missing object

```json
{
  "user_query": "Where is the stapler?",
  "parsed_object": "stapler",
  "retrieved_memory_id": null,
  "memory_record": null,
  "answer": "I do not remember seeing the stapler."
}
```

## Recall logging

Standalone CLI recall writes JSONL to:

```text
logs/recall.jsonl
```

Interactive backend recall writes JSONL to the isolated run folder:

```text
logs/runs/<run_id>/recall.jsonl
logs/latest/recall.jsonl
```

Each recall event includes:

```text
user_query
parsed_object
retrieved_memory_id
memory_retrieval_ms
llm_response_ms
used_llm
llm_error
answer
```

`latency.csv` now includes the recall latency columns:

```text
memory_retrieval_ms
llm_response_ms
```

## Pass/fail criteria for final evaluation/demo prep

Pass when all of the following are true:

1. Existing Milestone 3 backend still runs with engagement detection, optional YOLO object detection, SQLite memory writes, dedupe, Godot UDP, isolated logs, and analyzer support.
2. `python -m backend.conversation.recall_agent "Where is the cup?" --memory-db data/scene_memory.sqlite` returns the latest matching `normalized_label='cup'` memory.
3. Alias queries work for at least: `cellphone -> phone`, `computer -> laptop`, `screen -> monitor`, and `mug -> cup`.
4. Missing objects return a grounded non-memory answer and do not invent a location.
5. `--json` output contains `parsed_object`, `retrieved_memory_id`, `memory_record`, `memory_retrieval_ms`, `llm_response_ms`, and `answer`.
6. `--use-llm` improves phrasing only when a memory record exists, and falls back safely when Ollama is unavailable.
7. `--interactive-recall` lets you type a question while the backend loop runs and sends Godot a `recalling` command with `thinking`, `focus_glow`, and `speech_text`.
8. Runtime artifacts, frames, model files, caches, logs, and SQLite databases are covered by `.gitignore` and are not committed.

Fail if any recall answer states a location not present in the retrieved SQLite record.

```

