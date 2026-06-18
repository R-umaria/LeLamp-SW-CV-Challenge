# Milestone 4.1 Polish Patch Code Listing

## `.gitignore`

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

# Additional local runtime/model artifacts
*.sqlite
*.sqlite3
*.db
*.db-*
data/*.db
frontend_godot/.mono/
frontend_godot/export_presets.cfg

```

## `backend/conversation/query_parser.py`

```python

"""Deterministic object-query parser for grounded memory recall.

Milestone 4.1 keeps parsing deterministic and local. The parser extracts one
object target from short recall questions such as:
    - Where did you last see my phone?
    - Where is the cup?
    - Have you seen my mouse?
    - Did you see my spectacles that I left near the cup?

Important MVP rule:
    Nearby/context objects are not treated as the recall target when the user
    explicitly asks about a different possessive object. For example,
    "my spectacles ... near the cup" parses as ``glasses``, not ``cup``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Kept local to conversation code so the recall CLI does not import OpenCV,
# Ultralytics, or the object detector just to parse text.
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
    "spectacles": "glasses",
    "glasses": "glasses",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "around",
    "at",
    "by",
    "chance",
    "did",
    "do",
    "for",
    "have",
    "having",
    "i",
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
    "that",
    "the",
    "there",
    "was",
    "were",
    "where",
    "you",
}

# Clause boundaries keep relation/context phrases from being interpreted as the
# target object. In "my spectacles that I left near the cup", the target phrase
# ends before "that" and the nearby cup is ignored for target selection.
BOUNDARY_WORDS = {
    "around",
    "at",
    "behind",
    "beside",
    "by",
    "from",
    "i",
    "in",
    "inside",
    "left",
    "near",
    "next",
    "on",
    "over",
    "right",
    "that",
    "under",
    "which",
    "with",
    "you",
}

_BOUNDARY_REGEX = "|".join(sorted(re.escape(word) for word in BOUNDARY_WORDS))
_TARGET_TEXT = rf"(?P<object>[a-z0-9][a-z0-9\s'-]*?)(?=\s+(?:{_BOUNDARY_REGEX})\b|\s*$)"

OBJECT_PHRASE_PATTERNS = [
    re.compile(rf"\bwhere\s+(?:did\s+you\s+last\s+see|did\s+you\s+see|is|was|are|were)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bhave\s+you\s+seen\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\bdid\s+you\s+see\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\b(?:find|locate)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
    re.compile(rf"\b(?:was|were)\s+(?:i\s+)?(?:having|using|holding)\s+(?:my\s+|the\s+|a\s+|an\s+)?{_TARGET_TEXT}"),
]

POSSESSIVE_TARGET_PATTERN = re.compile(rf"\bmy\s+{_TARGET_TEXT}")


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
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
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

    # Strongest signal: explicit possessive target. This fixes questions like
    # "did you see my spectacles that I left near the cup?" by selecting
    # spectacles/glasses instead of the contextual cup.
    possessive_target = _extract_possessive_target(cleaned)
    if possessive_target:
        normalized = _normalize_with_aliases(possessive_target, alias_map)
        return ParsedObjectQuery(
            original_query=query,
            target_text=possessive_target,
            normalized_label=normalized or None,
            confidence=0.98,
            strategy="explicit_possessive",
        )

    # Next, parse a direct object from common recall question shapes. This can
    # return unsupported labels such as "stapler". The recall agent will then do
    # an exact memory lookup and answer that it does not remember seeing it.
    for pattern in OBJECT_PHRASE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        target = _sanitize_object_phrase(match.group("object"))
        if target:
            normalized = _normalize_with_aliases(target, alias_map)
            return ParsedObjectQuery(
                original_query=query,
                target_text=target,
                normalized_label=normalized or None,
                confidence=0.75,
                strategy="question_pattern",
            )

    # Known alias fallback for compact queries like "cup?" or "phone location".
    # Longest alias wins so "cell phone" wins before "phone".
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if _contains_phrase(cleaned, alias):
            return ParsedObjectQuery(
                original_query=query,
                target_text=alias,
                normalized_label=alias_map[alias],
                confidence=0.60,
                strategy="known_alias_fallback",
            )

    fallback = _last_content_token(cleaned.split())
    if fallback:
        normalized = _normalize_with_aliases(fallback, alias_map)
        return ParsedObjectQuery(
            original_query=query,
            target_text=fallback,
            normalized_label=normalized or None,
            confidence=0.35,
            strategy="fallback_last_token",
        )

    return ParsedObjectQuery(query, None, None, 0.0, "no_object_found")


def _extract_possessive_target(text: str) -> str | None:
    match = POSSESSIVE_TARGET_PATTERN.search(text)
    if not match:
        return None
    return _sanitize_object_phrase(match.group("object")) or None


def _normalize_with_aliases(label: str, alias_map: dict[str, str]) -> str:
    cleaned = clean_query_text(label)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -'")
    if cleaned in alias_map:
        return alias_map[cleaned]
    if cleaned.endswith("s") and cleaned[:-1] in alias_map:
        return alias_map[cleaned[:-1]]
    return cleaned


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _sanitize_object_phrase(phrase: str) -> str:
    cleaned = clean_query_text(phrase)
    cleaned = re.sub(r"\b(my|the|a|an|please)\b", " ", cleaned)
    cleaned = _trim_after_boundary(cleaned)
    tokens = [token for token in cleaned.split() if token not in STOPWORDS]
    return " ".join(tokens).strip(" -'")


def _trim_after_boundary(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for token in tokens:
        stripped = token.strip(" -'")
        if stripped in BOUNDARY_WORDS:
            break
        kept.append(stripped)
    return " ".join(token for token in kept if token)


def _last_content_token(tokens: Iterable[str]) -> str | None:
    for token in reversed(list(tokens)):
        token = token.strip(" -'")
        if token and token not in STOPWORDS:
            return token
    return None

```

## `backend/conversation/llm_client.py`

```python

"""Optional local LLM client for grounded recall phrasing.

This module uses only the Python standard library so the recall CLI works
without extra client packages. It targets Ollama and fails closed: if Ollama is
unavailable, the caller receives a structured diagnostic and can use a
 deterministic grounded template response.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaStatus:
    """Connectivity and model-availability diagnostic for Ollama."""

    ok: bool
    base_url: str
    model: str
    model_available: bool
    latency_ms: float
    error: str | None = None
    available_models: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "base_url": self.base_url,
            "model": self.model,
            "model_available": self.model_available,
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "available_models": list(self.available_models),
        }


@dataclass(frozen=True)
class LLMResponse:
    text: str
    attempted: bool
    used_llm: bool
    latency_ms: float
    error: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "attempted": self.attempted,
            "used_llm": self.used_llm,
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "fallback_reason": self.fallback_reason,
        }


class OllamaClient:
    """Small Ollama wrapper with timeout and deterministic generation settings."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_s: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)

    def check_connectivity(self) -> OllamaStatus:
        """Return whether Ollama is reachable and whether the requested model exists."""

        start = time.perf_counter()
        request = urllib.request.Request(url=f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_s, 3.0)) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            model_names = _extract_model_names(parsed)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            model_available = _model_name_matches(self.model, model_names)
            return OllamaStatus(
                ok=True,
                base_url=self.base_url,
                model=self.model,
                model_available=model_available,
                latency_ms=elapsed_ms,
                error=None if model_available else f"model_not_found: {self.model}",
                available_models=tuple(model_names),
            )
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return OllamaStatus(False, self.base_url, self.model, False, elapsed_ms, f"ollama_connection_error: {exc}")
        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return OllamaStatus(False, self.base_url, self.model, False, elapsed_ms, f"ollama_timeout: {exc}")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return OllamaStatus(False, self.base_url, self.model, False, elapsed_ms, f"ollama_status_error: {exc}")

    def generate(self, prompt: str) -> LLMResponse:
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 80,
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
                return LLMResponse(
                    text="",
                    attempted=True,
                    used_llm=False,
                    latency_ms=elapsed_ms,
                    error="ollama_returned_empty_response",
                    fallback_reason="empty_response",
                )
            return LLMResponse(text=text, attempted=True, used_llm=True, latency_ms=elapsed_ms)
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            body_text = _read_error_body(exc)
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_http_error_{exc.code}: {body_text or exc.reason}",
                fallback_reason="http_error",
            )
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_connection_error: {exc}",
                fallback_reason="connection_error",
            )
        except TimeoutError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_timeout: {exc}",
                fallback_reason="timeout",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return LLMResponse(
                text="",
                attempted=True,
                used_llm=False,
                latency_ms=elapsed_ms,
                error=f"ollama_error: {exc}",
                fallback_reason="runtime_error",
            )


def _extract_model_names(payload: dict[str, Any]) -> list[str]:
    models = payload.get("models", [])
    names: list[str] = []
    if isinstance(models, list):
        for model_info in models:
            if not isinstance(model_info, dict):
                continue
            name = str(model_info.get("name") or model_info.get("model") or "").strip()
            if name:
                names.append(name)
    return sorted(set(names))


def _model_name_matches(requested: str, available: list[str]) -> bool:
    requested = requested.strip()
    if not requested:
        return False
    for name in available:
        if name == requested:
            return True
        if name.startswith(requested + ":"):
            return True
        if requested.endswith(":latest") and name == requested.removesuffix(":latest"):
            return True
    return False


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip()[:300]
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Ollama connectivity for LeLamp grounded recall")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-model", type=str, default="llama3.2")
    parser.add_argument("--ollama-timeout", type=float, default=8.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OllamaClient(args.ollama_url, args.ollama_model, timeout_s=args.ollama_timeout)
    status = client.check_connectivity()
    if args.json:
        print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
    else:
        if status.ok and status.model_available:
            print(f"Ollama OK: {status.base_url} has model {status.model} ({status.latency_ms:.1f} ms)")
        elif status.ok:
            print(f"Ollama reachable, but model '{status.model}' was not found.")
            print("Available models: " + (", ".join(status.available_models) or "none"))
        else:
            print(f"Ollama unavailable: {status.error}")
    return 0 if status.ok and status.model_available else 1


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/conversation/recall_agent.py`

```python

"""Grounded memory-recall agent for Milestone 4.1.

CLI examples:
    python -m backend.conversation.recall_agent "Where did you last see my phone?"
    python -m backend.conversation.recall_agent "Where is the cup?" --json
    python -m backend.conversation.recall_agent "Have you seen my mouse?" --use-llm
    python -m backend.conversation.recall_agent --test-ollama --ollama-model llama3.2

The recall path remains bounded:
    text query -> deterministic object parser -> exact SQLite lookup ->
    deterministic or strictly grounded LLM phrasing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backend.conversation.llm_client import LLMResponse, OllamaClient, OllamaStatus
from backend.conversation.query_parser import ParsedObjectQuery, parse_object_query
from backend.memory.memory_store import MemoryRecord, MemoryStore


STRICT_GROUNDED_RECALL_PROMPT = """You are the voice of a LeLamp-inspired robotic lamp.

You must answer the user's object-location question using only the provided memory record.
Do not use outside knowledge.
Do not infer, guess, or invent a location.
Do not say where the object is now. Only say where it was last seen.
Do not mention objects that are not in the memory record.
Do not mention nearby objects or spatial relationships unless they are explicitly in the memory record.
Do not mention frame paths, files, IDs, JSON, logs, or implementation details.
If the memory record is missing or insufficient, say: "I do not remember seeing that."
Keep the answer to one short conversational sentence.

Required answer style:
I last saw your <object> on the <location> around <time>. My confidence was <confidence>.

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
    llm_requested: bool
    llm_attempted: bool
    llm_used: bool
    llm_error: str | None
    llm_fallback_reason: str | None
    ollama_connection_ok: bool | None
    timestamp: str

    @property
    def used_llm(self) -> bool:
        """Backward-compatible alias for Milestone 4 callers."""
        return self.llm_used

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
            "llm_requested": self.llm_requested,
            "llm_attempted": self.llm_attempted,
            "llm_used": self.llm_used,
            "used_llm": self.llm_used,
            "llm_error": self.llm_error,
            "llm_fallback_reason": self.llm_fallback_reason,
            "ollama_connection_ok": self.ollama_connection_ok,
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
        ollama_timeout_s: float = 8.0,
        log_paths: str | Path | Iterable[str | Path] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.memory_db = Path(memory_db)
        self.store = MemoryStore(self.memory_db)
        self.use_llm = bool(use_llm)
        self.logger = logger or logging.getLogger("lelamp")
        self.log_paths = _coerce_paths(log_paths)
        self.ollama: OllamaClient | None = None
        self.ollama_status: OllamaStatus | None = None

        if self.use_llm:
            self.ollama = OllamaClient(base_url=ollama_url, model=ollama_model, timeout_s=ollama_timeout_s)
            self.ollama_status = self.ollama.check_connectivity()
            if self.ollama_status.ok and self.ollama_status.model_available:
                self.logger.info(
                    "Ollama connectivity OK url=%s model=%s status_ms=%.3f",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    self.ollama_status.latency_ms,
                )
            elif self.ollama_status.ok:
                self.logger.warning(
                    "Ollama reachable but requested model is unavailable url=%s model=%s available_models=%s",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    list(self.ollama_status.available_models),
                )
            else:
                self.logger.warning(
                    "Ollama unavailable for recall url=%s model=%s error=%s",
                    self.ollama_status.base_url,
                    self.ollama_status.model,
                    self.ollama_status.error,
                )

    def answer(self, user_query: str) -> RecallResult:
        parsed = parse_object_query(user_query)
        retrieval_start = time.perf_counter()
        memory_record = None
        if parsed.normalized_label:
            memory_record = self.store.find_latest_by_normalized_label(parsed.normalized_label)
        memory_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fallback_answer = deterministic_recall_answer(parsed, memory_record)
        final_answer = fallback_answer

        llm_response: LLMResponse | None = None
        llm_attempted = False
        llm_used = False
        llm_error: str | None = None
        llm_fallback_reason: str | None = None
        ollama_connection_ok: bool | None = None

        if self.use_llm:
            ollama_connection_ok = False if self.ollama_status is None else (
                self.ollama_status.ok and self.ollama_status.model_available
            )

            # Safer MVP rule: only ask the LLM to phrase an answer when a
            # concrete memory exists. Missing-memory answers stay deterministic.
            if memory_record is None:
                llm_fallback_reason = "no_memory_record"
            elif self.ollama is None:
                llm_error = "ollama_client_not_initialized"
                llm_fallback_reason = "client_unavailable"
            elif self.ollama_status is not None and not self.ollama_status.ok:
                llm_error = self.ollama_status.error or "ollama_unavailable"
                llm_fallback_reason = "connectivity_check_failed"
            elif self.ollama_status is not None and not self.ollama_status.model_available:
                llm_error = self.ollama_status.error or f"model_not_found: {self.ollama.model}"
                llm_fallback_reason = "model_unavailable"
            else:
                prompt = build_grounded_prompt(user_query, parsed, memory_record)
                llm_response = self.ollama.generate(prompt)
                llm_attempted = llm_response.attempted
                llm_used = llm_response.used_llm
                llm_error = llm_response.error
                llm_fallback_reason = llm_response.fallback_reason
                if llm_response.used_llm and llm_response.text:
                    final_answer = _clean_llm_text(llm_response.text)
                else:
                    final_answer = fallback_answer

            if self.use_llm and not llm_used:
                self.logger.warning(
                    "LLM fallback used query=%r parsed_object=%s reason=%s error=%s attempted=%s",
                    user_query,
                    parsed.normalized_label,
                    llm_fallback_reason,
                    llm_error,
                    llm_attempted,
                )

        result = RecallResult(
            user_query=user_query,
            parsed_object=parsed.normalized_label,
            parsed=parsed,
            memory_record=memory_record,
            answer=final_answer,
            memory_retrieval_ms=memory_retrieval_ms,
            llm_response_ms=None if llm_response is None else llm_response.latency_ms,
            llm_requested=self.use_llm,
            llm_attempted=llm_attempted,
            llm_used=llm_used,
            llm_error=llm_error,
            llm_fallback_reason=llm_fallback_reason,
            ollama_connection_ok=ollama_connection_ok,
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
        )
        self._log_result(result)
        return result

    def _log_result(self, result: RecallResult) -> None:
        payload = result.to_dict()
        self.logger.info(
            "Recall query=%r parsed_object=%s memory_id=%s retrieval_ms=%.3f llm_attempted=%s llm_used=%s llm_ms=%s llm_error=%s fallback=%s answer=%r",
            result.user_query,
            result.parsed_object,
            result.memory_record.id if result.memory_record else None,
            result.memory_retrieval_ms,
            result.llm_attempted,
            result.llm_used,
            "" if result.llm_response_ms is None else f"{result.llm_response_ms:.3f}",
            result.llm_error,
            result.llm_fallback_reason,
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
        memory_record_json=json.dumps(_memory_record_for_prompt(memory_record), ensure_ascii=False, indent=2),
    )


def deterministic_recall_answer(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    display_label = _display_object_label(parsed, memory_record)

    if not parsed.normalized_label:
        return "I am not sure which object you mean. Ask me about a specific object, like your cup or phone."

    if memory_record is None:
        return f"I do not remember seeing your {display_label}."

    time_text = _friendly_time(memory_record.timestamp)
    return (
        f"I last saw your {display_label} on the {memory_record.location_label} "
        f"around {time_text}. My confidence was {memory_record.confidence:.2f}."
    )


def _memory_record_for_prompt(memory_record: MemoryRecord) -> dict:
    return {
        "object_label": memory_record.object_label,
        "normalized_label": memory_record.normalized_label,
        "location_label": memory_record.location_label,
        "confidence": round(float(memory_record.confidence), 3),
        "timestamp": memory_record.timestamp,
        "source": memory_record.source,
    }


def _display_object_label(parsed: ParsedObjectQuery, memory_record: MemoryRecord | None) -> str:
    if memory_record is not None:
        return memory_record.normalized_label or memory_record.object_label
    if parsed.normalized_label:
        return parsed.normalized_label
    if parsed.target_text:
        return parsed.target_text
    return "that object"


def _friendly_time(timestamp_text: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp_text)
    except ValueError:
        return timestamp_text
    return parsed.strftime("%H:%M:%S on %Y-%m-%d")


def _clean_llm_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
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
    parser.add_argument("--ollama-timeout", type=float, default=8.0, help="Ollama generate timeout in seconds")
    parser.add_argument("--test-ollama", action="store_true", help="Check Ollama connectivity/model availability and exit")
    parser.add_argument("--json", action="store_true", help="Print machine-readable recall result JSON")
    parser.add_argument("--debug", action="store_true", help="Print debug metadata such as frame_path outside the spoken answer")
    parser.add_argument("--log-path", type=str, default="logs/recall.jsonl", help="JSONL recall log path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test_ollama:
        client = OllamaClient(args.ollama_url, args.ollama_model, timeout_s=args.ollama_timeout)
        status = client.check_connectivity()
        if args.json:
            print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
        else:
            if status.ok and status.model_available:
                print(f"Ollama OK: {status.base_url} has model {status.model} ({status.latency_ms:.1f} ms)")
            elif status.ok:
                print(f"Ollama reachable, but model '{status.model}' was not found.")
                print("Available models: " + (", ".join(status.available_models) or "none"))
            else:
                print(f"Ollama unavailable: {status.error}")
        return 0 if status.ok and status.model_available else 1

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
        ollama_timeout_s=args.ollama_timeout,
        log_paths=args.log_path,
    )
    result = agent.answer(args.query)

    if args.use_llm and not result.llm_used:
        warning = (
            "Warning: --use-llm was requested, but the deterministic fallback was used. "
            f"reason={result.llm_fallback_reason or 'unknown'} error={result.llm_error or 'none'}"
        )
        print(warning, file=sys.stderr)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(
            f"parsed_object={result.parsed_object or 'none'} "
            f"memory_id={result.memory_record.id if result.memory_record else 'none'} "
            f"memory_retrieval_ms={result.memory_retrieval_ms:.3f} "
            f"llm_attempted={result.llm_attempted} "
            f"llm_used={result.llm_used} "
            f"llm_response_ms={'' if result.llm_response_ms is None else f'{result.llm_response_ms:.3f}'} "
            f"llm_error={result.llm_error or 'none'} "
            f"llm_fallback_reason={result.llm_fallback_reason or 'none'}"
        )
        if args.debug and result.memory_record is not None:
            print(f"frame_path={result.memory_record.frame_path or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

## `backend/conversation/test_query_parser.py`

```python

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

```

## `backend/main.py`

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
    parser.add_argument("--ollama-timeout", type=float, default=8.0, help="Ollama generate timeout in seconds for recall phrasing.")

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
            ollama_timeout_s=args.ollama_timeout,
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
        "Recall config interactive=%s use_llm=%s ollama_url=%s ollama_model=%s ollama_timeout=%.1fs",
        args.interactive_recall,
        args.use_llm,
        args.ollama_url,
        args.ollama_model,
        args.ollama_timeout,
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
            llm_attempted = ""
            llm_used = ""
            llm_error = ""
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
                    llm_attempted = recall_result.llm_attempted
                    llm_used = recall_result.llm_used
                    llm_error = recall_result.llm_error or ""
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
                        "Sent recall command to Godot state=recalling parsed_object=%s memory_id=%s llm_attempted=%s llm_used=%s",
                        recall_result.parsed_object,
                        recall_result.memory_record.id if recall_result.memory_record else None,
                        recall_result.llm_attempted,
                        recall_result.llm_used,
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
                    "llm_attempted": llm_attempted,
                    "llm_used": llm_used,
                    "llm_error": llm_error,
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

## `backend/evaluation/latency_logger.py`

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
        "llm_attempted",
        "llm_used",
        "llm_error",
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

## `frontend_godot/scripts/Main.gd`

```gdscript

extends Node3D

@export var response_visible_seconds: float = 8.0

@onready var udp_receiver: Node = $UdpCommandReceiver
@onready var lamp: Node = $LampRig
@onready var camera: Camera3D = $Camera3D
@onready var sun: DirectionalLight3D = $DirectionalLight3D

var _debug_label: Label
var _response_panel: PanelContainer
var _response_label: Label
var _receiver_status: String = "starting"
var _last_command: Dictionary = {}
var _last_packet_local_time: String = "never"
var _response_visible_until: float = 0.0
var _last_response_text: String = ""


func _ready() -> void:
	_setup_camera_and_light()
	_build_ui()

	udp_receiver.command_received.connect(_on_command_received)
	udp_receiver.receiver_status_changed.connect(_on_receiver_status_changed)
	udp_receiver.start()

	_last_command = _default_command()
	lamp.apply_command(_last_command)
	_refresh_debug_ui()
	_update_response_panel()


func _process(_delta: float) -> void:
	_refresh_debug_ui()
	_update_response_panel()


func _setup_camera_and_light() -> void:
	camera.position = Vector3(0.0, 2.0, 5.2)
	camera.look_at(Vector3(0.0, 1.35, 0.0), Vector3.UP)
	camera.fov = 45.0

	sun.rotation_degrees = Vector3(-45.0, 35.0, 0.0)
	sun.light_energy = 1.4


func _build_ui() -> void:
	var canvas: CanvasLayer = CanvasLayer.new()
	canvas.name = "LeLampCanvas"
	add_child(canvas)
	_build_debug_panel(canvas)
	_build_response_panel(canvas)


func _build_debug_panel(canvas: CanvasLayer) -> void:
	var panel: PanelContainer = PanelContainer.new()
	panel.name = "DebugPanel"
	panel.position = Vector2(12.0, 12.0)
	panel.custom_minimum_size = Vector2(410.0, 210.0)
	canvas.add_child(panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 10)
	margin.add_theme_constant_override("margin_top", 8)
	margin.add_theme_constant_override("margin_right", 10)
	margin.add_theme_constant_override("margin_bottom", 8)
	panel.add_child(margin)

	_debug_label = Label.new()
	_debug_label.name = "DebugLabel"
	_debug_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	margin.add_child(_debug_label)


func _build_response_panel(canvas: CanvasLayer) -> void:
	_response_panel = PanelContainer.new()
	_response_panel.name = "RecallResponsePanel"
	_response_panel.position = Vector2(440.0, 12.0)
	_response_panel.custom_minimum_size = Vector2(520.0, 150.0)
	_response_panel.visible = false
	canvas.add_child(_response_panel)

	var margin: MarginContainer = MarginContainer.new()
	margin.add_theme_constant_override("margin_left", 16)
	margin.add_theme_constant_override("margin_top", 12)
	margin.add_theme_constant_override("margin_right", 16)
	margin.add_theme_constant_override("margin_bottom", 12)
	_response_panel.add_child(margin)

	var vbox: VBoxContainer = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	margin.add_child(vbox)

	var title: Label = Label.new()
	title.text = "LeLamp recall"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	vbox.add_child(title)

	_response_label = Label.new()
	_response_label.name = "RecallResponseLabel"
	_response_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_response_label.text = ""
	vbox.add_child(_response_label)


func _on_receiver_status_changed(message: String) -> void:
	_receiver_status = message
	_refresh_debug_ui()


func _on_command_received(command: Dictionary) -> void:
	_last_command = command
	_last_packet_local_time = Time.get_datetime_string_from_system(false, true)
	lamp.apply_command(command)
	_maybe_show_recall_response(command)
	_refresh_debug_ui()


func _maybe_show_recall_response(command: Dictionary) -> void:
	var state_text: String = str(command.get("state", ""))
	var behavior: Dictionary = _dictionary_value(command, "behavior")
	var speech_value: Variant = behavior.get("speech_text", null)
	var speech_text: String = "" if speech_value == null else str(speech_value).strip_edges()
	if state_text == "recalling" and speech_text != "":
		_show_response_text(speech_text)


func _dictionary_value(source: Dictionary, key: String) -> Dictionary:
	var value: Variant = source.get(key, {})
	if typeof(value) == TYPE_DICTIONARY:
		return value as Dictionary
	return {}


func _show_response_text(text: String) -> void:
	_last_response_text = text
	_response_visible_until = Time.get_unix_time_from_system() + response_visible_seconds
	if _response_label != null:
		_response_label.text = text
	if _response_panel != null:
		_response_panel.visible = true


func _update_response_panel() -> void:
	if _response_panel == null:
		return
	var now: float = Time.get_unix_time_from_system()
	_response_panel.visible = _last_response_text != "" and now < _response_visible_until


func _refresh_debug_ui() -> void:
	if _debug_label == null:
		return

	var engagement: Dictionary = _dictionary_value(_last_command, "engagement")
	var behavior: Dictionary = _dictionary_value(_last_command, "behavior")
	var speech_value: Variant = behavior.get("speech_text", "")
	var speech_text: String = "" if speech_value == null else str(speech_value)

	var lines: Array[String] = []
	lines.append("LeLamp Milestone 4.1 Frontend")
	lines.append("UDP: %s" % _receiver_status)
	lines.append("State: %s" % str(_last_command.get("state", "unknown")))
	lines.append("Motion: %s" % str(behavior.get("motion", "none")))
	lines.append("Light: %s" % str(behavior.get("light", "none")))
	lines.append("Sound: %s" % str(behavior.get("sound", "none")))
	lines.append("Speech: %s" % speech_text)
	lines.append("Engagement: %s  confidence: %.2f" % [
		str(engagement.get("status", "unknown")),
		float(engagement.get("confidence", 0.0))
	])
	lines.append("Reason: %s" % str(engagement.get("reason", "none")))
	lines.append("Packet timestamp: %s" % str(_last_command.get("timestamp", "never")))
	lines.append("Local received: %s" % _last_packet_local_time)
	lines.append("Recall panel: %s" % ("visible" if _response_panel != null and _response_panel.visible else "hidden"))

	var text: String = ""
	for line: String in lines:
		if text != "":
			text += "\n"
		text += line
	_debug_label.text = text


func _default_command() -> Dictionary:
	return {
		"timestamp": "not connected yet",
		"state": "idle",
		"engagement": {
			"status": "absent",
			"confidence": 0.0,
			"reason": "waiting_for_udp",
		},
		"behavior": {
			"motion": "idle_breathe",
			"light": "dim_warm",
			"sound": null,
			"speech_text": null,
		},
		"memory": {
			"last_detected_objects": [],
		},
	}

```

## `docs/milestone4_1_polish.md`

```markdown

# Milestone 4.1: Grounded Recall Polish

Milestone 4.1 keeps the Milestone 4 architecture intact and only tightens recall diagnostics, deterministic query parsing, spoken answer formatting, and Godot display feedback.

## What changed

Backend:

- Added explicit Ollama diagnostics:
  - `llm_requested`
  - `llm_attempted`
  - `llm_used`
  - `llm_error`
  - `llm_fallback_reason`
  - `ollama_connection_ok`
- Added Ollama connectivity checks through:
  - `python -m backend.conversation.recall_agent --test-ollama --ollama-url http://localhost:11434 --ollama-model llama3.2`
  - `python -m backend.conversation.llm_client --ollama-url http://localhost:11434 --ollama-model llama3.2`
- Kept deterministic fallback as the default safe answer path.
- Updated the spoken/displayed deterministic response template:
  - `I last saw your {object} on the {location} around {time}. My confidence was {confidence}.`
- Removed `frame_path` from spoken/displayed sentences. It remains available in JSON/debug output through the retrieved memory record.
- Improved query parsing so possessive targets after `my` win over context objects. Example:
  - `By any chance, did you see my spectacles that I left near the cup?` parses as `glasses`, not `cup`.

Frontend:

- Added a visible recall response panel in Godot.
- When `state=recalling` and `behavior.speech_text` is non-empty, the answer is shown for several seconds.
- No voice input or TTS was added.

## Strict grounded LLM prompt

The LLM still receives a strict prompt that says it may only use the provided memory record. The prompt also forbids nearby-object inference unless such a relation is explicitly stored. The prompt does not include frame paths.

## PowerShell test commands

Parser regression tests:

```powershell
python -m unittest backend.conversation.test_query_parser -v
```

Ollama connectivity:

```powershell
python -m backend.conversation.recall_agent --test-ollama --ollama-url http://localhost:11434 --ollama-model llama3.2
python -m backend.conversation.recall_agent --test-ollama --ollama-url http://localhost:11434 --ollama-model llama3.2 --json
```

Deterministic recall:

```powershell
python -m backend.conversation.recall_agent "Where is my phone?" --memory-db data/scene_memory.sqlite
python -m backend.conversation.recall_agent "Where did you last see my cup?" --memory-db data/scene_memory.sqlite
python -m backend.conversation.recall_agent "Have you seen my mouse?" --memory-db data/scene_memory.sqlite
python -m backend.conversation.recall_agent "By any chance, did you see my spectacles that I left near the cup?" --memory-db data/scene_memory.sqlite
python -m backend.conversation.recall_agent "Where is the stapler?" --memory-db data/scene_memory.sqlite
```

LLM recall diagnostics:

```powershell
python -m backend.conversation.recall_agent "Where is my phone?" --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model llama3.2
python -m backend.conversation.recall_agent "Where is my phone?" --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model llama3.2 --json
```

Interactive backend recall with Godot:

```powershell
python -m backend.main --godot-udp --enable-objects --interactive-recall --memory-db data/scene_memory.sqlite --object-model yolov8n.pt --save-object-frames
```

## Expected outputs

If a memory exists:

```text
I last saw your phone on the left side of view around 12:31:07 on 2026-06-18. My confidence was 0.51.
parsed_object=phone memory_id=<uuid> memory_retrieval_ms=<ms> llm_attempted=False llm_used=False llm_response_ms= llm_error=none llm_fallback_reason=none
```

If the query contains a context object but asks about an unsupported target:

```text
I do not remember seeing your glasses.
parsed_object=glasses memory_id=none memory_retrieval_ms=<ms> llm_attempted=False llm_used=False llm_response_ms= llm_error=none llm_fallback_reason=none
```

If `--use-llm` is set but Ollama is unavailable or the model is missing:

```text
Warning: --use-llm was requested, but the deterministic fallback was used. reason=<reason> error=<error>
```

## Pass/fail criteria for final demo preparation

Pass:

1. Existing engagement detection, object detection, memory writes, Godot UDP, isolated run logs, and analyzer behavior still work.
2. Recall answers remain grounded in SQLite records.
3. `--use-llm` logs and prints whether Ollama was reachable, whether generation was attempted, whether the LLM was used, and why fallback happened.
4. `Where is my phone?` parses as `phone`.
5. `Where did you last see my cup?` parses as `cup`.
6. `Have you seen my mouse?` parses as `mouse`.
7. `By any chance, did you see my spectacles that I left near the cup?` parses as `glasses`, and does not answer about the cup unless glasses memory exists.
8. `Where is the stapler?` parses as `stapler` and returns a no-memory answer unless stapler memory exists.
9. Godot displays `behavior.speech_text` in the recall response panel when `state=recalling`.

Fail:

1. The system answers about a nearby/context object instead of the explicit requested target.
2. The LLM path hides a fallback reason.
3. Spoken/displayed answers include frame paths.
4. The frontend requires voice input, TTS, or a new dependency to show recall answers.
5. The command JSON protocol changes.

```
