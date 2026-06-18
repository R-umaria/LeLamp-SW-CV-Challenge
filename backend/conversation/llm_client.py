"""Local Ollama client for structured, grounded conversation responses.

Milestone 4.3 intentionally uses Ollama's ``/api/chat`` endpoint instead of
raw ``/api/generate`` phrasing. The model is constrained with a JSON schema and
all callers still validate the returned facts against SQLite memory before the
answer is trusted.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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
    """Result from a local LLM request.

    ``text`` is the raw assistant content. ``json_data`` is populated only when
    the raw content parsed as a JSON object. ``used_llm`` means Ollama returned a
    non-empty response; it does *not* mean the answer is trusted. Trust is decided
    later by the recall validator.
    """

    text: str
    attempted: bool
    used_llm: bool
    latency_ms: float
    error: str | None = None
    fallback_reason: str | None = None
    json_data: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "attempted": self.attempted,
            "used_llm": self.used_llm,
            "latency_ms": round(float(self.latency_ms), 3),
            "error": self.error,
            "fallback_reason": self.fallback_reason,
            "json_data": self.json_data,
        }


class OllamaClient:
    """Small Ollama wrapper using only the Python standard library."""

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

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        *,
        max_tokens: int = 180,
    ) -> LLMResponse:
        """Call Ollama ``/api/chat`` and request a JSON object matching ``schema``."""

        start = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "format": dict(schema),
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": int(max_tokens),
            },
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            message = parsed.get("message", {})
            text = ""
            if isinstance(message, dict):
                text = str(message.get("content", "")).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not text:
                return LLMResponse(
                    text="",
                    attempted=True,
                    used_llm=False,
                    latency_ms=elapsed_ms,
                    error="ollama_returned_empty_chat_content",
                    fallback_reason="empty_response",
                )
            json_data = _parse_json_object(text)
            if json_data is None:
                return LLMResponse(
                    text=text,
                    attempted=True,
                    used_llm=True,
                    latency_ms=elapsed_ms,
                    error="ollama_returned_invalid_json",
                    fallback_reason="invalid_json",
                    json_data=None,
                )
            return LLMResponse(
                text=text,
                attempted=True,
                used_llm=True,
                latency_ms=elapsed_ms,
                error=None,
                fallback_reason=None,
                json_data=json_data,
            )
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

    def generate(self, prompt: str) -> LLMResponse:
        """Backward-compatible helper; prefer ``chat_json`` for recall."""

        schema = {
            "type": "object",
            "properties": {"spoken_answer": {"type": "string"}},
            "required": ["spoken_answer"],
            "additionalProperties": False,
        }
        return self.chat_json([{"role": "user", "content": prompt}], schema)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Some models wrap JSON in text despite schema mode. Recover only when a
        # single clear object exists; otherwise reject so validation can log why.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


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
