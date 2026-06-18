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
