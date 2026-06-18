"""Browser chat server for Milestone 4.3.1.

This module intentionally uses only the Python standard library. It gives the
final demo a reliable browser input surface while keeping Python as the owner of
memory retrieval and LLM response generation. Godot remains display/embodiment:
a callback in ``backend.main`` sends the same existing UDP command shape after a
browser query is answered.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


@dataclass
class WebChatRequest:
    """Request object passed from the HTTP thread to the backend loop."""

    text: str
    created_at: float = field(default_factory=time.monotonic)
    reply_queue: "queue.Queue[dict[str, Any]]" = field(default_factory=lambda: queue.Queue(maxsize=1))

    def set_result(self, payload: dict[str, Any]) -> None:
        self.reply_queue.put(payload)


class WebChatServer:
    """Small threaded HTTP server for grounded recall chat."""

    def __init__(
        self,
        output_queue: "queue.Queue[object]",
        host: str = "127.0.0.1",
        port: int = 8765,
        request_timeout_s: float = 120.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_queue = output_queue
        self.host = host
        self.port = int(port)
        self.request_timeout_s = float(request_timeout_s)
        self.logger = logger or logging.getLogger("lelamp")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return

        parent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "LeLampWebChat/4.3.1"

            def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401 - stdlib override
                parent.logger.info("web_chat %s - " + fmt, self.address_string(), *args)

            def do_GET(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/index.html"}:
                    self._send_html(_HTML)
                    return
                if parsed.path == "/health":
                    self._send_json({"ok": True, "status": "backend connected", "url": parent.url})
                    return
                self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802 - stdlib API
                parsed = urlparse(self.path)
                if parsed.path != "/chat":
                    self._send_json({"ok": False, "error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json({"ok": False, "error": "invalid_content_length"}, status=HTTPStatus.BAD_REQUEST)
                    return
                if length <= 0 or length > 8192:
                    self._send_json({"ok": False, "error": "invalid_body_size"}, status=HTTPStatus.BAD_REQUEST)
                    return

                try:
                    raw = self.rfile.read(length).decode("utf-8")
                    payload = json.loads(raw)
                except Exception as exc:
                    self._send_json({"ok": False, "error": f"invalid_json: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                    return

                text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
                if not text:
                    self._send_json({"ok": False, "error": "empty_text"}, status=HTTPStatus.BAD_REQUEST)
                    return
                if len(text) > 1000:
                    self._send_json({"ok": False, "error": "text_too_long"}, status=HTTPStatus.BAD_REQUEST)
                    return

                request = WebChatRequest(text=text)
                parent.output_queue.put(request)
                try:
                    response_payload = request.reply_queue.get(timeout=parent.request_timeout_s)
                except queue.Empty:
                    parent.logger.warning("Browser chat request timed out text=%r", text)
                    self._send_json(
                        {
                            "ok": False,
                            "error": "backend_chat_timeout",
                            "answer": "The backend did not finish the recall request in time.",
                        },
                        status=HTTPStatus.GATEWAY_TIMEOUT,
                    )
                    return

                self._send_json(response_payload)

            def _send_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="web-chat-server", daemon=True)
        self._thread.start()
        self.logger.info("Browser chat server listening at %s", self.url)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.logger.info("Browser chat server stopped")
        self._server = None
        self._thread = None


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LeLamp Browser Chat</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #101114; color: #f4f4f5; }
    main { max-width: 920px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    .status { margin-bottom: 16px; color: #b7f7c1; font-size: 14px; }
    .panel { border: 1px solid #30323a; border-radius: 12px; background: #181a20; padding: 16px; }
    #history { min-height: 260px; max-height: 520px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
    .msg { padding: 10px 12px; border-radius: 10px; white-space: pre-wrap; line-height: 1.35; }
    .user { align-self: flex-end; max-width: 80%; background: #26395f; }
    .lamp { align-self: flex-start; max-width: 90%; background: #272a32; }
    .meta { margin-top: 8px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; color: #c7c7cc; white-space: pre-wrap; }
    form { display: flex; gap: 8px; margin-top: 16px; }
    input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #3a3d46; background: #0d0f13; color: #f4f4f5; font-size: 16px; }
    button { padding: 12px 16px; border-radius: 10px; border: 1px solid #626773; background: #e7e7ea; color: #111217; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: wait; }
    .hint { margin-top: 12px; color: #a6a6ad; font-size: 13px; }
    code { background: #22252d; padding: 1px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <main>
    <h1>LeLamp Browser Chat</h1>
    <div id="status" class="status">backend connected</div>
    <section class="panel">
      <div id="history"></div>
      <form id="form">
        <input id="text" autocomplete="off" placeholder="Where did you last see my phone?" autofocus />
        <button id="send" type="submit">Send</button>
      </form>
      <div class="hint">Use this browser page for the final demo. Do not type demo questions into the Godot text box or paste PowerShell into terminal recall.</div>
    </section>
  </main>
<script>
const historyEl = document.getElementById('history');
const formEl = document.getElementById('form');
const inputEl = document.getElementById('text');
const sendEl = document.getElementById('send');
const statusEl = document.getElementById('status');

function addMessage(kind, text, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + kind;
  const body = document.createElement('div');
  body.textContent = text;
  div.appendChild(body);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  historyEl.appendChild(div);
  historyEl.scrollTop = historyEl.scrollHeight;
}

function summarizeMeta(data) {
  const memory = data.memory_record || null;
  const retrieved = memory ? `${memory.normalized_label} @ ${memory.location_label} conf=${memory.confidence} time=${memory.timestamp}` : 'none';
  return [
    `answer_type=${data.answer_type || 'unknown'}`,
    `llm_used=${data.llm_used}`,
    `llm_attempted=${data.llm_attempted}`,
    `llm_error=${data.llm_error || 'none'}`,
    `llm_fallback_reason=${data.llm_fallback_reason || 'none'}`,
    `retrieved_memory=${retrieved}`
  ].join('\n');
}

formEl.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMessage('user', text);
  sendEl.disabled = true;
  statusEl.textContent = 'backend connected · waiting for recall answer...';
  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      addMessage('lamp', data.answer || 'Request failed.', `error=${data.error || response.status}`);
      statusEl.textContent = 'backend connected · last request failed';
    } else {
      addMessage('lamp', data.answer || '(empty answer)', summarizeMeta(data));
      statusEl.textContent = 'backend connected · last answer received';
    }
  } catch (err) {
    addMessage('lamp', 'Browser could not reach the backend chat endpoint.', `error=${err}`);
    statusEl.textContent = 'backend connection failed';
  } finally {
    sendEl.disabled = false;
    inputEl.focus();
  }
});
</script>
</body>
</html>
"""
