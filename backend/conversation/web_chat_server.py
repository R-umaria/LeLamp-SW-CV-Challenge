"""Non-blocking browser chat server for Milestone 4.3.2.

The browser is the final-demo text input surface. POST /chat enqueues a
question and returns immediately with a request_id. The UI then polls
GET /chat/result?request_id=... until the backend recall worker stores the
answer. Python remains the owner of recall, memory, and LLM calls; Godot only
receives the unchanged behavior command JSON for display/animation.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class WebChatRequest:
    """Request object passed from the HTTP thread to the backend loop."""

    text: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.monotonic)
    source: str = "browser"


class WebChatServer:
    """Small threaded HTTP server for non-blocking grounded recall chat."""

    def __init__(
        self,
        output_queue: "queue.Queue[object]",
        host: str = "127.0.0.1",
        port: int = 8765,
        request_timeout_s: float = 180.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_queue = output_queue
        self.host = host
        self.port = int(port)
        self.request_timeout_s = float(request_timeout_s)
        self.logger = logger or logging.getLogger("lelamp")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._results: dict[str, dict[str, Any]] = {}

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return

        parent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "LeLampWebChat/4.3.2"

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
                if parsed.path == "/chat/result":
                    params = parse_qs(parsed.query)
                    request_id = str((params.get("request_id") or [""])[0]).strip()
                    if not request_id:
                        self._send_json({"ok": False, "error": "missing_request_id"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    result = parent.get_result(request_id)
                    if result is None:
                        self._send_json({"ok": False, "error": "unknown_request_id"}, status=HTTPStatus.NOT_FOUND)
                        return
                    self._send_json(result)
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
                parent.mark_queued(request.request_id, text)
                parent.output_queue.put(request)
                parent.logger.info(
                    "recall_request_queued source=browser request_id=%s text=%r",
                    request.request_id,
                    text,
                )
                self._send_json(
                    {
                        "ok": True,
                        "status": "queued",
                        "request_id": request.request_id,
                        "answer": None,
                        "message": "Thinking...",
                    },
                    status=HTTPStatus.ACCEPTED,
                )

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

    def mark_queued(self, request_id: str, text: str) -> None:
        with self._lock:
            self._results[request_id] = {
                "ok": True,
                "request_id": request_id,
                "status": "queued",
                "answer": None,
                "user_query": text,
                "message": "Thinking...",
                "created_at": time.time(),
            }

    def mark_started(self, request_id: str) -> None:
        with self._lock:
            current = self._results.get(request_id)
            if current is not None and current.get("status") != "done":
                current["status"] = "running"
                current["message"] = "Thinking..."

    def set_result(self, request_id: str, payload: dict[str, Any]) -> None:
        final_payload = dict(payload)
        final_payload.setdefault("ok", True)
        final_payload["request_id"] = request_id
        final_payload["status"] = "done"
        with self._lock:
            previous = self._results.get(request_id, {})
            final_payload["created_at"] = previous.get("created_at", time.time())
            self._results[request_id] = final_payload
            self._prune_locked()

    def set_error(self, request_id: str, error: str, answer: str = "The recall request failed.") -> None:
        self.set_result(request_id, {"ok": False, "error": error, "answer": answer})

    def get_result(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            result = self._results.get(request_id)
            if result is None:
                return None
            return dict(result)

    def _prune_locked(self) -> None:
        if len(self._results) < 128:
            return
        cutoff = time.time() - max(300.0, self.request_timeout_s * 2.0)
        stale_ids = [rid for rid, payload in self._results.items() if float(payload.get("created_at", cutoff)) < cutoff]
        for rid in stale_ids:
            self._results.pop(rid, None)

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
    .pending { opacity: 0.78; font-style: italic; }
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
      <div class="hint">Final demo input lives here. Godot only displays the lamp state and recall answer.</div>
    </section>
  </main>
<script>
const historyEl = document.getElementById('history');
const formEl = document.getElementById('form');
const inputEl = document.getElementById('text');
const sendEl = document.getElementById('send');
const statusEl = document.getElementById('status');
const pollTimers = new Map();

function addMessage(kind, text, meta, extraClass) {
  const div = document.createElement('div');
  div.className = 'msg ' + kind + (extraClass ? ' ' + extraClass : '');
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
  return div;
}

function updateMessage(div, text, meta, extraClass) {
  div.className = 'msg lamp' + (extraClass ? ' ' + extraClass : '');
  div.innerHTML = '';
  const body = document.createElement('div');
  body.textContent = text;
  div.appendChild(body);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  historyEl.scrollTop = historyEl.scrollHeight;
}

function summarizeMeta(data) {
  const memory = data.memory_record || null;
  const retrieved = memory ? `${memory.normalized_label} @ ${memory.location_label} conf=${memory.confidence} time=${memory.timestamp}` : 'none';
  return [
    `request_id=${data.request_id || 'none'}`,
    `answer_type=${data.answer_type || 'unknown'}`,
    `llm_used=${data.llm_used}`,
    `llm_attempted=${data.llm_attempted}`,
    `llm_response_ms=${data.llm_response_ms ?? 'none'}`,
    `recall_worker_ms=${data.recall_worker_ms ?? 'none'}`,
    `llm_error=${data.llm_error || 'none'}`,
    `llm_fallback_reason=${data.llm_fallback_reason || 'none'}`,
    `retrieved_memory=${retrieved}`
  ].join('\n');
}

async function pollResult(requestId, pendingDiv, startedAt) {
  try {
    const response = await fetch(`/chat/result?request_id=${encodeURIComponent(requestId)}`);
    const data = await response.json();
    if (!response.ok || data.ok === false) {
      updateMessage(pendingDiv, data.answer || 'Request failed.', `request_id=${requestId}\nerror=${data.error || response.status}`);
      statusEl.textContent = 'backend connected · last request failed';
      pollTimers.delete(requestId);
      return;
    }
    if (data.status === 'queued' || data.status === 'running') {
      const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
      updateMessage(pendingDiv, `Thinking... (${elapsed}s)`, `request_id=${requestId}\nstatus=${data.status}`, 'pending');
      pollTimers.set(requestId, setTimeout(() => pollResult(requestId, pendingDiv, startedAt), 1000));
      return;
    }
    updateMessage(pendingDiv, data.answer || '(empty answer)', summarizeMeta(data));
    statusEl.textContent = 'backend connected · last answer received';
    pollTimers.delete(requestId);
  } catch (err) {
    updateMessage(pendingDiv, 'Browser could not reach the backend chat result endpoint.', `request_id=${requestId}\nerror=${err}`);
    statusEl.textContent = 'backend connection failed';
    pollTimers.delete(requestId);
  }
}

formEl.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMessage('user', text);
  sendEl.disabled = true;
  statusEl.textContent = 'backend connected · queued recall request';
  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok || data.ok === false || !data.request_id) {
      addMessage('lamp', data.answer || 'Request failed.', `error=${data.error || response.status}`);
      statusEl.textContent = 'backend connected · request failed';
      return;
    }
    const pendingDiv = addMessage('lamp', 'Thinking...', `request_id=${data.request_id}\nstatus=queued`, 'pending');
    pollResult(data.request_id, pendingDiv, Date.now());
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
