# Milestone 4.3.1 — Browser Chat + Homelab Ollama + Longer Timeout

## Purpose

Milestone 4.3.1 removes final-demo dependence on Godot text input and terminal interactive recall. The primary recall input surface is now a browser page served by the Python backend.

Python remains responsible for:

- SQLite memory retrieval
- deterministic intent and object parsing
- optional Ollama `/api/chat` usage
- structured LLM validation
- deterministic fallback when the LLM is unavailable or invalid
- sending the existing backend-to-Godot UDP command

Godot remains animation/display only. The backend-to-Godot command JSON shape is unchanged.

## New/changed runtime options

- `--enable-web-chat`
- `--web-chat-host 127.0.0.1`
- `--web-chat-port 8765`
- `--llm-timeout 90`
- `--llm-connect-timeout 5`
- `--ollama-keep-alive 1h`
- `--ollama-model qwen2.5:1.5b`
- `--ollama-url http://10.0.0.70:11434`

`--ollama-timeout` is still accepted as a backward-compatible alias for `--llm-timeout`, but the final demo command should use `--llm-timeout`.

## Browser chat behavior

Open:

```text
http://127.0.0.1:8765
```

The browser UI provides:

- text input
- Send button
- Enter-to-send
- chat history
- backend status line
- answer display
- `llm_used`
- `llm_attempted`
- `llm_error`
- `llm_fallback_reason`
- retrieved memory information when available

The browser sends:

```json
{
  "text": "Where did you last see my phone?"
}
```

to:

```text
POST /chat
```

The backend answers through the same `RecallAgent` path and sends this existing Godot command shape:

```json
{
  "timestamp": "...",
  "state": "recalling",
  "engagement": {
    "status": "...",
    "confidence": 0.0,
    "reason": "..."
  },
  "behavior": {
    "motion": "thinking",
    "light": "focus_glow",
    "sound": null,
    "speech_text": "..."
  },
  "memory": {
    "last_detected_objects": []
  }
}
```

## Interactive recall warning

`--interactive-recall` is now debug-only. Do not paste PowerShell commands into the backend recall prompt. The input thread prints a warning and ignores shell-looking lines, but the final demo should use browser chat instead.

## Ollama homelab setup

Use:

```text
http://10.0.0.70:11434
```

Use model:

```text
qwen2.5:1.5b
```

The client uses `/api/chat`, `stream=false`, `keep_alive="1h"`, and low temperature.

## PowerShell validation commands

Run parser, guardrail, UDP chat, and browser chat tests:

```powershell
python -m unittest `
  backend.conversation.test_query_parser `
  backend.conversation.test_intent_parser `
  backend.conversation.test_recall_llm_guardrails `
  backend.conversation.test_chat_udp_receiver `
  backend.conversation.test_web_chat_server `
  -v
```

Expected output:

```text
Ran 32 tests ... OK
```

Check homelab model availability:

```powershell
python -m backend.conversation.recall_agent --test-ollama --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --llm-timeout 90 --llm-connect-timeout 5 --ollama-keep-alive 1h --json
```

Expected successful JSON includes:

```json
{
  "ok": true,
  "base_url": "http://10.0.0.70:11434",
  "model": "qwen2.5:1.5b",
  "model_available": true,
  "error": null
}
```

Warm the model:

```powershell
python -m backend.conversation.recall_agent --warm-ollama --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --llm-timeout 90 --llm-connect-timeout 5 --ollama-keep-alive 1h --json
```

Expected successful JSON includes:

```json
{
  "ok": true,
  "model_available": true,
  "warm_up": {
    "attempted": true,
    "used_llm": true,
    "error": null
  }
}
```

Test recall with LLM required:

```powershell
python -m backend.conversation.recall_agent "Where did you last see my phone?" --memory-db data/scene_memory.sqlite --use-llm --llm-required --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --llm-timeout 90 --llm-connect-timeout 5 --ollama-keep-alive 1h --json
```

Expected success when matching SQLite memory exists and the LLM validates:

```json
{
  "answer_type": "memory_answer",
  "parsed_object": "phone",
  "llm_attempted": true,
  "llm_used": true,
  "llm_required_failed": false
}
```

Expected failure when `--llm-required` is set and the LLM times out, is unavailable, or fails validation:

```text
process exit code: 1
```

Run final backend demo:

```powershell
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242 --enable-objects --object-model yolov8n.pt --save-object-frames --memory-db data/scene_memory.sqlite --enable-web-chat --web-chat-host 127.0.0.1 --web-chat-port 8765 --use-llm --llm-required --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --llm-timeout 90 --llm-connect-timeout 5 --ollama-keep-alive 1h
```

## Final demo procedure

1. Start Godot first so it listens on UDP port `4242`.
2. Warm/check homelab Ollama with the commands above.
3. Start the backend with `--enable-web-chat` and the final demo command.
4. Open `http://127.0.0.1:8765` in a browser.
5. Show the webcam/object memory loop running.
6. Ask in the browser: `Where did you last see my phone?`
7. Confirm the browser shows `llm_used=true` and Godot displays the recall answer through `speech_text`.
8. Ask: `What objects did you detect?`
9. Confirm the answer lists recent unique SQLite objects and does not parse `detected` as an object.
10. Ask: `Was there any pen in the view?`
11. Confirm it parses `pen` and answers no memory unless a pen record exists.

## Pass/fail criteria

PASS if browser chat can ask questions while the backend is running.

PASS if Godot receives `state=recalling`, `behavior.motion=thinking`, `behavior.light=focus_glow`, and `behavior.speech_text` from browser chat questions.

PASS if homelab Ollama produces `llm_used=true` for normal memory questions with existing SQLite records.

PASS if `--llm-timeout` and `--llm-connect-timeout` are configurable and long enough for homelab inference.

PASS if `What objects did you detect?` returns recent SQLite objects.

FAIL if the backend must be stopped to ask recall questions.

FAIL if the final demo depends on typing inside the Godot UI text box.

FAIL if PowerShell script lines are interpreted as recall questions during the final demo.
