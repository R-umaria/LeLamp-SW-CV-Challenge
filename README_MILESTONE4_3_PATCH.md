# Milestone 4.3 Patch — Real Conversation + LLM Repair + Live Chat + Sleep Mode

This patch repairs the Milestone 4.2.1 recall path instead of adding another fallback-only layer.

## What changed

### Backend conversation path

- Replaced raw Ollama `/api/generate` phrasing with `/api/chat` structured output.
- Added `RECALL_RESPONSE_SCHEMA` validation with these fields:
  - `answer_type`
  - `target_object`
  - `location_label`
  - `confidence`
  - `timestamp`
  - `spoken_answer`
- Added strict validation so the LLM is accepted only when it matches retrieved SQLite memory.
- Added `--llm-required` to the recall CLI. If a memory answer is LLM-eligible and the LLM fails validation, the CLI returns nonzero.
- Added deterministic fallback, but fallback is logged as fallback and never counted as `llm_used=true`.

### Intent parsing

- Added `backend/conversation/intent_parser.py`.
- Supported intents:
  - `object_last_seen`
  - `list_recent_objects`
  - `unsupported`
- Fixed these regressions:
  - `Did you see if I had a stapler?` parses as `object_last_seen/stapler`, not `if`.
  - `What objects did you detect?` parses as `list_recent_objects`, not `detected`.
  - `What do you remember seeing?` parses as `list_recent_objects`.

### Live Godot chat

- Added `backend/conversation/chat_udp_receiver.py`.
- Backend listens for Godot chat JSON on UDP port `4243` by default:

```json
{
  "type": "chat_query",
  "timestamp": "2026-06-18T13:30:00",
  "text": "Where did you last see my phone?"
}
```

- Godot now has a live chat panel with:
  - chat history
  - text input
  - Send button
  - Enter-to-send
- Backend answers through the existing backend-to-Godot command JSON shape using:
  - `state=recalling`
  - `behavior.motion=thinking`
  - `behavior.light=focus_glow`
  - `behavior.speech_text=<grounded answer>`

### Godot stale-backend sleep mode

- Godot tracks the local time of the last backend command packet.
- If no backend command arrives for `60` seconds, Godot enters local sleep mode:
  - low resting pose
  - dim red light
  - no active searching/nodding
  - debug UI says `backend stale / sleeping`
- New backend command packets immediately exit sleep mode.
- This is local connection-health handling only; Godot still does not make engagement, memory, or recall decisions.

## Changed files

- `backend/main.py`
- `backend/conversation/__init__.py`
- `backend/conversation/chat_udp_receiver.py`
- `backend/conversation/intent_parser.py`
- `backend/conversation/llm_client.py`
- `backend/conversation/query_parser.py`
- `backend/conversation/recall_agent.py`
- `backend/conversation/test_chat_udp_receiver.py`
- `backend/conversation/test_intent_parser.py`
- `backend/conversation/test_query_parser.py`
- `backend/conversation/test_recall_llm_guardrails.py`
- `backend/memory/memory_store.py`
- `frontend_godot/scripts/Main.gd`
- `frontend_godot/scripts/LampController.gd`

## PowerShell validation commands

Run these from the project root.

### 1. Run parser/conversation tests

```powershell
python -m unittest `
  backend.conversation.test_query_parser `
  backend.conversation.test_intent_parser `
  backend.conversation.test_recall_llm_guardrails `
  backend.conversation.test_chat_udp_receiver `
  -v
```

Expected: all tests pass. This reproduces the previous bad LLM output and verifies it is rejected with `contradicts_retrieved_memory`.

### 2. Pull/check Ollama model

In one terminal, start Ollama if it is not already running:

```powershell
ollama serve
```

In another terminal:

```powershell
ollama pull llama3.2
python -m backend.conversation.llm_client --ollama-url http://localhost:11434 --ollama-model llama3.2 --json
```

Expected: `ok=true` and `model_available=true`.

### 3. Run recall with `--llm-required`

```powershell
python -m backend.conversation.recall_agent "Where did you last see my bottle?" `
  --memory-db data/scene_memory.sqlite `
  --use-llm `
  --llm-required `
  --ollama-url http://localhost:11434 `
  --ollama-model llama3.2 `
  --json
```

Expected when a matching phone row exists and Ollama returns valid grounded JSON:

- `answer_type="memory_answer"`
- `parsed_object="phone"`
- `retrieved_memory_id` is not null
- `llm_attempted=true`
- `llm_used=true`
- `llm_error=null`

If the LLM says it does not remember while the SQLite record exists, the CLI exits nonzero and logs `contradicts_retrieved_memory`.

### 4. Run backend with Godot chat UDP receiver

```powershell
python -m backend.main `
  --show-window `
  --godot-udp `
  --enable-objects `
  --interactive-recall `
  --enable-godot-chat `
  --chat-port 4243 `
  --use-llm `
  --llm-required `
  --ollama-url http://localhost:11434 `
  --ollama-model llama3.2 `
  --memory-db data/scene_memory.sqlite `
  --object-model yolov8n.pt `
  --save-object-frames
```

Then run the Godot project and ask in the chat panel:

```text
Where did you last see my phone?
```

### 5. Optional manual UDP chat packet test

Use this if you want to test the backend receiver without typing in Godot:

```powershell
$payload = @{
  type = "chat_query"
  timestamp = (Get-Date).ToString("s")
  text = "What objects did you detect?"
} | ConvertTo-Json -Compress

$client = New-Object System.Net.Sockets.UdpClient
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
[void]$client.Send($bytes, $bytes.Length, "127.0.0.1", 4243)
$client.Close()
```

## Demo pass/fail criteria

PASS if Godot chat can ask `Where did you last see my phone?` while the backend is running and receives a valid LLM-grounded answer with `llm_used=true` in `logs/latest/recall.jsonl`.

PASS if `What objects did you detect?` lists recent unique objects retrieved from SQLite.

PASS if stopping the backend causes Godot to enter local sleep mode after 60 seconds and a new backend command wakes it immediately.

FAIL if normal object recall only works by stopping the backend and running CLI commands.

FAIL if the LLM says no memory when SQLite has a matching record and that answer is accepted.

FAIL if `stapler` parses as `if` or `What objects did you detect?` parses as `detected`.
