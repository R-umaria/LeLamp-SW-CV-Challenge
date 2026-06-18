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
