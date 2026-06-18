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
