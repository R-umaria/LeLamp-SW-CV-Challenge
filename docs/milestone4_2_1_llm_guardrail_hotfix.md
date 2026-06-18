# Milestone 4.2.1 — LLM Grounding Guardrail Hotfix

## Problem observed

A recall run successfully retrieved a SQLite memory record for `phone` and Ollama returned text, so the result reported `llm_used=true`. However, the generated answer was:

```text
I do not remember seeing that.
```

That contradicted the retrieved memory record. The issue was not memory retrieval. The issue was that the recall agent accepted any non-empty Ollama response as used LLM output.

## Fix

The recall agent now validates every LLM answer before displaying it or sending it to Godot.

An LLM answer is accepted only if it:

- does not contradict the retrieved memory record,
- includes the stored `location_label`,
- refers to the target object,
- includes either the stored confidence or timestamp,
- does not leak frame paths, JSON, SQLite, memory IDs, or implementation details.

If validation fails:

- `llm_attempted=true`,
- `llm_used=false`,
- `llm_error=invalid_llm_response:<reason>`,
- `llm_fallback_reason=<reason>`,
- the deterministic grounded template is used.

This preserves the demo rule: the lamp may use Ollama for wording, but it must never display an ungrounded or contradictory response.

## Prompt change

The LLM prompt now explicitly states that a valid SQLite record was retrieved and includes the deterministic grounded answer as the preferred answer. The prompt no longer gives the model permission to say it does not remember when a record exists.

## Test commands

```powershell
python -m compileall -q backend
python -m unittest backend.conversation.test_query_parser backend.conversation.test_recall_llm_guardrails -v
python -m backend.conversation.recall_agent "Where did you last see my phone?" --memory-db data/scene_memory.sqlite --json
python -m backend.conversation.recall_agent "Where did you last see my phone?" --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model llama3.2:1b --json
```

## Expected behavior

If Ollama returns a valid grounded sentence:

```json
"llm_attempted": true,
"llm_used": true,
"llm_error": null,
"llm_fallback_reason": null
```

If Ollama returns a contradictory sentence such as `I do not remember seeing that.` when a memory record exists:

```json
"llm_attempted": true,
"llm_used": false,
"llm_error": "invalid_llm_response: contradicts_retrieved_memory",
"llm_fallback_reason": "contradicts_retrieved_memory",
"answer": "I last saw your phone on the center of view around 13:22:33 on 2026-06-18. My confidence was 0.73."
```

## Final demo recommendation

For the final demo, deterministic fallback remains the safest path. Use `--use-llm` only after confirming that the model produces validated grounded answers. A local LLM is optional polish, not a dependency for the challenge demo.
