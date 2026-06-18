# Milestone 4.2 Final Demo Polish

Milestone 4.2 is a stability and presentation patch. It does not change the backend command JSON protocol and does not add voice input.

## What changed

### Godot frontend

- Moved the default `Camera3D` to a front-three-quarter view of the lamp.
- Preserved all 6-DOF animation axes.
- Added `FillLight3D` so arm joints and lamp gestures remain readable.
- Added a small `FrontLookMarker` and `LookDirectionTip` on the lamp head so it is obvious where the lamp is looking.
- Kept the Milestone 4.1 recall answer panel visible during `state=recalling` commands.

### Backend

No backend code changes are required for Milestone 4.2. The existing Milestone 4.1 recall diagnostics remain the source of truth for LLM status:

- `llm_requested`
- `llm_attempted`
- `llm_used`
- `llm_error`
- `llm_fallback_reason`
- `llm_response_ms`
- `ollama_connection_ok`

## Ollama PowerShell verification sequence

Run these commands from the project root unless noted otherwise.

### 1. Check whether the Ollama CLI is installed

```powershell
ollama --version
```

Expected if installed:

```text
ollama version is printed
```

If PowerShell says `ollama` is not recognized, install Ollama first and reopen the terminal.

### 2. Make sure the Ollama service is running

Usually Ollama runs as a background service. If not, start it in a separate terminal:

```powershell
ollama serve
```

Leave that terminal open.

### 3. Pull a small demo model

```powershell
ollama pull llama3.2:1b
ollama list
```

Expected:

```text
llama3.2:1b appears in the model list
```

### 4. Test LeLamp Ollama connectivity

Either command is acceptable:

```powershell
python -m backend.conversation.llm_client --ollama-url http://localhost:11434 --ollama-model llama3.2:1b
```

```powershell
python -m backend.conversation.recall_agent --test-ollama --ollama-url http://localhost:11434 --ollama-model llama3.2:1b
```

Expected success:

```text
Ollama OK: http://localhost:11434 has model llama3.2:1b
```

Expected fallback case:

```text
Ollama unavailable: <connection error>
```

or:

```text
Ollama reachable, but model 'llama3.2:1b' was not found.
```

### 5. Confirm recall can report `llm_used=True`

Make sure `data/scene_memory.sqlite` contains at least one `phone` memory record. Then run:

```powershell
python -m backend.conversation.recall_agent "Where did you last see my phone?" --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model llama3.2:1b --json
```

Expected success keys:

```json
{
  "parsed_object": "phone",
  "llm_requested": true,
  "llm_attempted": true,
  "llm_used": true,
  "llm_error": null,
  "llm_fallback_reason": null
}
```

If Ollama is unavailable, expected fallback keys:

```json
{
  "llm_requested": true,
  "llm_attempted": false,
  "llm_used": false,
  "llm_fallback_reason": "connectivity_check_failed"
}
```

The answer remains grounded either way. The LLM is only allowed to phrase a retrieved SQLite memory record. It is not allowed to invent an object location.

## Backend final demo command

Start Godot first. Then run this from the project root:

```powershell
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242 --enable-objects --object-model yolov8n.pt --save-object-frames --interactive-recall --memory-db data/scene_memory.sqlite
```

Use deterministic recall for the most stable demo. Add the LLM flags only after `--test-ollama` succeeds:

```powershell
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242 --enable-objects --object-model yolov8n.pt --save-object-frames --interactive-recall --memory-db data/scene_memory.sqlite --use-llm --ollama-url http://localhost:11434 --ollama-model llama3.2:1b
```

## Recommended final demo script

1. Start Godot and verify the lamp is visible from the front-three-quarter view.
2. Start the backend with the final demo command.
3. Look at the webcam/lamp.
   - Expected backend state: `engaged`.
   - Expected Godot behavior: `attentive_nod`, steady warm light.
4. Look away.
   - Expected backend state: `disengaged` after the configured stability window.
   - Expected Godot behavior: `searching_glance`, slow pulse.
5. Stay disengaged.
   - Expected backend state: `seeking_attention` after sustained disengagement.
   - Expected Godot behavior: `curious_tilt`, soft pulse.
6. Place or show desk objects such as phone, cup, mouse, laptop, or monitor.
   - Expected backend logs: YOLO detections and SQLite memory writes.
   - Expected command JSON: `memory.last_detected_objects` populated.
7. In the interactive recall prompt, type:

```text
Where did you last see my phone?
```

8. Verify recall behavior.
   - Expected backend retrieval: latest SQLite `phone` record.
   - Expected command JSON: `state=recalling`, `motion=thinking`, `light=focus_glow`, `speech_text=<grounded answer>`.
   - Expected Godot behavior: thinking pose, focus glow, visible recall answer panel.

## Final sanity checklist

### Godot visual readiness

- [ ] Lamp is viewed from the front-three-quarter angle.
- [ ] Head, shade, light cone, and front marker are visible.
- [ ] Arm joints are visible during motion.
- [ ] `idle_breathe` reads as subtle breathing.
- [ ] `attentive_nod` reads as engagement.
- [ ] `searching_glance` reads as looking around.
- [ ] `curious_tilt` plus `soft_pulse` reads as attention-seeking.
- [ ] `thinking` plus `focus_glow` reads as recall/thinking.
- [ ] Recall answer panel remains visible long enough to read.

### Backend recall readiness

- [ ] `python -m backend.conversation.recall_agent --test-ollama ...` gives a clear success or fallback diagnostic.
- [ ] Deterministic fallback works when Ollama is unavailable.
- [ ] `frame_path` is not included in spoken/displayed answers.
- [ ] `frame_path` remains present in JSON/debug metadata.
- [ ] The LLM never receives a prompt unless a real SQLite memory record exists.
- [ ] Missing-object answers say the lamp does not remember seeing the object.

### Demo stability readiness

- [ ] Engagement detection still transitions `engaged -> disengaged -> seeking_attention`.
- [ ] Object detection does not block engagement updates badly enough to break the demo.
- [ ] SQLite memory writes are visible in logs.
- [ ] Interactive recall can be typed while the backend is running.
- [ ] Godot receives recall commands over UDP.
- [ ] `logs/latest/commands.jsonl`, `logs/latest/runtime.log`, and `logs/latest/latency.csv` are created.

## Pass/fail criteria

PASS if the final demo shows all four challenge requirements: engagement detection, attention-seeking behavior, memory formation, and grounded memory recall.

PASS if Godot clearly shows the lamp's face/front marker, light cone, gestures, and recall answer panel.

PASS if `--use-llm` produces `llm_used=True` when Ollama and the requested model are running, and produces an explicit fallback reason otherwise.

PASS if deterministic fallback remains grounded and acceptable when Ollama is not running.

FAIL if the lamp still appears to face away from the demo camera.

FAIL if recall answers mention a location not present in SQLite memory.

FAIL if the command JSON protocol changes.

FAIL if the demo depends on voice input or an unavailable LLM.
