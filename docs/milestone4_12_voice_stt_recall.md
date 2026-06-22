# Milestone 4.12: Voice STT Recall

This patch adds a bounded speech-to-text path for Lumos voice recall.

## Goal

Replace the browser-chat surface for the demo-critical recall question path:

> "Did you see my phone?"

The first voice milestone does **not** add a general LLM conversation agent. It only converts speech into text and routes accepted transcripts into the already-grounded recall worker. The recall worker still retrieves from SQLite memory and the Godot frontend still receives the same controlled `recall_point` / `recall_not_found` command payloads.

## Data flow

```text
microphone -> audio capture -> VAD -> active speaker awareness
          -> gated utterance segmenter -> local STT worker
          -> transcript -> RecallWorker -> SceneMemory SQLite
          -> Godot recall command / pointing behavior
```

## Safety / scope decision

STT is gated by active speaker awareness. Lumos records only when:

1. microphone audio is active,
2. a visible speaker is likely talking,
3. the speaker appears to be talking to Lumos,
4. confidence is above `--stt-speaker-min-confidence`.

This prevents background speech from constantly triggering recall queries.

## Local STT backend

The default backend is `faster-whisper` with `tiny.en` on CPU/int8:

```powershell
python -m pip install -r backend/requirements.txt
```

The first STT run may download the Whisper model. After that, it runs locally.

## Recommended run command

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-audio --enable-speaker-awareness --enable-stt --preview-flip-horizontal --speaker-fusion-interval 0.25 --speaker-frame-width 640 --speaker-policy-hold-s 1.15 --stt-model-size tiny.en
```

Then ask:

```text
Did you see my phone?
```

If a phone has been stored in scene memory, Lumos should point toward the last remembered location. If no phone memory exists, it should answer that it does not remember seeing it.

## Useful STT flags

```text
--enable-stt
--stt-backend faster_whisper
--stt-model-size tiny.en
--stt-device cpu
--stt-compute-type int8
--stt-min-utterance-s 0.55
--stt-max-utterance-s 6.0
--stt-end-silence-s 0.85
--stt-speaker-min-confidence 0.45
--stt-cooldown-s 1.25
```

For a better transcript at the cost of speed, try:

```text
--stt-model-size base.en
```

For NVIDIA GPU systems with CUDA configured, try:

```text
--stt-device cuda --stt-compute-type float16
```

## Logs

Each run writes:

```text
logs/runs/<run_id>/stt_events.jsonl
logs/latest/stt_events.jsonl
```

Important events:

```text
stt_utterance
stt_transcript
voice_recall_transcript_queued
recall_request_queued source=voice_stt
```

## Limitations

- This patch transcribes only short utterances, not continuous conversation.
- It does not use an LLM to interpret arbitrary commands yet.
- Whisper model loading can take time on the first utterance.
- False VAD triggers can still happen in noisy rooms, but the active-speaker gate and transcript filters reduce accidental recall.
- If the transcript is wrong, the deterministic recall parser may ask for the wrong object or reject the query.

## Next milestone

Milestone 4.13 should add a bounded voice understanding layer:

```text
STT transcript -> deterministic intent parser -> optional LLM intent rewriter -> bounded action/recall commands
```

The LLM should never directly control motion. It should only map natural speech into the existing safe command interfaces.
