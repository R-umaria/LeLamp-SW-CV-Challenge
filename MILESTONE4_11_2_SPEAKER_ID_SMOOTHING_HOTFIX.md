# Milestone 4.11.2: Speaker ID + Listening Smoothing Hotfix

## Why this patch exists

The 4.11.1 run confirmed that active speaker awareness now works, but the logs exposed three UX issues:

1. A single visible user could be exposed as `person_4` or `person_5` after face tracking dropped/reacquired.
2. Listening behavior could flicker between `active_listen/listening_blue` and normal `attentive_follow/steady_warm` during short speech-confidence dips.
3. Speech heard without an identified visible mouth needed to trigger sound-seeking at a lower confidence threshold, especially when stereo DOA is available.

## Changes

### Presentation IDs

Internal face tracks still keep continuity for mouth-motion smoothing, but public speaker IDs are now presentation labels:

- one visible person -> `person_1`
- two visible people -> `person_1`, `person_2` left-to-right
- no visible speaker -> `active_track_id=null`

This prevents confusing `person_5` labels in a single-user demo. `person_2` appears only when two credible visible face tracks are present.

### Secondary face filtering

Secondary Haar face candidates must exceed `secondary_face_min_area_ratio` before becoming visible speaker tracks. This suppresses common false positives on monitors, chairs, wall texture, or windows.

CLI tuning:

```powershell
--speaker-secondary-min-area 0.018
```

Raise it to `0.025` if false `person_2` tracks appear. Lower it only if a real second person is farther away.

### Listening behavior hold

Speaker-aware behavior is now held briefly after a high-confidence override. This makes the lamp stay in listening mode through short syllable gaps instead of blinking blue for one command and immediately returning to face-follow.

CLI tuning:

```powershell
--speaker-policy-hold-s 1.15
```

Increase to `1.5` for a calmer demo. Decrease to `0.6` for faster recovery.

### Sound-seeking confidence threshold

When speech is detected but no visible speaker is identified, the sound-seeking behavior now uses a separate lower threshold:

```powershell
--speaker-seek-min-confidence 0.30
```

This allows `sound_seek_left/right/center` to trigger when speech is heard elsewhere, without requiring the higher confidence used for “this visible user is talking to Lumos.”

### VAD confidence smoothing

The VAD now carries recent confidence through short speech gaps. This avoids confusing states such as:

```json
{"speech_detected": true, "audio_confidence": 0.0}
```

## Speech understanding note

This milestone detects who is probably speaking and whether they appear to be talking to Lumos. It does **not** transcribe the words. Understanding spoken content requires a separate speech-to-text module, such as local Whisper, Vosk, or platform speech recognition. The recommended next milestone is optional local STT gated by `speaking_to_robot=True`.

## Recommended command

Single mic:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --preview-flip-horizontal --speaker-fusion-interval 0.25 --speaker-frame-width 640 --speaker-policy-hold-s 1.15 --speaker-secondary-min-area 0.018
```

Stereo DOA:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --enable-doa --mic-distance-m 0.08 --preview-flip-horizontal --speaker-fusion-interval 0.25 --speaker-frame-width 640 --speaker-policy-hold-s 1.15 --speaker-seek-min-confidence 0.30 --speaker-secondary-min-area 0.018
```

## Tests

```text
python -m compileall -q backend
python -m pytest -q
55 passed
```
