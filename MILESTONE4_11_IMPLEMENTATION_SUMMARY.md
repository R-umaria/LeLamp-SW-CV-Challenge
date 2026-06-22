# Milestone 4.11 Implementation Summary

Added optional Active Speaker Awareness + Directional Listening.

Key properties:

- Disabled by default; existing demo path is preserved.
- Audio capture uses a non-blocking `sounddevice` worker.
- VAD uses RMS energy over a rolling noise floor.
- Stereo DOA uses pure NumPy GCC-PHAT when enabled and stereo input is available.
- Face tracks are temporary `person_N` labels, not identity recognition.
- Active speaker fusion combines VAD, mouth motion, engagement, and optional DOA.
- Speaker policy emits bounded deterministic behavior overrides only above confidence threshold.
- Command protocol is extended only by optional top-level `speaker` field.
- Godot adds listening and bounded sound-seeking embodiment skills.
- Pure logic tests pass.

Run tests:

```bash
python -m pytest -q
```

Run single-mic demo:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --preview-flip-horizontal
```

Run stereo DOA demo:

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --enable-doa --mic-distance-m 0.08 --preview-flip-horizontal
```
