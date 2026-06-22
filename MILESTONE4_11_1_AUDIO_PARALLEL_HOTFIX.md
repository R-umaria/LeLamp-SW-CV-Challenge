# Milestone 4.11.1 Audio + Parallel Speaker Awareness Hotfix

## Problem found in user logs

The run logs showed:

```text
speaker_awareness_disabled_reason reason=sounddevice_unavailable: No module named 'sounddevice'
```

So microphone capture never started. As a direct consequence, VAD never produced a `VoiceActivityResult`, and active speaker fusion repeatedly emitted:

```text
reason=no_voice_activity_result
```

The same logs also showed speaker fusion running inside the webcam loop with repeated `fusion_ms` around 50-140 ms, which explains preview lag.

## Fixes implemented

- Added `backend/perception/speaker_awareness_worker.py`.
- Moved face tracking, mouth motion, and active-speaker fusion to a latest-frame-only worker thread.
- Added frame downscaling for speaker analysis with `--speaker-frame-width`.
- Added `--speaker-fusion-interval` so speaker fusion can run at a controlled rate.
- Throttled voice/DOA logs to reduce disk/console I/O load.
- Improved audio capture startup:
  - logs install hint when `sounddevice` is missing,
  - tries stereo first for DOA,
  - falls back to mono VAD if stereo fails,
  - tries device default sample rate if requested rate fails.
- Added `--list-audio-devices` to diagnose microphone input.
- Anchored speaker face tracking to the engagement detector's selected face bbox to reduce false positive speaker tracks.

## Required setup

```powershell
python -m pip install -r backend/requirements.txt
```

Then:

```powershell
python -m backend.main --list-audio-devices
```

## Recommended run

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --preview-flip-horizontal --speaker-fusion-interval 0.25 --speaker-frame-width 640
```

## Stereo DOA run

```powershell
python -m backend.main --show-window --godot-udp --enable-objects --enable-web-chat --enable-audio --enable-speaker-awareness --enable-doa --mic-distance-m 0.08 --preview-flip-horizontal --speaker-fusion-interval 0.25 --speaker-frame-width 640
```

## Validation in this environment

```text
python -m compileall -q backend
python -m pytest -q
53 passed
```

Real microphone, stereo DOA, webcam, and Godot runtime cannot be fully validated in this sandbox.
