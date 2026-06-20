# Milestone 4.5: Small CV Preview Window

This patch keeps Python as the single owner of webcam perception and makes the optional OpenCV preview less intrusive during the Godot demo.

## Why not put the webcam directly inside Godot yet?

A true single-window demo is possible, but the safe implementation is to stream Python's already-processed preview into Godot. Directly opening the webcam from Godot is risky for this project because Python already needs the camera for engagement detection, object memory, and evaluation. On many systems only one process can reliably own the webcam at a time.

For the final challenge demo, the recommended path is:

1. Keep Python as the perception owner.
2. Keep Godot as the embodiment/simulator owner.
3. Make Python's preview small, movable, and display-only.
4. Add a future Python-to-Godot preview stream only after the perception/memory path remains stable.

## New behavior

The OpenCV preview now defaults to about half-size. With the default `640x480` camera input, the preview is about `320x240`, so it should no longer cover the Godot simulator.

The preview rendering is display-only:

- it does not change the frame used by engagement detection,
- it does not change object-memory coordinates,
- it does not change Godot face-follow coordinates,
- it does not change the backend-to-Godot command protocol.

## New CLI flags

```bash
python -m backend.main --godot-udp --enable-web-chat --show-window
```

Useful preview controls:

```bash
# Smaller preview, default behavior
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-scale 0.50

# Very small preview
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-scale 0.35

# Exact preview size
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-width 320 --preview-height 240

# Move preview to the upper-left corner
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-x 20 --preview-y 20

# Toggle horizontal display flip if your preview looks mirrored/reversed
python -m backend.main --godot-udp --enable-web-chat --show-window --preview-flip-horizontal

# Hide Python preview entirely for the cleanest Godot-only visual demo
python -m backend.main --godot-udp --enable-web-chat --no-window
```

## Mirroring note

The uploaded code did not intentionally mirror the camera feed with `cv2.flip()`. This patch adds an explicit display-only flip switch. Use `--preview-flip-horizontal` only when the preview looks reversed on your machine. The robot-follow behavior stays grounded in the original camera coordinates either way.

## Files changed

- `backend/main.py`
- `backend/utils/preview_window.py`
- `docs/milestone4_5_preview_window.md`
