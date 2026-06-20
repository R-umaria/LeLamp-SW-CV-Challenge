# Milestone 4.7: Vertical Follow + Recall Pointing

## Goal

This patch improves Lumos in two demo-visible areas:

1. **Full-screen face following**: Lumos now uses both `face_x_norm` and `face_y_norm` from the backend command protocol. Horizontal face position controls base/wrist yaw. Vertical face position controls shoulder pitch, wrist pitch, and lamp-head tilt.
2. **Embodied memory recall**: when a grounded recall answer finds an object in memory, Lumos points toward the remembered screen region and brightens its lamp. If no recent memory exists, Lumos performs a depressed “no” motion.

Python remains the intelligence layer. Godot only renders bounded motion/light skills and optional display hints.

## Backend changes

### Tracking range

`EngagementConfig` and backend CLI defaults were widened:

- `--center-tolerance-x`: `0.46`
- `--center-tolerance-y`: `0.44`

This keeps a visible user trackable across nearly the full webcam frame, instead of entering attention-seeking when the face moves slightly outside the previous central box.

### Recall lookback

Main backend recall now defaults to a 24-hour memory window:

```bash
--recall-lookback-hours 24
```

Use `--recall-lookback-hours 0` for all-time recall.

### Recall feedback protocol

Completed recall commands may now include:

```json
"memory": {
  "last_detected_objects": [],
  "recall_target": {
    "found": true,
    "object_label": "phone",
    "location_label": "lower left side of view",
    "confidence": 0.88,
    "timestamp": "2026-06-20T20:37:44",
    "bbox": [10, 300, 50, 50],
    "point_x_norm": 0.18,
    "point_y_norm": 0.82
  }
}
```

For no-memory answers:

```json
"recall_target": {
  "found": false,
  "object_label": "phone",
  "reason": "no_recent_memory"
}
```

## Godot changes

### Full-screen follow

`frontend_godot/scripts/LampController.gd` now maintains separate smoothed follow signals:

- `_smoothed_face_follow_x_deg`
- `_smoothed_face_follow_y_deg`

The horizontal signal drives:

- base yaw
- wrist yaw

The vertical signal drives:

- shoulder pitch
- wrist pitch
- lamp-head tilt

### Recall pointing

New motion/light skills:

- `recall_point` + `pointer_spot`: lean and brighten toward the remembered object region.
- `recall_not_found` + `sad_dim`: lean down and perform a wrist/head “no” shake.

## Recommended run command

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --enable-objects --enable-web-chat --preview-flip-horizontal --recall-lookback-hours 24
```

## Validation

Validated in this patch:

- `python -m compileall -q backend`
- `python -m backend.main --help`
- `python -m unittest backend.conversation.test_query_parser backend.conversation.test_intent_parser backend.conversation.test_recall_llm_guardrails -v`
- Manual smoke test for 24-hour recall feedback payloads.

Godot editor execution was not available in this environment, so the Godot scripts are syntax-reviewed but not editor-run here.
