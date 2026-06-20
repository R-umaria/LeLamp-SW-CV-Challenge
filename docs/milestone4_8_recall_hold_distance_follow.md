# Milestone 4.8: Recall Hold + Distance-Aware Attentive Tracking

This patch extends the Milestone 4.7 vertical-follow and recall-pointing behavior.

## Goals

1. Keep Lumos focused on the recalled object location for a visible duration instead of only a brief instant.
2. Return automatically to normal attentive head tracking after recall feedback finishes.
3. Make pre-sleep scanning cover the same left/right extremes as head tracking for about 3–6 seconds.
4. Make attentive tracking adjust apparent distance from the user using the detected face box size.

## Recall feedback hold

The backend now exposes:

```bash
--recall-point-hold-seconds 5.0
```

When a recall answer completes, the backend sends the final recall command immediately, then continues sending the same bounded feedback motion for the hold window:

- Found memory: `motion=recall_point`, `light=pointer_spot`
- No recent memory: `motion=recall_not_found`, `light=sad_dim`

Only the first command includes the answer text. Follow-up hold commands keep the pose and light active without repeatedly re-triggering speech or the response panel.

After the hold expires, the backend stops overriding the normal FSM output. If the user is still engaged, Lumos naturally returns to `attentive_follow` and resumes horizontal/vertical face tracking.

## Distance-aware attentive tracking

The engagement protocol now includes `engagement.face_area_ratio` whenever a face is detected. Godot uses that value only as an embodiment hint:

- Small face area: Lumos slides slightly toward the camera.
- Very large face area: Lumos slides slightly away from the camera.
- Comfortable face area: Lumos stays near its home position.

This is intentionally implemented in Godot because it is expressive body placement, not a perception or memory decision. Gesture approach/retreat and recall pointing suppress automatic distance correction so deliberate behaviors remain clear.

## Wider pre-sleep scan

`sleepy_search_then_rest` now uses wall-clock elapsed time rather than globally slowed animation time. This ensures the pre-sleep scan reaches the far left/right regions before Lumos curls into the sleep posture.

## Primary files changed

- `backend/main.py`
- `backend/perception/engagement_detector.py`
- `frontend_godot/scripts/LampController.gd`
- `frontend_godot/scripts/Main.gd`
- `frontend_godot/scripts/lumos/MotionSkillLibrary.gd`

## Recommended run command

```powershell
python -m backend.main --show-window --godot-udp --enable-gestures --enable-objects --enable-web-chat --preview-flip-horizontal --recall-lookback-hours 24 --recall-point-hold-seconds 5
```

## Validation

Validated Python syntax and recall parser/guardrail tests:

```bash
python -m compileall -q backend
python -m backend.main --help
python -m unittest backend.conversation.test_query_parser backend.conversation.test_intent_parser backend.conversation.test_recall_llm_guardrails -v
```

Godot scripts were edited statically in this environment. Open the project in Godot to confirm runtime animation feel and tune exported values on `LampRig` if needed.
