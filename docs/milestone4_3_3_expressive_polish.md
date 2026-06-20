# Milestone 4.3.3 — Expressive Motion + Scene Polish + Small Intent Fix

Implemented:

- Added exact list-recent-object intent coverage for:
  - `What objects do you see?`
  - `What can you see?`
  - `What are you seeing?`
  - `What objects did you detect?`
  - `What do you remember seeing?`
- Kept `How are you?` unsupported.
- Added additive optional `engagement.face_x_norm` and `engagement.face_y_norm` protocol fields when a stable face is available.
- Updated Godot procedural lamp animations for idle, engaged, disengaged, seeking attention, recalling/thinking, and stale sleep.
- Added subtle Godot face-following from `face_x_norm` without moving perception, memory, behavior policy, or LLM decisions into Godot.
- Added a lightweight procedural desk/room backdrop, soft ambient lighting, and debug display of stale/sleep and face-follow fields.
- Kept browser chat as the only final-demo text input path.

Not added:

- Voice input.
- Speaker ID.
- Emotion detection.
- New AI dependencies.
- New backend-to-Godot command envelope.


## Scene polish patch update

This patch keeps the backend, browser chat, object memory, LLM recall, UDP command shape, and behavior policy unchanged. The update is limited to the Godot presentation layer.

Changed in `frontend_godot/scripts/Main.gd`:

- Replaced the plain demo backdrop with a runtime `RoomRoot` containing off-white walls, left wall, raised ceiling, wider floor/baseboard trim, plain wooden table, right-side drawer block, and large horizontal back-wall window.
- Added a pale sky panel, dark window frame/sill, lightweight grey building blocks, and low-poly green tree silhouettes behind the lamp.
- Rescaled and positioned the existing `LampRig` on the tabletop so the lamp is larger and easier to read in the final demo while still facing the front-right camera view.
- Moved the `Camera3D` to a right-front table-corner perspective that shows the table surface, lamp full body, and wide window.
- Added softer daylight/fill lighting and a brighter ambient world background to remove the black/debug look.
- Kept the debug panel small, preserved the recall response panel, and did not add or expose a Godot text input box.

Validation focus:

- The scene should resemble the reference composition: large plain wood tabletop in foreground, wide horizontal window in back, enlarged lamp centered-left on the table, taller room volume, and front-right diagonal camera angle.
- Existing motions should remain readable: `idle_breathe`, `attentive_nod`, `searching_glance`, `curious_tilt`/`soft_pulse`, `thinking`/`focus_glow`, `scanning`, and `sleep`.
- Godot remains an animation/display frontend only.
