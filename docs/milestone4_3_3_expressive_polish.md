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


## Milestone 4.3.5 frontend polish update

Scope: Godot presentation layer only. Backend perception, state management, behavior policy, UDP command shape, object memory, browser chat, and LLM recall were not changed.

Changed in `frontend_godot/scripts/Main.gd`:

- Increased the procedural window height by enlarging the sky/glass panels and moving the top frame higher.
- Added four table legs beneath the plain wooden desk.
- Zoomed the camera closer to the lamp while preserving the right/front table-corner viewpoint.
- Updated the debug label to identify this as the 4.3.5 frontend polish scene.

Changed in `frontend_godot/scripts/LampController.gd`:

- Audited the light pipeline and found that prior behavior color changes were applied to the emissive head/body material and marker, while `LampSpotLight.light_color` stayed at the default color.
- Synchronized `LampSpotLight.light_color` with the current lamp head/emission color.
- Synchronized `VisibleLightCone` RGB with the current lamp head/emission color while preserving alpha/energy pulsing.
- Added distinct, bounded light colors for warm, attention pulse, scanning, focus/recall, and sleep states.

Validation focus:

- Window appears taller in the back wall.
- Table has visible legs and remains plain/untextured.
- Lamp is larger in frame because the camera is closer/narrower.
- When backend commands change `behavior.light`, the lamp body/emission, visible cone, and actual `SpotLight3D` emitted color change together.
