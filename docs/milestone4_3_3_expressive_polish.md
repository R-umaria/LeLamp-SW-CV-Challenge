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
