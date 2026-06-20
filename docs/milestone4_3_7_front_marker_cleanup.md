# Milestone 4.3.7 Front Marker Cleanup

This frontend-only patch removes the temporary visual orientation markers from the editable Godot lamp rig.

## Removed nodes

From `frontend_godot/scenes/LampRig.tscn`:

- `FrontLookMarker`
- `LookDirectionTip`

These were debug/editor helpers used to show the lamp head's forward direction after converting the procedural rig into an editable `.tscn` hierarchy. They are not needed for the final demo and made the lamp look like it had floating spheres in front of its head.

## Runtime script cleanup

`frontend_godot/scripts/LampController.gd` no longer binds or animates marker materials. The script still controls:

- 6-DOF lamp motion
- lamp head material color
- actual `SpotLight3D.light_color`
- visible light cone color
- behavior command mapping

## Backend impact

No backend files were changed. Python remains responsible for perception, state, memory, recall, logging, and command generation. Godot remains the expressive embodiment layer.
