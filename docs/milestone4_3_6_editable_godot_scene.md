# Milestone 4.3.6 — Editable Godot Scene Refactor

Scope: Godot/frontend refactor only. Backend perception, engagement state machine, object memory, browser chat, UDP command shape, and LLM recall were not changed.

## Goal

Convert the visually polished procedural simulator into editor-authored `.tscn` scenes so the room, table, window, camera, lights, and lamp rig can be adjusted directly in the Godot 3D editor.

This makes the frontend closer to a normal 3D-modeling workflow: select a node, move/rotate/scale it with the Godot viewport gizmos, save the scene, and rerun the demo.

## New editable scene structure

```text
frontend_godot/
  scenes/
    Main.tscn        # top-level composition and runtime receiver
    Room.tscn        # walls, floor, ceiling, trim
    Desk.tscn        # plain table, legs, drawer block, supports
    WindowWall.tscn  # tall window, glass, frame, outdoor silhouettes
    LampRig.tscn     # editor-visible 6-DOF lamp mesh/joint hierarchy
  scripts/
    Main.gd
    UdpCommandReceiver.gd
    LampController.gd
```

## What changed

- Removed runtime procedural construction of the room, desk, window, outdoor backdrop, and world environment from `Main.gd`.
- Added `Room.tscn`, `Desk.tscn`, and `WindowWall.tscn` as editable scene instances under `Main.tscn`.
- Converted `LampRig.tscn` from an empty script host into an editor-visible hierarchy containing the base, joints, arms, head, shade, spotlight, cone, and front marker.
- Updated `LampController.gd` so it animates existing nodes instead of generating the lamp geometry at runtime.
- Kept runtime material binding for the lamp head, spotlight, visible cone, and front marker so behavior-driven light/color changes still work.
- Added `apply_demo_framing_on_start` to `Main.gd`. It defaults to `false`, so camera/lamp/light transforms saved in the editor are not overwritten when the scene starts.

## What remains code-controlled

The following should remain script-controlled because they are part of the bounded embodiment interface:

- UDP command receiving.
- Behavior-to-animation mapping.
- DOF joint animation.
- Face-follow yaw hint.
- Lamp head/body emission color.
- Actual `SpotLight3D.light_color` and intensity.
- Visible cone color/alpha.
- Debug panel and recall response panel.

## What is now editor-controlled

The following can be changed directly in the Godot editor:

- Room width/depth/ceiling height.
- Desk size, legs, panels, and position.
- Window height/width/frame/sill/glass.
- Outdoor skyline/tree silhouettes.
- Lamp overall position, rotation, and scale in `Main.tscn`.
- Camera position, rotation, and FOV.
- Static environment lights.
- Static preview materials for room/table/window/lamp parts.

## Safe tinkering rules

Recommended to edit freely:

- `Main.tscn` instance transforms.
- `Room.tscn`, `Desk.tscn`, and `WindowWall.tscn` mesh sizes, positions, scales, and materials.
- `Camera3D`, `DirectionalLight3D`, `FillLight3D`, `WindowSoftDaylight`, and `TabletopWarmBounce`.

Avoid renaming these nodes unless you also update `LampController.gd` paths:

```text
BaseYaw_DOF1
ShoulderPitch_DOF2
ElbowPitch_DOF3
WristPitch_DOF4
WristYaw_DOF5
LampHeadTilt_DOF6
LampHead
LampShade
LampSpotLight
VisibleLightCone
FrontLookMarker
LookDirectionTip
```

The controller uses those names to find and animate the 6-DOF chain.

## Validation focus

- Open `frontend_godot/project.godot` in Godot.
- Confirm `Main.tscn` shows separate editable instances for `Room`, `WindowWall`, `Desk`, and `LampRig`.
- Confirm selecting `Desk` or `WindowWall` exposes individual child mesh nodes in the scene tree.
- Run the frontend and send test commands:

```powershell
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

Expected behavior:

- UDP commands still drive the lamp.
- The lamp still cycles through idle, engaged, disengaged, seeking-attention, scanning, recalling, and sleep states.
- The spotlight color, visible cone color, and lamp head emission still change together.
- Editor-authored room/table/window/camera transforms are preserved at runtime.
