# Lumos Godot Frontend

This is the lightweight Godot embodiment layer for the Lumos challenge prototype. It receives bounded backend behavior commands over UDP and maps them to controlled 6-DOF placeholder lamp animations.

Godot does **not** make engagement, memory, recall, or behavior decisions. The Python backend remains the intelligence layer.

## Recommended version

Use Godot 4.6.3 stable or newer Godot 4.x stable. The project uses basic Godot 4 nodes, primitive meshes, GDScript, and `PacketPeerUDP`; it does not require C#/.NET.

## Milestone 4.3.6 editable scene refactor

Milestone 4.3.6 converts the previously procedural visual simulator into editable `.tscn` scenes. This makes the frontend much easier to tinker with in the Godot 3D viewport: room, table, window, lamp placement, camera, and static lights are now normal scene-tree nodes instead of geometry created inside `Main.gd` at runtime.

The behavior architecture is unchanged. Python still owns perception, state, behavior selection, memory, and recall. Godot remains a bounded animation/display frontend.

## Folder layout

```text
frontend_godot/
  project.godot
  scenes/
    Main.tscn        # top-level scene composition
    Room.tscn        # editable walls, ceiling, floor, baseboards
    Desk.tscn        # editable plain desk, legs, supports, drawer block
    WindowWall.tscn  # editable tall window, frame, glass, outdoor silhouettes
    LampRig.tscn     # editable 6-DOF lamp mesh/joint hierarchy
  scripts/
    Main.gd
    UdpCommandReceiver.gd
    LampController.gd
  assets/
```

## Scene hierarchy

`Main.tscn`:

```text
Main (Node3D) [Main.gd]
├── Room (instance of Room.tscn)
├── WindowWall (instance of WindowWall.tscn)
├── Desk (instance of Desk.tscn)
├── LampRig (instance of LampRig.tscn) [LampController.gd]
├── Camera3D
├── DirectionalLight3D
├── FillLight3D
├── WindowSoftDaylight
├── TabletopWarmBounce
├── SoftAmbientWorld
└── UdpCommandReceiver (Node) [UdpCommandReceiver.gd]
```

Runtime-created UI:

```text
Main
└── LumosCanvas (CanvasLayer)
    ├── DebugPanel (PanelContainer)
    │   └── DebugLabel (Label)
    └── RecallResponsePanel (PanelContainer)
        └── RecallResponseLabel (Label)
```

Editor-authored lamp rig under `LampRig.tscn`:

```text
LampRig (Node3D) [LampController.gd]
└── BaseYaw_DOF1
    ├── Base
    └── ShoulderPitch_DOF2
        ├── ShoulderJoint
        ├── UpperArm
        └── ElbowPitch_DOF3
            ├── ElbowJoint
            ├── LowerArm
            └── WristPitch_DOF4
                ├── WristPitchJoint
                └── WristYaw_DOF5
                    └── LampHeadTilt_DOF6
                        ├── LampHead
                        ├── LampShade
                        ├── LampSpotLight
                        └── VisibleLightCone
```

The six controlled placeholder degrees of freedom are base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist yaw, and lamp head tilt.

## How to tinker visually

1. Open `frontend_godot/project.godot` in Godot.
2. Open `scenes/Main.tscn`.
3. Use the scene tree to select `Room`, `Desk`, `WindowWall`, `LampRig`, `Camera3D`, or the lights.
4. Use the 3D viewport move/rotate/scale gizmos to adjust composition.
5. Save the scene and press **F5**.

Important: `Main.gd` has `apply_demo_framing_on_start = false` by default. Leave it false when tinkering, otherwise runtime code will overwrite the saved camera/lamp/light transforms.

Safe to edit freely:

- Room, desk, and window child mesh transforms/materials.
- LampRig overall position, rotation, and scale in `Main.tscn`.
- Camera transform and FOV.
- Static lights and world environment.

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
```

## Incoming UDP protocol

The receiver expects one JSON object per UDP packet, preserving the backend command protocol:

```json
{
  "timestamp": "2026-06-17T15:30:00",
  "state": "seeking_attention",
  "engagement": {
    "status": "disengaged",
    "confidence": 0.82,
    "reason": "head_turned_away"
  },
  "behavior": {
    "motion": "curious_tilt",
    "light": "soft_pulse",
    "sound": "gentle_chime",
    "speech_text": null
  },
  "memory": {
    "last_detected_objects": []
  }
}
```

Optional normalized face hints are supported when Python sends them:

```json
"engagement": {
  "status": "engaged",
  "confidence": 0.92,
  "reason": "face_centered",
  "face_x_norm": 0.50,
  "face_y_norm": 0.48
}
```

If these fields are missing, Godot falls back to centered procedural animations.

## Behavior mapping

| Backend command | Godot placeholder animation |
| --- | --- |
| `idle` / `idle_breathe` | Subtle breathing, tiny sway, soft warm dim lamp |
| `engaged` / `attentive_nod` | Forward attentive pose, small nod, warm steady light, optional face-follow |
| `disengaged` / `searching_glance` | Slower side-to-side search, head scan, slow light pulse |
| `seeking_attention` / `curious_tilt` + `soft_pulse` | Asymmetric tilt, anticipation motion, soft pulse, tiny restrained bounce |
| `scanning` | Wider base sweep and scan light |
| `recalling` / `thinking` + `focus_glow` | Immediate thinking pose, head down, focus glow, small waiting oscillation, visible answer panel |
| local stale `sleep` / `sleep_rest` + `sleep_red` | Folded pet-like rest pose, very dim red light, no active search/nod |

## Run the frontend

1. Open Godot.
2. Click **Import**.
3. Select `frontend_godot/project.godot`.
4. Open the imported project.
5. Press **F5** or click **Run Project**.
6. Confirm the debug panel says `Listening on udp://0.0.0.0:4242`.
7. Confirm the lamp is visible from the camera view and that the lamp head and visible light cone point toward the camera.

## Test without webcam

From the project root:

```powershell
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

Expected result: the debug panel cycles through `idle`, `engaged`, `disengaged`, `seeking_attention`, `scanning`, `recalling`, and `sleep`; the lamp changes motion/light behavior on each packet; the recall answer panel appears during the `recalling` command; face-follow hints subtly yaw the lamp when present.

## Connect the real backend

Start Godot first, then run this from the project root:

```powershell
python -m backend.main --godot-udp --enable-objects --enable-web-chat --use-llm --ollama-url http://10.0.0.70:11434 --ollama-model qwen2.5:1.5b --llm-timeout 90 --llm-connect-timeout 5 --show-window
```

Open browser chat at:

```text
http://127.0.0.1:8765
```

The backend writes isolated logs under `logs/runs/<run_id>/` and mirrors the latest run to `logs/latest/` unless `--no-latest` is used.

## Scope boundaries

Included:

- UDP receive loop in Godot.
- Editor-authored primitive 6-DOF lamp rig.
- Controlled placeholder motion and light animations.
- Editable room, large table, tall window, and stylized outdoor skyline scenes.
- Front/right table-corner demo camera.
- Visible light cone only; debug front/look markers removed.
- Visible debug UI.
- Visible recall response panel.

Not included:

- Voice input.
- Text-to-speech.
- Inverse kinematics.
- Imported 3D art/model.
- Real servo control.


## Milestone 4.3.7 cleanup note

The editable lamp rig no longer includes the two debug orientation markers that appeared as small floating geometry in front of the lamp head:

- `FrontLookMarker`
- `LookDirectionTip`

These nodes were useful while checking the lamp's facing direction, but they are not required for UDP behavior, light color changes, recall display, or animation. The final scene now keeps only the actual `LampSpotLight` and translucent `VisibleLightCone`.

## Milestone 4.9 physical motion smoothing

The frontend now treats the lamp as a physical actuator system instead of only a visual animation. `LampController.gd` still receives the same bounded backend motion names, but it converts target pose changes into safe transitions using:

- quintic S-curve blending between behavior states;
- per-joint velocity, acceleration, and jerk limits;
- slew-limited face-follow input;
- profiled root/forward-back movement;
- slew-limited light color and brightness changes.

Useful inspector parameters on `LampRig`:

- `physical_motion_enabled`: turn the physical profile on/off.
- `transition_blend_seconds`: base transition time between behavior states.
- `servo_max_velocity_degrees_per_second`: max joint speed.
- `servo_max_acceleration_degrees_per_second2`: max joint acceleration.
- `servo_max_jerk_degrees_per_second3`: max change in acceleration.
- `face_follow_max_slew_degrees_per_second`: max face-tracking command change.
- `light_color_slew_per_second` and `light_energy_slew_per_second`: lamp output ramp rates.

For final demo recording, keep `physical_motion_enabled` enabled. If the lamp feels too conservative, raise velocity/acceleration/jerk slightly; if it feels jerky, lower them or raise `transition_blend_seconds`.
