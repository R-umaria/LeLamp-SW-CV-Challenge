# LeLamp Godot Frontend

This is the lightweight Godot embodiment layer for the LeLamp challenge prototype. It receives bounded backend behavior commands over UDP and maps them to controlled 6-DOF placeholder lamp animations.

Godot does **not** make engagement, memory, recall, or behavior decisions. The Python backend remains the intelligence layer.

## Recommended version

Use Godot 4.6.3 stable or newer Godot 4.x stable. The project uses basic Godot 4 nodes, primitive meshes, GDScript, and `PacketPeerUDP`; it does not require C#/.NET.

## Milestone 4.2 visual polish

Milestone 4.2 changes the default camera to a front-three-quarter view. The lamp head and visible light cone point along the rig's local `-Z` axis, so the camera is now placed on that front side with a slight side offset. This makes the head, front marker, light cone, arm joints, and recall answer panel visible in the final demo.

The 6-DOF animation axes are unchanged. The polish is primarily camera placement, lighting, and a small visible front/look marker on the lamp head.

## Folder layout

```text
frontend_godot/
  project.godot
  scenes/
    Main.tscn
    LampRig.tscn
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
├── UdpCommandReceiver (Node) [UdpCommandReceiver.gd]
├── LampRig (instance of LampRig.tscn)
├── Camera3D
├── DirectionalLight3D
└── FillLight3D
```

Runtime-created UI:

```text
Main
└── LeLampCanvas (CanvasLayer)
    ├── DebugPanel (PanelContainer)
    │   └── DebugLabel (Label)
    └── RecallResponsePanel (PanelContainer)
        └── RecallResponseLabel (Label)
```

Runtime-created lamp rig under `LampRig`:

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
                        ├── VisibleLightCone
                        ├── FrontLookMarker
                        └── LookDirectionTip
```

The six controlled placeholder degrees of freedom are base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist yaw, and lamp head tilt.

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

Milestone 4.2 does not change this schema.

## Behavior mapping

| Backend command | Godot placeholder animation |
| --- | --- |
| `idle` / `idle_breathe` | Small breathing motion, dim lamp |
| `engaged` / `attentive_nod` | Centered pose with nodding head |
| `disengaged` / `searching_glance` | Sweeping base and wrist glance |
| `seeking_attention` / `curious_tilt` + `soft_pulse` | Curious tilt and brighter pulsing light |
| `scanning` | Wider base sweep and scan light |
| `recalling` / `thinking` + `focus_glow` | Thinking pose, focus glow, and visible recall answer panel |

## Run the frontend

1. Open Godot.
2. Click **Import**.
3. Select `frontend_godot/project.godot`.
4. Open the imported project.
5. Press **F5** or click **Run Project**.
6. Confirm the debug panel says `Listening on udp://0.0.0.0:4242`.
7. Confirm the lamp is visible from a front-three-quarter view and the light cone/front marker points toward the camera.

## Test without webcam

From the project root:

```powershell
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

Expected result: the debug panel cycles through `idle`, `engaged`, `disengaged`, `seeking_attention`, `scanning`, and `recalling`; the lamp changes motion/light behavior on each packet; the recall answer panel appears during the `recalling` command.

## Connect the real backend

Start Godot first, then run this from the project root:

```powershell
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242 --enable-objects --object-model yolov8n.pt --save-object-frames --interactive-recall --memory-db data/scene_memory.sqlite
```

The backend writes isolated logs under `logs/runs/<run_id>/` and mirrors the latest run to `logs/latest/` unless `--no-latest` is used.

## Scope boundaries

Included:

- UDP receive loop in Godot.
- Primitive 6-DOF lamp rig.
- Controlled placeholder motion and light animations.
- Front-three-quarter demo camera.
- Visible front/look marker and light cone.
- Visible debug UI.
- Visible recall response panel.

Not included:

- Voice input.
- Text-to-speech.
- Inverse kinematics.
- Imported 3D art/model.
- Real servo control.
