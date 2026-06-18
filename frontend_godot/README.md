# LeLamp Milestone 2 Godot Frontend

This is a lightweight Godot embodiment layer for the LeLamp challenge prototype. It receives backend behavior commands over UDP and maps the bounded command fields to placeholder 6-DOF lamp animations.

Godot does **not** make engagement, memory, or behavior decisions. The Python backend remains the intelligence layer.

## Recommended version

Use Godot 4.6.3 stable or newer Godot 4.x stable. The project uses basic Godot 4 nodes, primitive meshes, GDScript, and `PacketPeerUDP`; it does not require C#/.NET.

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
└── DirectionalLight3D
```

Runtime-created UI:

```text
Main
└── DebugCanvas (CanvasLayer)
    └── DebugPanel (PanelContainer)
        └── MarginContainer
            └── DebugLabel (Label)
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
                        └── VisibleLightCone
```

The six controlled placeholder degrees of freedom are base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist yaw, and lamp head tilt.

## Incoming UDP protocol

The receiver expects one JSON object per UDP packet, preserving the existing backend command protocol:

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

## Behavior mapping

| Backend command | Godot placeholder animation |
| --- | --- |
| `idle` / `idle_breathe` | Small breathing motion, dim lamp |
| `engaged` / `attentive_nod` | Centered pose with nodding head |
| `disengaged` / `searching_glance` | Sweeping base and wrist glance |
| `seeking_attention` / `curious_tilt` + `soft_pulse` | Curious tilt and brighter pulsing light |
| `scanning` | Wider base sweep and scan light |
| `recalling` / `thinking` | Slight tilted thinking pose and focus glow |

## Run the frontend

1. Open Godot.
2. Click **Import**.
3. Select `lelamp_challenge/frontend_godot/project.godot`.
4. Open the imported project.
5. Press **F5** or click **Run Project**.
6. Confirm the debug panel says `Listening on udp://0.0.0.0:4242`.

## Test without webcam

From the project root:

```bash
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

PowerShell:

```powershell
python -m backend.tools.send_test_commands --count 24 --interval 1.0
```

Expected result: the debug panel cycles through `idle`, `engaged`, `disengaged`, `seeking_attention`, `scanning`, and `recalling`, and the lamp changes motion/light behavior on each packet.

## Connect the real backend

Start Godot first, then from the project root run the backend with UDP enabled:

```bash
python -m backend.main --show-window --godot-udp --godot-host 127.0.0.1 --godot-port 4242
```

Tuned dark-room command with Godot enabled:

```bash
python -m backend.main --show-window --godot-udp --smoothing-window 9 --min-state-dwell 1.0 --exit-disengaged-frames 7 --exit-absent-frames 10 --min-candidate-area-ratio 0.012 --min-face-area-ratio 0.022
```

The backend still writes isolated logs under `logs/runs/<run_id>/` and mirrors the latest run to `logs/latest/` unless `--no-latest` is used.

## Scope boundaries

Included in Milestone 2:

- UDP receive loop in Godot.
- Primitive 6-DOF lamp rig.
- Controlled placeholder motion and light animations.
- Visible debug UI.
- Python UDP sender and webcam-free test sender.

Not included yet:

- Inverse kinematics.
- Imported 3D art/model.
- Object detection.
- Memory storage.
- LLM recall.
- Real servo control.
