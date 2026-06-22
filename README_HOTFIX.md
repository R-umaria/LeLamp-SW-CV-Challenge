# Lumos Godot Hotfix: MotionSkillLibrary.gd

This hotfix fixes Godot errors:

```text
Function "_active_listen()" not found in base self
Function "_listening_attentive()" not found in base self
Function "_sound_seek()" not found in base self
```

Copy this file into your existing Milestone 4.11 project, replacing:

```text
frontend_godot/scripts/lumos/MotionSkillLibrary.gd
```

The bug was that the new motion names were mapped in `target_for()`, but the actual static helper functions were not appended to the script.
