extends RefCounted

# LightSkillLibrary keeps emissive body color, visible light cone, and SpotLight3D
# color synchronized. Output array shape is [r, g, b, energy].


static func normalize_light(light: String) -> String:
	match light:
		"steady_warm", "warm", "engaged_warm":
			return "steady_warm"
		"pink", "excited", "excited_pink":
			return "excited_pink"
		"happy", "happy_gold":
			return "happy_gold"
		"music", "dance_color":
			return "dance_color"
		"sad_blue", "upset_blue":
			return "upset_blue"
		"thinking", "focus_glow":
			return "focus_glow"
		"pointing", "pointer_spot", "recall_pointer":
			return "pointer_spot"
		"sad", "sad_dim", "recall_sad":
			return "sad_dim"
		"sleep", "sleep_red":
			return "sleep_red"
		_:
			return light


static func values_for(light: String, t: float) -> PackedFloat32Array:
	var skill: String = normalize_light(light)
	match skill:
		"sleep_red":
			var sleep_pulse: float = 0.5 + 0.5 * sin(t * 0.45)
			return _values(0.55, 0.035, 0.03, 0.050 + sleep_pulse * 0.020)
		"dim_warm":
			var dim_pulse: float = 0.5 + 0.5 * sin(t * 0.52)
			return _values(1.0, 0.86, 0.48, 0.30 + dim_pulse * 0.065)
		"steady_warm":
			return _values(1.0, 0.82, 0.40, 1.08)
		"excited_pink":
			var excited_pulse: float = 0.5 + 0.5 * sin(t * 1.85)
			return _values(1.0, 0.34, 0.66, 1.18 + excited_pulse * 0.18)
		"slow_pulse":
			var slow_pulse: float = 0.5 + 0.5 * sin(t * 0.82)
			return _values(1.0, 0.72, 0.30, 0.50 + slow_pulse * 0.42)
		"soft_pulse":
			var soft_pulse: float = 0.5 + 0.5 * sin(t * 1.15)
			return _values(1.0, 0.62, 0.24, 0.74 + soft_pulse * 0.44)
		"scan_sweep":
			var scan_pulse: float = 0.5 + 0.5 * sin(t * 1.60)
			return _values(0.62, 0.80, 1.0, 0.76 + scan_pulse * 0.36)
		"focus_glow":
			var focus_pulse: float = 0.5 + 0.5 * sin(t * 0.70)
			return _values(0.74, 0.76, 1.0, 0.96 + focus_pulse * 0.18)
		"pointer_spot":
			var pointer_pulse: float = 0.5 + 0.5 * sin(t * 1.05)
			return _values(1.0, 0.88, 0.46, 1.42 + pointer_pulse * 0.28)
		"sad_dim":
			var sad_pulse: float = 0.5 + 0.5 * sin(t * 0.32)
			return _values(0.24, 0.34, 0.62, 0.20 + sad_pulse * 0.08)
		"happy_gold":
			var happy_pulse: float = 0.5 + 0.5 * sin(t * 1.45)
			return _values(1.0, 0.78, 0.20, 1.02 + happy_pulse * 0.30)
		"dance_color":
			var r: float = 0.82 + 0.18 * sin(t * 1.28)
			var g: float = 0.45 + 0.20 * sin(t * 1.28 + 2.0)
			var b: float = 0.86 + 0.12 * sin(t * 1.28 + 4.0)
			return _values(r, g, b, 1.10)
		"upset_blue":
			var upset_pulse: float = 0.5 + 0.5 * sin(t * 0.36)
			return _values(0.24, 0.36, 0.72, 0.30 + upset_pulse * 0.08)
		_:
			return _values(1.0, 0.86, 0.48, 0.62)


static func _values(red: float, green: float, blue: float, energy: float) -> PackedFloat32Array:
	return PackedFloat32Array([red, green, blue, energy])
