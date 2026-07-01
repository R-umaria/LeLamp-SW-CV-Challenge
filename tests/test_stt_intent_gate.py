from backend.audio.stt_intent_gate import STTIntentGate, parse_wake_words
from backend.utils.config import SpeechToTextConfig


def test_wake_word_query_accepted():
    gate = STTIntentGate(SpeechToTextConfig(require_wake_word=True, wake_words=parse_wake_words("Lumos,hey Lumos"), intent_cooldown_s=0.0))
    decision = gate.evaluate("Lumos, where is my phone?", confidence=0.9, now=10.0)
    assert decision.accepted is True
    assert decision.directed_to_lumos is True


def test_random_background_rejected():
    gate = STTIntentGate(SpeechToTextConfig(require_wake_word=True, wake_words=("lumos",), intent_cooldown_s=0.0))
    decision = gate.evaluate("I think the game starts later", confidence=0.9, now=10.0)
    assert decision.accepted is False
    assert decision.reason == "missing_wake_word"


def test_clear_memory_query_without_wake_word_can_be_accepted_when_allowed():
    gate = STTIntentGate(SpeechToTextConfig(require_wake_word=False, intent_cooldown_s=0.0))
    decision = gate.evaluate("Where did you last see my laptop?", confidence=0.8, now=10.0)
    assert decision.accepted is True
    assert decision.clear_memory_query is True


def test_duplicate_cooldown_rejects_repeat():
    gate = STTIntentGate(SpeechToTextConfig(require_wake_word=False, intent_cooldown_s=2.0))
    assert gate.evaluate("Where is my phone?", confidence=0.8, now=10.0).accepted is True
    second = gate.evaluate("Where is my phone?", confidence=0.8, now=10.5)
    assert second.accepted is False
