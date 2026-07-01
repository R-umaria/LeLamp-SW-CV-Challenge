from backend.audio.audio_output import AudioOutputWorker
from backend.utils.config import AudioOutputConfig


def test_audio_output_disabled_does_not_enqueue():
    worker = AudioOutputWorker(AudioOutputConfig(enabled=False, tts_enabled=False))
    assert worker.request_cue("gentle_chime") is False
    assert worker.request_speech("Hello") is False
