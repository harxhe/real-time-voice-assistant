import numpy as np
from faster_whisper import WhisperModel

# Load once at import time — model stays resident so repeated calls are fast.
# base.en + int8 keeps memory low and runs comfortably on CPU.
_model = WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe a mono float32 audio array and return the plain text.

    If the audio is empty or contains only silence the function returns an
    empty string rather than raising. The return value is always a plain
    string — no timestamps, no JSON, no extra structure.

    audio  : mono float32 numpy array at sample_rate Hz
    sample_rate : must be 16000 (faster-whisper resamples internally if needed,
                  but the contract here is 16kHz in)
    """
    if audio is None or len(audio) == 0:
        return ""

    audio = np.asarray(audio, dtype=np.float32)

    # A chunk that's pure noise or silence tends to produce hallucinated filler
    # text from Whisper (e.g. "Thank you." / "you" / music notes). Guard against
    # that by checking RMS energy — below this threshold the audio is effectively
    # silent and we return early rather than letting Whisper hallucinate.
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.001:
        return ""

    segments, _info = _model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,          # built-in VAD suppresses silence/noise segments
        vad_parameters={"min_silence_duration_ms": 300},
    )

    parts = [seg.text.strip() for seg in segments]
    return " ".join(parts).strip()
