import numpy as np
import torch

# Load once at import time — no point reloading on every call.
_model, _utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True,
)
_model.eval()

(
    _get_speech_timestamps,
    _save_audio,
    _read_audio,
    _VADIterator,
    _collect_chunks,
) = _utils

# Only 8000 and 16000 Hz are supported by the model.
_SUPPORTED_RATES = {8000, 16000}

# Silero needs exactly 512 samples per call at 16kHz, 256 at 8kHz (~32ms either way).
_CHUNK_SIZE = {16000: 512, 8000: 256}

_SPEECH_THRESHOLD = 0.5   # model confidence to call a chunk "speech"
_SILENCE_MS = 600         # how long silence must hold before we consider the utterance done
_MAX_UTTERANCE_S = 30     # hard cap so we don't buffer forever


def is_speech(audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
    """Return True if the chunk contains speech.

    The model is stateful — call this on consecutive chunks in order for
    reliable results. A single isolated chunk fed cold will often score low.

    audio_chunk should be ~512 samples at 16kHz (32ms). Shorter chunks get
    zero-padded; longer ones get truncated.
    """
    if sample_rate not in _SUPPORTED_RATES:
        raise ValueError(f"sample_rate must be 8000 or 16000, got {sample_rate}")

    chunk_size = _CHUNK_SIZE[sample_rate]
    chunk = audio_chunk.astype(np.float32)

    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
    else:
        chunk = chunk[:chunk_size]

    tensor = torch.from_numpy(chunk).unsqueeze(0)

    with torch.no_grad():
        confidence = _model(tensor, sample_rate).item()

    return confidence >= _SPEECH_THRESHOLD


def detect_end_of_speech(audio_stream) -> np.ndarray:
    """Read from audio_stream and return the utterance once the user stops talking.

    audio_stream should yield mono float32 numpy arrays at 16kHz, ~512 samples
    each. The function blocks until it sees speech followed by _SILENCE_MS of
    continuous silence, then returns everything it buffered from the first
    speech chunk onwards (trailing silence not included).

    Returns an empty array if no speech was detected before the stream ended.
    """
    sample_rate = 16000
    silence_samples_needed = int(_SILENCE_MS / 1000 * sample_rate)
    max_samples = int(_MAX_UTTERANCE_S * sample_rate)

    utterance_buffer: list[np.ndarray] = []
    silence_buffer: list[np.ndarray] = []
    speech_started = False
    silence_sample_count = 0

    _model.reset_states()

    for chunk in audio_stream:
        chunk = np.asarray(chunk, dtype=np.float32)

        if is_speech(chunk, sample_rate=sample_rate):
            speech_started = True

            # If there was a short pause, fold it back into the utterance —
            # it was probably just a breath or a natural gap mid-sentence.
            utterance_buffer.extend(silence_buffer)
            silence_buffer.clear()
            silence_sample_count = 0

            utterance_buffer.append(chunk)

            if sum(len(c) for c in utterance_buffer) >= max_samples:
                break

        else:
            if speech_started:
                silence_buffer.append(chunk)
                silence_sample_count += len(chunk)

                if silence_sample_count >= silence_samples_needed:
                    break

    if not utterance_buffer:
        return np.array([], dtype=np.float32)

    return np.concatenate(utterance_buffer)
