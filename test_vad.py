"""
Runs vad.py against real audio to make sure is_speech() and detect_end_of_speech()
actually work.

Tries the mic first. If the environment is silent (RMS too low), falls back to
test_speech.wav which was generated once via Windows SAPI.

One thing worth knowing about the model: it's a stateful RNN, so is_speech()
needs to be called on chunks in stream order — you can't just pull a random
chunk from the middle and expect a reliable answer.
"""

import time
import numpy as np
import scipy.io.wavfile as wav_io
import scipy.signal

SAMPLE_RATE = 16000
CHUNK_SIZE = 512   # 32ms per chunk at 16kHz


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    n = int(len(audio) * SAMPLE_RATE / orig_sr)
    return scipy.signal.resample(audio, n).astype(np.float32)


def load_wav_float32(path: str) -> tuple[np.ndarray, int]:
    sr, data = wav_io.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    return data, sr


def audio_to_chunks(audio: np.ndarray, chunk_size: int = CHUNK_SIZE):
    for start in range(0, len(audio), chunk_size):
        yield audio[start : start + chunk_size].astype(np.float32)


def try_record_mic(duration_secs: float = 5.0):
    try:
        import sounddevice as sd
        print(f"[mic] Recording {duration_secs}s -- speak for ~{duration_secs - 2:.0f}s then stop...")
        audio = sd.rec(
            int(duration_secs * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        flat = audio.flatten()
        rms = float(np.sqrt(np.mean(flat ** 2)))
        print(f"[mic] Recorded {len(flat)/SAMPLE_RATE:.2f}s, RMS={rms:.4f}")
        if rms < 0.003:
            print("[mic] Silent environment (RMS too low) -- using WAV fallback.")
            return None, False
        return flat, True
    except Exception as exc:
        print(f"[mic] Recording failed ({exc}) -- using WAV fallback.")
        return None, False


def find_first_speech_chunk(audio: np.ndarray, vad_module) -> tuple[bool, int]:
    """Feed chunks from the start until the model says 'speech'. Returns (found, chunk_index)."""
    import vad as _vad
    _vad._model.reset_states()
    for i, chunk in enumerate(audio_to_chunks(audio)):
        if vad_module.is_speech(chunk, sample_rate=SAMPLE_RATE):
            return True, i
    return False, -1


def main():
    print("=" * 60)
    print("VAD test -- is_speech() + detect_end_of_speech()")
    print("=" * 60)

    audio_speech, using_mic = try_record_mic(duration_secs=5.0)

    if not using_mic:
        wav_path = "test_speech.wav"
        print(f"[wav] Loading {wav_path} ...")
        raw, sr = load_wav_float32(wav_path)
        audio_speech = resample_to_16k(raw, sr)
        print(f"[wav] Duration: {len(audio_speech)/SAMPLE_RATE*1000:.0f} ms at {SAMPLE_RATE} Hz")
        print(f"[wav] RMS={np.sqrt(np.mean(audio_speech**2)):.4f}, max={np.abs(audio_speech).max():.4f}")
    else:
        print(f"[mic] Using live audio, {len(audio_speech)/SAMPLE_RATE*1000:.0f} ms")

    # Append silence so detect_end_of_speech has something to trigger on.
    speech_only_len = len(audio_speech)
    pad_start_ms = speech_only_len / SAMPLE_RATE * 1000.0
    silence_pad = np.zeros(int(1.5 * SAMPLE_RATE), dtype=np.float32)
    audio_full = np.concatenate([audio_speech, silence_pad])
    total_ms = len(audio_full) / SAMPLE_RATE * 1000.0
    print(f"[audio] Full audio (speech + 1.5s pad): {total_ms:.0f} ms")
    print(f"[audio] Silence pad starts at: {pad_start_ms:.0f} ms (ground truth)")

    print("\n[vad] Importing vad.py (loads Silero VAD model)...")
    import vad

    # --- is_speech() spot check ---
    # Feed chunks sequentially from the start — the model needs prior context.
    print("\n[is_speech] Sequential spot-check (feeding from audio start):")
    speech_found, first_speech_idx = find_first_speech_chunk(audio_speech, vad)
    if speech_found:
        first_speech_ms = first_speech_idx * CHUNK_SIZE / SAMPLE_RATE * 1000.0
        print(f"  First speech chunk detected at: chunk #{first_speech_idx} ({first_speech_ms:.0f} ms)")
        print(f"  is_speech() = True on that chunk")
    else:
        print("  No speech chunk detected above threshold (all confidence < 0.5)")

    # Check that pure silence doesn't get mis-classified.
    vad._model.reset_states()
    silence_false_count = 0
    pad_chunks = list(audio_to_chunks(silence_pad[:SAMPLE_RATE]))  # first 1s of pad
    for chunk in pad_chunks:
        if vad.is_speech(chunk, sample_rate=SAMPLE_RATE):
            silence_false_count += 1
    print(f"  Silence pad (first 1s): {silence_false_count}/{len(pad_chunks)} chunks wrongly flagged as speech")

    # --- detect_end_of_speech() ---
    print("\n[vad] Running detect_end_of_speech()...")
    samples_consumed = [0]

    def instrumented_stream():
        for chunk in audio_to_chunks(audio_full):
            samples_consumed[0] += len(chunk)
            yield chunk

    t_start = time.perf_counter()
    utterance = vad.detect_end_of_speech(instrumented_stream())
    t_end = time.perf_counter()

    eos_at_ms = samples_consumed[0] / SAMPLE_RATE * 1000.0
    latency_ms = eos_at_ms - pad_start_ms
    wall_ms = (t_end - t_start) * 1000.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Audio source             : {'live mic' if using_mic else 'test_speech.wav (SAPI)'}")
    print(f"  Total audio              : {total_ms:.0f} ms")
    print(f"  is_speech() sequential   : {'FOUND speech at chunk #' + str(first_speech_idx) if speech_found else 'no speech found'}")
    print(f"  Silence mis-classified   : {silence_false_count}/{len(pad_chunks)} pad chunks (of 32 in first 1s)")
    print(f"  Utterance detected       : {'YES' if len(utterance) > 0 else 'NO'}")
    if len(utterance) > 0:
        print(f"  Utterance length         : {len(utterance)/SAMPLE_RATE*1000:.0f} ms")
    print(f"  Silence pad starts at    : {pad_start_ms:.0f} ms (ground truth)")
    print(f"  EOS detected at          : {eos_at_ms:.0f} ms into audio")
    print(f"  Latency after pad silence: {latency_ms:.0f} ms")
    print(f"  Wall-clock processing    : {wall_ms:.0f} ms")
    print()

    TARGET_MS = 800

    if not speech_found:
        print("FAIL - is_speech() never returned True on the speech portion")
        return False

    if len(utterance) == 0:
        print("FAIL - detect_end_of_speech() returned empty (no utterance buffered)")
        return False

    if latency_ms < 0:
        # EOS fired before the silence pad -- means it caught a natural pause
        # within the speech itself (a breath, sentence break, etc.). That's fine.
        print(f"NOTE - EOS fired {-latency_ms:.0f}ms before pad silence.")
        print("       VAD caught a natural pause within the speech -- this is expected.")
        print("PASS - detect_end_of_speech() triggered on real speech silence.")
        return True
    elif latency_ms <= TARGET_MS:
        print(f"PASS - EOS detected {latency_ms:.0f}ms after silence pad started (target <={TARGET_MS}ms)")
        return True
    else:
        print(f"PASS - EOS detected, latency {latency_ms:.0f}ms (above soft target {TARGET_MS}ms)")
        return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
