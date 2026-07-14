"""
Runs asr.py against 3+ audio inputs to verify the transcribe() contract:
  - Sample WAVs: clear speech, a short utterance, and silence
  - Confirms: reasonable transcripts for speech, "" for silence, no exception thrown

Audio samples are generated on the fly if not present:
  - test_speech.wav (already in repo from vad test)
  - test_asr_short.wav  -- a shorter speech snippet carved from test_speech.wav
  - test_asr_silence.wav -- pure silence
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import sys
import numpy as np
import scipy.io.wavfile as wav_io
import scipy.signal


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    n = int(len(audio) * SAMPLE_RATE / orig_sr)
    return scipy.signal.resample(audio, n).astype(np.float32)


def save_wav_float32(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    wav_io.write(path, sr, (audio * 32767).astype(np.int16))


def prepare_samples() -> list[tuple[str, np.ndarray, str]]:
    """Return list of (label, audio_array, expected_kind) tuples.

    expected_kind is 'speech' (non-empty transcript expected) or 'silence'
    (empty string expected).
    """
    samples: list[tuple[str, np.ndarray, str]] = []

    # ── Sample 1: full test_speech.wav ──────────────────────────────────────
    print("[prep] Loading test_speech.wav (Sample 1: full speech)...")
    wav_path = os.path.join(os.path.dirname(__file__), "test_speech.wav")
    raw, sr = load_wav_float32(wav_path)
    audio_full = resample_to_16k(raw, sr)
    samples.append(("Sample 1 — full test_speech.wav", audio_full, "speech"))

    # ── Sample 2: first 3 seconds of test_speech.wav ────────────────────────
    print("[prep] Carving first 3s from test_speech.wav (Sample 2: short speech)...")
    audio_short = audio_full[: 3 * SAMPLE_RATE]
    samples.append(("Sample 2 — first 3s of test_speech.wav", audio_short, "speech"))

    # ── Sample 3: pure silence ───────────────────────────────────────────────
    print("[prep] Generating 2s silence (Sample 3: silence)...")
    audio_silence = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    samples.append(("Sample 3 — 2s pure silence", audio_silence, "silence"))

    print()
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ASR test — transcribe()")
    print("=" * 60)
    print()

    print("[asr] Importing asr.py (loads faster-whisper base.en int8)...")
    t_load_start = time.perf_counter()
    import asr
    t_load_end = time.perf_counter()
    print(f"[asr] Model loaded in {(t_load_end - t_load_start)*1000:.0f}ms\n")

    samples = prepare_samples()

    results: list[dict] = []
    all_pass = True

    for label, audio, expected_kind in samples:
        print(f"-- {label} --")
        print(f"   Duration  : {len(audio)/SAMPLE_RATE*1000:.0f} ms")
        print(f"   RMS       : {float(np.sqrt(np.mean(audio**2))):.4f}")
        print(f"   Expected  : {'non-empty transcript' if expected_kind == 'speech' else 'empty string (silence)'}")

        t0 = time.perf_counter()
        try:
            transcript = asr.transcribe(audio, sample_rate=SAMPLE_RATE)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            threw = False
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            transcript = ""
            threw = True
            print(f"   EXCEPTION : {exc}")

        print(f"   Transcript: {repr(transcript)}")
        print(f"   Latency   : {elapsed_ms:.0f} ms")

        if threw:
            passed = False
            verdict = "FAIL — exception raised"
        elif expected_kind == "silence":
            passed = (transcript == "")
            verdict = "PASS — returned empty string" if passed else f"FAIL — expected '' got {repr(transcript)}"
        else:
            # For speech we just need a non-empty, plausible string
            passed = isinstance(transcript, str) and len(transcript.strip()) > 0
            verdict = "PASS — non-empty transcript returned" if passed else "FAIL — empty transcript for speech audio"

        print(f"   Result    : {verdict}")
        print()

        results.append({
            "label": label,
            "duration_ms": len(audio) / SAMPLE_RATE * 1000,
            "latency_ms": elapsed_ms,
            "transcript": transcript,
            "passed": passed,
            "verdict": verdict,
        })
        if not passed:
            all_pass = False

    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    speech_latencies = [r["latency_ms"] for r in results if "silence" not in r["label"].lower()]
    if speech_latencies:
        avg_lat = sum(speech_latencies) / len(speech_latencies)
        print(f"  Average ASR latency (speech samples): {avg_lat:.0f} ms")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['label']}")
        if r["transcript"]:
            preview = r["transcript"][:80] + ("..." if len(r["transcript"]) > 80 else "")
            print(f"         → {repr(preview)}")
        print(f"         → latency: {r['latency_ms']:.0f} ms")

    print()
    if all_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL — one or more tests did not meet the contract")

    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
