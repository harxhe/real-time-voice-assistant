"""
Quick mic -> ASR test.
Records from your default microphone, then runs transcribe() and prints the result.

Usage:
    python test_asr_mic.py            # records for 5 seconds
    python test_asr_mic.py 8          # records for 8 seconds
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np

SAMPLE_RATE = 16000
RECORD_SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def record_mic(duration: float) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("[error] sounddevice not installed. Run: pip install sounddevice")
        raise

    print(f"[mic] Recording for {duration}s — speak now...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    # Live countdown
    for remaining in range(int(duration), 0, -1):
        print(f"      {remaining}s remaining...", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print()  # clear countdown line
    flat = audio.flatten()
    rms = float(np.sqrt(np.mean(flat ** 2)))
    print(f"[mic] Done. Duration: {len(flat)/SAMPLE_RATE:.1f}s  RMS: {rms:.4f}")
    return flat


def main():
    print("=" * 60)
    print(f"Mic -> ASR test  ({RECORD_SECS}s recording)")
    print("=" * 60)

    audio = record_mic(RECORD_SECS)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.003:
        print("[warn] Very low RMS — mic may be muted or volume too low.")

    print("\n[asr] Loading model and transcribing...")
    import asr

    t0 = time.perf_counter()
    transcript = asr.transcribe(audio, sample_rate=SAMPLE_RATE)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"Transcript : {repr(transcript) if transcript else '(empty — silence or too quiet)'}")
    print(f"Latency    : {latency_ms:.0f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
