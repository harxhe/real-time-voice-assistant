"""
Validates the pre-generated filler audio library.

Checks that every expected WAV file exists under fillers/, is a valid WAV
(correct header, sample rate, mono channel), contains non-silent audio, and
that durations are reasonable (each filler is at least 300ms and under 6s).
"""

import os
import struct
import sys

# Resolve project root (tests/ is one level below it)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILLERS_DIR = os.path.join(PROJECT_ROOT, "fillers")

EXPECTED_SAMPLE_RATE = 22050
MIN_DURATION_MS = 300
MAX_DURATION_MS = 6000

EXPECTED_FILES = {
    "ack_0.wav": "Mm-hmm.",
    "ack_1.wav": "Got it.",
    "ack_2.wav": "One sec.",
    "ack_3.wav": "Sure, let me see.",
    "thinking_0.wav": "Let me think about that for a moment.",
    "thinking_1.wav": "Give me just a second here.",
    "thinking_2.wav": "Good question — working on it.",
    "extended_0.wav": "This one's taking a little longer than usual, still on it.",
    "extended_1.wav": "Almost there, thanks for hanging on.",
    "recovery_0.wav": "I'm having some trouble...",
    "recovery_1.wav": "I couldn't quite get there...",
}


def _read_wav_info(path: str):
    """Return (sample_rate, num_channels, num_samples) from a WAV file header."""
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a RIFF file")
        f.read(4)  # chunk size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError("Not a WAVE file")

        # Find fmt chunk
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                raise ValueError("fmt chunk not found")
            (chunk_size,) = struct.unpack("<I", f.read(4))
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = \
                    struct.unpack("<HHIIHH", fmt_data[:16])
                break
            else:
                f.read(chunk_size)

        # Find data chunk
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                raise ValueError("data chunk not found")
            (chunk_size,) = struct.unpack("<I", f.read(4))
            if chunk_id == b"data":
                break
            else:
                f.read(chunk_size)

    bytes_per_sample = bits_per_sample // 8
    num_samples = chunk_size // (num_channels * bytes_per_sample)
    return sample_rate, num_channels, num_samples


def _rms_from_wav(path: str) -> float:
    """Return approximate RMS of the audio in a 16-bit PCM WAV file."""
    import array as arr

    with open(path, "rb") as f:
        data = f.read()

    # Find 'data' chunk
    idx = data.find(b"data")
    if idx == -1:
        return 0.0
    chunk_size = struct.unpack_from("<I", data, idx + 4)[0]
    pcm_start = idx + 8
    pcm_bytes = data[pcm_start: pcm_start + chunk_size]

    samples = arr.array("h", pcm_bytes)
    if not samples:
        return 0.0
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    return rms / 32768.0  # normalise to 0..1


def main():
    print("=" * 60)
    print("Filler audio library validation")
    print(f"Directory: {FILLERS_DIR}")
    print("=" * 60)

    if not os.path.isdir(FILLERS_DIR):
        print(f"\nFAIL: fillers/ directory not found at {FILLERS_DIR}")
        sys.exit(1)

    pass_count = 0
    fail_count = 0
    results = []

    for filename in sorted(EXPECTED_FILES.keys()):
        path = os.path.join(FILLERS_DIR, filename)
        row = {"file": filename, "status": None, "notes": []}

        if not os.path.exists(path):
            row["status"] = "FAIL"
            row["notes"].append("file missing")
            results.append(row)
            fail_count += 1
            continue

        file_size = os.path.getsize(path)
        if file_size < 44:  # minimum valid WAV header
            row["status"] = "FAIL"
            row["notes"].append(f"file too small ({file_size} bytes)")
            results.append(row)
            fail_count += 1
            continue

        try:
            sample_rate, num_channels, num_samples = _read_wav_info(path)
        except Exception as exc:
            row["status"] = "FAIL"
            row["notes"].append(f"invalid WAV: {exc}")
            results.append(row)
            fail_count += 1
            continue

        duration_ms = int(num_samples / sample_rate * 1000) if sample_rate else 0
        row["duration_ms"] = duration_ms
        row["sample_rate"] = sample_rate
        row["channels"] = num_channels

        if sample_rate != EXPECTED_SAMPLE_RATE:
            row["notes"].append(
                f"wrong sample rate {sample_rate} (expected {EXPECTED_SAMPLE_RATE})"
            )
        if num_channels != 1:
            row["notes"].append(f"expected mono, got {num_channels} channels")
        if duration_ms < MIN_DURATION_MS:
            row["notes"].append(
                f"too short ({duration_ms}ms, min {MIN_DURATION_MS}ms)"
            )
        if duration_ms > MAX_DURATION_MS:
            row["notes"].append(
                f"suspiciously long ({duration_ms}ms, max {MAX_DURATION_MS}ms)"
            )

        rms = _rms_from_wav(path)
        row["rms"] = rms
        if rms < 0.005:
            row["notes"].append(f"near-silent audio (RMS={rms:.4f})")

        if row["notes"]:
            row["status"] = "FAIL"
            fail_count += 1
        else:
            row["status"] = "PASS"
            pass_count += 1

        results.append(row)

    # Print results table
    print(f"\n{'File':<20} {'Status':<6} {'Rate':>6} {'Ch':>3} {'Duration':>10} {'RMS':>8}  Notes")
    print("-" * 80)
    for row in results:
        sr   = str(row.get("sample_rate", "-"))
        ch   = str(row.get("channels", "-"))
        dur  = f"{row.get('duration_ms', '-')}ms" if "duration_ms" in row else "-"
        rms  = f"{row.get('rms', 0):.4f}" if "rms" in row else "-"
        notes = "; ".join(row["notes"]) if row["notes"] else ""
        print(f"{row['file']:<20} {row['status']:<6} {sr:>6} {ch:>3} {dur:>10} {rms:>8}  {notes}")

    print()
    print(f"Result: {pass_count} PASS, {fail_count} FAIL  (out of {len(EXPECTED_FILES)} expected files)")
    print()

    if fail_count == 0:
        print("ALL PASS")
    else:
        print("SOME CHECKS FAILED — see table above")
        sys.exit(1)


if __name__ == "__main__":
    main()
