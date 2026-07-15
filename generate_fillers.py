"""
Pre-generates all filler audio clips and saves them as WAV files under fillers/.
Run this once when setting up the project or if the fillers need to be regenerated.
Do not call this at runtime — the live pipeline loads these cached files directly.
"""

import os
import struct
import sys

import numpy as np

# Ensure the project root is on sys.path so tts can be imported from here
sys.path.insert(0, os.path.dirname(__file__))

import tts as _tts

SAMPLE_RATE = 22050  # matches en_US-lessac-medium model config

FILLERS = {
    "ack": [
        "Uh-huh.",
        "Got it.",
        "One sec.",
        "Sure, let me see.",
    ],
    "thinking": [
        "Let me think about that for a moment.",
        "Give me just a second here.",
        "Good question — working on it.",
    ],
    "extended": [
        "This one's taking a little longer than usual, still on it.",
        "Almost there, thanks for hanging on.",
    ],
    "recovery": [
        "I'm having some trouble with that one — want to try asking it a different way?",
        "I couldn't quite get there this time. Mind repeating that or trying something else?",
    ],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "fillers")


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write a mono float32 numpy array to a 16-bit PCM WAV file."""
    # Clamp and convert to int16
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    data_bytes = pcm.tobytes()

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_chunk_size = len(data_bytes)
    riff_chunk_size = 36 + data_chunk_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_chunk_size))
        f.write(b"WAVE")
        # fmt sub-chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # sub-chunk size
        f.write(struct.pack("<H", 1))            # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # data sub-chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_chunk_size))
        f.write(data_bytes)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = sum(len(v) for v in FILLERS.values())
    generated = 0

    for category, phrases in FILLERS.items():
        for idx, phrase in enumerate(phrases):
            filename = f"{category}_{idx}.wav"
            out_path = os.path.join(OUTPUT_DIR, filename)

            print(f"  [{generated + 1}/{total}] {filename}: \"{phrase}\"", end=" ... ", flush=True)
            audio = _tts.synthesize(phrase)

            if audio.size == 0:
                print("WARNING: empty audio returned, skipping.")
                continue

            _write_wav(out_path, audio, SAMPLE_RATE)
            duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
            print(f"done  ({duration_ms}ms, {len(audio)} samples)")
            generated += 1

    print(f"\nGenerated {generated}/{total} filler clips in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
