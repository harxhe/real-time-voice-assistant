"""Plays every filler clip sequentially so you can audition them all."""

import os
import struct
import time

import numpy as np
import sounddevice as sd

FILLERS_DIR = os.path.join(os.path.dirname(__file__), "fillers")
SAMPLE_RATE = 22050

# Ordered for a natural listen-through
CLIPS = [
    ("ack_0.wav",       'ack       | "Uh-huh."'),
    ("ack_1.wav",       'ack       | "Got it."'),
    ("ack_2.wav",       'ack       | "One sec."'),
    ("ack_3.wav",       'ack       | "Sure, let me see."'),
    ("thinking_0.wav",  'thinking  | "Let me think about that for a moment."'),
    ("thinking_1.wav",  'thinking  | "Give me just a second here."'),
    ("thinking_2.wav",  'thinking  | "Good question — working on it."'),
    ("extended_0.wav",  'extended  | "This one\'s taking a little longer than usual, still on it."'),
    ("extended_1.wav",  'extended  | "Almost there, thanks for hanging on."'),
    ("recovery_0.wav",  'recovery  | "I\'m having some trouble with that one..."'),
    ("recovery_1.wav",  'recovery  | "I couldn\'t quite get there this time..."'),
]


def _load_wav(path: str):
    """Read a 16-bit PCM WAV and return float32 numpy array + sample rate."""
    with open(path, "rb") as f:
        data = f.read()

    idx = data.find(b"fmt ")
    fmt_data = data[idx + 8: idx + 24]
    _, num_channels, sr, _, _, bits = struct.unpack("<HHIIHH", fmt_data)

    idx = data.find(b"data")
    chunk_size = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = np.frombuffer(data[idx + 8: idx + 8 + chunk_size], dtype=np.int16)

    if num_channels > 1:
        pcm = pcm[::num_channels]  # take left channel if stereo

    return pcm.astype(np.float32) / 32768.0, sr


def main():
    print(f"\nPlaying {len(CLIPS)} filler clips from: {FILLERS_DIR}\n")
    print("-" * 64)

    for filename, label in CLIPS:
        path = os.path.join(FILLERS_DIR, filename)
        if not os.path.exists(path):
            print(f"  MISSING: {filename}")
            continue

        audio, sr = _load_wav(path)
        duration_ms = int(len(audio) / sr * 1000)

        print(f"  > {label}")
        print(f"     ({duration_ms}ms)", end="  ", flush=True)

        sd.play(audio, samplerate=sr)
        sd.wait()

        print("done")
        time.sleep(0.25)  # short gap between clips

    print("-" * 64)
    print("All clips played.\n")


if __name__ == "__main__":
    main()
