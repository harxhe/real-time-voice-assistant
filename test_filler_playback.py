import os
import sys
import sounddevice as sd
import soundfile as sf
import time

def test_playback():
    fillers_dir = os.path.join(os.path.dirname(__file__), "fillers")
    test_file = os.path.join(fillers_dir, "ack_0.wav")
    
    if not os.path.exists(test_file):
        print(f"File not found: {test_file}")
        return
        
    print(f"Reading {test_file}...")
    data, fs = sf.read(test_file)
    print(f"Read {len(data)} samples at {fs}Hz. Max amplitude: {max(abs(data))}")
    
    print("Playing normally with sd.play...")
    sd.play(data, fs)
    sd.wait()
    print("Done playing normally.")
    
    time.sleep(1)
    
    print("Playing padded with 0.5s of silence at the start...")
    import numpy as np
    silence = np.zeros(int(fs * 0.5), dtype=data.dtype)
    padded_data = np.concatenate([silence, data])
    sd.play(padded_data, fs)
    sd.wait()
    print("Done playing padded.")

if __name__ == "__main__":
    test_playback()
