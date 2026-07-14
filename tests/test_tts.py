import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import sounddevice as sd
import tts

def mock_llm_stream():
    """Simulates an LLM token generator."""
    tokens = [
        "Hello", " there!", " This", " is", " a", " streaming", " test.\n",
        "It", " synthesizes", " audio", " sentence", " by", " sentence.", " ",
        "Does", " it", " work?"
    ]
    for t in tokens:
        yield t
        time.sleep(0.05) # Simulate ~50ms LLM token delay

def main():
    print("=" * 60)
    print("TTS test -- synthesize_stream() via Piper")
    print("=" * 60)
    
    print("[tts] Initializing model (cold load)...")
    # Trigger a cold load so we can measure streaming latency separately
    tts.synthesize("Warmup.")
    print("[tts] Ready.\n")
    
    t_start = time.perf_counter()
    t_first_chunk = None
    chunks_received = 0
    total_samples = 0
    
    print("[test] Starting mock LLM text stream...")
    print("-" * 60)
    
    try:
        # Expected: 3 sentences -> 3 audio chunks
        for audio_chunk in tts.synthesize_stream(mock_llm_stream()):
            if t_first_chunk is None:
                t_first_chunk = time.perf_counter()
            chunks_received += 1
            total_samples += len(audio_chunk)
            print(f"  -> Received audio chunk {chunks_received}: {len(audio_chunk)} samples")
            
            # Play the chunk live so the user can hear the test
            sd.play(audio_chunk, samplerate=22050)
            sd.wait() # wait for this chunk to finish before the next one

    except Exception as exc:
        print(f"\n[error] synthesize_stream() raised: {exc}")
        return False
        
    t_end = time.perf_counter()
    print("-" * 60)
    print()
    
    total_ms = (t_end - t_start) * 1000
    first_chunk_ms = (t_first_chunk - t_start) * 1000 if t_first_chunk else None
    audio_seconds = total_samples / 22050
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Chunks received        : {chunks_received}")
    print(f"  Time-to-first-audio    : {first_chunk_ms:.0f} ms" if first_chunk_ms else "  Time-to-first-audio: N/A")
    print(f"  Total stream time      : {total_ms:.0f} ms")
    print(f"  Total audio generated  : {total_samples} samples ({audio_seconds:.2f} sec)")
    print()
    
    passed = True
    
    if chunks_received < 3:
        print(f"FAIL -- expected at least 3 chunks (3 sentences), got {chunks_received}")
        passed = False
    else:
        print(f"PASS -- {chunks_received} chunks received; sentence-by-sentence streaming confirmed")
        
    if total_samples == 0:
        print("FAIL -- empty audio generated")
        passed = False
        
    print()
    if passed:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
        
    return passed

if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
