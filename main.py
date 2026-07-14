import queue
import time
import sounddevice as sd
import numpy as np

import vad
import asr
import llm
import tts

def mic_stream(q: queue.Queue):
    """Generator that yields audio chunks from the microphone queue."""
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk

def run_pipeline():
    history = []
    print("=" * 60)
    print("Real-Time Voice Assistant (Day 1 Sequential)")
    print("=" * 60)
    
    print("[main] Warming up models...")
    vad.is_speech(np.zeros(512, dtype=np.float32))
    asr.transcribe(np.zeros(16000, dtype=np.float32))
    tts.synthesize("Warm up.")
    
    print("[main] Loading LLM into memory (this may take a minute on first run)...")
    for _ in llm.generate_stream("Warm up.", []):
        pass

    print("[main] Ready! Press Ctrl+C to exit.\n")
    
    while True:
        try:
            # 1. Listen (VAD)
            print("\n[ Assistant is listening... ]")
            q = queue.Queue()
            def callback(indata, frames, time_info, status):
                q.put(indata.copy().flatten())
                
            stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', blocksize=512, callback=callback)
            with stream:
                utterance = vad.detect_end_of_speech(mic_stream(q))
                
            if len(utterance) == 0:
                continue
                
            # 2. Transcribe (ASR)
            t0 = time.perf_counter()
            transcript = asr.transcribe(utterance, sample_rate=16000)
            if not transcript:
                continue
            
            print(f"\nUser: {transcript}")
            print(f"(ASR Latency: {(time.perf_counter()-t0)*1000:.0f}ms)")
            
            # 3. Generate (LLM)
            print("Assistant: ", end="", flush=True)
            token_stream = llm.generate_stream(transcript, history)
            
            full_response_parts = []
            def token_interceptor(gen):
                for token in gen:
                    print(token, end="", flush=True)
                    full_response_parts.append(token)
                    yield token
                    
            # 4. Synthesize and Play (TTS)
            audio_stream = tts.synthesize_stream(token_interceptor(token_stream))
            for audio_chunk in audio_stream:
                sd.play(audio_chunk, samplerate=22050)
                sd.wait() 
                
            print() 
            
            # 5. Update History
            history.append({"role": "user", "content": transcript})
            history.append({"role": "assistant", "content": "".join(full_response_parts)})
            
        except KeyboardInterrupt:
            print("\n\n[main] Exiting gracefully...")
            break
        except Exception as e:
            print(f"\n[main] Error: {e}")
            break

if __name__ == "__main__":
    run_pipeline()
