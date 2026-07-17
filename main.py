import queue
import time
import sounddevice as sd
import numpy as np

import vad
import asr
import llm
import tts
import turn_manager

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
    print("Real-Time Voice Assistant")
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
            
            # 3. Generate and Play (Orchestrated by Turn Manager)
            import threading
            barge_in_event = threading.Event()
            stop_barge_in_listener = threading.Event()
            
            def barge_in_listener():
                q2 = queue.Queue()
                def callback2(indata, frames, time_info, status):
                    q2.put(indata.copy().flatten())
                # Open a new stream just for barge-in monitoring
                stream2 = sd.InputStream(samplerate=16000, channels=1, dtype='float32', blocksize=512, callback=callback2)
                with stream2:
                    vad._model.reset_states()
                    while not stop_barge_in_listener.is_set():
                        try:
                            chunk = q2.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if vad.is_speech(chunk, sample_rate=16000):
                            barge_in_event.set()
                            break
            
            listener_thread = threading.Thread(target=barge_in_listener, daemon=True)
            listener_thread.start()
            
            print(f"Assistant: \n", end="", flush=True)
            t0_turn = time.perf_counter()
            tokens = turn_manager.run_turn(transcript, history, barge_in_event=barge_in_event)
            
            stop_barge_in_listener.set()
            listener_thread.join()
            
            print(f"\n(Turn E2E Latency: {(time.perf_counter()-t0_turn)*1000:.0f}ms)")
            
            # 4. Update History
            if tokens:
                full_response = "".join(tokens)
                print(f"Response: {full_response}")
                history.append({"role": "user", "content": transcript})
                history.append({"role": "assistant", "content": full_response})
            else:
                print("Response: [Interrupted or Failed]")
            
        except KeyboardInterrupt:
            print("\n\n[main] Exiting gracefully...")
            break
        except Exception as e:
            print(f"\n[main] Error: {e}")
            break

if __name__ == "__main__":
    run_pipeline()
