import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import sounddevice as sd

# Headless mode for tests
def _noop_play(audio, samplerate=22050, **kwargs):
    pass

def _noop_wait():
    pass

def _noop_stop():
    pass

sd.play = _noop_play
sd.wait = _noop_wait
sd.stop = _noop_stop

import turn_manager
import llm

def test_forced_escalation():
    print("============================================================")
    print("Test: Forced escalation to cloud (local LLM fails)")
    print("============================================================")
    
    # 1. Force local LLM to fail immediately
    def _failing_local_stream(*args, **kwargs):
        raise ConnectionError("Simulated Ollama connection failure")
        yield  # Make it a generator
        
    original_generate_stream = llm.generate_stream
    llm.generate_stream = _failing_local_stream
    
    try:
        t0 = time.perf_counter()
        
        # 2. Run a turn. The local LLM will fail instantly.
        # The turn_manager will play ack and thinking fillers, then at 2.0s
        # it should escalate to cloud and get a real response from Groq.
        tokens = turn_manager.run_turn(
            transcript="Hello, this is a test. Reply with 'Cloud test successful'.",
            history=[]
        )
        
        t1 = time.perf_counter()
        
        print(f"\nTurn complete in {t1-t0:.2f}s")
        response_text = "".join(tokens)
        print(f"Response tokens ({len(tokens)}): {response_text!r}")
        
        # Verification
        if len(tokens) > 0:
            print("\n  PASS  Cloud path responded with tokens")
        else:
            print("\n  FAIL  No tokens received from cloud path")
            sys.exit(1)
            
    finally:
        # Restore
        llm.generate_stream = original_generate_stream

def test_forced_total_failure():
    print("\n============================================================")
    print("Test: Forced total failure (local and cloud both fail)")
    print("============================================================")
    
    def _failing_stream(*args, **kwargs):
        raise ConnectionError("Simulated connection failure")
        yield  # Make it a generator
        
    original_generate_stream = llm.generate_stream
    original_generate_stream_cloud = llm.generate_stream_cloud
    llm.generate_stream = _failing_stream
    llm.generate_stream_cloud = _failing_stream
    
    try:
        t0 = time.perf_counter()
        
        # 2. Run a turn. Both LLMs will fail instantly.
        # The turn_manager will play ack and thinking fillers, then at 2.0s
        # it will escalate to cloud (which fails). Then at ~5.0s it will play recovery.
        tokens = turn_manager.run_turn(
            transcript="Hello, this is a test.",
            history=[]
        )
        
        t1 = time.perf_counter()
        
        print(f"\nTurn complete in {t1-t0:.2f}s")
        print(f"Response tokens ({len(tokens)})")
        
        # Verification
        if len(tokens) == 0:
            print("\n  PASS  Warm recovery triggered, no generic error returned")
        else:
            print("\n  FAIL  Tokens received despite total failure")
            sys.exit(1)
            
    finally:
        # Restore
        llm.generate_stream = original_generate_stream
        llm.generate_stream_cloud = original_generate_stream_cloud

if __name__ == "__main__":
    test_forced_escalation()
    test_forced_total_failure()
