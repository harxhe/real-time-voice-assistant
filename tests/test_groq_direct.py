import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import llm

def test_groq():
    try:
        print("Starting request...")
        t0 = time.time()
        t1 = None
        for token in llm.generate_stream_cloud("Reply with exactly: 'Hello, Groq!'", []):
            if t1 is None:
                t1 = time.time()
                print(f"Time to first token: {t1-t0:.2f}s")
            print(f"Token: {token!r}")
        print(f"Finished in {time.time()-t0:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_groq()
