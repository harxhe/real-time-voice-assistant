"""
Interactive test for llm.py
Type a prompt and see the streamed response from the local Ollama model.
Type 'quit' or 'exit' to stop.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import sys

def main():
    print("=" * 60)
    print("Interactive LLM test -- generate_stream() via Ollama")
    print("=" * 60)
    print("Loading llm.py...")
    import llm
    print("Done. Type 'quit' to exit.\n")
    
    history = []
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
            if not prompt.strip():
                continue
                
            print("\nAssistant: ", end="", flush=True)
            
            t_start = time.perf_counter()
            t_first_token = None
            chunks = 0
            
            for chunk in llm.generate_stream(prompt, history):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                chunks += 1
                print(chunk, end="", flush=True)
                
            t_end = time.perf_counter()
            
            # Note: We aren't preserving history in this simple test script, 
            # but you could append to `history` here if you wanted to test multi-turn.
            
            print(f"\n\n[Stats: {chunks} chunks | First token: {(t_first_token - t_start)*1000:.0f}ms | Total: {(t_end - t_start)*1000:.0f}ms]")
            
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"\nError: {exc}")
            break

if __name__ == "__main__":
    main()
