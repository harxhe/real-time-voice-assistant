"""
Tests generate_stream() per Spec Section 9:
  - Sends a hardcoded prompt to the local Ollama model (phi3:mini)
  - Confirms tokens arrive as multiple chunks, not one blocking return
  - Logs time-to-first-token and total latency

Run after starting Ollama:
    ollama serve          (in a separate terminal, if not already running)
    python -X utf8 test_llm.py
"""

import time
import sys


def main():
    print("=" * 60)
    print("LLM test -- generate_stream() via Ollama")
    print("=" * 60)
    print()

    print("[llm] Importing llm.py...")
    import llm
    print("[llm] Import OK\n")

    prompt = "In two sentences, what is the speed of light and why does it matter?"
    history = []

    print(f"[test] Prompt : {repr(prompt)}")
    print(f"[test] History: {history}")
    print()
    print("[test] Streaming response (tokens printed as they arrive):")
    print("-" * 60)

    chunks: list[str] = []
    t_start = time.perf_counter()
    t_first_token: float | None = None

    try:
        for chunk in llm.generate_stream(prompt, history):
            if t_first_token is None:
                t_first_token = time.perf_counter()
            chunks.append(chunk)
            print(chunk, end="", flush=True)
    except Exception as exc:
        print(f"\n\n[error] generate_stream() raised: {exc}")
        print()
        print("Is Ollama running?  Start it with:  ollama serve")
        print("Is phi3:mini pulled?  Pull it with: ollama pull phi3:mini")
        return False

    t_end = time.perf_counter()
    print()
    print("-" * 60)
    print()

    total_ms = (t_end - t_start) * 1000
    first_token_ms = (t_first_token - t_start) * 1000 if t_first_token else None

    full_response = "".join(chunks)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Chunk count        : {len(chunks)}")
    print(f"  Time-to-first-token: {first_token_ms:.0f} ms" if first_token_ms else "  Time-to-first-token: N/A (no tokens)")
    print(f"  Total stream time  : {total_ms:.0f} ms")
    print(f"  Response length    : {len(full_response)} chars")
    print()

    passed = True

    if len(chunks) <= 1:
        print("FAIL -- only 1 chunk received; generate_stream() is not truly streaming")
        passed = False
    else:
        print(f"PASS -- {len(chunks)} chunks received; streaming confirmed")

    if not full_response.strip():
        print("FAIL -- empty response")
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
