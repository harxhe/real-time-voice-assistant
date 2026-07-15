"""
Runs 5 scripted end-to-end conversations through the streaming pipeline
(ASR → LLM streaming → TTS streaming) and logs per-stage latency.

Each prompt is synthesized to audio via TTS, then fed through ASR so the
test exercises the full pipeline without requiring a live microphone. This
makes runs deterministic and repeatable. Metrics captured per turn:
  - ASR latency  (synthesized WAV → transcript)
  - LLM first-token latency (request sent → first token received)
  - TTS first-chunk latency (request sent → first audio chunk ready)
  - End-to-end total (user utterance done → first audio chunk ready)

Requires Ollama to be running with the configured local model:
    python -X utf8 test_e2e_streaming.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import scipy.io.wavfile as wav_io
import scipy.signal

# Scripted prompts covering a range of short, conversational query types.

SCRIPTED_PROMPTS = [
    "What time is it right now?",
    "Tell me a fun fact about space.",
    "How do I boil an egg perfectly?",
    "What's the capital of France?",
    "Give me a motivational quote.",
]

SAMPLE_RATE_WAV = 16000   # faster-whisper expects 16kHz

# ---------------------------------------------------------------------------

def synthesize_prompt_audio(text: str) -> np.ndarray:
    """
    Generate speech for a prompt using the project TTS so we have a real
    WAV to feed into ASR (same path as live mic would produce).
    Returns mono float32 at 16kHz.
    """
    import tts as _tts
    audio_22k = _tts.synthesize(text)   # returns float32 @ 22050 Hz

    # Resample 22050 → 16000 for ASR
    n_out = int(len(audio_22k) * SAMPLE_RATE_WAV / 22050)
    audio_16k = scipy.signal.resample(audio_22k, n_out).astype(np.float32)
    return audio_16k


def run_turn(prompt_text: str, history: list[dict], turn_num: int) -> dict:
    """
    Run one full turn: ASR on synthesized audio → LLM stream → TTS stream.
    Returns a dict of latency measurements.
    """
    import asr as _asr
    import llm as _llm
    import tts as _tts

    print(f"\n  Turn {turn_num}: {repr(prompt_text)}")
    print("  " + "-" * 50)

    # --- ASR ----------------------------------------------------------------
    audio = synthesize_prompt_audio(prompt_text)
    t_asr_start = time.perf_counter()
    transcript = _asr.transcribe(audio, sample_rate=SAMPLE_RATE_WAV)
    t_asr_end = time.perf_counter()
    asr_ms = (t_asr_end - t_asr_start) * 1000

    if not transcript:
        transcript = prompt_text   # fallback: if ASR returns empty, use literal text
        print(f"  [warn] ASR returned empty — using raw prompt text")

    print(f"  ASR transcript  : {repr(transcript)}")
    print(f"  ASR latency     : {asr_ms:.0f} ms")

    # --- LLM + TTS streaming ------------------------------------------------
    t_llm_start = time.perf_counter()
    t_first_token: float | None = None
    t_first_chunk: float | None = None

    response_parts: list[str] = []
    chunks_received = 0

    def token_stream_with_timing():
        nonlocal t_first_token
        for token in _llm.generate_stream(transcript, history):
            if t_first_token is None:
                t_first_token = time.perf_counter()
            response_parts.append(token)
            yield token

    print("  LLM+TTS output  : ", end="", flush=True)

    for audio_chunk in _tts.synthesize_stream(token_stream_with_timing()):
        if t_first_chunk is None:
            t_first_chunk = time.perf_counter()
        chunks_received += 1
        print(".", end="", flush=True)   # dot per audio chunk; no playback

    t_total_end = time.perf_counter()
    print()

    llm_first_token_ms = (t_first_token - t_llm_start) * 1000 if t_first_token else None
    tts_first_chunk_ms = (t_first_chunk - t_llm_start) * 1000 if t_first_chunk else None
    total_ms = (t_total_end - t_llm_start) * 1000   # transcript-ready → all audio done

    # end-to-end = asr + llm/tts pipeline
    e2e_ms = asr_ms + total_ms

    full_response = "".join(response_parts)
    print(f"  Response        : {repr(full_response[:80])}{'...' if len(full_response)>80 else ''}")
    print(f"  LLM 1st token   : {llm_first_token_ms:.0f} ms" if llm_first_token_ms else "  LLM 1st token   : N/A")
    print(f"  TTS 1st chunk   : {tts_first_chunk_ms:.0f} ms" if tts_first_chunk_ms else "  TTS 1st chunk   : N/A")
    print(f"  TTS chunks      : {chunks_received}")
    print(f"  ASR latency     : {asr_ms:.0f} ms")
    print(f"  E2E (ASR→audio) : {e2e_ms:.0f} ms")

    history.append({"role": "user", "content": transcript})
    history.append({"role": "assistant", "content": full_response})

    return {
        "turn": turn_num,
        "prompt": prompt_text,
        "asr_ms": round(asr_ms),
        "llm_first_token_ms": round(llm_first_token_ms) if llm_first_token_ms else None,
        "tts_first_chunk_ms": round(tts_first_chunk_ms) if tts_first_chunk_ms else None,
        "total_e2e_ms": round(e2e_ms),
    }


def main():
    print("=" * 60)
    print("E2E Streaming Pipeline — per-turn latency benchmark")
    print("=" * 60)

    print("\n[warmup] Importing and warming up all models...")
    import asr as _asr
    import llm as _llm
    import tts as _tts
    import vad as _vad

    _vad.is_speech(np.zeros(512, dtype=np.float32))
    _asr.transcribe(np.zeros(16000, dtype=np.float32))
    _tts.synthesize("Warm up.")
    print("[warmup] Warming LLM (first token might be slow)...")
    for _ in _llm.generate_stream("Ready?", []):
        pass
    print("[warmup] All models ready.\n")

    history: list[dict] = []
    results: list[dict] = []

    for i, prompt in enumerate(SCRIPTED_PROMPTS, start=1):
        result = run_turn(prompt, history, i)
        results.append(result)

    # --- Summary table -------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS — Per-turn latency breakdown")
    print("=" * 60)
    print()
    print(f"{'Turn':<5} {'ASR':>7} {'LLM 1st tok':>12} {'TTS 1st chunk':>14} {'E2E':>8}")
    print(f"{'':-<5} {'':-<7} {'':-<12} {'':-<14} {'':-<8}")

    asr_vals = []
    llm_vals = []
    tts_vals = []
    e2e_vals = []

    for r in results:
        asr_s  = f"{r['asr_ms']}ms"
        llm_s  = f"{r['llm_first_token_ms']}ms" if r['llm_first_token_ms'] else "N/A"
        tts_s  = f"{r['tts_first_chunk_ms']}ms" if r['tts_first_chunk_ms'] else "N/A"
        e2e_s  = f"{r['total_e2e_ms']}ms"
        print(f"{r['turn']:<5} {asr_s:>7} {llm_s:>12} {tts_s:>14} {e2e_s:>8}  {repr(r['prompt'][:35])}")

        asr_vals.append(r['asr_ms'])
        if r['llm_first_token_ms']:
            llm_vals.append(r['llm_first_token_ms'])
        if r['tts_first_chunk_ms']:
            tts_vals.append(r['tts_first_chunk_ms'])
        e2e_vals.append(r['total_e2e_ms'])

    print()
    avg_asr = sum(asr_vals) / len(asr_vals) if asr_vals else 0
    avg_llm = sum(llm_vals) / len(llm_vals) if llm_vals else 0
    avg_tts = sum(tts_vals) / len(tts_vals) if tts_vals else 0
    avg_e2e = sum(e2e_vals) / len(e2e_vals) if e2e_vals else 0

    print(f"{'AVG':<5} {avg_asr:>6.0f}ms {avg_llm:>11.0f}ms {avg_tts:>13.0f}ms {avg_e2e:>7.0f}ms")

    print()
    print("Pipeline averages:")
    print(f"  ASR             : {avg_asr:.0f}ms")
    print(f"  LLM 1st token   : {avg_llm:.0f}ms")
    print(f"  TTS 1st chunk   : {avg_tts:.0f}ms")
    print(f"  E2E             : {avg_e2e:.0f}ms")

    # Bottleneck identification
    stages = {"ASR": avg_asr, "LLM (first token)": avg_llm, "TTS (first chunk)": avg_tts}
    bottleneck = max(stages, key=lambda k: stages[k])
    print(f"\nCurrent bottleneck stage: {bottleneck} ({stages[bottleneck]:.0f}ms avg)")

    print()
    print("OVERALL: PASS" if all(r["llm_first_token_ms"] for r in results) else "OVERALL: PARTIAL (some turns had no LLM tokens)")


if __name__ == "__main__":
    main()
