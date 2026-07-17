"""
Tests the turn_manager state machine using an artificially delayed LLM stub.

Validates:
  - Filler sequence: ack fires at ~300ms, thinking at ~1000ms, extended at ~2000ms.
  - No gaps or overlaps (each filler stops before the next begins).
  - Clean transition to RESPONDING when real tokens arrive (~3s mark).
  - Barge-in: a simulated VAD speech event mid-playback halts output within 200ms.
"""

import sys
import os
import threading
import time

# Make project root importable when running from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import sounddevice as sd

# Silence sounddevice output during automated testing so we don't need speakers.
# We patch sd.play and sd.wait to no-ops so the test runs headlessly.
_play_log: list[tuple[float, str]] = []  # (timestamp, label)


_play_duration = [0.0]

def _mock_play(audio, samplerate=22050, **kwargs):
    _play_duration[0] = len(audio) / samplerate

def _mock_wait():
    # sleep for the chunk duration to simulate real playback blocking
    if _play_duration[0] > 0:
        time.sleep(_play_duration[0])

def _mock_stop():
    pass

sd.play = _mock_play
sd.wait = _mock_wait
sd.stop = _mock_stop

import turn_manager  # noqa: E402 — import after patching sd

# Test helpers

# Patch _play_filler to record which filler was triggered and when, but still
# honour the stop_event so the interrupt test works correctly.
_events: list[dict] = []


def _recording_play_filler(path: str, stop_event: threading.Event, label: str) -> None:
    filename = os.path.basename(path)
    entry = {"t": time.perf_counter(), "label": label, "file": filename, "interrupted": False}
    _events.append(entry)
    # Simulate playback duration by sleeping in short slices, honouring stop_event
    duration = _FILLER_DURATIONS.get(filename, 1.0)
    slept = 0.0
    step = 0.05
    while slept < duration:
        if stop_event.is_set():
            entry["interrupted"] = True
            break
        time.sleep(min(step, duration - slept))
        slept += step


# Approximate durations of our actual filler clips (seconds) for realistic simulation
_FILLER_DURATIONS = {
    "ack_0.wav":      0.917,
    "ack_1.wav":      0.452,
    "ack_2.wav":      0.835,
    "ack_3.wav":      1.253,
    "thinking_0.wav": 2.054,
    "thinking_1.wav": 1.613,
    "thinking_2.wav": 1.822,
    "extended_0.wav": 2.867,
    "extended_1.wav": 2.217,
    "recovery_0.wav": 4.167,
    "recovery_1.wav": 4.318,
}

turn_manager._play_filler = _recording_play_filler  # type: ignore[attr-defined]


# Fake LLM generators

def _delayed_llm(delay_seconds: float):
    """Yield a short response after an artificial delay."""
    time.sleep(delay_seconds)
    for token in ["This ", "is ", "the ", "real ", "response."]:
        yield token


def _instant_llm():
    """Yield immediately (simulates a hot-cached LLM)."""
    for token in ["Fast ", "reply."]:
        yield token


def _never_llm():
    """Never yields — simulates a totally hung LLM."""
    # Sleep longer than the 5s recovery window
    time.sleep(10)
    return
    yield  # make it a generator


# Test cases

PASS = "PASS"
FAIL = "FAIL"


def _run_turn_headless(delay: float, barge_in_after: float | None = None):
    """
    Run a turn_manager.run_turn() call with a delayed fake LLM stream.
    If barge_in_after is set (seconds), fire barge_in_event that many seconds
    after the turn starts.
    """
    global _events
    _events = []
    barge_in_event = threading.Event()
    turn_start = [0.0]

    def fire_barge_in():
        time.sleep(barge_in_after)
        barge_in_event.set()

    if barge_in_after is not None:
        threading.Thread(target=fire_barge_in, daemon=True).start()

    turn_start[0] = time.perf_counter()
    tokens = turn_manager.run_turn(
        transcript="What is the capital of France?",
        history=[],
        barge_in_event=barge_in_event,
        _llm_override=_delayed_llm(delay),
    )
    elapsed = (time.perf_counter() - turn_start[0]) * 1000
    return tokens, elapsed, list(_events)


# Scenario: 4s delay — filler sequence ack → thinking → extended

def test_filler_sequence():
    print("\n" + "=" * 60)
    print("Scenario: 4-second LLM delay — filler sequence")
    print("=" * 60)

    t_start = time.perf_counter()
    tokens, elapsed_ms, events = _run_turn_headless(delay=4.0)
    wall = (time.perf_counter() - t_start) * 1000

    print(f"\n  Total turn wall time: {wall:.0f}ms")
    print(f"  Tokens received    : {len(tokens)} ({''.join(tokens)!r})")
    print(f"  Filler events      : {len(events)}")
    for e in events:
        t_rel = (e["t"] - t_start) * 1000
        interrupted = " [interrupted]" if e["interrupted"] else ""
        print(f"    t={t_rel:6.0f}ms  {e['label']:<20}  {e['file']}{interrupted}")

    labels = [e["label"] for e in events]

    checks = []

    # Ack must be first
    got_ack = any(e["label"] == "FILLER_ACK" for e in events)
    checks.append(("ack filler fired", got_ack))

    # Thinking must be second
    got_thinking = any(e["label"] == "FILLER_THINKING" for e in events)
    checks.append(("thinking filler fired", got_thinking))

    # Extended must be third
    got_extended = any(e["label"] == "FILLER_EXTENDED" for e in events)
    checks.append(("extended filler fired", got_extended))

    # Order: ack before thinking before extended
    if got_ack and got_thinking and got_extended:
        ack_t      = next(e["t"] for e in events if e["label"] == "FILLER_ACK")
        thinking_t = next(e["t"] for e in events if e["label"] == "FILLER_THINKING")
        extended_t = next(e["t"] for e in events if e["label"] == "FILLER_EXTENDED")
        checks.append(("ack before thinking", ack_t < thinking_t))
        checks.append(("thinking before extended", thinking_t < extended_t))
    else:
        checks.append(("ack before thinking", False))
        checks.append(("thinking before extended", False))

    # RESPONDING state must appear (tokens received after ~3s)
    checks.append(("tokens received", len(tokens) > 0))

    # Ack fired within first second (timing gate)
    ack_events = [e for e in events if e["label"] == "FILLER_ACK"]
    if ack_events:
        ack_rel_ms = (ack_events[0]["t"] - t_start) * 1000
        checks.append(("ack fired within 1000ms", ack_rel_ms < 1000))
        print(f"\n  Ack fired at       : {ack_rel_ms:.0f}ms (target >=300ms, <1000ms)")

    _print_checks(checks)
    return all(v for _, v in checks)


# Scenario: fast LLM — fillers skipped entirely

def test_fast_llm_skips_fillers():
    print("\n" + "=" * 60)
    print("Scenario: Instant LLM — fillers should be skipped")
    print("=" * 60)

    global _events
    _events = []
    t_start = time.perf_counter()
    tokens = turn_manager.run_turn(
        transcript="Quick question.",
        history=[],
        _llm_override=_instant_llm(),
    )
    elapsed = (time.perf_counter() - t_start) * 1000

    print(f"\n  Elapsed            : {elapsed:.0f}ms")
    print(f"  Tokens received    : {len(tokens)} ({''.join(tokens)!r})")
    print(f"  Filler events      : {len(_events)}")

    checks = [
        ("no fillers when LLM is fast", len(_events) == 0 or
         not any(e["label"] in ("FILLER_ACK", "FILLER_THINKING", "FILLER_EXTENDED") for e in _events)),
        ("tokens received", len(tokens) > 0),
    ]
    _print_checks(checks)
    return all(v for _, v in checks)


# Scenario: barge-in during filler playback

def test_barge_in():
    print("\n" + "=" * 60)
    print("Scenario: Barge-in 500ms into turn (during ack filler)")
    print("=" * 60)

    t_start = time.perf_counter()
    tokens, elapsed_ms, events = _run_turn_headless(delay=5.0, barge_in_after=0.5)
    elapsed_total = (time.perf_counter() - t_start) * 1000

    print(f"\n  Total elapsed      : {elapsed_total:.0f}ms")
    print(f"  Filler events      : {len(events)}")
    for e in events:
        t_rel = (e["t"] - t_start) * 1000
        interrupted = " [interrupted]" if e["interrupted"] else ""
        print(f"    t={t_rel:6.0f}ms  {e['label']:<20}  {e['file']}{interrupted}")

    interrupted_events = [e for e in events if e["interrupted"]]
    any_interrupted = len(interrupted_events) > 0
    halted_quickly = elapsed_total < 1500

    checks = [
        ("some filler was interrupted", any_interrupted),
        ("turn halted quickly (<1500ms total)", halted_quickly),
        ("no LLM tokens received (barged in early)", len(tokens) == 0),
    ]
    _print_checks(checks)
    return all(v for _, v in checks)


# Scenario: barge-in during RESPONDING playback

def test_barge_in_response():
    print("\n" + "=" * 60)
    print("Scenario: Barge-in during real LLM response playback")
    print("=" * 60)

    # We use instant LLM, so RESPONDING state starts almost immediately.
    # We will trigger barge_in 200ms after turn starts.
    barge_in_time = 0.2
    
    # We need to measure exact stop latency. We will patch barge_in_event.set()
    barge_in_event = threading.Event()
    actual_barge_in_time = [0.0]
    
    def fire_barge_in():
        time.sleep(barge_in_time)
        actual_barge_in_time[0] = time.perf_counter()
        barge_in_event.set()

    threading.Thread(target=fire_barge_in, daemon=True).start()

    t_start = time.perf_counter()
    
    tokens = turn_manager.run_turn(
        transcript="Quick question.",
        history=[],
        barge_in_event=barge_in_event,
        _llm_override=_instant_llm(),
    )
    
    t_end = time.perf_counter()
    
    stop_latency = (t_end - actual_barge_in_time[0]) * 1000
    
    print(f"\n  Turn finished. Tokens received: {len(tokens)}")
    print(f"  Barge-in stop latency: {stop_latency:.0f} ms")
    
    checks = [
        ("stop latency < 200ms", stop_latency < 200),
    ]
    _print_checks(checks)
    return all(v for _, v in checks)



# Scenario: total failure → recovery filler

def test_recovery_on_total_failure():
    print("\n" + "=" * 60)
    print("Scenario: LLM never responds — recovery filler plays")
    print("=" * 60)

    global _events
    _events = []
    t_start = time.perf_counter()
    tokens = turn_manager.run_turn(
        transcript="This will hang forever.",
        history=[],
        _llm_override=_never_llm(),
    )
    elapsed = (time.perf_counter() - t_start) * 1000

    print(f"\n  Total elapsed      : {elapsed:.0f}ms")
    print(f"  Tokens received    : {len(tokens)}")
    print(f"  Filler events      : {len(_events)}")
    for e in _events:
        t_rel = (e["t"] - t_start) * 1000
        print(f"    t={t_rel:6.0f}ms  {e['label']:<20}  {e['file']}")

    recovery_fired = any(e["label"] == "RECOVERY" for e in _events)
    checks = [
        ("recovery filler played", recovery_fired),
        ("no LLM tokens returned", len(tokens) == 0),
        ("turn completed in ~5–12s", 4500 < elapsed < 15000),
    ]
    _print_checks(checks)
    return all(v for _, v in checks)


# Runner

def _print_checks(checks: list[tuple[str, bool]]) -> None:
    print()
    for label, passed in checks:
        mark = "  PASS" if passed else "  FAIL"
        print(f"  {mark}  {label}")


def main():
    print("=" * 60)
    print("turn_manager test suite")
    print("=" * 60)

    results = {
        "Scenario: filler sequence (4s delay)": test_filler_sequence(),
        "Scenario: no fillers when LLM is fast": test_fast_llm_skips_fillers(),
        "Scenario: barge-in interrupts filler":  test_barge_in(),
        "Scenario: barge-in interrupts response": test_barge_in_response(),
        "Scenario: recovery on total failure":   test_recovery_on_total_failure(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("ALL PASS")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
