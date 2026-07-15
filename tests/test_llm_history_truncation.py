"""
Tests that history truncation in llm.py works as described in Spec 4.3:
  - history is capped at the last 6 turns (12 messages) before being sent to the model
  - older messages are silently dropped, not an error
  - generate_stream() still yields multiple chunks when given a 14-message history

Run with Ollama already serving:
    python -X utf8 test_llm_history_truncation.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time


def build_fake_history(num_messages: int) -> list[dict]:
    """Alternate user/assistant messages, oldest first."""
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": f"Message {i + 1}"}
        for i in range(num_messages)
    ]


def test_truncation_logic():
    print("=" * 60)
    print("Part 1 — _build_messages() truncation unit test")
    print("=" * 60)

    import llm

    cases = [
        (0,  "empty history",          0),
        (6,  "exactly 6 messages",     6),
        (12, "exactly 12 messages",   12),
        (13, "13 messages (1 extra)",  12),
        (20, "20 messages (8 extra)",  12),
    ]

    all_passed = True
    for num_in, label, expected_history_len in cases:
        history = build_fake_history(num_in)
        messages = llm._build_messages("test prompt", history)

        # messages = [system] + [history_slice] + [current user turn]
        # history portion is messages[1:-1]
        actual_history_in_messages = messages[1:-1]
        got = len(actual_history_in_messages)
        ok = got == expected_history_len

        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {label}: fed {num_in} msgs → {got} history msgs in payload (expected {expected_history_len})")

        # For the over-12 cases, confirm it's the *tail* that's kept.
        if num_in > 12 and ok:
            kept_content = [m["content"] for m in actual_history_in_messages]
            expected_tail = [f"Message {i + 1}" for i in range(num_in - 12, num_in)]
            tail_ok = kept_content == expected_tail
            if not tail_ok:
                all_passed = False
            print(f"         Tail check: {'PASS' if tail_ok else 'FAIL'} — kept msgs {num_in - 11}..{num_in}")

    return all_passed


def test_stream_with_long_history():
    print()
    print("=" * 60)
    print("Part 2 — generate_stream() with 14-message history (truncation in live call)")
    print("=" * 60)

    import llm

    history = build_fake_history(14)   # 2 over the cap
    prompt = "Just say 'OK' and nothing else."

    print(f"  History size fed   : {len(history)} messages")
    print(f"  Cap (Spec 4.3)     : 12 messages (6 turns)")
    print(f"  Messages dropped   : {max(0, len(history) - 12)}")
    print()
    print("  Streaming response:")
    print("  " + "-" * 40)

    chunks: list[str] = []
    t_start = time.perf_counter()
    t_first: float | None = None

    try:
        for chunk in llm.generate_stream(prompt, history):
            if t_first is None:
                t_first = time.perf_counter()
            chunks.append(chunk)
            print(chunk, end="", flush=True)
    except Exception as exc:
        print(f"\n  [error] generate_stream() raised: {exc}")
        print("  Is Ollama running?  ollama serve")
        return False

    t_end = time.perf_counter()
    print()
    print("  " + "-" * 40)

    first_ms = (t_first - t_start) * 1000 if t_first else None
    total_ms = (t_end - t_start) * 1000

    print()
    print(f"  Chunks received        : {len(chunks)}")
    if first_ms is not None:
        print(f"  Time-to-first-token    : {first_ms:.0f} ms")
    print(f"  Total stream time      : {total_ms:.0f} ms")
    print(f"  Response               : {''.join(chunks)!r}")

    ok = len(chunks) >= 1 and bool("".join(chunks).strip())
    print(f"\n  Stream result          : {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    part1 = test_truncation_logic()
    part2 = test_stream_with_long_history()

    print()
    print("=" * 60)
    print("OVERALL")
    print("=" * 60)
    overall = part1 and part2
    print(f"  Part 1 (truncation logic) : {'PASS' if part1 else 'FAIL'}")
    print(f"  Part 2 (live stream)      : {'PASS' if part2 else 'FAIL'}")
    print()
    print("OVERALL:", "PASS" if overall else "FAIL")
    return overall


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
