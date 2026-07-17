"""
Turn Manager — orchestrates one conversational turn.

Runs the LLM call concurrently with a timer-based filler injection system.
When the LLM starts yielding tokens, any active filler is stopped immediately
and playback transitions to the real response.
"""

import os
import struct
import threading
import time
from typing import Iterator

import numpy as np
import sounddevice as sd

import llm
import tts

# Paths

_FILLERS_DIR = os.path.join(os.path.dirname(__file__), "fillers")
_SAMPLE_RATE = 22050

# States (internal — not exposed to callers)

_STATE_THINKING     = "THINKING"
_STATE_FILLER_ACK   = "FILLER_ACK"
_STATE_FILLER_THINK = "FILLER_THINKING"
_STATE_FILLER_EXT   = "FILLER_EXTENDED"
_STATE_RESPONDING   = "RESPONDING"
_STATE_INTERRUPTED  = "INTERRUPTED"
_STATE_RECOVERY     = "RECOVERY"

# Filler rotation helpers

_last_filler_index: dict[str, int] = {}


def _pick_filler(category: str) -> str | None:
    """
    Return the path of the next filler clip for the given category, rotating
    through available clips so the same one is never played twice in a row.
    Returns None if no clips exist for that category.
    """
    clips = sorted(
        f for f in os.listdir(_FILLERS_DIR)
        if f.startswith(f"{category}_") and f.endswith(".wav")
    )
    if not clips:
        return None

    last = _last_filler_index.get(category, -1)
    next_idx = (last + 1) % len(clips)
    _last_filler_index[category] = next_idx
    return os.path.join(_FILLERS_DIR, clips[next_idx])


# WAV loading

def _load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file and return (float32 array, sample_rate)."""
    with open(path, "rb") as f:
        data = f.read()

    idx = data.find(b"fmt ")
    fmt_data = data[idx + 8: idx + 24]
    _, num_channels, sr, _, _, bits = struct.unpack("<HHIIHH", fmt_data)

    idx = data.find(b"data")
    chunk_size = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = np.frombuffer(data[idx + 8: idx + 8 + chunk_size], dtype=np.int16)

    if num_channels > 1:
        pcm = pcm[::num_channels]

    return pcm.astype(np.float32) / 32768.0, sr


# Interruptible filler player

_CHUNK_FRAMES = 2048  # ~93ms per chunk at 22050 Hz — small enough to check stop flag promptly


def _play_filler(path: str, stop_event: threading.Event, label: str) -> None:
    """
    Play a WAV filler clip in small chunks.  Stops immediately when stop_event
    is set — the current ~93ms chunk finishes and then playback halts.
    """
    audio, sr = _load_wav(path)
    offset = 0
    while offset < len(audio):
        if stop_event.is_set():
            sd.stop()
            break
        chunk = audio[offset: offset + _CHUNK_FRAMES]
        sd.play(chunk, samplerate=sr)
        sd.wait()
        offset += _CHUNK_FRAMES


# LLM worker

def _collect_llm_stream(
    gen: Iterator[str],
    token_queue: "list[str | None]",
    lock: threading.Lock,
    first_token_event: threading.Event,
    done_event: threading.Event,
    stop_filler_event: threading.Event | None,
    my_id: str,
    winner_box: list[str | None],
) -> None:
    """
    Drain the LLM token generator on a background thread.
    - Uses winner_box to ensure only the first thread to yield a token continues.
    - Puts None as a sentinel when the stream is exhausted, then sets done_event.
    """
    try:
        for token in gen:
            with lock:
                if winner_box[0] is not None and winner_box[0] != my_id:
                    return  # Another stream won, exit silently
                if winner_box[0] is None:
                    winner_box[0] = my_id

                token_queue.append(token)

            if not first_token_event.is_set():
                first_token_event.set()
                if stop_filler_event:
                    stop_filler_event.set()
    except Exception as e:
        print(f"  [turn_manager] LLM stream error ({my_id}): {e}")
        return  # Do not set done_event or append None if we errored out before winning

    # Only the winner should terminate the stream
    with lock:
        if winner_box[0] == my_id or winner_box[0] is None:
            winner_box[0] = my_id
            token_queue.append(None)  # sentinel
            done_event.set()


def _drain_token_queue(
    token_queue: "list[str | None]",
    lock: threading.Lock,
    done_event: threading.Event,
) -> Iterator[str]:
    """
    Yield tokens from the shared queue as they become available.
    Blocks lightly between polls.  Stops when it sees the None sentinel.
    """
    while True:
        with lock:
            if token_queue:
                token = token_queue.pop(0)
                if token is None:
                    return
                yield token
            elif done_event.is_set():
                return
        time.sleep(0.005)


# Public API

def run_turn(
    transcript: str,
    history: list[dict],
    barge_in_event: threading.Event | None = None,
    _llm_override: Iterator[str] | None = None,
) -> list[str]:
    """
    Orchestrate one conversational turn.

    Parameters
    ----------
    transcript      : The user's transcribed utterance.
    history         : Conversation history (list of role/content dicts).
    barge_in_event  : Optional threading.Event; set by the VAD thread when the
                      user starts speaking mid-playback.  If None, barge-in is
                      disabled for this turn.
    _llm_override   : Internal hook used by tests to inject a fake LLM stream
                      (e.g. one with an artificial delay) without touching llm.py.

    Returns
    -------
    A list of response tokens (the full LLM response, in order).
    """

    t0 = time.perf_counter()

    # Shared state between filler thread and main thread
    stop_filler   = threading.Event()   # set → stop whatever filler is playing
    first_token   = threading.Event()   # set → LLM has started streaming
    llm_done      = threading.Event()   # set → LLM stream fully exhausted
    token_queue: list[str | None] = []
    lock          = threading.Lock()
    state         = [_STATE_THINKING]   # mutable wrapper so nested fns can write it
    winner_box: list[str | None] = [None] # tracks which LLM stream won

    def elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000

    # Start LLM in the background
    raw_gen = _llm_override if _llm_override is not None else llm.generate_stream(transcript, history)

    llm_thread = threading.Thread(
        target=_collect_llm_stream,
        args=(raw_gen, token_queue, lock, first_token, llm_done, stop_filler, "local", winner_box),
        daemon=True,
    )
    llm_thread.start()

    # Monitor barge-in
    if barge_in_event:
        def monitor_barge_in():
            barge_in_event.wait()
            stop_filler.set()
        threading.Thread(target=monitor_barge_in, daemon=True).start()

    collected_tokens: list[str] = []

    # Filler scheduling loop (runs on this thread)

    def play_if_not_interrupted(category: str, new_state: str) -> bool:
        """
        Play a filler clip if tokens haven't arrived yet.
        Returns True if the filler ran (even if interrupted mid-way).
        Returns False if tokens had already arrived before we started.
        """
        if stop_filler.is_set():
            return False
        path = _pick_filler(category)
        if path is None:
            return False
        state[0] = new_state
        print(f"  [turn_manager] {new_state}: playing {os.path.basename(path)}")
        _play_filler(path, stop_filler, new_state)
        return True

    # 300 ms gate — wait, then play ack if no tokens yet
    _wait_or_token(stop_filler, 0.300)
    if not stop_filler.is_set():
        play_if_not_interrupted("ack", _STATE_FILLER_ACK)

    # 1000 ms gate
    remaining_ms = 1000 - elapsed_ms()
    if remaining_ms > 0:
        _wait_or_token(stop_filler, remaining_ms / 1000)
    if not stop_filler.is_set():
        play_if_not_interrupted("thinking", _STATE_FILLER_THINK)

    # 2000 ms gate
    remaining_ms = 2000 - elapsed_ms()
    if remaining_ms > 0:
        _wait_or_token(stop_filler, remaining_ms / 1000)
    if not stop_filler.is_set():
        # Trigger cloud escalation in parallel
        print("  [turn_manager] FILLER_EXTENDED: escalating to cloud LLM")
        _escalate_to_cloud(transcript, history, token_queue, lock, first_token, llm_done, winner_box, stop_filler)
        play_if_not_interrupted("extended", _STATE_FILLER_EXT)

    # Wait up to ~5s total for tokens or give up (recovery)
    remaining_ms = 5000 - elapsed_ms()
    if remaining_ms > 0 and not stop_filler.is_set():
        _wait_or_token(stop_filler, remaining_ms / 1000)

    # Handle barge-in if the user spoke mid-filler
    if barge_in_event and barge_in_event.is_set():
        stop_filler.set()
        sd.stop()
        state[0] = _STATE_INTERRUPTED
        print(f"  [turn_manager] {_STATE_INTERRUPTED}: barge-in detected, halting playback")
        return collected_tokens

    # Handle recovery if no tokens arrived within ~5s
    if not first_token.is_set():
        state[0] = _STATE_RECOVERY
        path = _pick_filler("recovery")
        if path:
            print(f"  [turn_manager] {_STATE_RECOVERY}: playing {os.path.basename(path)}")
            _play_filler(path, threading.Event(), _STATE_RECOVERY)  # not interruptible at this point
        return collected_tokens

    # Tokens are arriving, so stop any filler and switch to RESPONDING
    stop_filler.set()
    sd.stop()
    state[0] = _STATE_RESPONDING
    print(f"  [turn_manager] {_STATE_RESPONDING}: real tokens arriving, elapsed={elapsed_ms():.0f}ms")

    # Stream tokens through TTS
    def token_gen() -> Iterator[str]:
        for tok in _drain_token_queue(token_queue, lock, llm_done):
            collected_tokens.append(tok)
            yield tok

    audio_stream = tts.synthesize_stream(token_gen())
    for audio_chunk in audio_stream:
        if state[0] == _STATE_INTERRUPTED:
            break
            
        offset = 0
        while offset < len(audio_chunk):
            if barge_in_event and barge_in_event.is_set():
                sd.stop()
                state[0] = _STATE_INTERRUPTED
                print(f"  [turn_manager] {_STATE_INTERRUPTED}: barge-in during response")
                break
                
            chunk = audio_chunk[offset: offset + _CHUNK_FRAMES]
            sd.play(chunk, samplerate=22050)
            sd.wait()
            offset += _CHUNK_FRAMES

    return collected_tokens


# Internal helpers

def _wait_or_token(event: threading.Event, timeout: float) -> None:
    """Block until the event fires or the timeout expires, whichever is first."""
    event.wait(timeout=max(0.0, timeout))


def _escalate_to_cloud(
    transcript: str,
    history: list[dict],
    token_queue: "list[str | None]",
    lock: threading.Lock,
    first_token: threading.Event,
    llm_done: threading.Event,
    winner_box: list[str | None],
    stop_filler: threading.Event,
) -> None:
    """
    Start a cloud LLM stream. The local thread is left to finish (or time out) harmlessly.
    The `winner_box` ensures only the first stream to yield tokens gets to write to the queue.
    """
    try:
        cloud_gen = llm.generate_stream_cloud(transcript, history)
    except EnvironmentError as exc:
        print(f"  [turn_manager] cloud escalation skipped: {exc}")
        return

    cloud_thread = threading.Thread(
        target=_collect_llm_stream,
        args=(cloud_gen, token_queue, lock, first_token, llm_done, stop_filler, "cloud", winner_box),
        daemon=True,
    )
    cloud_thread.start()
