# Real-Time Voice Conversational AI Assistant

An offline-first, real-time, audio-in/audio-out conversational assistant.

## Architecture

```
Mic → VAD → Streaming ASR → Turn Manager ──→ Streaming LLM ──→ Streaming TTS → Speaker
                                   │
                                   ├─ (if LLM slow) → Filler Audio Cache → Speaker
                                   └─ (if local LLM fails/escalates) → Cloud LLM fallback
```

The system is orchestrated by a **Turn Manager** that handles state transitions, filler injections, and barge-in (allowing users to interrupt the assistant mid-sentence).

## Offline/Online Split

This prototype is designed to be **offline-first**:
- **Offline:** Voice Activity Detection (Silero), Automatic Speech Recognition (faster-whisper), Primary LLM (Ollama with phi3:mini), and Text-to-Speech (Piper) run entirely locally on your hardware.
- **Online:** A cloud fallback LLM (Groq API using llama-3.1-8b-instant) is available but is *only* triggered if the local LLM fails or takes longer than 2 seconds to generate the first token.

## Measured Latency

Based on our final pipeline testing:
- **ASR Latency:** ~300ms - 900ms (depending on query length)
- **Local LLM Time-to-First-Token (TTFT):** ~2100ms
- **Filler Injection:** 
  - `ack` filler triggers at 300ms
  - `thinking` filler triggers at 1000ms
  - `extended` filler and Cloud escalation triggers at 2000ms
- **Barge-in Latency:** ~88ms from speech detection to audio cutoff

*Note: Because of the aggressive filler injection, the system never has "dead air" even if the local LLM takes 2+ seconds to respond.*

## Explicit Scope Cuts

The following features were explicitly cut from this prototype:
- Multi-user or multi-session support.
- Persistent memory across sessions (history is capped to the last 6 turns).
- Deployment packaging (e.g., Docker, mobile app).
- Noise robustness or accent tuning beyond default model capabilities.
- A graphical user interface (GUI).
- Dynamic cloud escalation (escalation is strictly timer-based, not based on query complexity).

## Production & Scaling

For details on how this local-first prototype can be adapted for concurrent usage, edge devices, and fully scaled production deployments, please refer to the [Scaling and Production Plan](SCALING.md).
