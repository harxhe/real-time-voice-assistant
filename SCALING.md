# Scaling and Production Plan

This document outlines the steps required to take this offline-first prototype into a robust production environment.

## 1. Cloud-Native vs. Edge Deployment
Currently, the pipeline is entirely local. For production, the architecture must be split based on the target deployment environment:
- **Edge Devices (Smart Speakers, Mobile):** The VAD and wakeword detection must remain on-device for privacy and latency. ASR, LLM, and TTS should be offloaded to a cloud backend via WebSockets or WebRTC to minimize battery and compute usage.
- **Local-First Privacy mode:** If the goal is a fully private appliance, the current hardware stack requires a dedicated GPU (e.g., NVIDIA Jetson Orin) to maintain sub-second latency for the LLM and TTS components.

## 2. Infrastructure & Concurrency
The current `turn_manager.py` uses basic Python threading and blocking queues. To support multiple concurrent users (e.g., as a backend service):
- Replace `sounddevice` and local PyAudio blocking calls with asynchronous WebSocket streams (e.g., FastAPI + WebSockets).
- Rewrite the Turn Manager to use `asyncio` for non-blocking I/O and state transitions.
- Deploy the ASR, LLM, and TTS models as separate scalable microservices (e.g., vLLM for the language model, Triton Inference Server for Whisper).

## 3. Dynamic Escalation & Routing
The prototype uses a hardcoded 2-second timer to escalate from the local LLM to the cloud. In production:
- Implement query classification: route complex queries (e.g., math, current events) directly to the cloud LLM immediately after ASR.
- Use a semantic cache (e.g., Redis + vector search) to return instant answers for frequently asked questions without hitting the LLM.

## 4. Acoustic Echo Cancellation (AEC)
The current barge-in implementation relies on the user wearing headphones to prevent the microphone from picking up the assistant's own TTS output. In a production physical device:
- A DSP-level Acoustic Echo Cancellation (AEC) module must be added *before* the VAD stage.
- The TTS output stream must be fed back into the AEC as a reference signal to successfully subtract the assistant's voice from the microphone input.

## 5. Session Management & Memory
To support long-term usage:
- Integrate a database (e.g., PostgreSQL) to persist user sessions and preferences.
- Implement a vector database (e.g., Pinecone, Milvus) for long-term memory retrieval, replacing the current naive 6-turn sliding window.
