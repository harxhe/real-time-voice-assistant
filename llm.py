import os
from typing import Iterator
from openai import OpenAI

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_LOCAL_MODEL = "phi3:mini"
_CLOUD_MODEL = "llama-3.1-8b-instant"

_SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Keep responses conversational and concise — "
    "2 to 4 sentences unless the user clearly asks for more detail. Avoid lists, markdown, or "
    "formatting since your output will be spoken aloud, not read."
)

_local_client: OpenAI | None = None
_cloud_client: OpenAI | None = None


def _get_local_client() -> OpenAI:
    global _local_client
    if _local_client is None:
        _local_client = OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")
    return _local_client


def _get_cloud_client() -> OpenAI:
    global _cloud_client
    if _cloud_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                            break
                            
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it or add it to .env before using the cloud fallback path."
            )
        _cloud_client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    return _cloud_client


def _build_messages(prompt: str, history: list[dict]) -> list[dict]:
    truncated = history[-12:] if len(history) > 12 else history
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(truncated)
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_stream(prompt: str, history: list[dict]) -> Iterator[str]:
    messages = _build_messages(prompt, history)
    stream = _get_local_client().chat.completions.create(
        model=_LOCAL_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_stream_cloud(prompt: str, history: list[dict]) -> Iterator[str]:
    messages = _build_messages(prompt, history)
    stream = _get_cloud_client().chat.completions.create(
        model=_CLOUD_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
