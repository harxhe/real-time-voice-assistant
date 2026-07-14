import os
import re
from typing import Iterator

import numpy as np

# Piper TTS will be imported lazily
PiperVoice = None

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "en_US-lessac-medium.onnx")
_voice = None


def _get_voice():
    global _voice, PiperVoice
    if _voice is None:
        try:
            from piper import PiperVoice
        except ImportError:
            raise ImportError("piper-tts is not installed. Please run: pip install piper-tts")
            
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {_MODEL_PATH}. "
                "Please download the .onnx and .json config into the models/ directory."
            )
        _voice = PiperVoice.load(_MODEL_PATH)
    return _voice


def synthesize(text: str) -> np.ndarray:
    """Non-streaming version for pre-generating filler clips."""
    voice = _get_voice()
    arrays = []
    for chunk in voice.synthesize(text):
        arrays.append(chunk.audio_float_array)
    return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)


def synthesize_stream(text_stream: Iterator[str]) -> Iterator[np.ndarray]:
    """
    Synthesize audio sentence-by-sentence as text arrives.
    
    Reads tokens from the LLM generator, buffers them until a sentence boundary
    is reached, synthesizes that sentence, and yields the resulting audio chunk.
    """
    voice = _get_voice()
    buffer = ""
    
    # Matches sentence endings (., ?, !) optionally followed by quotes, then at least one whitespace character.
    sentence_end_re = re.compile(r'([.?!]["\']?\s+)')
    
    for token in text_stream:
        buffer += token
        
        # Continually check if the buffer contains a full sentence
        while True:
            match = sentence_end_re.search(buffer)
            if not match:
                break
                
            end_idx = match.end()
            sentence = buffer[:end_idx].strip()
            buffer = buffer[end_idx:]
            
            if sentence:
                # Synthesize this sentence
                for chunk in voice.synthesize(sentence):
                    yield chunk.audio_float_array
                    
    # Flush any remaining text in the buffer after the stream ends
    buffer = buffer.strip()
    if buffer:
        for chunk in voice.synthesize(buffer):
            yield chunk.audio_float_array
