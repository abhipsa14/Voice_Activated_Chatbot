"""
Raspberry Pi Voice Chatbot – Central Configuration
=====================================================
All tuneable settings in one place.  Every value is overridable
via an environment variable of the same name.

Optimised for Raspberry Pi 4 (2 GB RAM) with sub-second cached
response latency.
"""

import os
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
AUDIO_DIR = BASE_DIR / "audio"
CACHE_DIR = BASE_DIR / ".cache"
TTS_CACHE_DIR = BASE_DIR / ".tts_cache"

# Auto-create directories
for _d in (MODELS_DIR, AUDIO_DIR, CACHE_DIR, TTS_CACHE_DIR):
    _d.mkdir(exist_ok=True)

# ── Knowledge base ────────────────────────────────────────────────────
KB_TXT_FILE = BASE_DIR / "uit.txt"
KB_JSON_FILE = BASE_DIR / "knowledge_base.json"

# ── Wake word ──────────────────────────────────────────────────────────
WAKE_WORD = os.environ.get("WAKE_WORD", "hey uit")
WAKE_WORD_ALTERNATIVES = [
    "okay uit", "ok u i t", "okay u i t",
    "okay united", "ok united", "hey uit", "hey u i t",
]

# ── Voice Activity Detection (VAD) ────────────────────────────────────
SILENCE_THRESHOLD = int(os.environ.get("SILENCE_THRESHOLD", "450"))
SILENCE_DURATION = float(os.environ.get("SILENCE_DURATION", "1.2"))  # s
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 30          # ms per frame (WebRTC VAD wants 10/20/30)
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION_MS // 1000  # 480 samples
VAD_AGGRESSIVENESS = 2          # 0-3  (higher = more aggressive)
PRE_SPEECH_BUFFER_MS = 300      # ms of audio kept BEFORE speech onset
MAX_RECORDING_SECONDS = 15      # safety cap
WAKE_WORD_MAX_SECONDS = 3       # short clip for wake-word check

# ── Speech-to-Text (whisper.cpp) ──────────────────────────────────────
WHISPER_CPP_BIN = os.environ.get(
    "WHISPER_CPP_BIN",
    str(BASE_DIR / "whisper.cpp" / "main"),
)
WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL",
    str(MODELS_DIR / "ggml-tiny.en.bin"),
)
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "4"))

# ── LLM (Ollama) ──────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "tinyllama")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "150"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_SYSTEM_PROMPT = os.environ.get("LLM_SYSTEM_PROMPT", (
    "You are 'UIT Bot', the official voice assistant for United Institute "
    "of Technology, Prayagraj. Answer questions about the college. "
    "Keep answers short (1-3 sentences) because they are spoken aloud. "
    "Be friendly and helpful. If unsure, say you don't know."
))

# ── RAG pipeline ──────────────────────────────────────────────────────
RAG_CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_CONFIDENCE", "0.15"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# ── Caching (multi-level) ─────────────────────────────────────────────
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
MEMORY_CACHE_SIZE = int(os.environ.get("MEMORY_CACHE_SIZE", "500"))
MEMORY_CACHE_TTL = int(os.environ.get("MEMORY_CACHE_TTL", "86400"))  # 24 h

REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() in ("1", "true", "yes")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL = int(os.environ.get("REDIS_TTL", "604800"))  # 7 days

# ── Conversation history ──────────────────────────────────────────────
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "6"))

# ── Text-to-Speech (Piper) ────────────────────────────────────────────
PIPER_MODEL = os.environ.get(
    "PIPER_MODEL",
    str(MODELS_DIR / "en_US-lessac-medium.onnx"),
)
PIPER_CONFIG = os.environ.get(
    "PIPER_CONFIG",
    str(MODELS_DIR / "en_US-lessac-medium.onnx.json"),
)
PIPER_BIN = os.environ.get("PIPER_BIN", "piper")

# ── Audio playback ────────────────────────────────────────────────────
AUDIO_PLAYER = os.environ.get("AUDIO_PLAYER", "aplay")
TEMP_WAV = str(AUDIO_DIR / "temp.wav")
TTS_OUTPUT_WAV = str(AUDIO_DIR / "tts_output.wav")
