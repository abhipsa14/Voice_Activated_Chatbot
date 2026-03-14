# United Institute Technology, Prayagraj Voice Chatbot

A Voice-enabled fully offline voice assistant for **United Institute of Technology, Prayagraj** — runs on a Raspberry Pi 4/5.
**No cloud, no API keys, no internet required after setup.**

```
You speak → Whisper STT → RAG + TinyLlama LLM → Piper TTS → Pi speaks back
```

---

## ✨ Features

- **Wake word activation** — Say **"Ok UIT"** to activate (like Alexa)
- **Instant answers** — RAG retrieval over UIT knowledge base (~0 ms for cached answers)
- **LLM fallback** — Ollama + TinyLlama for questions outside the knowledge base
- **3-tier caching** — L1 memory + L2 disk + L3 Redis for sub-second latency
- **TTS audio caching** — Repeated responses play back instantly (0 ms)
- **Improved VAD** — WebRTC VAD with ring-buffer pre-capture (no clipped words)
- **Adaptive noise calibration** — Auto-adjusts to ambient noise levels
- **Hands-free operation** — Always-on daemon mode via systemd
- **Raspberry Pi optimised** — Everything runs on Pi 4 (2 GB RAM)

---

## Hardware Requirements

| Component | Minimum |
|-----------|---------|
| Raspberry Pi | Pi 4 (2 GB RAM) or Pi 5 |
| Microphone | USB microphone or USB audio adapter |
| Speaker | 3.5 mm jack, USB, or HDMI |
| Storage | 16 GB microSD (32 GB recommended) |

---

## Quick Start

```bash
git clone https://github.com/you/chatbot
cd chatbot
chmod +x setup.sh
./setup.sh
```

After setup finishes:

```bash
source .venv/bin/activate
ollama serve &        # start Ollama in background
python -m src.main
```

Say **"Ok UIT"** to activate, then ask your question.

---

## How the Pipeline Works

```
┌──────────┐   WAV   ┌────────────┐  text  ┌─────────────┐  reply  ┌──────────┐
│  Mic/VAD │ ──────► │ Whisper STT│ ─────► │ RAG + LLM   │ ──────► │ Piper TTS│
│ (WebRTC) │         │ (whisper.  │        │ + 3-tier $  │         │ + cache  │
└──────────┘         │    cpp)    │        └─────────────┘         └──────────┘
                     └────────────┘
```

### 1 · Voice Activity Detection (`src/vad.py`)
- **WebRTC VAD** (Google's algorithm) — primary, fast, accurate
- **Ring-buffer pre-capture** — keeps 300 ms before speech onset (no clipped words)
- **Adaptive noise calibration** — auto-adjusts threshold at startup
- Falls back to RMS energy detection if `webrtcvad` unavailable

### 2 · Speech-to-Text (`src/stt.py`)
- Calls **whisper.cpp** C++ binary as subprocess (~5-10× faster than Python Whisper on ARM)
- Model: `ggml-tiny.en.bin` (~75 MB, ~1-2 s on Pi 4)
- Falls back to Python Whisper → Google Speech API

### 3 · Answer Engine (`src/llm.py`)
- **Stage 1 — RAG**: TF-IDF + fuzzy keyword retrieval over UIT knowledge base (instant)
- **Stage 2 — LLM**: Falls back to Ollama (TinyLlama 1.1B Q4) for out-of-KB questions
- **3-tier cache**: L1 memory (0 ms) → L2 disk (1-5 ms) → L3 Redis (1-2 ms)
- Rolling conversation history (last 6 turns)

### 4 · Text-to-Speech (`src/tts.py`)
- **Piper** neural TTS (~100 ms on Pi 4) — primary
- **TTS audio cache** — every response saved as WAV, replayed instantly on repeat
- Falls back to Edge TTS → espeak

---

## Configuration (`config/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `WAKE_WORD` | `"ok uit"` | Phrase to trigger the bot |
| `SILENCE_THRESHOLD` | `450` | RMS energy to detect silence |
| `SILENCE_DURATION` | `1.2` | Seconds of silence to stop recording |
| `OLLAMA_MODEL` | `"tinyllama"` | Any Ollama model you've pulled |
| `LLM_MAX_TOKENS` | `150` | Keep short for fast TTS |
| `CACHE_ENABLED` | `True` | Toggle multi-level caching |
| `RAG_CONFIDENCE_THRESHOLD` | `0.15` | Min score for RAG match |
| `MAX_HISTORY_TURNS` | `6` | Conversation memory depth |
| `PRE_SPEECH_BUFFER_MS` | `300` | Audio kept before speech onset |

All settings are overridable via environment variables.

---

## Run Modes

| Command | Description |
|---------|-------------|
| `python -m src.main` | Voice mode with "Ok UIT" wake word |
| `python -m src.main --no-wake-word` | Always listening (no wake word) |
| `python -m src.main --text` | Text-only mode (type/read) |
| `python -m src.main --list-mics` | List available microphones |
| `python -m src.main --mic 2` | Use specific microphone index |

---

## Project Structure

```
chatbot/
├── setup.sh                  # One-click Raspberry Pi installer
├── requirements.txt          # Python dependencies
├── config/
│   ├── __init__.py
│   └── config.py             # All settings in one place
├── src/
│   ├── __init__.py
│   ├── main.py               # Main voice loop (Alexa-like)
│   ├── vad.py                # Voice activity detection (WebRTC + ring buffer)
│   ├── stt.py                # whisper.cpp STT wrapper
│   ├── llm.py                # RAG + Ollama LLM + 3-tier cache
│   ├── tts.py                # Piper TTS + audio cache
│   └── cache.py              # Multi-level cache manager
├── parse_txt.py              # Converts uit.txt → knowledge_base.json
├── knowledge_base.json       # Auto-generated Q&A data
├── uit.txt                   # Source knowledge base
├── models/                   # Whisper + Piper models
├── audio/                    # Temp WAV files (auto-created)
├── .cache/                   # Disk cache (auto-created)
├── .tts_cache/               # Cached TTS audio (auto-created)
└── systemd/
    └── voicebot.service      # Auto-start on boot
```

---

## Performance on Raspberry Pi 4 (2 GB)

| Stage | Cached | Cold |
|-------|--------|------|
| VAD + recording | Real-time | Real-time |
| Whisper STT (tiny.en) | — | ~1.5 s |
| RAG retrieval (cached) | < 1 ms | ~5 ms |
| LLM TinyLlama (cached) | < 1 ms | ~3-6 s |
| Piper TTS (cached) | < 1 ms | ~100-300 ms |
| **Total (cached, RAG)** | **< 1.5 s** | — |
| **Total (cold, LLM)** | — | **~5-8 s** |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No audio input | `arecord -l` to list devices; use `--mic <index>` |
| Ollama not responding | Run `ollama serve` first; check `curl localhost:11434` |
| Whisper binary not found | Run `cd whisper.cpp && make` |
| Redis connection refused | `sudo systemctl start redis-server` |
| Very slow responses | Use smaller model or reduce `LLM_MAX_TOKENS` |
| Wake word not detected | Lower `SILENCE_THRESHOLD` in config, or speak louder |

---

## Run on Boot (systemd)

```bash
sudo cp systemd/voicebot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voicebot
journalctl -u voicebot -f    # watch logs
```

---

## 📝 License

This project is for educational purposes at United Institute of Technology, Prayagraj.
