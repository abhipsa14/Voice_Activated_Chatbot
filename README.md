# 🎓 UIT Prayagraj Chatbot

An AI-powered Q&A chatbot for **United Institute of Technology, Prayagraj**. It converts text content into a structured knowledge base and answers questions using TF-IDF + fuzzy matching. Supports voice input/output and a wake-word mode ("Hey UIT") for hands-free operation on Raspberry Pi.

---

## ✨ Features

- **Text-to-JSON parsing** — Converts Q&A content from `uit.txt` into structured JSON
- **Smart matching** — TF-IDF + cosine similarity + fuzzy keyword matching with synonyms
- **Handles general questions** — No need for exact phrasing; similar/approximate questions work
- **Human-like voice** — Microsoft Edge neural TTS voices (Indian English, US, UK)
- **Accurate speech recognition** — OpenAI Whisper on desktop; Google Speech API on Pi
- **Wake word detection** — Say "Hey UIT" to activate (always-on daemon mode)
- **Raspberry Pi ready** — Auto-detects ARM, skips incompatible libraries, runs as systemd service

---

## 📁 Project Structure

```
chatbot/
├── chatbot.py            # Main chatbot with CLI and voice modes
├── parse_txt.py          # Converts uit.txt → knowledge_base.json
├── knowledge_base.json   # Auto-generated structured Q&A data
├── uit.txt               # Source Q&A text file
├── tts_module.py         # Text-to-Speech (Edge TTS + pyttsx3 fallback)
├── stt_module.py         # Speech-to-Text (Whisper on desktop, Google on Pi)
├── wake_word.py          # Wake word detector ("Hey UIT")
├── setup_pi.sh           # One-command Raspberry Pi installer
├── uit-chatbot.service   # systemd service file for auto-start
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quick Start (Windows / Linux / Mac)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/chatbot.git
cd chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Activate it:
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Install dependencies

**Text-only mode** — no extra installs needed (uses Python standard library).

**For voice features (desktop / Windows / x86 Linux):**

```bash
pip install edge-tts pygame pyttsx3 openai-whisper SpeechRecognition PyAudio
```

**For voice features (Raspberry Pi / ARM):**

```bash
sudo apt-get install -y python3-pyaudio portaudio19-dev espeak flac ffmpeg
pip install edge-tts pygame pyttsx3 SpeechRecognition PyAudio
```

> ⚠️ **Do NOT install `openai-whisper` on Raspberry Pi** — PyTorch causes an "Illegal instruction" crash on ARM. The chatbot auto-detects ARM and uses Google Speech API instead.

### 4. Run the chatbot

```bash
python chatbot.py
```

The knowledge base (`knowledge_base.json`) is auto-generated from `uit.txt` on first run.

---

## 🎮 Run Modes

| Command | Description |
|---|---|
| `python chatbot.py` | Text-only mode (type questions, read answers) |
| `python chatbot.py --voice` | Full voice mode (speak into mic + hear answers) |
| `python chatbot.py --tts` | Type questions, hear answers spoken aloud |
| `python chatbot.py --stt` | Speak questions, read answers on screen |
| `python chatbot.py --daemon` | Always-on: waits for "Hey UIT" wake word |
| `python chatbot.py --daemon --wake-word "hello bot"` | Custom wake word |

---

## 🎤 Voice Commands (during chat)

| Command | Action |
|---|---|
| `categories` | List all available Q&A topics |
| `voices` | Show available voice presets |
| `voice indian_male` | Switch TTS voice (e.g., indian_male, us_female, uk_male) |
| `type` | Switch to keyboard input temporarily (in voice mode) |
| `quit` / `exit` / `bye` | Exit the chatbot |

### Available Voice Presets

| Preset | Voice |
|---|---|
| `indian_female` | en-IN-NeerjaNeural (default) |
| `indian_male` | en-IN-PrabhatNeural |
| `us_female` | en-US-JennyNeural |
| `us_male` | en-US-GuyNeural |
| `uk_female` | en-GB-SoniaNeural |
| `uk_male` | en-GB-RyanNeural |

---

## 🍓 Raspberry Pi Setup

### Option A: One-command installer

```bash
# Copy files to Pi (from your PC)
scp -r chatbot/ pi@<pi-ip>:/home/pi/chatbot/

# SSH into Pi and run
ssh pi@<pi-ip>
cd ~/chatbot
chmod +x setup_pi.sh
./setup_pi.sh
```

The script installs all system packages, Python dependencies, and sets up the systemd service automatically.

> ℹ️ Whisper is **not** installed on Pi (PyTorch is incompatible with ARM). STT and wake-word detection use Google Speech API instead (requires internet).

### Option B: Manual setup

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip \
    python3-pyaudio portaudio19-dev espeak flac ffmpeg

# Python setup
python3 -m venv .venv
source .venv/bin/activate
# ⚠️ Do NOT install openai-whisper on Pi — it crashes on ARM
pip install edge-tts pygame pyttsx3 SpeechRecognition PyAudio

# Run
python chatbot.py --daemon
```

### Systemd Service (auto-start on boot)

```bash
# Install service
sudo cp uit-chatbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable uit-chatbot
sudo systemctl start uit-chatbot

# Manage service
sudo systemctl status uit-chatbot    # Check status
sudo systemctl stop uit-chatbot      # Stop
sudo systemctl restart uit-chatbot   # Restart
journalctl -u uit-chatbot -f         # View live logs
```

> **Note:** Edit `uit-chatbot.service` to update paths and username if your setup differs from the defaults.

---

## 🔧 Customization

### Add your own Q&A content

1. Edit `uit.txt` — add questions and answers in this format:
   ```
   Q36. Your question here?
   A: Your answer here.
   ```
2. Delete `knowledge_base.json` (it will be regenerated)
3. Restart the chatbot

### Change the wake word

```bash
python chatbot.py --daemon --wake-word "hello assistant"
```

### Adjust matching sensitivity

In `chatbot.py`, modify `UITChatbot.SIMILARITY_THRESHOLD` (lower = more lenient, higher = stricter):
```python
SIMILARITY_THRESHOLD = 0.08  # default
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `edge-tts` | Neural TTS with human-like voices (needs internet) |
| `pygame` | Audio playback for TTS output |
| `pyttsx3` | Offline TTS fallback (espeak/SAPI5) |
| `openai-whisper` | Offline speech recognition — **desktop only** (crashes on Pi ARM) |
| `SpeechRecognition` | Microphone input framework |
| `PyAudio` | Microphone hardware access |

---

## 📝 License

This project is for educational purposes at United Institute of Technology, Prayagraj.
