#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
#  UIT Prayagraj Voice Chatbot – One-Click Raspberry Pi Installer
# ════════════════════════════════════════════════════════════════
#
#  This script installs everything needed to run the voice chatbot:
#    1. System packages (portaudio, espeak, ffmpeg, redis, build tools)
#    2. Builds whisper.cpp from source (fast offline STT)
#    3. Downloads Whisper tiny.en model (~75 MB)
#    4. Installs Ollama + pulls TinyLlama 1.1B
#    5. Downloads Piper TTS voice model
#    6. Creates Python venv + installs pip packages
#    7. Generates knowledge base from uit.txt
#    8. Warms caches for instant first-run responses
#    9. Installs systemd service for auto-start on boot
#
#  Usage:
#    chmod +x setup.sh && ./setup.sh
#
#  Run as your normal user (NOT root). Uses sudo where needed.
# ════════════════════════════════════════════════════════════════

set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${INSTALL_DIR}/.venv"
MODELS_DIR="${INSTALL_DIR}/models"
WHISPER_DIR="${INSTALL_DIR}/whisper.cpp"
SERVICE_NAME="voicebot"
CURRENT_USER="$(whoami)"

echo "════════════════════════════════════════════════════════════"
echo "  🎓 UIT Prayagraj Voice Chatbot – Raspberry Pi Setup"
echo "════════════════════════════════════════════════════════════"
echo "  Directory : ${INSTALL_DIR}"
echo "  User      : ${CURRENT_USER}"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── 1. System dependencies ────────────────────────────────────────────
echo "📦 [1/9] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    python3-pyaudio portaudio19-dev \
    espeak-ng espeak flac ffmpeg wget curl git \
    build-essential cmake \
    libsdl2-mixer-2.0-0 libsdl2-2.0-0 \
    redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
echo "✅ System packages installed."
echo ""

# ── 2. Build whisper.cpp ──────────────────────────────────────────────
echo "🔨 [2/9] Building whisper.cpp..."
if [ ! -d "${WHISPER_DIR}" ]; then
    git clone https://github.com/ggerganov/whisper.cpp.git "${WHISPER_DIR}"
fi
cd "${WHISPER_DIR}"
if [ ! -f "${WHISPER_DIR}/main" ]; then
    make -j$(nproc)
fi
cd "${INSTALL_DIR}"
echo "✅ whisper.cpp built."
echo ""

# ── 3. Download Whisper model ─────────────────────────────────────────
echo "📥 [3/9] Downloading Whisper tiny.en model..."
mkdir -p "${MODELS_DIR}"
WHISPER_MODEL="${MODELS_DIR}/ggml-tiny.en.bin"
if [ ! -f "${WHISPER_MODEL}" ]; then
    wget -q --show-progress \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" \
        -O "${WHISPER_MODEL}"
fi
echo "✅ Whisper model ready ($(du -h "${WHISPER_MODEL}" | cut -f1))."
echo ""

# ── 4. Install Ollama + TinyLlama ─────────────────────────────────────
echo "🧠 [4/9] Installing Ollama + TinyLlama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   Starting Ollama..."
    ollama serve &
    sleep 5
fi
ollama pull tinyllama
echo "✅ Ollama + TinyLlama ready."
echo ""

# ── 5. Download Piper TTS voice ───────────────────────────────────────
echo "🔊 [5/9] Downloading Piper TTS voice..."
PIPER_ONNX="${MODELS_DIR}/en_US-lessac-medium.onnx"
PIPER_JSON="${MODELS_DIR}/en_US-lessac-medium.onnx.json"
if [ ! -f "${PIPER_ONNX}" ]; then
    wget -q --show-progress \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
        -O "${PIPER_ONNX}"
fi
if [ ! -f "${PIPER_JSON}" ]; then
    wget -q --show-progress \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        -O "${PIPER_JSON}"
fi
echo "✅ Piper voice ready."
echo ""

# ── 6. Python virtual environment + packages ─────────────────────────
echo "🐍 [6/9] Setting up Python environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
pip install --quiet -r "${INSTALL_DIR}/requirements.txt"
echo "✅ Python packages installed."
echo ""

# ── 7. Generate knowledge base ────────────────────────────────────────
echo "📄 [7/9] Generating knowledge base..."
if [ ! -f "${INSTALL_DIR}/knowledge_base.json" ]; then
    "${VENV_DIR}/bin/python" "${INSTALL_DIR}/parse_txt.py"
else
    echo "   knowledge_base.json already exists."
fi
echo "✅ Knowledge base ready."
echo ""

# ── 8. Warm caches ────────────────────────────────────────────────────
echo "🔥 [8/9] Pre-warming caches..."
mkdir -p "${INSTALL_DIR}/.tts_cache" "${INSTALL_DIR}/.cache" "${INSTALL_DIR}/audio"
"${VENV_DIR}/bin/python" -c "
import sys; sys.path.insert(0, '${INSTALL_DIR}')
from src.llm import LLMEngine
e = LLMEngine()
e.warm_cache()
print('Cache warmed.')
"
echo "✅ Caches warmed."
echo ""

# ── 9. Systemd service ───────────────────────────────────────────────
echo "⚙️  [9/9] Installing systemd service..."
sudo usermod -aG audio "${CURRENT_USER}" 2>/dev/null || true
systemctl --user enable pulseaudio.service 2>/dev/null || true
systemctl --user start pulseaudio.service 2>/dev/null || true
sudo loginctl enable-linger "${CURRENT_USER}" 2>/dev/null || true

USER_UID="$(id -u)"
TEMP_SERVICE="/tmp/${SERVICE_NAME}.service"
sed -e "s|User=pi|User=${CURRENT_USER}|g" \
    -e "s|Group=pi|Group=${CURRENT_USER}|g" \
    -e "s|WorkingDirectory=.*|WorkingDirectory=${INSTALL_DIR}|g" \
    -e "s|ExecStart=.*|ExecStart=${VENV_DIR}/bin/python -m src.main --no-wake-word|g" \
    -e "s|/run/user/1000/|/run/user/${USER_UID}/|g" \
    "${INSTALL_DIR}/systemd/voicebot.service" > "${TEMP_SERVICE}"

sudo cp "${TEMP_SERVICE}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
echo "✅ Service installed and enabled."
echo ""

# ── Done ──────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  ▶ Quick start:"
echo "    source .venv/bin/activate"
echo "    ollama serve &"
echo "    python -m src.main"
echo ""
echo "  ▶ Run as service:"
echo "    sudo systemctl start ${SERVICE_NAME}"
echo "    journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "  Say \"ok uit\" to activate, then ask your question."
echo "════════════════════════════════════════════════════════════"
