#!/usr/bin/env bash
# ============================================================
#  UIT Prayagraj Chatbot – Raspberry Pi Installer
# ============================================================
#  This script:
#    1. Installs system dependencies (espeak, portaudio, ffmpeg, flac)
#    2. Creates a Python virtual environment
#    3. Installs Python packages (edge-tts, pygame, SpeechRecognition, etc.)
#    4. Installs & enables the systemd service
#
#  NOTE: Whisper / PyTorch is NOT installed on ARM (causes "Illegal instruction").
#        STT and wake-word use Google Speech API on the Pi instead.
#
#  Usage:
#    chmod +x setup_pi.sh
#    ./setup_pi.sh
#
#  Run as your normal user (NOT root). The script uses sudo where needed.
# ============================================================

set -e  # Exit on any error

# ── Configuration ──────────────────────────────────────────────────────
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="uit-chatbot"
SERVICE_FILE="${INSTALL_DIR}/uit-chatbot.service"
VENV_DIR="${INSTALL_DIR}/.venv"
CURRENT_USER="$(whoami)"

echo "============================================================"
echo "  🎓 UIT Prayagraj Chatbot – Raspberry Pi Setup"
echo "============================================================"
echo "  Install directory : ${INSTALL_DIR}"
echo "  User              : ${CURRENT_USER}"
echo "============================================================"
echo ""

# ── Step 1: System dependencies ───────────────────────────────────────
echo "📦 [1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-pip \
    python3-pyaudio portaudio19-dev espeak flac ffmpeg wget unzip \
    libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0
echo "✅ System dependencies installed."
echo ""

# ── Step 2: Python virtual environment ────────────────────────────────
echo "🐍 [2/5] Setting up Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "   Created ${VENV_DIR}"
else
    echo "   Virtual environment already exists."
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet
echo "✅ Virtual environment ready."
echo ""

# ── Step 3: Python packages ──────────────────────────────────────────
echo "📚 [3/5] Installing Python packages..."
# NOTE: openai-whisper is NOT installed — PyTorch causes "Illegal instruction" on ARM.
# STT uses Google Speech API (online) on the Pi instead.
pip install edge-tts pygame pyttsx3 SpeechRecognition PyAudio --quiet
echo "✅ Python packages installed."
echo ""

# ── Step 4: Generate knowledge base if missing ──────────────────────
if [ ! -f "${INSTALL_DIR}/knowledge_base.json" ]; then
    echo "📄 [4/5] Generating knowledge_base.json from uit.txt..."
    "${VENV_DIR}/bin/python" "${INSTALL_DIR}/parse_txt.py"
else
    echo "✅ [4/5] knowledge_base.json already exists."
fi

# ── Step 6: Install systemd service ─────────────────────────────────
echo "⚙️  [5/5] Installing systemd service..."

# Add user to audio group for mic/speaker access
sudo usermod -aG audio "${CURRENT_USER}" 2>/dev/null || true

# Update service file with actual paths and username
TEMP_SERVICE="/tmp/${SERVICE_NAME}.service"
sed -e "s|User=pi|User=${CURRENT_USER}|g" \
    -e "s|Group=pi|Group=${CURRENT_USER}|g" \
    -e "s|WorkingDirectory=.*|WorkingDirectory=${INSTALL_DIR}|g" \
    -e "s|ExecStart=.*|ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/chatbot.py --voice|g" \
    -e "s|/run/user/1000/|/run/user/$(id -u)/|g" \
    "${SERVICE_FILE}" > "${TEMP_SERVICE}"

sudo cp "${TEMP_SERVICE}" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
echo "✅ Service installed and enabled."
echo ""

# ── Done ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  ✅ Setup complete!"
echo "============================================================"
echo ""
echo "  Start the chatbot service:"
echo "    sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "  Check status:"
echo "    sudo systemctl status ${SERVICE_NAME}"
echo ""
echo "  View live logs:"
echo "    journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "  Stop the service:"
echo "    sudo systemctl stop ${SERVICE_NAME}"
echo ""
echo "  Run manually (for testing):"
echo "    cd ${INSTALL_DIR}"
echo "    source .venv/bin/activate"
echo "    python chatbot.py --voice"
echo ""
echo "  The chatbot will auto-start on every boot."
echo "============================================================"
