#!/usr/bin/env bash
# ============================================================
#  UIT Prayagraj Chatbot – Raspberry Pi Installer
# ============================================================
#  This script:
#    1. Installs system dependencies (espeak, portaudio, flac)
#    2. Creates a Python virtual environment
#    3. Installs Python packages (pyttsx3, SpeechRecognition, etc.)
#    4. Downloads the Vosk offline speech model (~40 MB)
#    5. Installs & enables the systemd service
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
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_ZIP="vosk-model-small-en-us-0.15.zip"
VOSK_MODEL_DIR="vosk-model-small-en-us-0.15"
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
    python3-pyaudio portaudio19-dev espeak flac wget unzip
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
pip install pyttsx3 SpeechRecognition PyAudio vosk --quiet
echo "✅ Python packages installed."
echo ""

# ── Step 4: Vosk offline model ───────────────────────────────────────
echo "🎤 [4/5] Setting up Vosk offline speech model..."
cd "${INSTALL_DIR}"
if [ -d "${VOSK_MODEL_DIR}" ]; then
    echo "   Vosk model already exists at ${VOSK_MODEL_DIR}"
else
    if [ ! -f "${VOSK_MODEL_ZIP}" ]; then
        echo "   Downloading Vosk model (~40 MB)..."
        wget -q --show-progress "${VOSK_MODEL_URL}" -O "${VOSK_MODEL_ZIP}"
    fi
    echo "   Extracting model..."
    unzip -q -o "${VOSK_MODEL_ZIP}"
    rm -f "${VOSK_MODEL_ZIP}"
    echo "   Model extracted to ${VOSK_MODEL_DIR}"
fi
echo "✅ Vosk model ready."
echo ""

# ── Step 5: Generate knowledge base if missing ──────────────────────
if [ ! -f "${INSTALL_DIR}/knowledge_base.json" ]; then
    echo "📄 Generating knowledge_base.json from uit.txt..."
    "${VENV_DIR}/bin/python" "${INSTALL_DIR}/parse_txt.py"
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
