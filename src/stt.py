"""
Speech-to-Text – whisper.cpp wrapper
======================================
Calls the whisper.cpp C++ binary as a subprocess for blazing-fast
offline transcription on ARM (~1-2 s for tiny.en on Pi 4).

Fallback chain:  whisper.cpp → Python Whisper → Google Speech API
"""

import os
import re
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path so config.config resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    WHISPER_CPP_BIN, WHISPER_MODEL, WHISPER_LANGUAGE,
    WHISPER_THREADS, TEMP_WAV,
)


def _find_whisper_cpp() -> str | None:
    """Locate the whisper.cpp binary."""
    # Explicit config
    if os.path.isfile(WHISPER_CPP_BIN) and os.access(WHISPER_CPP_BIN, os.X_OK):
        return WHISPER_CPP_BIN
    # PATH
    for name in ("whisper-cpp", "whisper.cpp", "main"):
        found = shutil.which(name)
        if found:
            return found
    # Project tree
    p = Path(WHISPER_CPP_BIN)
    if p.exists():
        return str(p)
    return None


def _find_model() -> str | None:
    """Locate a GGML model file."""
    if os.path.isfile(WHISPER_MODEL):
        return WHISPER_MODEL
    models_dir = Path(WHISPER_MODEL).parent
    if models_dir.is_dir():
        for f in sorted(models_dir.glob("ggml-*.bin")):
            return str(f)
    return None


class STTEngine:
    """
    Speech-to-text engine.

    Usage:
        stt = STTEngine()
        text = stt.transcribe("audio/temp.wav")
    """

    def __init__(self):
        self.backend = None
        self._whisper_bin = None
        self._whisper_model = None
        self._py_whisper = None

        # 1. whisper.cpp (fastest on ARM)
        self._whisper_bin = _find_whisper_cpp()
        self._whisper_model = _find_model()
        if self._whisper_bin and self._whisper_model:
            self.backend = "whisper.cpp"
            print(f"🎤 STT: whisper.cpp")
            print(f"   bin   = {self._whisper_bin}")
            print(f"   model = {self._whisper_model}")
            return

        # 2. Python openai-whisper
        try:
            import whisper
            self._py_whisper = whisper.load_model("tiny.en")
            self.backend = "whisper-python"
            print("🎤 STT: Python Whisper (tiny.en)")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️  Python Whisper failed: {e}")

        # 3. Google Speech API (online)
        try:
            import speech_recognition  # noqa: F401
            self.backend = "google"
            print("🎤 STT: Google Speech API (online)")
            return
        except ImportError:
            pass

        print("❌ No STT backend available.")
        print("   Build whisper.cpp  or  pip install openai-whisper")

    @property
    def available(self) -> bool:
        return self.backend is not None

    def transcribe(self, wav_path: str | None = None) -> str | None:
        """Transcribe a WAV file → text."""
        wav_path = wav_path or TEMP_WAV
        if not os.path.isfile(wav_path):
            print(f"❌ File not found: {wav_path}")
            return None

        if self.backend == "whisper.cpp":
            return self._via_cpp(wav_path)
        if self.backend == "whisper-python":
            return self._via_python(wav_path)
        if self.backend == "google":
            return self._via_google(wav_path)
        return None

    # ── Backends ───────────────────────────────────────────────────────

    def _via_cpp(self, wav_path: str) -> str | None:
        cmd = [
            self._whisper_bin,
            "-m", self._whisper_model,
            "-f", wav_path,
            "-l", WHISPER_LANGUAGE,
            "-t", str(WHISPER_THREADS),
            "--no-timestamps",
            "-nt",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                print(f"⚠️  whisper.cpp: {r.stderr.strip()}")
                return None
            text = self._clean_cpp_output(r.stdout)
            return text if text else None
        except subprocess.TimeoutExpired:
            print("⚠️  whisper.cpp timed out")
            return None
        except FileNotFoundError:
            print(f"❌ Binary not found: {self._whisper_bin}")
            return None
        except Exception as e:
            print(f"❌ whisper.cpp: {e}")
            return None

    @staticmethod
    def _clean_cpp_output(raw: str) -> str:
        lines = raw.strip().splitlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("whisper_", "main:", "system_info:")):
                continue
            line = re.sub(r"^\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", line)
            cleaned.append(line)
        return " ".join(cleaned).strip()

    def _via_python(self, wav_path: str) -> str | None:
        try:
            result = self._py_whisper.transcribe(
                wav_path, language=WHISPER_LANGUAGE, fp16=False,
            )
            text = result.get("text", "").strip()
            return text if text else None
        except FileNotFoundError:
            print("❌ Whisper: ffmpeg not found! openai-whisper needs ffmpeg.")
            print("   Install via:  winget install Gyan.FFmpeg")
            print("   Then restart your terminal so the PATH update takes effect.")
            return None
        except Exception as e:
            print(f"❌ Whisper: {e}")
            return None

    def _via_google(self, wav_path: str) -> str | None:
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as src:
                audio = recognizer.record(src)
            text = recognizer.recognize_google(audio)
            return text.strip() if text else None
        except Exception as e:
            print(f"❌ Google STT: {e}")
            return None


# ── Module-level convenience ───────────────────────────────────────────
_engine: STTEngine | None = None


def transcribe(wav_path: str | None = None) -> str | None:
    global _engine
    if _engine is None:
        _engine = STTEngine()
    return _engine.transcribe(wav_path)


if __name__ == "__main__":
    print("=== STT Test ===")
    e = STTEngine()
    if e.available:
        if os.path.isfile(TEMP_WAV):
            print(f"Transcribing {TEMP_WAV} ...")
            print(f"Result: {e.transcribe()}")
        else:
            print(f"No file at {TEMP_WAV}. Run vad.py first.")
    else:
        print("No backend available.")
