"""
Speech-to-Text module for UIT Chatbot (Raspberry Pi compatible)
---------------------------------------------------------------
Supports multiple recognition backends:
  1. Vosk   – fully OFFLINE, best for Raspberry Pi (recommended)
  2. Google – online, no API key needed (fallback)

Microphone input via PyAudio.

Setup on Raspberry Pi:
    sudo apt-get install python3-pyaudio portaudio19-dev espeak
    pip install SpeechRecognition pyaudio vosk

    # Download a small Vosk model for offline use:
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip -d vosk-model-small-en-us

Usage:
    from stt_module import listen, STTEngine

    # Quick one-liner
    text = listen()

    # Or use the engine for control
    engine = STTEngine(backend="vosk", model_path="vosk-model-small-en-us")
    text = engine.listen()
"""

import os
import json
import platform
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Check available libraries ──────────────────────────────────────────
try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer

    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class STTEngine:
    """Wrapper for Speech-to-Text with Vosk (offline) and Google (online) backends."""

    BACKENDS = ("vosk", "google")

    def __init__(
        self,
        backend: str = "auto",
        model_path: str | None = None,
        energy_threshold: int = 300,
        pause_threshold: float = 1.0,
        timeout: float = 5.0,
        phrase_time_limit: float = 10.0,
    ):
        """
        Args:
            backend:            "vosk", "google", or "auto" (tries vosk first).
            model_path:         Path to Vosk model directory (for offline mode).
            energy_threshold:   Minimum audio energy to consider for recording.
            pause_threshold:    Seconds of silence before a phrase is considered complete.
            timeout:            Max seconds to wait for speech to start.
            phrase_time_limit:  Max seconds for a single phrase.
        """
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

        if not SR_AVAILABLE:
            print("⚠️  SpeechRecognition not installed.")
            print("   Install: pip install SpeechRecognition")
            self.recognizer = None
            self.backend = None
            return

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True

        # Resolve backend
        if backend == "auto":
            if VOSK_AVAILABLE and self._find_vosk_model(model_path):
                self.backend = "vosk"
            else:
                self.backend = "google"
                if not VOSK_AVAILABLE:
                    print("ℹ️  Vosk not installed, using Google (online) STT.")
        else:
            self.backend = backend

        # Load Vosk model if needed
        self.vosk_model = None
        if self.backend == "vosk":
            vosk_path = self._find_vosk_model(model_path)
            if vosk_path:
                print(f"🎤 Loading Vosk model from: {vosk_path}")
                self.vosk_model = VoskModel(str(vosk_path))
            else:
                print("⚠️  Vosk model not found, falling back to Google STT.")
                self.backend = "google"

        print(f"🎤 STT backend: {self.backend}")

    def _find_vosk_model(self, model_path: str | None = None) -> Path | None:
        """Search for a Vosk model directory."""
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        # Search in project directory for any vosk-model-* folder
        candidates.extend(sorted(BASE_DIR.glob("vosk-model*")))
        candidates.extend(sorted(Path.home().glob("vosk-model*")))

        for path in candidates:
            if path.is_dir() and (path / "conf" / "model.conf").exists():
                return path
            # Some models have mfcc.conf directly
            if path.is_dir() and any(path.glob("*.conf")):
                return path
            if path.is_dir() and (path / "am" ).is_dir():
                return path
        return None

    @property
    def available(self) -> bool:
        return self.recognizer is not None and self.backend is not None

    def listen(self, prompt: str = "🎤 Listening...") -> str | None:
        """
        Listen to microphone and return recognised text, or None on failure.

        Returns:
            Recognised text string, or None if nothing was understood.
        """
        if not self.available:
            print("❌ STT not available.")
            return None

        try:
            with sr.Microphone() as source:
                print(prompt)
                # Adjust for ambient noise briefly
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )
        except sr.WaitTimeoutError:
            print("⏱️  No speech detected (timeout).")
            return None
        except OSError as e:
            print(f"❌ Microphone error: {e}")
            if platform.system() == "Linux":
                print("   Try: sudo apt-get install portaudio19-dev python3-pyaudio")
            return None

        return self._recognise(audio)

    def listen_from_file(self, audio_path: str) -> str | None:
        """Recognise speech from a WAV/FLAC/AIFF file (useful for testing)."""
        if not self.available:
            return None
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            return self._recognise(audio)
        except Exception as e:
            print(f"❌ Error reading audio file: {e}")
            return None

    def _recognise(self, audio) -> str | None:
        """Run recognition on an audio segment."""
        try:
            if self.backend == "vosk" and self.vosk_model:
                return self._recognise_vosk(audio)
            else:
                return self._recognise_google(audio)
        except sr.UnknownValueError:
            print("🤷 Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"❌ Recognition service error: {e}")
            return None
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return None

    def _recognise_vosk(self, audio) -> str | None:
        """Offline recognition using Vosk."""
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        rec = KaldiRecognizer(self.vosk_model, 16000)
        rec.AcceptWaveform(raw_data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        return text if text else None

    def _recognise_google(self, audio) -> str | None:
        """Online recognition using Google Speech API (free tier)."""
        text = self.recognizer.recognize_google(audio)
        return text.strip() if text else None


# ── Module-level singleton for convenience ─────────────────────────────
_default_engine: STTEngine | None = None


def _get_engine() -> STTEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = STTEngine()
    return _default_engine


def listen(prompt: str = "🎤 Listening...") -> str | None:
    """Convenience function: listen and return text using default engine."""
    return _get_engine().listen(prompt)


def is_available() -> bool:
    """Check if STT is available on this system."""
    return _get_engine().available


if __name__ == "__main__":
    print("Testing Speech-to-Text module...")
    print("Speak something into your microphone:\n")

    if not SR_AVAILABLE:
        print("❌ Install SpeechRecognition: pip install SpeechRecognition")
    else:
        engine = STTEngine()
        if engine.available:
            text = engine.listen()
            if text:
                print(f"\n✅ You said: \"{text}\"")
            else:
                print("\n⚠️  No speech detected or could not understand.")
        else:
            print("❌ STT engine not available. Check dependencies.")
