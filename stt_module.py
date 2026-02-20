"""
Speech-to-Text module for UIT Chatbot (Raspberry Pi compatible)
---------------------------------------------------------------
Primary:  OpenAI Whisper — highly accurate, runs OFFLINE, handles accents well
Fallback: Google Speech API (online, free tier)

Whisper is far more accurate than Vosk / Google for Indian English accents
and noisy environments. The 'base' model runs well on Raspberry Pi 4.

Setup:
    pip install openai-whisper SpeechRecognition PyAudio

    On Raspberry Pi:
        sudo apt-get install python3-pyaudio portaudio19-dev ffmpeg
        pip install openai-whisper SpeechRecognition PyAudio

Usage:
    from stt_module import listen, STTEngine

    text = listen()

    engine = STTEngine(backend="whisper", model_size="base")
    text = engine.listen()
"""

import io
import json
import os
import wave
import platform
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── ARM / Raspberry Pi detection ───────────────────────────────────────
_MACHINE = platform.machine().lower()
IS_ARM = _MACHINE.startswith("arm") or _MACHINE.startswith("aarch")

# ── Check available libraries ──────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# Whisper / PyTorch causes "Illegal instruction" on Raspberry Pi ARM.
# Skip the import entirely on ARM to prevent a hard crash.
WHISPER_AVAILABLE = False
if not IS_ARM:
    try:
        import whisper as openai_whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        pass
else:
    openai_whisper = None  # placeholder so references don't error

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class STTEngine:
    """Speech-to-Text with Whisper (offline, accurate) and Google (online) backends."""

    BACKENDS = ("whisper", "google", "vosk")

    # Whisper model sizes — pick based on your hardware:
    #   tiny   (~39 MB)  — fastest, decent accuracy
    #   base   (~74 MB)  — good balance for Pi 4
    #   small  (~244 MB) — better accuracy, slower on Pi
    WHISPER_MODELS = ("tiny", "base", "small", "medium")

    def __init__(
        self,
        backend: str = "auto",
        model_size: str = "base",
        model_path: str | None = None,
        energy_threshold: int = 300,
        pause_threshold: float = 1.5,
        timeout: float = 7.0,
        phrase_time_limit: float = 15.0,
        language: str = "en",
    ):
        """
        Args:
            backend:            "whisper", "google", "vosk", or "auto".
            model_size:         Whisper model size ("tiny", "base", "small").
            model_path:         Path to Vosk model directory (if using vosk).
            energy_threshold:   Min audio energy to trigger recording.
            pause_threshold:    Seconds of silence to end a phrase.
            timeout:            Max seconds to wait for speech to start.
            phrase_time_limit:  Max seconds for a single phrase.
            language:           Language code for Whisper (default "en").
        """
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.language = language
        self.whisper_model = None
        self.vosk_model = None

        if not SR_AVAILABLE:
            print("⚠️  SpeechRecognition not installed.")
            print("   Install: pip install SpeechRecognition PyAudio")
            self.recognizer = None
            self.backend = None
            return

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True

        # Resolve backend
        if backend == "auto":
            if IS_ARM and not WHISPER_AVAILABLE:
                # On Raspberry Pi, prefer Vosk (offline) > Google (online)
                if VOSK_AVAILABLE and self._find_vosk_model(model_path):
                    self.backend = "vosk"
                else:
                    self.backend = "google"
                print("ℹ️  ARM detected — Whisper/PyTorch skipped (incompatible).")
            elif WHISPER_AVAILABLE:
                self.backend = "whisper"
            elif VOSK_AVAILABLE and self._find_vosk_model(model_path):
                self.backend = "vosk"
            else:
                self.backend = "google"
        else:
            if backend == "whisper" and IS_ARM and not WHISPER_AVAILABLE:
                print("⚠️  Whisper is not available on ARM/Raspberry Pi.")
                print("   Falling back to Google STT.")
                self.backend = "google"
            else:
                self.backend = backend

        # Load model for chosen backend
        if self.backend == "whisper":
            if WHISPER_AVAILABLE:
                print(f"🎤 Loading Whisper model '{model_size}'...")
                try:
                    self.whisper_model = openai_whisper.load_model(model_size)
                    print(f"🎤 STT: Whisper ({model_size}) — offline, high accuracy")
                except Exception as e:
                    print(f"⚠️  Whisper load failed: {e}")
                    self.backend = "google"
            else:
                print("⚠️  Whisper not installed. Install: pip install openai-whisper")
                self.backend = "google"

        if self.backend == "vosk":
            vosk_path = self._find_vosk_model(model_path)
            if vosk_path and VOSK_AVAILABLE:
                self.vosk_model = VoskModel(str(vosk_path))
                print(f"🎤 STT: Vosk (offline) from {vosk_path}")
            else:
                print("⚠️  Vosk model not found, using Google STT.")
                self.backend = "google"

        if self.backend == "google":
            print("🎤 STT: Google (online)")

    def _find_vosk_model(self, model_path: str | None = None) -> Path | None:
        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend(sorted(BASE_DIR.glob("vosk-model*")))
        candidates.extend(sorted(Path.home().glob("vosk-model*")))
        for path in candidates:
            if path.is_dir() and (
                (path / "conf" / "model.conf").exists()
                or any(path.glob("*.conf"))
                or (path / "am").is_dir()
            ):
                return path
        return None

    @property
    def available(self) -> bool:
        return self.recognizer is not None and self.backend is not None

    def listen(self, prompt: str = "🎤 Listening...") -> str | None:
        """Listen via microphone and return recognised text, or None."""
        if not self.available:
            print("❌ STT not available.")
            return None

        try:
            with sr.Microphone() as source:
                print(prompt)
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
        """Recognise speech from a WAV/FLAC/AIFF file."""
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
        """Run recognition on captured audio."""
        try:
            if self.backend == "whisper" and self.whisper_model:
                return self._recognise_whisper(audio)
            elif self.backend == "vosk" and self.vosk_model:
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

    def _recognise_whisper(self, audio) -> str | None:
        """Offline recognition using OpenAI Whisper."""
        # Save audio to a temp WAV file for Whisper
        tmp_path = os.path.join(tempfile.gettempdir(), "stt_whisper_input.wav")
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_data)

        result = self.whisper_model.transcribe(
            tmp_path,
            language=self.language,
            fp16=False,  # Pi doesn't have GPU, use fp32
        )
        text = result.get("text", "").strip()

        # Clean up
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return text if text else None

    def _recognise_vosk(self, audio) -> str | None:
        """Offline recognition using Vosk."""
        raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        rec = KaldiRecognizer(self.vosk_model, 16000)
        rec.AcceptWaveform(raw_data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        return text if text else None

    def _recognise_google(self, audio) -> str | None:
        """Online recognition using Google Speech API."""
        text = self.recognizer.recognize_google(audio)
        return text.strip() if text else None


# ── Module-level convenience ───────────────────────────────────────────
_default_engine: STTEngine | None = None


def _get_engine() -> STTEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = STTEngine()
    return _default_engine


def listen(prompt: str = "🎤 Listening...") -> str | None:
    """Listen and return text using default engine."""
    return _get_engine().listen(prompt)


def is_available() -> bool:
    return _get_engine().available


if __name__ == "__main__":
    print("=== STT Module Test ===")
    print(f"  Whisper : {WHISPER_AVAILABLE}")
    print(f"  Vosk    : {VOSK_AVAILABLE}")
    print(f"  SpeechRecognition : {SR_AVAILABLE}")
    print()

    engine = STTEngine()
    if engine.available:
        print("Speak something into your microphone:\n")
        text = engine.listen()
        if text:
            print(f'\n✅ You said: "{text}"')
        else:
            print("\n⚠️  No speech detected or could not understand.")
    else:
        print("❌ STT not available. Install: pip install openai-whisper SpeechRecognition PyAudio")
