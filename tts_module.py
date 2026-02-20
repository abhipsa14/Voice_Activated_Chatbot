"""
Text-to-Speech module for UIT Chatbot (Raspberry Pi compatible)
---------------------------------------------------------------
Primary:  edge-tts  — Microsoft Edge neural voices, very human-like, FREE
Fallback: pyttsx3   — works fully offline (robotic but reliable)

edge-tts produces natural, expressive speech using neural voice models.
Indian English voices are available for a local feel.

Usage:
    from tts_module import speak, TTSEngine

    speak("Hello from UIT Prayagraj!")

    engine = TTSEngine(voice="indian_female")
    engine.speak("Welcome to UIT chatbot")
"""

import asyncio
import os
import re
import platform
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".tts_cache"

# ── Check available libraries ──────────────────────────────────────────
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ── Natural voice presets ──────────────────────────────────────────────
VOICE_PRESETS = {
    # Indian English
    "indian_female": "en-IN-NeerjaNeural",
    "indian_male": "en-IN-PrabhatNeural",
    # US English
    "us_female": "en-US-JennyNeural",
    "us_male": "en-US-GuyNeural",
    "us_aria": "en-US-AriaNeural",
    # UK English
    "uk_female": "en-GB-SoniaNeural",
    "uk_male": "en-GB-RyanNeural",
}

DEFAULT_VOICE = "en-IN-NeerjaNeural"


def _clean_text(text: str) -> str:
    """Remove emoji and special characters for cleaner speech."""
    clean = re.sub(r"[^\w\s.,;:!?'\"\-()/@]", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _get_or_create_event_loop():
    """Get or create an asyncio event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class TTSEngine:
    """Human-like TTS using Edge neural voices with pyttsx3 fallback."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        fallback_rate: int = 160,
    ):
        """
        Args:
            voice:          Edge TTS voice name or preset key (e.g. "indian_female").
            rate:           Speech rate adjustment ("+10%", "-5%", "+0%").
            volume:         Volume adjustment ("+0%", "-10%").
            pitch:          Pitch adjustment ("+0Hz", "+5Hz").
            fallback_rate:  pyttsx3 words-per-minute for offline fallback.
        """
        self.voice = VOICE_PRESETS.get(voice, voice)
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.use_edge = EDGE_TTS_AVAILABLE
        self._fallback_engine = None
        self._fallback_rate = fallback_rate
        self._pygame_inited = False

        CACHE_DIR.mkdir(exist_ok=True)

        if self.use_edge:
            print(f"🔊 TTS: Edge Neural Voice ({self.voice})")
            self._init_pygame()
        elif PYTTSX3_AVAILABLE:
            print("🔊 TTS: pyttsx3 offline fallback")
            self._init_pyttsx3()
        else:
            print("⚠️  No TTS engine available.")
            print("   Install: pip install edge-tts pygame")

    def _init_pygame(self):
        """Initialize pygame mixer for audio playback."""
        if PYGAME_AVAILABLE and not self._pygame_inited:
            try:
                pygame.mixer.init(frequency=24000)
                self._pygame_inited = True
            except Exception as e:
                print(f"⚠️  pygame mixer init failed: {e}")

    def _init_pyttsx3(self):
        """Initialize pyttsx3 as offline fallback."""
        try:
            self._fallback_engine = pyttsx3.init()
            self._fallback_engine.setProperty("rate", self._fallback_rate)
            self._fallback_engine.setProperty("volume", 1.0)
            voices = self._fallback_engine.getProperty("voices")
            for v in voices:
                if "zira" in v.name.lower() or "female" in v.name.lower():
                    self._fallback_engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"⚠️  pyttsx3 init error: {e}")
            self._fallback_engine = None

    @property
    def available(self) -> bool:
        return self.use_edge or self._fallback_engine is not None

    def speak(self, text: str) -> None:
        """Convert text to natural speech and play through speakers."""
        if not text or not text.strip():
            return
        clean = _clean_text(text)
        if not clean:
            return

        if self.use_edge:
            try:
                self._speak_edge(clean)
                return
            except Exception as e:
                print(f"⚠️  Edge TTS error ({e}), using fallback...")
                if PYTTSX3_AVAILABLE and not self._fallback_engine:
                    self._init_pyttsx3()

        if self._fallback_engine:
            self._speak_pyttsx3(clean)
        else:
            print(f"[TTS disabled] {clean}")

    def _speak_edge(self, text: str) -> None:
        """Generate and play speech using Edge TTS neural voices."""
        audio_file = CACHE_DIR / "tts_output.mp3"

        async def _generate():
            communicate = edge_tts.Communicate(
                text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch,
            )
            await communicate.save(str(audio_file))

        loop = _get_or_create_event_loop()
        loop.run_until_complete(_generate())
        self._play_audio(str(audio_file))

    def _play_audio(self, filepath: str) -> None:
        """Play an audio file."""
        if self._pygame_inited:
            try:
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
                return
            except Exception:
                pass

        # System fallback
        system = platform.system()
        if system == "Windows":
            os.system(f'start /min /wait wmplayer "{filepath}"')
        elif system == "Linux":
            os.system(
                f'mpg123 -q "{filepath}" 2>/dev/null || '
                f'ffplay -nodisp -autoexit -loglevel quiet "{filepath}" 2>/dev/null || '
                f'aplay "{filepath}" 2>/dev/null'
            )
        elif system == "Darwin":
            os.system(f'afplay "{filepath}"')

    def _speak_pyttsx3(self, text: str) -> None:
        """Offline fallback speech."""
        if self._fallback_engine:
            self._fallback_engine.say(text)
            self._fallback_engine.runAndWait()

    def set_voice(self, voice: str) -> None:
        """Change voice (preset name or Edge TTS voice ID)."""
        self.voice = VOICE_PRESETS.get(voice, voice)
        print(f"🔊 Voice changed to: {self.voice}")

    def set_rate(self, rate: str) -> None:
        self.rate = rate

    def set_volume(self, volume: str) -> None:
        self.volume = volume

    def set_pitch(self, pitch: str) -> None:
        self.pitch = pitch

    def list_voices(self) -> list[str]:
        """Return available preset voice names."""
        return list(VOICE_PRESETS.keys())

    def stop(self) -> None:
        if self._pygame_inited:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        if self._fallback_engine:
            self._fallback_engine.stop()


# ── Module-level convenience ───────────────────────────────────────────
_default_engine: TTSEngine | None = None


def _get_engine() -> TTSEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = TTSEngine()
    return _default_engine


def speak(text: str) -> None:
    """Speak text using the default engine."""
    _get_engine().speak(text)


def is_available() -> bool:
    return _get_engine().available


if __name__ == "__main__":
    print("=== TTS Module Test ===")
    print(f"  edge-tts : {EDGE_TTS_AVAILABLE}")
    print(f"  pygame   : {PYGAME_AVAILABLE}")
    print(f"  pyttsx3  : {PYTTSX3_AVAILABLE}")
    print()

    engine = TTSEngine()
    if engine.available:
        print(f"Voice presets: {engine.list_voices()}")
        print("\nSpeaking test phrases...")
        engine.speak("Hello! Welcome to United Institute of Technology Prayagraj.")
        engine.speak("I can answer your questions about UIT. Just ask me anything.")
        print("✅ TTS test complete.")
    else:
        print("❌ No TTS available. Install: pip install edge-tts pygame")
