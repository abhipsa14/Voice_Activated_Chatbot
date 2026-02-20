"""
Text-to-Speech module for UIT Chatbot (Raspberry Pi compatible)
---------------------------------------------------------------
Uses pyttsx3 which works offline and leverages:
  - espeak on Linux / Raspberry Pi
  - SAPI5 on Windows
  - NSSpeechSynthesizer on macOS

Usage:
    from tts_module import speak, TTSEngine

    # Quick one-liner
    speak("Hello from UIT Prayagraj!")

    # Or use the engine for more control
    engine = TTSEngine(rate=150, volume=0.9)
    engine.speak("Welcome to UIT chatbot")
"""

import platform

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class TTSEngine:
    """Wrapper around pyttsx3 for text-to-speech."""

    def __init__(self, rate: int = 160, volume: float = 1.0, voice_index: int = 0):
        """
        Args:
            rate:        Speech speed in words per minute (default 160, good for Pi).
            volume:      Volume level 0.0 to 1.0.
            voice_index: Index of the voice to use (0 = default).
        """
        if not PYTTSX3_AVAILABLE:
            print("⚠️  pyttsx3 not installed. Install with: pip install pyttsx3")
            if platform.system() == "Linux":
                print("   Also install espeak: sudo apt-get install espeak")
            self.engine = None
            return

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        # Select voice
        voices = self.engine.getProperty("voices")
        if voices and 0 <= voice_index < len(voices):
            self.engine.setProperty("voice", voices[voice_index].id)

    @property
    def available(self) -> bool:
        return self.engine is not None

    def speak(self, text: str) -> None:
        """Convert text to speech and play through speakers."""
        if not self.available:
            print(f"[TTS disabled] {text}")
            return

        # Clean text of emoji/special chars for cleaner speech
        import re
        clean = re.sub(r"[^\w\s.,;:!?'\"-]", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()

        if clean:
            self.engine.say(clean)
            self.engine.runAndWait()

    def set_rate(self, rate: int) -> None:
        """Change speech rate (words per minute)."""
        if self.available:
            self.engine.setProperty("rate", rate)

    def set_volume(self, volume: float) -> None:
        """Change volume (0.0 to 1.0)."""
        if self.available:
            self.engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def list_voices(self) -> list[str]:
        """Return list of available voice names."""
        if not self.available:
            return []
        voices = self.engine.getProperty("voices")
        return [v.name for v in voices]

    def set_voice_by_name(self, name: str) -> bool:
        """Set voice by partial name match. Returns True if found."""
        if not self.available:
            return False
        voices = self.engine.getProperty("voices")
        for v in voices:
            if name.lower() in v.name.lower():
                self.engine.setProperty("voice", v.id)
                return True
        return False

    def stop(self) -> None:
        """Stop any ongoing speech."""
        if self.available:
            self.engine.stop()


# ── Module-level singleton for convenience ─────────────────────────────
_default_engine: TTSEngine | None = None


def _get_engine() -> TTSEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = TTSEngine()
    return _default_engine


def speak(text: str) -> None:
    """Convenience function: speak text using the default engine."""
    _get_engine().speak(text)


def is_available() -> bool:
    """Check if TTS is available on this system."""
    return _get_engine().available


if __name__ == "__main__":
    print("Testing Text-to-Speech module...")
    if is_available():
        engine = TTSEngine()
        print(f"Available voices: {engine.list_voices()}")
        speak("Hello! Welcome to United Institute of Technology Prayagraj chatbot.")
        speak("I can answer your questions about UIT.")
        print("✅ TTS test complete.")
    else:
        print("❌ TTS not available. Install pyttsx3: pip install pyttsx3")
