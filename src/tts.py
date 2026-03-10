"""
Text-to-Speech – Piper TTS with audio caching
================================================
Uses Piper neural TTS (~100 ms on Pi 4) with automatic audio caching
so repeated responses play back in 0 ms.

Fallback chain:  piper-tts (Python) → piper (binary) → espeak

Audio cache:
  Every spoken sentence is cached as a WAV file keyed by MD5 hash.
  Subsequent calls with the same text skip synthesis entirely and
  play the cached file → effectively 0 ms TTS latency.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

# Ensure project root is on sys.path so config.config resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    PIPER_MODEL, PIPER_BIN,
    TTS_OUTPUT_WAV, AUDIO_PLAYER, AUDIO_DIR, TTS_CACHE_DIR,
)
from src.cache import CacheManager

# ── Optional imports ───────────────────────────────────────────────────
PIPER_PY_OK = False
try:
    from piper import PiperVoice
    PIPER_PY_OK = True
except ImportError:
    pass

EDGE_TTS_OK = False
try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    pass

PYGAME_OK = False
try:
    import pygame
    PYGAME_OK = True
except ImportError:
    pass

ESPEAK_OK = shutil.which("espeak") is not None or shutil.which("espeak-ng") is not None


def _find_piper_bin() -> str | None:
    if shutil.which(PIPER_BIN):
        return shutil.which(PIPER_BIN)
    for cand in ["/usr/local/bin/piper", "/usr/bin/piper",
                 os.path.expanduser("~/piper/piper"),
                 os.path.expanduser("~/.local/bin/piper")]:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _clean(text: str) -> str:
    """Strip emoji and special chars for cleaner speech."""
    t = re.sub(r"[^\w\s.,;:!?'\"\-()/@]", "", text)
    return re.sub(r"\s+", " ", t).strip()


class TTSEngine:
    """
    Neural TTS with automatic audio caching.

    Usage:
        tts = TTSEngine()
        tts.speak("Hello from UIT Prayagraj!")
    """

    # Edge TTS Indian voice
    _EDGE_VOICE = "en-IN-PrabhatNeural"

    def __init__(self):
        self.backend = None
        self._piper_voice = None
        self._piper_bin = None
        self._cache = CacheManager()
        self._pygame_init = False

        Path(AUDIO_DIR).mkdir(exist_ok=True)
        TTS_CACHE_DIR.mkdir(exist_ok=True)

        # 1. Piper Python
        if PIPER_PY_OK and os.path.isfile(PIPER_MODEL):
            try:
                self._piper_voice = PiperVoice.load(PIPER_MODEL)
                self.backend = "piper-python"
                print(f"🔊 TTS: Piper (Python) — {os.path.basename(PIPER_MODEL)}")
                return
            except Exception as e:
                print(f"⚠️  Piper Python: {e}")

        # 2. Piper binary
        self._piper_bin = _find_piper_bin()
        if self._piper_bin and os.path.isfile(PIPER_MODEL):
            self.backend = "piper-binary"
            print(f"🔊 TTS: Piper (binary) — {os.path.basename(PIPER_MODEL)}")
            return

        # 3. Edge TTS (needs internet, but high quality)
        if EDGE_TTS_OK:
            self.backend = "edge-tts"
            print(f"🔊 TTS: Edge TTS ({self._EDGE_VOICE})")
            self._init_pygame()
            return

        # 4. espeak (robotic but always available)
        if ESPEAK_OK:
            self.backend = "espeak"
            print("🔊 TTS: espeak (fallback)")
            return

        print("❌ No TTS engine available.")
        print("   pip install piper-tts   OR   sudo apt install espeak")

    @property
    def available(self) -> bool:
        return self.backend is not None

    def speak(self, text: str) -> None:
        """Speak text, using cached audio when available."""
        if not text or not text.strip():
            return
        clean = _clean(text)
        if not clean:
            return

        # ── Check audio cache ──────────────────────────────────────
        cached_path = self._cache.get_audio(text)
        if cached_path:
            print(f"   ⚡ [TTS CACHE HIT]")
            self._play(cached_path)
            return

        print(f"   🔊 [TTS generating...]")

        # ── Generate audio ─────────────────────────────────────────
        if self.backend == "piper-python":
            self._speak_piper_py(clean, text)
        elif self.backend == "piper-binary":
            self._speak_piper_bin(clean, text)
        elif self.backend == "edge-tts":
            self._speak_edge(clean, text)
        elif self.backend == "espeak":
            self._speak_espeak(clean)
        else:
            print(f"   [TTS OFF] {clean}")

    # ── Backends ───────────────────────────────────────────────────────

    def _speak_piper_py(self, clean: str, original: str) -> None:
        out = self._cache.audio_cache_path(original, ".wav")
        try:
            with wave.open(out, "wb") as wf:
                self._piper_voice.synthesize(clean, wf)
            self._play(out)
        except Exception as e:
            print(f"   ⚠️  Piper: {e}")
            if ESPEAK_OK:
                self._speak_espeak(clean)

    def _speak_piper_bin(self, clean: str, original: str) -> None:
        out = self._cache.audio_cache_path(original, ".wav")
        try:
            r = subprocess.run(
                [self._piper_bin, "--model", PIPER_MODEL,
                 "--output_file", out],
                input=clean, capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and os.path.isfile(out):
                self._play(out)
            else:
                print(f"   ⚠️  Piper binary: {r.stderr.strip()}")
                if ESPEAK_OK:
                    self._speak_espeak(clean)
        except Exception as e:
            print(f"   ⚠️  Piper binary: {e}")
            if ESPEAK_OK:
                self._speak_espeak(clean)

    def _speak_edge(self, clean: str, original: str) -> None:
        import asyncio
        out = self._cache.audio_cache_path(original, ".mp3")

        async def _gen():
            comm = edge_tts.Communicate(clean, voice=self._EDGE_VOICE)
            await comm.save(out)

        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_gen())
            loop.close()
            self._play(out)
        except Exception as e:
            print(f"   ⚠️  Edge TTS: {e}")
            if ESPEAK_OK:
                self._speak_espeak(clean)

    def _speak_espeak(self, clean: str) -> None:
        cmd = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
        try:
            subprocess.run([cmd, "-s", "150", clean],
                           capture_output=True, timeout=30)
        except Exception as e:
            print(f"   ⚠️  espeak: {e}")

    # ── Audio playback ─────────────────────────────────────────────────

    def _init_pygame(self):
        if PYGAME_OK and not self._pygame_init:
            try:
                pygame.mixer.init(frequency=24000)
                self._pygame_init = True
            except Exception:
                pass

    def _play(self, filepath: str) -> None:
        system = platform.system()

        # pygame (cross-platform, good for mp3)
        if self._pygame_init:
            try:
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
                return
            except KeyboardInterrupt:
                pygame.mixer.music.stop()
                return
            except Exception:
                pass

        if system == "Linux":
            for cmd in [
                [AUDIO_PLAYER, filepath],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
                ["paplay", filepath],
                ["mpg123", "-q", filepath],
            ]:
                try:
                    subprocess.run(cmd, capture_output=True, timeout=30)
                    return
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
        elif system == "Darwin":
            subprocess.run(["afplay", filepath], timeout=30,
                           capture_output=True)
        elif system == "Windows":
            try:
                ps = f"(New-Object Media.SoundPlayer '{filepath}').PlaySync()"
                subprocess.run(["powershell", "-c", ps],
                               capture_output=True, timeout=30)
            except Exception:
                os.system(f'start /min /wait wmplayer "{filepath}"')

    def stop(self):
        if self._pygame_init:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass


# ── Module-level convenience ───────────────────────────────────────────
_engine: TTSEngine | None = None


def speak(text: str) -> None:
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    _engine.speak(text)


def is_available() -> bool:
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine.available


if __name__ == "__main__":
    print("=== TTS Test ===\n")
    e = TTSEngine()
    if e.available:
        print(f"Backend: {e.backend}")
        e.speak("Hello! Welcome to United Institute of Technology Prayagraj.")
        e.speak("I can answer questions about UIT. Just ask me anything.")
        # second call should be a cache hit
        e.speak("Hello! Welcome to United Institute of Technology Prayagraj.")
        print("✅ Done.")
    else:
        print("❌ No TTS available.")
