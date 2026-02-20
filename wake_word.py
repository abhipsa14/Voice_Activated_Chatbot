"""
Wake Word Detection module for UIT Chatbot
-------------------------------------------
Continuously listens for a wake phrase (default: "hey uit") using
lightweight audio monitoring + Whisper tiny model.

How it works:
  1. Monitors microphone for audio energy spikes (someone speaking)
  2. Captures a short audio clip (~2-3 seconds)
  3. Transcribes it with Whisper tiny (very fast, ~0.5s on Pi 4)
  4. Checks if the wake phrase is present
  5. If yes — signals the chatbot to start listening for a question

This approach uses no API keys and runs fully offline.

Usage:
    from wake_word import WakeWordDetector

    detector = WakeWordDetector(wake_phrase="hey uit")
    detector.wait_for_wake_word()  # blocks until wake word detected
"""

import os
import re
import wave
import struct
import tempfile
import time
import platform
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WakeWordDetector:
    """
    Lightweight wake word detector using energy-based VAD + Whisper tiny.
    Optimized for Raspberry Pi — low CPU usage while idle.
    """

    # Audio settings
    FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024  # frames per buffer

    # Detection settings
    SILENCE_THRESHOLD = 500     # RMS energy threshold for "someone is speaking"
    MIN_SPEECH_CHUNKS = 3       # minimum chunks above threshold to trigger
    MAX_RECORD_SECONDS = 3      # max seconds to record for wake word check
    COOLDOWN_SECONDS = 1.0      # pause after a detection before listening again

    def __init__(
        self,
        wake_phrase: str = "hey uit",
        alternative_phrases: list[str] | None = None,
        sensitivity: float = 0.6,
        model_size: str = "tiny",
    ):
        """
        Args:
            wake_phrase:         Primary wake phrase to listen for.
            alternative_phrases: Additional accepted phrases (e.g. ["ok uit", "hello uit"]).
            sensitivity:         0.0-1.0, lower = more strict matching.
            model_size:          Whisper model ("tiny" recommended for wake word).
        """
        self.wake_phrase = wake_phrase.lower().strip()
        self.alternatives = [p.lower().strip() for p in (alternative_phrases or [])]
        self.all_phrases = [self.wake_phrase] + self.alternatives
        self.sensitivity = sensitivity
        self.whisper_model = None
        self._audio = None

        if not PYAUDIO_AVAILABLE:
            print("⚠️  PyAudio not installed. Wake word detection unavailable.")
            print("   Install: pip install PyAudio")
            return

        if not WHISPER_AVAILABLE:
            print("⚠️  Whisper not installed. Wake word detection unavailable.")
            print("   Install: pip install openai-whisper")
            return

        # Load tiny Whisper model (fast enough for wake word, ~39 MB)
        print(f"🔔 Loading Whisper '{model_size}' for wake word detection...")
        try:
            self.whisper_model = openai_whisper.load_model(model_size)
        except Exception as e:
            print(f"⚠️  Failed to load Whisper model: {e}")
            return

        # Adjust threshold based on sensitivity
        self.SILENCE_THRESHOLD = int(300 + (1 - sensitivity) * 700)

        print(f"🔔 Wake word: \"{self.wake_phrase}\"")
        if self.alternatives:
            print(f"   Alternatives: {self.alternatives}")

    @property
    def available(self) -> bool:
        return PYAUDIO_AVAILABLE and self.whisper_model is not None

    def _rms(self, data: bytes) -> float:
        """Calculate root-mean-square energy of audio chunk."""
        count = len(data) // 2
        if count == 0:
            return 0
        shorts = struct.unpack(f"<{count}h", data)
        sum_sq = sum(s * s for s in shorts)
        return (sum_sq / count) ** 0.5

    def _record_speech_segment(self, stream) -> bytes | None:
        """
        Wait for speech, then record until silence returns.
        Returns raw audio bytes or None if timeout.
        """
        frames = []
        speech_chunks = 0
        silent_chunks = 0
        max_chunks = int(self.RATE / self.CHUNK * self.MAX_RECORD_SECONDS)
        recording = False

        for _ in range(max_chunks * 3):  # overall timeout
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
            except Exception:
                continue

            energy = self._rms(data)

            if energy > self.SILENCE_THRESHOLD:
                speech_chunks += 1
                silent_chunks = 0
                if speech_chunks >= self.MIN_SPEECH_CHUNKS:
                    recording = True
            else:
                silent_chunks += 1

            if recording:
                frames.append(data)
                # Stop after enough silence following speech
                if silent_chunks > int(self.RATE / self.CHUNK * 0.8):
                    break
                # Safety: don't record forever
                if len(frames) > max_chunks:
                    break

        if not frames or len(frames) < self.MIN_SPEECH_CHUNKS:
            return None

        return b"".join(frames)

    def _transcribe_audio(self, raw_audio: bytes) -> str:
        """Transcribe raw audio bytes using Whisper tiny."""
        tmp_path = os.path.join(tempfile.gettempdir(), "wake_word_check.wav")
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(raw_audio)

        try:
            result = self.whisper_model.transcribe(
                tmp_path, language="en", fp16=False,
                no_speech_threshold=0.5,
            )
            text = result.get("text", "").lower().strip()
        except Exception:
            text = ""

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return text

    def _is_wake_phrase(self, text: str) -> bool:
        """Check if transcribed text contains the wake phrase (fuzzy)."""
        text = re.sub(r"[^\w\s]", "", text.lower())

        for phrase in self.all_phrases:
            phrase_clean = re.sub(r"[^\w\s]", "", phrase)
            # Exact substring
            if phrase_clean in text:
                return True
            # Fuzzy: check if all words of the phrase appear in text
            phrase_words = phrase_clean.split()
            text_words = text.split()
            if all(
                any(self._word_similar(pw, tw) for tw in text_words)
                for pw in phrase_words
            ):
                return True

        return False

    @staticmethod
    def _word_similar(a: str, b: str) -> bool:
        """Check if two words are similar (Levenshtein distance <= 2 or substring)."""
        if a == b:
            return True
        if a in b or b in a:
            return True
        # Simple Levenshtein
        if abs(len(a) - len(b)) > 2:
            return False
        if len(a) < 3 or len(b) < 3:
            return a == b
        d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            d[i][0] = i
        for j in range(len(b) + 1):
            d[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        return d[len(a)][len(b)] <= 2

    def wait_for_wake_word(self) -> bool:
        """
        Block until the wake word is detected.
        Returns True when detected, False if unavailable.
        """
        if not self.available:
            print("❌ Wake word detection not available.")
            return False

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        try:
            while True:
                raw = self._record_speech_segment(stream)
                if raw is None:
                    continue

                text = self._transcribe_audio(raw)
                if text:
                    # Debug: uncomment to see what was heard
                    # print(f"   [heard: \"{text}\"]")
                    if self._is_wake_phrase(text):
                        return True

                time.sleep(0.1)

        except KeyboardInterrupt:
            return False
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def check_audio(self, text: str) -> bool:
        """Check if already-transcribed text contains the wake phrase."""
        return self._is_wake_phrase(text)


if __name__ == "__main__":
    print("=== Wake Word Detector Test ===")
    print('Say "Hey UIT" to trigger detection...\n')

    detector = WakeWordDetector(
        wake_phrase="hey uit",
        alternative_phrases=["ok uit", "hello uit", "hi uit"],
    )
    if detector.available:
        print("Listening for wake word... (Ctrl+C to stop)\n")
        detected = detector.wait_for_wake_word()
        if detected:
            print("\n🔔 Wake word detected! Chatbot would activate now.")
    else:
        print("❌ Cannot test — missing dependencies.")
