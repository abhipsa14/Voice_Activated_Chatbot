"""
Voice Activity Detection – Improved
======================================
High-precision audio capture optimised for Raspberry Pi.

Key improvements over a basic VAD:
  1. **Ring-buffer pre-capture** – keeps the last 300 ms of audio BEFORE
     speech is detected so the first word is never clipped.
  2. **WebRTC VAD** (primary) with RMS energy fallback – Google's
     production-grade voice activity classifier.
  3. **Adaptive silence threshold** – auto-calibrates from 1 s of ambient
     noise at startup so it works in quiet rooms AND noisy environments.
  4. **Hangover timer** – prevents premature silence detection on short
     pauses between words.
  5. **Short-mode** for wake-word detection (max 3 s clip).
  6. **Long-mode** for questions (max 15 s clip).

Saves captured audio to config.TEMP_WAV (16 kHz mono PCM16 WAV).
"""

import collections
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Ensure project root is on sys.path so config.config resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    SAMPLE_RATE, CHANNELS, CHUNK_SIZE, CHUNK_DURATION_MS,
    SILENCE_THRESHOLD, SILENCE_DURATION,
    VAD_AGGRESSIVENESS, TEMP_WAV, AUDIO_DIR,
    PRE_SPEECH_BUFFER_MS, MAX_RECORDING_SECONDS,
    WAKE_WORD_MAX_SECONDS,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _rms(data: bytes) -> float:
    """Root-mean-square energy of PCM-16 mono audio."""
    count = len(data) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"<{count}h", data)
    return math.sqrt(sum(s * s for s in shorts) / count)


def _calibrate_noise(stream, duration: float = 1.0) -> float:
    """
    Read `duration` seconds of ambient noise and return a threshold
    ~1.8× the observed RMS so that normal background noise doesn't
    trigger recording.
    """
    n_chunks = int(SAMPLE_RATE / CHUNK_SIZE * duration)
    energies = []
    for _ in range(n_chunks):
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            energies.append(_rms(data))
        except Exception:
            pass

    if not energies:
        return SILENCE_THRESHOLD

    avg = sum(energies) / len(energies)
    # Threshold = mean + 1.8× std-dev, with a minimum floor
    std = math.sqrt(sum((e - avg) ** 2 for e in energies) / len(energies))
    threshold = avg + 1.8 * std
    return max(threshold, SILENCE_THRESHOLD * 0.5)


# ── VAD Recorder ──────────────────────────────────────────────────────

class VADRecorder:
    """
    Records speech from the microphone using high-precision VAD.

    Usage:
        recorder = VADRecorder()
        wav_path = recorder.record()          # question mode (max 15 s)
        wav_path = recorder.record_short()    # wake-word mode (max 3 s)
    """

    def __init__(self, mic_index: int | None = None):
        self.mic_index = mic_index
        self._vad = None
        self._adaptive_threshold = SILENCE_THRESHOLD
        self._calibrated = False

        # Number of frames to keep before speech onset
        self._pre_buffer_frames = max(
            1, int(PRE_SPEECH_BUFFER_MS / CHUNK_DURATION_MS)
        )

        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio not installed. Cannot record audio.")
            print("   sudo apt-get install python3-pyaudio portaudio19-dev")
            return

        if WEBRTC_AVAILABLE:
            self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            print(f"🎙️  VAD: WebRTC (aggressiveness={VAD_AGGRESSIVENESS})")
        else:
            print(f"🎙️  VAD: RMS energy (threshold ≈{SILENCE_THRESHOLD})")

        print(f"   Pre-buffer: {PRE_SPEECH_BUFFER_MS} ms "
              f"({self._pre_buffer_frames} frames)")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return PYAUDIO_AVAILABLE

    def record(self, prompt: str = "🎤 Listening...") -> str | None:
        """Record speech (question mode, up to MAX_RECORDING_SECONDS)."""
        return self._record_internal(
            prompt, max_seconds=MAX_RECORDING_SECONDS
        )

    def record_short(self, prompt: str = "🎙️  Listening for wake word...") -> str | None:
        """Record a short clip (wake-word mode, up to 3 s)."""
        return self._record_internal(
            prompt, max_seconds=WAKE_WORD_MAX_SECONDS
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _record_internal(self, prompt: str, max_seconds: float) -> str | None:
        if not self.available:
            return None

        print(prompt)
        pa = pyaudio.PyAudio()

        stream_kw = dict(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        if self.mic_index is not None:
            stream_kw["input_device_index"] = self.mic_index

        stream = pa.open(**stream_kw)

        try:
            # Adaptive noise calibration (only once, or re-calibrate
            # if it's been a while).
            if not self._calibrated:
                sys.stdout.write("   🔇 Calibrating noise... ")
                sys.stdout.flush()
                self._adaptive_threshold = _calibrate_noise(stream, 1.0)
                self._calibrated = True
                print(f"threshold={self._adaptive_threshold:.0f}")

            frames = self._capture(stream, max_seconds)
        except KeyboardInterrupt:
            print("\n⏹️  Recording interrupted.")
            frames = []
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        if not frames:
            return None

        # ── Save WAV ────────────────────────────────────────────────
        Path(AUDIO_DIR).mkdir(exist_ok=True)
        with wave.open(TEMP_WAV, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        duration = len(frames) * CHUNK_SIZE / SAMPLE_RATE
        print(f"   ✅ Captured {duration:.1f} s → {TEMP_WAV}")
        return TEMP_WAV

    def _capture(self, stream, max_seconds: float) -> list[bytes]:
        """
        Core capture loop with ring-buffer pre-capture and hangover.
        """
        # Ring buffer keeps last N frames before speech onset
        ring = collections.deque(maxlen=self._pre_buffer_frames)

        frames: list[bytes] = []
        is_speaking = False
        voiced_count = 0
        silence_start: float | None = None

        # Number of consecutive voiced frames needed to trigger
        trigger_frames = 3
        # Extra silence budget after last voiced frame ("hangover")
        hangover = SILENCE_DURATION

        wait_timeout = 10.0  # max wait for speech to start
        wait_start = time.time()
        max_frames = int(SAMPLE_RATE / CHUNK_SIZE * max_seconds)

        while True:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except Exception:
                continue

            voiced = self._is_voiced(data)

            if not is_speaking:
                # ── Pre-speech: waiting for trigger ────────────────
                ring.append(data)

                if voiced:
                    voiced_count += 1
                else:
                    voiced_count = 0

                if voiced_count >= trigger_frames:
                    # Speech started! Flush ring buffer first
                    is_speaking = True
                    frames.extend(ring)
                    ring.clear()
                    silence_start = None
                    sys.stdout.write("   🔴 Speaking...")
                    sys.stdout.flush()

                elif time.time() - wait_start > wait_timeout:
                    return []

            else:
                # ── Recording speech ──────────────────────────────
                frames.append(data)

                if voiced:
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= hangover:
                        print(" ⏹️")
                        break

                if len(frames) >= max_frames:
                    print(" ⏹️ (max length)")
                    break

        return frames

    def _is_voiced(self, data: bytes) -> bool:
        """Classify a frame as speech or silence."""
        if self._vad is not None:
            try:
                return self._vad.is_speech(data, SAMPLE_RATE)
            except Exception:
                pass
        # RMS fallback with adaptive threshold
        return _rms(data) > self._adaptive_threshold


# ── Module-level convenience ───────────────────────────────────────────
_recorder: VADRecorder | None = None


def record(prompt: str = "🎤 Listening...") -> str | None:
    global _recorder
    if _recorder is None:
        _recorder = VADRecorder()
    return _recorder.record(prompt)


def record_short(prompt: str = "🎙️  Listening for wake word...") -> str | None:
    global _recorder
    if _recorder is None:
        _recorder = VADRecorder()
    return _recorder.record_short(prompt)


if __name__ == "__main__":
    print("=== VAD Recorder Test ===\n")
    r = VADRecorder()
    if r.available:
        path = r.record("Speak something...")
        if path:
            print(f"\n✅ Saved to: {path}")
        else:
            print("\n❌ No speech captured.")
    else:
        print("❌ PyAudio unavailable.")
