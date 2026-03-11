"""
Speech-to-Text – whisper.cpp wrapper + contextual post-processing
====================================================================
Calls the whisper.cpp C++ binary as a subprocess for blazing-fast
offline transcription on ARM (~1-2 s for tiny.en on Pi 4).

Fallback chain:  whisper.cpp → Python Whisper → Google Speech API

Post-processing pipeline (inspired by Alexa contextual ASR):
  1. Domain-vocabulary prompt conditioning (Whisper initial_prompt)
  2. Phonetic / fuzzy replacement rules for common misrecognitions
  3. Knowledge-base derived corrections
  4. Context-aware biasing using recent conversation history
"""

import json
import os
import re
import subprocess
import shutil
import sys
import tempfile
from collections import deque
from pathlib import Path

# Ensure project root is on sys.path so config.config resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    WHISPER_CPP_BIN, WHISPER_MODEL, WHISPER_LANGUAGE,
    WHISPER_THREADS, TEMP_WAV, WHISPER_INITIAL_PROMPT,
    STT_CONTEXT_ENABLED, STT_CONTEXT_WINDOW, KB_JSON_FILE,
)


# ── Contextual Post-Processor ─────────────────────────────────────────

class ContextualPostProcessor:
    """
    Corrects common Whisper misrecognitions using:
      - Static phonetic replacement rules
      - Knowledge-base derived vocabulary
      - Context-aware biasing from recent conversation history
    """

    # Ordered replacement rules: (pattern, replacement)
    # Patterns are case-insensitive and matched as whole words or phrases
    STATIC_CORRECTIONS = [
        # Wake-word misrecognitions → UIT
        (r"\bu\s*i\s*tried\b", "UIT"),
        (r"\byou\s+it\b", "UIT"),
        (r"\bu\s+i\s+t\b", "UIT"),
        (r"\bu\s+i\s+d\b", "UIT"),
        (r"\bu\s+i\s+t\s+e\b", "UIT"),
        (r"\byou\s+eye\s+tea\b", "UIT"),
        (r"\byou\s+i\s+t\b", "UIT"),
        (r"\bu\s+it\b", "UIT"),
        (r"\buit\b", "UIT"),
        # Acronyms
        (r"\ba\s*k\s*t\s*u\b", "AKTU"),
        (r"\ba\s*i\s*c\s*t\s*e\b", "AICTE"),
        (r"\be\s*r\s*p\b", "ERP"),
        (r"\bh\s*o\s*d\b", "HOD"),
        # Place name
        (r"\bprayag\s*raj\b", "Prayagraj"),
        (r"\bpray\s*a\s*graj\b", "Prayagraj"),
        # Common name corrections
        (r"\bsanjay\s+shrivastava\b", "Sanjay Srivastava"),
        (r"\bsanjay\s+srivastav\b", "Sanjay Srivastava"),
        (r"\babhishek\s+malvia\b", "Abhishek Malviya"),
        (r"\babhishek\s+malvya\b", "Abhishek Malviya"),
        (r"\bshruti\s+sharma\b", "Shruti Sharma"),
        (r"\bamit\s+kumar\s+tiwari\b", "Amit Kumar Tiwari"),
    ]

    def __init__(self):
        self._kb_terms: list[str] = []
        self._kb_corrections: list[tuple[re.Pattern, str]] = []
        self._load_kb_vocabulary()

    def _load_kb_vocabulary(self):
        """Extract proper nouns and domain terms from knowledge_base.json."""
        kb_path = str(KB_JSON_FILE)
        if not os.path.isfile(kb_path):
            return
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Post-processor: cannot read KB: {e}")
            return

        names = set()
        for entry in entries:
            # Extract proper nouns from answers (capitalized multi-word terms)
            answer = entry.get("answer", "")
            # Match capitalized names like "Prof. Sanjay Srivastava"
            for m in re.finditer(
                r"(?:(?:Prof\.|Dr\.|Mr\.|Mrs\.|Ms\.)\s+)?"
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", answer
            ):
                names.add(m.group(0).strip())
            # Extract category keywords
            cat = entry.get("category", "")
            if cat:
                self._kb_terms.extend(
                    w for w in cat.split() if len(w) > 3
                )

        self._kb_terms = list(set(self._kb_terms))

        # Filter out names starting with common articles/words
        skip_starts = {"the", "a", "an", "yes", "no", "all", "are", "is"}

        # Build fuzzy correction patterns for extracted names
        for name in names:
            # Skip if the name starts with a common article
            first_word = name.split()[0].lower().rstrip(".")
            if first_word in skip_starts:
                continue
            # Create a pattern that matches the name with flexible spacing
            parts = name.replace(".", r"\.").split()
            if len(parts) >= 2:
                # Allow minor spelling variations (just flexible spacing)
                pattern = r"\b" + r"\s+".join(
                    re.escape(p) for p in parts
                ) + r"\b"
                self._kb_corrections.append(
                    (re.compile(pattern, re.IGNORECASE), name)
                )

    def process(self, text: str, context_keywords: list[str] | None = None) -> str:
        """
        Apply corrections to transcribed text.

        Args:
            text: Raw transcription from Whisper / Google
            context_keywords: Keywords from recent conversation history
                              used to bias corrections
        Returns:
            Corrected text
        """
        if not text:
            return text

        result = text

        # 1. Apply static phonetic correction rules
        for pattern, replacement in self.STATIC_CORRECTIONS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # 2. Apply KB-derived name corrections
        for pattern, replacement in self._kb_corrections:
            result = pattern.sub(replacement, result)

        # 3. Context-aware biasing
        if context_keywords:
            result = self._apply_context_bias(result, context_keywords)

        # Clean up extra whitespace
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _apply_context_bias(self, text: str, context_keywords: list[str]) -> str:
        """
        Use recent conversation context to resolve ambiguities.

        If the user recently discussed a topic (e.g., "placements"),
        bias corrections toward that domain.
        """
        text_lower = text.lower()

        # Context-aware substitutions
        context_set = {k.lower() for k in context_keywords}

        # If recently talking about placements, bias toward placement terms
        if context_set & {"placement", "placements", "recruit", "companies"}:
            text = re.sub(
                r"\bplacement\s+sell\b", "Placement Cell",
                text, flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\btraining\s+and\s+placements?\s+sell\b",
                "Training and Placement Cell",
                text, flags=re.IGNORECASE,
            )

        # If recently talking about HODs / departments
        if context_set & {"hod", "department", "computer", "electronics"}:
            text = re.sub(
                r"\bcomputer\s+signs\b", "Computer Science",
                text, flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\belectric\s+onyx\b", "Electronics",
                text, flags=re.IGNORECASE,
            )

        # If recently talking about leadership / principal
        if context_set & {"principal", "dean", "director", "leadership"}:
            text = re.sub(
                r"\bprincipal\b", "Principal",
                text, flags=re.IGNORECASE,
            )
            text = re.sub(
                r"\bdean\s+academic\b", "Dean Academics",
                text, flags=re.IGNORECASE,
            )

        return text


# ── Whisper binary / model discovery ───────────────────────────────────

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


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a transcription for context."""
    if not text:
        return []
    # Remove common stop words and keep significant terms
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "have", "has", "had", "will", "would",
        "can", "could", "may", "might", "shall", "should", "must",
        "i", "me", "my", "we", "our", "you", "your", "he", "she",
        "it", "they", "them", "his", "her", "its", "their",
        "this", "that", "these", "those", "what", "which", "who",
        "how", "when", "where", "why", "and", "or", "but", "not",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "about", "tell", "me", "please", "know", "want", "need",
        "there", "here", "yes", "no", "ok", "okay", "hey",
    }
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [w for w in words if w not in stop_words]


# ── STT Engine ─────────────────────────────────────────────────────────

class STTEngine:
    """
    Speech-to-text engine with contextual post-processing.

    Usage:
        stt = STTEngine()
        text = stt.transcribe("audio/temp.wav")
    """

    def __init__(self):
        self.backend = None
        self._whisper_bin = None
        self._whisper_model = None
        self._py_whisper = None

        # Contextual post-processing
        self._post_processor = ContextualPostProcessor()
        self._history: deque[str] = deque(maxlen=STT_CONTEXT_WINDOW)

        # 1. whisper.cpp (fastest on ARM)
        self._whisper_bin = _find_whisper_cpp()
        self._whisper_model = _find_model()
        if self._whisper_bin and self._whisper_model:
            self.backend = "whisper.cpp"
            print(f"🎤 STT: whisper.cpp")
            print(f"   bin   = {self._whisper_bin}")
            print(f"   model = {self._whisper_model}")
            if STT_CONTEXT_ENABLED:
                print(f"   ✅ Contextual post-processing enabled")
            return

        # 2. Python openai-whisper
        try:
            import whisper
            self._py_whisper = whisper.load_model("tiny.en")
            self.backend = "whisper-python"
            print("🎤 STT: Python Whisper (tiny.en)")
            if STT_CONTEXT_ENABLED:
                print(f"   ✅ Contextual post-processing enabled")
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
            if STT_CONTEXT_ENABLED:
                print(f"   ✅ Contextual post-processing enabled")
            return
        except ImportError:
            pass

        print("❌ No STT backend available.")
        print("   Build whisper.cpp  or  pip install openai-whisper")

    @property
    def available(self) -> bool:
        return self.backend is not None

    def transcribe(self, wav_path: str | None = None) -> str | None:
        """Transcribe a WAV file → text (with contextual post-processing)."""
        wav_path = wav_path or TEMP_WAV
        if not os.path.isfile(wav_path):
            print(f"❌ File not found: {wav_path}")
            return None

        # Get raw transcription from backend
        if self.backend == "whisper.cpp":
            raw = self._via_cpp(wav_path)
        elif self.backend == "whisper-python":
            raw = self._via_python(wav_path)
        elif self.backend == "google":
            raw = self._via_google(wav_path)
        else:
            return None

        if not raw:
            return None

        # Apply contextual post-processing
        if STT_CONTEXT_ENABLED:
            # Gather context keywords from recent history
            context_keywords = []
            for past in self._history:
                context_keywords.extend(_extract_keywords(past))

            corrected = self._post_processor.process(raw, context_keywords)

            if corrected != raw:
                print(f"   📝 Post-processed: \"{raw}\" → \"{corrected}\"")

            # Add to history
            self._history.append(corrected)
            return corrected if corrected else None

        return raw

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
            "--prompt", WHISPER_INITIAL_PROMPT,
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
                wav_path,
                language=WHISPER_LANGUAGE,
                fp16=False,
                initial_prompt=WHISPER_INITIAL_PROMPT,
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
