"""
Raspberry Pi Voice Chatbot – Main Loop
=========================================
Alexa-like voice assistant for UIT Prayagraj.

Pipeline:
  "Ok UIT" wake word → VAD capture → Whisper STT
  → RAG/LLM answer → Piper TTS → speaker

Usage:
    python -m src.main                     # voice + wake word
    python -m src.main --no-wake-word      # always listen
    python -m src.main --text              # text-only (no mic)
"""

import argparse
import os
import re
import sys
import time

# Ensure project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import WAKE_WORD, WAKE_WORD_ALTERNATIVES


# ── Argument parsing ───────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(description="UIT Voice Chatbot")
    p.add_argument("--text", action="store_true",
                   help="Text-only mode (no mic/speaker)")
    p.add_argument("--no-wake-word", action="store_true",
                   help="Skip wake word — always listen")
    p.add_argument("--wake-word", type=str, default=WAKE_WORD,
                   help=f"Custom wake phrase (default: '{WAKE_WORD}')")
    p.add_argument("--mic", type=int, default=None,
                   help="Microphone device index")
    p.add_argument("--list-mics", action="store_true",
                   help="List microphones and exit")
    return p.parse_args()


# ── Wake-word matching (fuzzy) ─────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _wake_match(text: str, wake: str) -> bool:
    """Fuzzy check: does *text* contain the wake phrase?"""
    tn = _normalise(text)
    phrases = [wake] + WAKE_WORD_ALTERNATIVES
    for phrase in phrases:
        pn = _normalise(phrase)
        if pn in tn:
            return True
        # All words of the phrase appear in text
        pw = pn.split()
        tw = tn.split()
        if all(any(p in t or t in p for t in tw) for p in pw):
            return True
    return False


# ── Banner ─────────────────────────────────────────────────────────────

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║              🎓  UIT Prayagraj Voice Assistant               ║
╠══════════════════════════════════════════════════════════════╣
║  Pipeline : Mic → VAD → Whisper STT → RAG/LLM → Piper TTS  ║
║  Wake word: "{wake}"                                         
║  Mode     : {mode}                                           
║                                                              ║
║  Say "{wake}" to activate, then ask your question.           
║  Say "quit" or "exit" to stop.  Ctrl+C to force quit.       
╚══════════════════════════════════════════════════════════════╝
"""


# ── Voice loop ─────────────────────────────────────────────────────────

def run_voice(args):
    from src.vad import VADRecorder
    from src.stt import STTEngine
    from src.llm import LLMEngine
    from src.tts import TTSEngine

    print("🔄 Initialising pipeline...\n")
    vad = VADRecorder(mic_index=args.mic)
    stt = STTEngine()
    llm = LLMEngine()
    tts = TTSEngine()

    if not vad.available:
        print("❌ Microphone not available. Exiting.")
        return
    if not stt.available:
        print("❌ No STT backend. Exiting.")
        return

    # Warm cache
    print("\n🔥 Warming caches...")
    llm.warm_cache()

    use_wake = not args.no_wake_word
    mode = f"Wake Word (\"{args.wake_word}\")" if use_wake else "Always Listening"
    print(_BANNER.format(wake=args.wake_word, mode=mode))

    if tts.available:
        tts.speak("UIT chatbot is ready. Say Ok UIT to wake me up.")
    print("✅ Pipeline ready.\n")

    while True:
        try:
            # ── Wake word phase ────────────────────────────────────
            if use_wake:
                print(f"💤 Waiting for \"{args.wake_word}\"...\n")
                wav = vad.record_short()
                if not wav:
                    continue
                heard = stt.transcribe(wav)
                if not heard:
                    continue
                if not _wake_match(heard, args.wake_word):
                    print(f"   [heard: \"{heard}\"] — not wake word\n")
                    continue
                print(f"🔔 Wake word detected! (\"{heard}\")")
                if tts.available:
                    tts.speak("Yes? How can I help?")

            # ── Capture question ───────────────────────────────────
            wav = vad.record("🎤 What's your question?")
            if not wav:
                if tts.available:
                    tts.speak("I didn't catch that. Say Ok UIT to try again.")
                print()
                continue

            # ── Transcribe ─────────────────────────────────────────
            t0 = time.time()
            question = stt.transcribe(wav)
            stt_ms = (time.time() - t0) * 1000

            if not question:
                if tts.available:
                    tts.speak("Sorry, couldn't understand. Please try again.")
                print()
                continue

            print(f"\n   You: {question}  ({stt_ms:.0f} ms STT)")

            # Exit commands
            if question.lower().strip() in (
                "quit","exit","stop","bye","shutdown","turn off",
                "shut down","goodbye"
            ):
                msg = "Goodbye! Have a great day!"
                print(f"   Bot: {msg}")
                if tts.available:
                    tts.speak(msg)
                break

            # ── Get answer ─────────────────────────────────────────
            t0 = time.time()
            answer = llm.chat(question)
            llm_ms = (time.time() - t0) * 1000
            print(f"   Bot: {answer}  ({llm_ms:.0f} ms LLM)")

            # ── Speak answer ───────────────────────────────────────
            if tts.available:
                t0 = time.time()
                tts.speak(answer)
                tts_ms = (time.time() - t0) * 1000
                print(f"   ({tts_ms:.0f} ms TTS)")

            total = stt_ms + llm_ms
            print(f"   ─── Total pipeline: {total:.0f} ms ───\n")

        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            if tts.available:
                tts.speak("Goodbye!")
            break

    print(f"\n📊 Session stats: {llm.get_stats()}")


# ── Text loop ──────────────────────────────────────────────────────────

def run_text(args):
    from src.llm import LLMEngine

    print("🔄 Initialising...\n")
    llm = LLMEngine()
    llm.warm_cache()

    print("\n" + "=" * 56)
    print("  🎓 UIT Prayagraj Chatbot (Text Mode)")
    print("  Type your question.  'quit' to exit.  'stats' for cache.")
    print("=" * 56 + "\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        if not q:
            continue
        lo = q.lower()
        if lo in ("quit","exit","q","bye"):
            print("👋 Goodbye!")
            break
        if lo == "clear":
            llm.clear_history()
            print("🧹 History cleared.\n")
            continue
        if lo == "stats":
            print(f"📊 {llm.get_stats()}\n")
            continue

        t0 = time.time()
        ans = llm.chat(q)
        ms = (time.time() - t0) * 1000
        print(f"Bot: {ans}  ({ms:.0f} ms)\n")

    print(f"\n📊 {llm.get_stats()}")


# ── List mics ──────────────────────────────────────────────────────────

def list_mics():
    try:
        import speech_recognition as sr
        names = sr.Microphone.list_microphone_names()
        print(f"\n🎙️ Found {len(names)} device(s):\n")
        for i, n in enumerate(names):
            print(f"  [{i}] {n}")
        print(f"\n💡 Usage: python -m src.main --mic <index>")
    except ImportError:
        print("❌ SpeechRecognition not installed.")
    except Exception as e:
        print(f"❌ {e}")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    args = _args()
    if args.list_mics:
        list_mics()
        return
    if args.text:
        run_text(args)
    else:
        run_voice(args)


if __name__ == "__main__":
    main()
