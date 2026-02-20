"""
UIT Prayagraj Chatbot
---------------------
An interactive Q&A chatbot that loads knowledge from a JSON file
(generated from uit.txt) and uses TF-IDF + cosine-similarity + fuzzy
matching to find the best answer for any user question.

Supports a wake-word mode ("Hey UIT") for always-on hands-free operation
on Raspberry Pi.

Usage:
    python chatbot.py              # text-only mode
    python chatbot.py --voice      # voice mode (mic + speaker)
    python chatbot.py --tts        # text input, spoken answers
    python chatbot.py --stt        # voice input, text answers
    python chatbot.py --daemon     # always-on: waits for "Hey UIT" wake word
"""

import json
import re
import math
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "knowledge_base.json"
TXT_FILE = BASE_DIR / "uit.txt"

# ── Stop-words (small built-in list so we don't need nltk) ──────────────
STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "is", "am",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "may", "might", "can",
    "could", "a", "an", "the", "and", "but", "or", "nor", "not", "so", "if",
    "of", "at", "by", "for", "with", "about", "to", "from", "in", "on", "up",
    "out", "into", "over", "after", "before", "between", "under", "again",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "some", "any", "no",
    "own", "same", "than", "too", "very", "just", "also", "what", "which",
    "who", "whom", "tell", "please", "know", "want", "need", "like",
}

# ── Synonym / alias mapping for domain-specific terms ──────────────────
SYNONYMS = {
    "principal": ["principal", "head", "director", "chief", "leader", "boss"],
    "placement": ["placement", "placements", "recruit", "recruitment", "job", "jobs", "career", "hiring", "hire", "package", "salary", "offer"],
    "hostel": ["hostel", "hostels", "accommodation", "stay", "boarding", "dormitory", "dorm", "rooms", "living"],
    "library": ["library", "books", "reading", "study", "journal", "journals", "ebook"],
    "wifi": ["wifi", "wi-fi", "internet", "network", "broadband", "connectivity", "online"],
    "lab": ["lab", "labs", "laboratory", "laboratories", "computer", "computers", "system", "systems", "practical"],
    "admission": ["admission", "admissions", "enroll", "enrollment", "join", "apply", "application", "entrance", "counseling", "counselling", "seat"],
    "ragging": ["ragging", "bully", "bullying", "harassment", "anti-ragging", "antiragging", "safe", "safety"],
    "sports": ["sports", "games", "playground", "indoor", "outdoor", "cricket", "football", "gym", "fitness", "athletic"],
    "research": ["research", "innovation", "startup", "startups", "project", "projects", "paper", "papers", "publish"],
    "erp": ["erp", "attendance", "registration", "portal", "online", "record", "records"],
    "cctv": ["cctv", "security", "surveillance", "camera", "guard", "safe", "safety", "monitor"],
    "seminar": ["seminar", "seminars", "auditorium", "hall", "conference", "event", "events"],
    "club": ["club", "clubs", "society", "societies", "cultural", "technical", "fest", "festival", "extracurricular"],
    "workshop": ["workshop", "workshops", "fdp", "seminar", "training", "program", "programme", "course"],
    "result": ["result", "results", "exam", "examination", "marks", "grade", "grades", "cgpa", "sgpa", "score", "performance", "pass", "fail"],
    "uit": ["uit", "united", "institute", "college", "campus", "prayagraj", "allahabad"],
    "aktu": ["aktu", "university", "kalam", "abdul", "affiliated", "affiliation"],
    "aicte": ["aicte", "approved", "approval", "recognized", "recognition", "accredited"],
    "faculty": ["faculty", "teacher", "teachers", "professor", "professors", "staff", "hod", "dean", "academic"],
    "smart": ["smart", "classroom", "classrooms", "digital", "projector", "board"],
    "internship": ["internship", "internships", "intern", "industrial", "training", "experience"],
    "discipline": ["discipline", "rules", "conduct", "behavior", "behaviour", "strict", "policy"],
    "counseling": ["counseling", "counselling", "guidance", "mentor", "mentoring", "support", "help"],
}


def tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, remove stop-words."""
    tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def expand_with_synonyms(tokens: list[str]) -> list[str]:
    """Add synonym variants so queries like 'job' also match 'placement'."""
    expanded = list(tokens)
    for token in tokens:
        for _key, group in SYNONYMS.items():
            if token in group:
                expanded.extend(group)
    return list(set(expanded))


# ── Fuzzy string helpers ───────────────────────────────────────────────
def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def word_similarity(a: str, b: str) -> float:
    """Return 0.0-1.0 similarity between two words."""
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein(a, b)
    return max(0.0, 1.0 - dist / max_len)


# ── TF-IDF helpers ─────────────────────────────────────────────────────
def build_tfidf(documents: list[list[str]]):
    """Return (tfidf_vectors, idf_dict) for a corpus of tokenised docs."""
    n = len(documents)
    df: Counter = Counter()
    for doc in documents:
        df.update(set(doc))

    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}

    vectors = []
    for doc in documents:
        tf = Counter(doc)
        doc_len = len(doc) or 1
        vec = {term: (count / doc_len) * idf.get(term, 0) for term, count in tf.items()}
        vectors.append(vec)

    return vectors, idf


def cosine_sim(vec_a: dict, vec_b: dict) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Chatbot class ──────────────────────────────────────────────────────
class UITChatbot:
    SIMILARITY_THRESHOLD = 0.08  # Lower threshold for more general matching

    def __init__(self, json_path: Path = JSON_FILE):
        self.kb = self._load_kb(json_path)
        self.doc_tokens = [
            tokenize(item["question"] + " " + item["answer"]) for item in self.kb
        ]
        self.tfidf_vectors, self.idf = build_tfidf(self.doc_tokens)

    @staticmethod
    def _load_kb(path: Path) -> list[dict]:
        if not path.exists():
            from parse_txt import parse_txt_to_json
            kb = parse_txt_to_json(TXT_FILE)
            path.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
            return kb
        return json.loads(path.read_text(encoding="utf-8"))

    def _query_vector(self, text: str) -> dict:
        tokens = expand_with_synonyms(tokenize(text))
        tf = Counter(tokens)
        doc_len = len(tokens) or 1
        return {term: (count / doc_len) * self.idf.get(term, 1) for term, count in tf.items()}

    def _keyword_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """
        Fuzzy keyword overlap score: for each query token, find the best
        matching token in the document using word similarity.
        Returns 0.0-1.0.
        """
        doc_toks = self.doc_tokens[doc_idx]
        if not query_tokens or not doc_toks:
            return 0.0

        total = 0.0
        for qt in query_tokens:
            best = max((word_similarity(qt, dt) for dt in doc_toks), default=0.0)
            total += best

        return total / len(query_tokens)

    def get_answer(self, user_input: str) -> str:
        if not user_input.strip():
            return "Please ask a question about UIT Prayagraj."

        query_tokens = tokenize(user_input)
        expanded_tokens = expand_with_synonyms(query_tokens)
        query_vec = self._query_vector(user_input)

        scored = []
        for idx, dv in enumerate(self.tfidf_vectors):
            tfidf_score = cosine_sim(query_vec, dv)
            kw_score = self._keyword_score(expanded_tokens, idx)

            # Combined score: 60% TF-IDF + 40% fuzzy keyword
            combined = 0.6 * tfidf_score + 0.4 * kw_score
            scored.append((combined, tfidf_score, kw_score, idx))

        scored.sort(reverse=True, key=lambda x: x[0])

        best_combined, best_tfidf, best_kw, best_idx = scored[0]

        if best_combined < self.SIMILARITY_THRESHOLD:
            return (
                "I'm not sure about that. Could you rephrase your question? "
                "I can help with topics like admissions, placements, facilities, "
                "faculty, hostel, sports, and more at UIT Prayagraj."
            )

        entry = self.kb[best_idx]
        response = f"📌 {entry['answer']}"

        # Include related answer if second-best is close
        if len(scored) > 1:
            second_combined = scored[1][0]
            second_idx = scored[1][3]
            if second_combined > self.SIMILARITY_THRESHOLD and (best_combined - second_combined) < 0.08:
                response += f"\n\nℹ️  Related: {self.kb[second_idx]['answer']}"

        return response

    def list_categories(self) -> list[str]:
        return sorted({item["category"] for item in self.kb})


# ── Interactive CLI ────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="UIT Prayagraj Chatbot")
    parser.add_argument(
        "--voice", action="store_true",
        help="Enable full voice mode (speech-to-text input + text-to-speech output)",
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="Enable text-to-speech only (type questions, hear answers)",
    )
    parser.add_argument(
        "--stt", action="store_true",
        help="Enable speech-to-text only (speak questions, read answers)",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Always-on mode: listen for wake word 'Hey UIT', then answer",
    )
    parser.add_argument(
        "--wake-word", type=str, default="hey uit",
        help="Custom wake word/phrase (default: 'hey uit')",
    )
    return parser.parse_args()


def run_daemon(bot, tts_engine, stt_engine, wake_phrase: str):
    """
    Always-on daemon mode:
      1. Wait passively for the wake word ("Hey UIT")
      2. Play an acknowledgement sound / speak "Yes?"
      3. Listen for the actual question
      4. Answer it with voice
      5. Go back to waiting
    """
    from wake_word import WakeWordDetector

    detector = WakeWordDetector(
        wake_phrase=wake_phrase,
        alternative_phrases=["ok uit", "hello uit", "hi uit", "hey you it"],
    )

    if not detector.available:
        print("❌ Wake word detection not available. Install: pip install openai-whisper PyAudio")
        return

    print("=" * 60)
    print("  🎓  UIT Prayagraj Chatbot — DAEMON MODE")
    print(f"  Wake word: \"{wake_phrase}\"")
    print("  Say the wake word, then ask your question.")
    print("  Say 'stop' or 'shutdown' to exit.")
    print("  Press Ctrl+C to force quit.")
    print("=" * 60)

    print(f"\n✅ Loaded {len(bot.kb)} Q&A entries.")

    if tts_engine:
        tts_engine.speak("UIT chatbot is ready. Say Hey UIT to wake me up.")

    print("\n💤 Waiting for wake word...\n")

    while True:
        try:
            # Step 1: Wait for wake word
            detected = detector.wait_for_wake_word()
            if not detected:
                continue

            # Step 2: Acknowledge
            print("🔔 Wake word detected!")
            if tts_engine:
                tts_engine.speak("Yes? How can I help you?")
            else:
                print("Bot: Yes? How can I help you?")

            # Step 3: Listen for the question
            question = None
            if stt_engine:
                question = stt_engine.listen("🎤 Listening for your question...")
                if question:
                    print(f"You: {question}")

            if not question:
                if tts_engine:
                    tts_engine.speak("I didn't catch that. Say Hey UIT to try again.")
                print("💤 Waiting for wake word...\n")
                continue

            # Check for shutdown commands
            lower = question.lower().strip()
            if lower in ("stop", "shutdown", "quit", "exit", "bye", "turn off", "shut down"):
                farewell = "Goodbye! Shutting down."
                print(f"Bot: {farewell}")
                if tts_engine:
                    tts_engine.speak(farewell)
                break

            # Step 4: Answer the question
            answer = bot.get_answer(question)
            print(f"\nBot: {answer}\n")
            if tts_engine:
                tts_engine.speak(answer)

            print("💤 Waiting for wake word...\n")

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            if tts_engine:
                tts_engine.speak("Goodbye!")
            break


def main():
    args = parse_args()

    # Daemon mode implies full voice
    if args.daemon:
        args.voice = True

    use_tts = args.voice or args.tts
    use_stt = args.voice or args.stt

    # ── Initialise TTS ─────────────────────────────────────────────────
    tts_engine = None
    if use_tts:
        try:
            from tts_module import TTSEngine
            tts_engine = TTSEngine(voice="indian_female")
            if not tts_engine.available:
                print("⚠️  TTS unavailable — falling back to text output.")
                tts_engine = None
        except ImportError:
            print("⚠️  tts_module not found — falling back to text output.")

    # ── Initialise STT ─────────────────────────────────────────────────
    stt_engine = None
    if use_stt:
        try:
            from stt_module import STTEngine
            stt_engine = STTEngine()
            if not stt_engine.available:
                print("⚠️  STT unavailable — falling back to text input.")
                stt_engine = None
        except ImportError:
            print("⚠️  stt_module not found — falling back to text input.")

    # ── Load chatbot ───────────────────────────────────────────────────
    bot = UITChatbot()

    # ── Daemon mode ────────────────────────────────────────────────────
    if args.daemon:
        run_daemon(bot, tts_engine, stt_engine, args.wake_word)
        return

    # ── Banner ─────────────────────────────────────────────────────────
    mode = "TEXT"
    if use_tts and use_stt:
        mode = "VOICE (speak + listen)"
    elif use_tts:
        mode = "TTS (type + listen)"
    elif use_stt:
        mode = "STT (speak + read)"

    print("=" * 60)
    print("  🎓  UIT Prayagraj Chatbot")
    print("  Ask any question about United Institute of Technology")
    print(f"  Mode: {mode}")
    print("  Type 'categories' to see topics | 'quit' to exit")
    if tts_engine:
        print("  Type 'voices' to see voice options | 'voice <name>' to change")
    if stt_engine:
        print("  Say 'quit' or 'exit' to stop | 'type' for keyboard")
    print("=" * 60)

    print(f"\n✅ Loaded {len(bot.kb)} Q&A entries.\n")

    if tts_engine:
        tts_engine.speak("UIT Prayagraj chatbot is ready. Ask me anything.")

    while True:
        # ── Get user input (mic or keyboard) ───────────────────────────
        user_input = None
        if stt_engine:
            user_input = stt_engine.listen("🎤 Listening... (say 'type' for keyboard)")
            if user_input:
                print(f"You (voice): {user_input}")
            else:
                try:
                    user_input = input("You (keyboard): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye! 👋")
                    break
        else:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")
                break

        if not user_input:
            continue

        lower = user_input.lower().strip()

        # ── Commands ───────────────────────────────────────────────────
        if lower in ("quit", "exit", "bye", "q", "stop"):
            farewell = "Goodbye!"
            print(farewell, "👋")
            if tts_engine:
                tts_engine.speak(farewell)
            break

        if lower == "type":
            try:
                user_input = input("You (keyboard): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")
                break
            if not user_input:
                continue

        if lower == "categories":
            print("\n📂 Available categories:")
            cats = bot.list_categories()
            for cat in cats:
                print(f"   • {cat}")
            print()
            if tts_engine:
                tts_engine.speak("Available categories are: " + ", ".join(cats))
            continue

        if lower == "voices" and tts_engine:
            print("\n🔊 Available voice presets:")
            for v in tts_engine.list_voices():
                print(f"   • {v}")
            print("\n   Usage: type 'voice indian_male' to switch")
            print()
            continue

        if lower.startswith("voice ") and tts_engine:
            voice_name = user_input.split(" ", 1)[1].strip()
            tts_engine.set_voice(voice_name)
            tts_engine.speak(f"Voice changed. This is how I sound now.")
            continue

        # ── Get answer ─────────────────────────────────────────────────
        answer = bot.get_answer(user_input)
        print(f"\nBot: {answer}\n")

        if tts_engine:
            tts_engine.speak(answer)


if __name__ == "__main__":
    main()
