"""
UIT Prayagraj Chatbot
---------------------
An interactive Q&A chatbot that loads knowledge from a JSON file
(generated from uit.txt) and uses TF-IDF + cosine-similarity to
find the best matching answer for any user question.

Usage:
    python chatbot.py              # text-only mode
    python chatbot.py --voice      # voice mode (speak answers + listen via mic)
    python chatbot.py --tts        # text input, spoken answers
    python chatbot.py --stt        # voice input, text answers
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
    "principal": ["principal", "head"],
    "placement": ["placement", "placements", "recruit", "recruitment", "job", "jobs", "career"],
    "hostel": ["hostel", "hostels", "accommodation", "stay", "boarding"],
    "library": ["library", "books", "reading"],
    "wifi": ["wifi", "wi-fi", "internet", "network"],
    "lab": ["lab", "labs", "laboratory", "laboratories", "computer"],
    "admission": ["admission", "admissions", "enroll", "enrollment", "join"],
    "ragging": ["ragging", "bully", "bullying", "harassment"],
    "sports": ["sports", "games", "playground", "indoor", "outdoor"],
    "research": ["research", "innovation", "startup", "startups"],
    "erp": ["erp", "attendance", "registration"],
    "cctv": ["cctv", "security", "surveillance", "camera"],
    "seminar": ["seminar", "seminars", "auditorium", "hall"],
    "club": ["club", "clubs", "society", "societies", "cultural", "technical"],
    "workshop": ["workshop", "workshops", "fdp", "seminar"],
    "result": ["result", "results", "exam", "examination", "marks", "grade"],
    "uit": ["uit", "united institute", "united institute of technology"],
    "aktu": ["aktu", "university", "kalam", "abdul kalam"],
    "aicte": ["aicte", "approved", "approval"],
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
    SIMILARITY_THRESHOLD = 0.15

    def __init__(self, json_path: Path = JSON_FILE):
        self.kb = self._load_kb(json_path)
        self.doc_tokens = [
            tokenize(item["question"] + " " + item["answer"]) for item in self.kb
        ]
        self.tfidf_vectors, self.idf = build_tfidf(self.doc_tokens)

    @staticmethod
    def _load_kb(path: Path) -> list[dict]:
        if not path.exists():
            # Auto-generate from txt if JSON missing
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

    def get_answer(self, user_input: str) -> str:
        if not user_input.strip():
            return "Please ask a question about UIT Prayagraj."

        query_vec = self._query_vector(user_input)
        scored = [
            (cosine_sim(query_vec, dv), idx)
            for idx, dv in enumerate(self.tfidf_vectors)
        ]
        scored.sort(reverse=True)

        best_score, best_idx = scored[0]

        if best_score < self.SIMILARITY_THRESHOLD:
            return (
                "Sorry, I don't have enough information to answer that. "
                "Please ask something related to UIT Prayagraj."
            )

        entry = self.kb[best_idx]

        # If second-best is very close, include it too
        response = f"📌 {entry['answer']}"
        if len(scored) > 1:
            second_score, second_idx = scored[1]
            if second_score > self.SIMILARITY_THRESHOLD and (best_score - second_score) < 0.05:
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
    return parser.parse_args()


def main():
    args = parse_args()
    use_tts = args.voice or args.tts
    use_stt = args.voice or args.stt

    # ── Initialise TTS ─────────────────────────────────────────────────
    tts_engine = None
    if use_tts:
        try:
            from tts_module import TTSEngine
            tts_engine = TTSEngine(rate=160, volume=1.0)
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
    if stt_engine:
        print("  Say 'quit' or 'exit' to stop | 'type' for keyboard")
    print("=" * 60)

    bot = UITChatbot()
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
                # Fallback to keyboard on failed recognition
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
            # Temporarily use keyboard even in voice mode
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

        # ── Get answer ─────────────────────────────────────────────────
        answer = bot.get_answer(user_input)
        print(f"\nBot: {answer}\n")

        if tts_engine:
            tts_engine.speak(answer)


if __name__ == "__main__":
    main()
