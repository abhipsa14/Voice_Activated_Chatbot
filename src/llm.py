"""
LLM + RAG Hybrid Engine
=========================
Two-stage answer pipeline for sub-second latency:

  Stage 1 – RAG lookup:  TF-IDF + fuzzy keyword match over the
            UIT knowledge base.  If a high-confidence match is found
            the answer is returned instantly (no LLM call needed).

  Stage 2 – Ollama LLM:  For queries that fall outside the knowledge
            base, the question is forwarded to TinyLlama via Ollama's
            local HTTP API.

Both stages use the multi-level cache (L1 memory → L2 disk → L3 Redis)
so repeat questions return in <1 ms.

Conversation history is maintained as a rolling window (last 6 turns).
"""

import json
import math
import os
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path

try:
    import requests as _requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# Ensure project root is on sys.path so config.config resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    OLLAMA_HOST, OLLAMA_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, LLM_SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    KB_JSON_FILE, KB_TXT_FILE,
    RAG_CONFIDENCE_THRESHOLD, RAG_TOP_K,
)
from src.cache import CacheManager


# ═══════════════════════════════════════════════════════════════════════
# Stop-words, synonyms, helpers  (kept minimal to save RAM on Pi)
# ═══════════════════════════════════════════════════════════════════════

STOP = {
    "i","me","my","we","our","you","your","he","she","it","its",
    "they","them","their","this","that","these","those","is","am",
    "are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","shall","should","may","might","can",
    "could","a","an","the","and","but","or","nor","not","so","if",
    "of","at","by","for","with","about","to","from","in","on","up",
    "out","into","over","after","before","between","under","again",
    "then","once","here","there","when","where","why","how","all",
    "each","every","both","few","more","most","some","any","no",
    "own","same","than","too","very","just","also","what","which",
    "who","whom","tell","please","know","want","need","like",
}

SYNONYMS = {
    "principal": ["head","director","chief","leader"],
    "placement": ["placements","recruit","recruitment","job","jobs",
                   "career","hiring","package","salary","offer"],
    "hostel": ["hostels","accommodation","dormitory","dorm","boarding"],
    "library": ["books","reading","journal","ebook"],
    "wifi": ["wi-fi","internet","network","broadband","connectivity"],
    "lab": ["labs","laboratory","laboratories","computer","practical"],
    "admission": ["admissions","enroll","enrollment","apply",
                   "application","entrance","counseling","seat"],
    "ragging": ["bully","bullying","harassment","anti-ragging","safety"],
    "sports": ["games","playground","indoor","outdoor","cricket",
               "football","gym","fitness"],
    "research": ["innovation","startup","project","paper","publish"],
    "erp": ["attendance","registration","portal","record"],
    "cctv": ["security","surveillance","camera","guard","monitor"],
    "club": ["clubs","society","societies","cultural","technical",
             "fest","festival","extracurricular"],
    "workshop": ["workshops","fdp","seminar","training","programme"],
    "result": ["results","exam","examination","marks","grade","cgpa",
               "sgpa","score","performance"],
    "uit": ["united","institute","college","campus","prayagraj"],
    "aktu": ["university","kalam","affiliated","affiliation"],
    "faculty": ["teacher","professor","staff","hod","dean","academic"],
}


def _tok(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 1]


def _expand(tokens: list[str]) -> list[str]:
    expanded = set(tokens)
    for t in tokens:
        for _key, group in SYNONYMS.items():
            if t == _key or t in group:
                expanded.add(_key)
                expanded.update(group)
    return list(expanded)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _word_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    mx = max(len(a), len(b))
    return max(0.0, 1.0 - _levenshtein(a, b) / mx) if mx else 1.0


# ═══════════════════════════════════════════════════════════════════════
# RAG Retriever  (lightweight TF-IDF + fuzzy keyword, no ML dependency)
# ═══════════════════════════════════════════════════════════════════════

class _RAG:
    """In-process RAG over the UIT knowledge base."""

    def __init__(self, kb: list[dict]):
        self.kb = kb
        self.doc_tokens = [_tok(e["question"] + " " + e["answer"]) for e in kb]

        # Build TF-IDF
        n = len(self.doc_tokens)
        df: Counter = Counter()
        for doc in self.doc_tokens:
            df.update(set(doc))
        self.idf = {t: math.log((n+1)/(c+1))+1 for t, c in df.items()}
        self.vectors = []
        for doc in self.doc_tokens:
            tf = Counter(doc)
            dl = len(doc) or 1
            self.vectors.append(
                {t: (c/dl)*self.idf.get(t,0) for t, c in tf.items()}
            )

    def query(self, text: str, top_k: int = RAG_TOP_K
              ) -> list[tuple[dict, float]]:
        """Return [(kb_entry, score), ...] ranked by relevance."""
        q_tokens = _tok(text)
        q_expanded = _expand(q_tokens)
        q_tf = Counter(q_expanded)
        q_len = len(q_expanded) or 1
        q_vec = {t: (c/q_len)*self.idf.get(t,1) for t, c in q_tf.items()}

        scored = []
        for idx, dv in enumerate(self.vectors):
            # TF-IDF cosine
            common = set(q_vec) & set(dv)
            if not common:
                tfidf = 0.0
            else:
                dot = sum(q_vec[k]*dv[k] for k in common)
                ma = math.sqrt(sum(v*v for v in q_vec.values()))
                mb = math.sqrt(sum(v*v for v in dv.values()))
                tfidf = dot/(ma*mb) if ma and mb else 0.0

            # Fuzzy keyword score
            dtoks = self.doc_tokens[idx]
            if q_expanded and dtoks:
                kw = sum(max((_word_sim(qt,dt) for dt in dtoks), default=0)
                         for qt in q_expanded) / len(q_expanded)
            else:
                kw = 0.0

            score = 0.55*tfidf + 0.45*kw
            scored.append((idx, score))

        scored.sort(key=lambda x: -x[1])
        return [(self.kb[i], s) for i, s in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════
# Main LLM Engine
# ═══════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    Hybrid RAG + LLM engine with multi-level caching.

    Usage:
        engine = LLMEngine()
        answer = engine.chat("Who is the principal?")
    """

    # Greeting patterns
    _GREET_RE = re.compile(
        r"^(hi|hello|hey|hii|hiii|good\s*(morning|evening|afternoon|night)|"
        r"namaste|howdy|sup|what'?s\s*up)\b",
        re.IGNORECASE,
    )
    _GREETING_REPLY = (
        "Hello! I'm UIT Bot, the voice assistant for United Institute of "
        "Technology, Prayagraj. Ask me anything about UIT!"
    )

    def __init__(self):
        self.cache = CacheManager()
        self._history: deque[dict] = deque(maxlen=MAX_HISTORY_TURNS * 2)
        self._rag: _RAG | None = None
        self._ollama_ok = False

        # Load knowledge base → build RAG index
        kb = self._load_kb()
        if kb:
            self._rag = _RAG(kb)
            print(f"📚 RAG: indexed {len(kb)} Q&A entries")

        # Check Ollama
        if REQUESTS_OK:
            self._ollama_ok = self._ping_ollama()
            status = "✅ connected" if self._ollama_ok else "⚠️  not reachable"
            print(f"🧠 LLM: Ollama ({OLLAMA_MODEL}) — {status}")
        else:
            print("⚠️  'requests' not installed — LLM disabled")

        # Print cache status
        cs = self.cache.stats()
        print(f"🗄️  Cache: L1={cs['l1_size']}, L2={cs['l2_size']}, "
              f"Redis={'✅' if cs['redis'] else '❌'}")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._rag is not None or self._ollama_ok

    def chat(self, user_input: str) -> str:
        """
        Main entry point.  Checks cache → greetings → RAG → LLM.
        """
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you repeat?"

        query = user_input.strip()

        # 0. Cache check (all three levels)
        cached = self.cache.get(query)
        if cached:
            print(f"   ⚡ [CACHE HIT]")
            return cached

        # 1. Greeting
        if self._GREET_RE.match(query):
            self.cache.put(query, self._GREETING_REPLY)
            return self._GREETING_REPLY

        # 2. RAG lookup
        if self._rag:
            results = self._rag.query(query)
            if results:
                best_entry, best_score = results[0]
                print(f"   📊 [RAG] top score = {best_score:.4f}")

                if best_score >= RAG_CONFIDENCE_THRESHOLD:
                    answer = f"📌 {best_entry['answer']}"

                    self.cache.put(query, answer)
                    self._add_history(query, answer)
                    return answer

                print(f"   📊 [RAG] below threshold, trying LLM...")

        # 3. Ollama LLM
        if self._ollama_ok:
            answer = self._call_ollama(query)
            if answer:
                self.cache.put(query, answer)
                self._add_history(query, answer)
                return answer

        # 4. Nothing available
        fallback = (
            "I'm not sure about that. Could you rephrase? I can help with "
            "admissions, placements, facilities, faculty, hostel, sports, "
            "and more at UIT Prayagraj."
        )
        return fallback

    def clear_history(self):
        self._history.clear()

    def get_stats(self) -> dict:
        return self.cache.stats()

    def warm_cache(self):
        """Pre-populate cache with all KB Q&A pairs for instant replies."""
        if not self._rag:
            return
        count = 0
        for entry in self._rag.kb:
            q = entry["question"]
            a = f"📌 {entry['answer']}"
            if not self.cache.get(q):
                self.cache.put(q, a)
                count += 1
        print(f"   🔥 Warmed {count} entries into cache")

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _load_kb() -> list[dict]:
        if KB_JSON_FILE.exists():
            try:
                return json.loads(KB_JSON_FILE.read_text("utf-8"))
            except Exception:
                pass
        # Generate from txt
        if KB_TXT_FILE.exists():
            try:
                from parse_txt import parse_txt_to_json
                kb = parse_txt_to_json(KB_TXT_FILE)
                KB_JSON_FILE.write_text(
                    json.dumps(kb, indent=2, ensure_ascii=False), "utf-8"
                )
                return kb
            except Exception as e:
                print(f"⚠️  KB parse failed: {e}")
        return []

    @staticmethod
    def _ping_ollama() -> bool:
        try:
            r = _requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _call_ollama(self, query: str) -> str | None:
        msgs = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        msgs.extend(list(self._history))
        msgs.append({"role": "user", "content": query})

        t0 = time.time()
        try:
            r = _requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": msgs,
                    "stream": False,
                    "options": {
                        "num_predict": LLM_MAX_TOKENS,
                        "temperature": LLM_TEMPERATURE,
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            reply = r.json().get("message", {}).get("content", "").strip()
            elapsed = time.time() - t0
            print(f"   🧠 [LLM] {elapsed:.1f}s")
            return reply if reply else None
        except Exception as e:
            print(f"   ❌ Ollama error: {e}")
            return None

    def _add_history(self, query: str, answer: str):
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": answer})


# ── Module-level convenience ───────────────────────────────────────────
_engine: LLMEngine | None = None


def chat(message: str) -> str:
    global _engine
    if _engine is None:
        _engine = LLMEngine()
    return _engine.chat(message)


if __name__ == "__main__":
    print("=== LLM + RAG Test ===\n")
    e = LLMEngine()
    e.warm_cache()
    print("\nType a question ('quit' to exit):\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit","exit","q"):
            break
        if q.lower() == "stats":
            print(f"📊 {e.get_stats()}\n")
            continue
        print(f"Bot: {e.chat(q)}\n")
