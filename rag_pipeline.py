"""
RAG (Retrieval-Augmented Generation) Pipeline for UIT Chatbot
--------------------------------------------------------------
Upgrades the chatbot from basic TF-IDF matching to a semantic
retrieval pipeline with intelligent answer generation.

Architecture:
  1. Document Chunking — split knowledge base into overlapping semantic chunks
  2. Dual Retrieval   — combine TF-IDF (lexical) + sentence embeddings (semantic)
  3. Re-Ranking        — fuse scores from both retrievers for best results
  4. Answer Generation — template-based contextual answer synthesis
                         (no external LLM required — runs fully offline)

Why not a full LLM?
  - This runs on Raspberry Pi — no GPU, limited RAM
  - The knowledge base is small and factual (no creative generation needed)
  - Template-based generation is deterministic and fast

Dependencies:
  pip install sentence-transformers   # for semantic embeddings

  If sentence-transformers is unavailable, the pipeline falls back to
  TF-IDF-only mode (still better than before due to improved chunking
  and re-ranking).
"""

import json
import math
import re
import os
import hashlib
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"

# ── Check for sentence-transformers ────────────────────────────────────
SEMANTIC_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SEMANTIC_AVAILABLE = True
except ImportError:
    pass

# ── Stop-words (same as chatbot.py for consistency) ────────────────────
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


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, remove stop-words."""
    tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ═══════════════════════════════════════════════════════════════════════
# Document Chunking
# ═══════════════════════════════════════════════════════════════════════
class DocumentChunk:
    """A chunk of knowledge with metadata for retrieval."""

    __slots__ = ("text", "question", "answer", "category", "chunk_id",
                 "source_index", "tokens")

    def __init__(self, text: str, question: str, answer: str,
                 category: str, source_index: int, chunk_id: int):
        self.text = text
        self.question = question
        self.answer = answer
        self.category = category
        self.source_index = source_index
        self.chunk_id = chunk_id
        self.tokens = _tokenize(text)

    def __repr__(self):
        return f"Chunk({self.chunk_id}, cat={self.category}, q='{self.question[:40]}...')"


def chunk_knowledge_base(kb: list[dict]) -> list[DocumentChunk]:
    """
    Create retrieval chunks from the knowledge base.

    Each Q&A pair becomes multiple overlapping chunks:
      1. question-only chunk (for matching question-style queries)
      2. answer-only chunk (for matching fact-seeking queries)
      3. combined Q+A chunk (for broad matching)
      4. category + question chunk (for category-aware retrieval)
    """
    chunks = []
    chunk_id = 0

    for idx, entry in enumerate(kb):
        q = entry.get("question", "").strip()
        a = entry.get("answer", "").strip()
        cat = entry.get("category", "General").strip()

        if not q and not a:
            continue

        # Chunk 1: Question only
        chunks.append(DocumentChunk(
            text=q,
            question=q, answer=a, category=cat,
            source_index=idx, chunk_id=chunk_id,
        ))
        chunk_id += 1

        # Chunk 2: Answer only
        chunks.append(DocumentChunk(
            text=a,
            question=q, answer=a, category=cat,
            source_index=idx, chunk_id=chunk_id,
        ))
        chunk_id += 1

        # Chunk 3: Combined Q+A
        combined = f"{q} {a}"
        chunks.append(DocumentChunk(
            text=combined,
            question=q, answer=a, category=cat,
            source_index=idx, chunk_id=chunk_id,
        ))
        chunk_id += 1

        # Chunk 4: Category-enriched
        cat_enriched = f"{cat}: {q} {a}"
        chunks.append(DocumentChunk(
            text=cat_enriched,
            question=q, answer=a, category=cat,
            source_index=idx, chunk_id=chunk_id,
        ))
        chunk_id += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════
# TF-IDF Retriever (Lexical)
# ═══════════════════════════════════════════════════════════════════════
class TFIDFRetriever:
    """Fast lexical retrieval using TF-IDF + cosine similarity."""

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.idf: dict[str, float] = {}
        self.vectors: list[dict[str, float]] = []
        self._build_index()

    def _build_index(self):
        n = len(self.chunks)
        df: Counter = Counter()
        for chunk in self.chunks:
            df.update(set(chunk.tokens))

        self.idf = {
            term: math.log((n + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

        for chunk in self.chunks:
            tf = Counter(chunk.tokens)
            doc_len = len(chunk.tokens) or 1
            vec = {
                term: (count / doc_len) * self.idf.get(term, 0)
                for term, count in tf.items()
            }
            self.vectors.append(vec)

    def _query_vector(self, query: str) -> dict[str, float]:
        tokens = _tokenize(query)
        tf = Counter(tokens)
        doc_len = len(tokens) or 1
        return {
            term: (count / doc_len) * self.idf.get(term, 1)
            for term, count in tf.items()
        }

    @staticmethod
    def _cosine_sim(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Return top-k chunks ranked by TF-IDF cosine similarity."""
        qvec = self._query_vector(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            score = self._cosine_sim(qvec, self.vectors[i])
            scored.append((chunk, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# Semantic Retriever (Embedding-based)
# ═══════════════════════════════════════════════════════════════════════
class SemanticRetriever:
    """
    Semantic retrieval using sentence-transformers.

    Uses 'all-MiniLM-L6-v2' (22 MB) — fast enough for Raspberry Pi,
    accurate enough for short factual queries.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.model = None
        self.embeddings = None
        self._load_model()
        self._build_index()

    def _cache_path(self) -> Path:
        """Path to cached embeddings file."""
        content_hash = hashlib.md5(
            "|".join(c.text for c in self.chunks).encode()
        ).hexdigest()[:12]
        return CACHE_DIR / f"embeddings_{content_hash}.pkl"

    def _load_model(self):
        if not SEMANTIC_AVAILABLE:
            return
        try:
            print(f"🧠 Loading semantic model '{self.MODEL_NAME}'...")
            self.model = SentenceTransformer(self.MODEL_NAME)
            print("🧠 Semantic retriever ready.")
        except Exception as e:
            print(f"⚠️  Failed to load semantic model: {e}")
            self.model = None

    def _build_index(self):
        if self.model is None:
            return

        cache_path = self._cache_path()

        # Try loading from cache
        if cache_path.exists():
            try:
                import joblib
                self.embeddings = joblib.load(cache_path)
                print(f"🧠 Loaded cached embeddings ({len(self.chunks)} chunks)")
                return
            except Exception:
                pass

        # Encode all chunks
        texts = [c.text for c in self.chunks]
        print(f"🧠 Encoding {len(texts)} chunks...")
        self.embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=32,
        )

        # Save to cache
        CACHE_DIR.mkdir(exist_ok=True)
        try:
            import joblib
            joblib.dump(self.embeddings, cache_path)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self.model is not None and self.embeddings is not None

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Return top-k chunks ranked by semantic similarity."""
        if not self.available:
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=False)
        similarities = st_util.cos_sim(query_embedding, self.embeddings)[0]

        scored = [
            (self.chunks[i], float(similarities[i]))
            for i in range(len(self.chunks))
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# Hybrid Re-Ranker
# ═══════════════════════════════════════════════════════════════════════
class HybridReRanker:
    """
    Fuses results from TF-IDF and semantic retrievers using
    Reciprocal Rank Fusion (RRF).
    """

    RRF_K = 60  # standard RRF constant

    @classmethod
    def fuse(
        cls,
        tfidf_results: list[tuple[DocumentChunk, float]],
        semantic_results: list[tuple[DocumentChunk, float]],
        tfidf_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Reciprocal Rank Fusion of two result lists.

        Args:
            tfidf_results:   List of (chunk, score) from TF-IDF.
            semantic_results: List of (chunk, score) from semantic search.
            tfidf_weight:    Weight for TF-IDF scores (default 0.4).
            semantic_weight: Weight for semantic scores (default 0.6).

        Returns:
            Fused, re-ranked list of (chunk, fused_score).
        """
        scores: dict[int, float] = {}  # chunk_id → fused score
        chunk_map: dict[int, DocumentChunk] = {}

        # RRF for TF-IDF results
        for rank, (chunk, score) in enumerate(tfidf_results):
            chunk_map[chunk.chunk_id] = chunk
            rrf = tfidf_weight / (cls.RRF_K + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + rrf

        # RRF for semantic results
        for rank, (chunk, score) in enumerate(semantic_results):
            chunk_map[chunk.chunk_id] = chunk
            rrf = semantic_weight / (cls.RRF_K + rank + 1)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + rrf

        # Sort by fused score
        fused = [
            (chunk_map[cid], score)
            for cid, score in sorted(scores.items(), key=lambda x: -x[1])
        ]
        return fused


# ═══════════════════════════════════════════════════════════════════════
# Answer Generator (Template-based, no LLM required)
# ═══════════════════════════════════════════════════════════════════════
class AnswerGenerator:
    """
    Generates natural, contextual answers from retrieved chunks.

    Uses template-based synthesis — no external LLM needed.
    Combines top relevant chunks into a coherent response.
    """

    # Confidence thresholds
    HIGH_CONFIDENCE = 0.015   # top result is very relevant
    MEDIUM_CONFIDENCE = 0.008 # top result is somewhat relevant
    LOW_CONFIDENCE = 0.004    # barely relevant

    @classmethod
    def generate(
        cls,
        query: str,
        ranked_results: list[tuple[DocumentChunk, float]],
        max_chunks: int = 3,
    ) -> tuple[str, float]:
        """
        Generate an answer from ranked retrieval results.

        Args:
            query:          User's question.
            ranked_results: Re-ranked (chunk, score) list.
            max_chunks:     Maximum number of chunks to include.

        Returns:
            (answer_text, confidence_score)
        """
        if not ranked_results:
            return cls._no_answer(), 0.0

        top_chunk, top_score = ranked_results[0]

        if top_score < cls.LOW_CONFIDENCE:
            return cls._no_answer(), top_score

        # Deduplicate by source KB entry
        seen_sources = set()
        unique_results = []
        for chunk, score in ranked_results:
            if chunk.source_index not in seen_sources:
                seen_sources.add(chunk.source_index)
                unique_results.append((chunk, score))
            if len(unique_results) >= max_chunks:
                break

        # Build response
        primary = unique_results[0]
        primary_chunk, primary_score = primary

        if primary_score >= cls.HIGH_CONFIDENCE:
            # High confidence — give the direct answer
            response = f"📌 {primary_chunk.answer}"

            # Add related info if second result is close
            if len(unique_results) > 1:
                second_chunk, second_score = unique_results[1]
                if second_score >= cls.MEDIUM_CONFIDENCE:
                    if second_chunk.answer != primary_chunk.answer:
                        response += f"\n\nℹ️  Related: {second_chunk.answer}"

        elif primary_score >= cls.MEDIUM_CONFIDENCE:
            # Medium confidence — provide answer with qualification
            response = f"📌 {primary_chunk.answer}"

            # Include context from additional results
            additions = []
            for chunk, score in unique_results[1:]:
                if score >= cls.LOW_CONFIDENCE and chunk.answer != primary_chunk.answer:
                    additions.append(chunk.answer)

            if additions:
                response += "\n\nYou might also find these helpful:"
                for add in additions[:2]:
                    response += f"\n  • {add}"

        else:
            # Low confidence — hedged answer
            response = (
                f"I'm not entirely sure, but here's what I found:\n\n"
                f"📌 {primary_chunk.answer}\n\n"
                f"Could you rephrase your question if this isn't what you were looking for?"
            )

        return response, primary_score

    @staticmethod
    def _no_answer() -> str:
        return (
            "I'm not sure about that. Could you rephrase your question? "
            "I can help with topics like admissions, placements, facilities, "
            "faculty, hostel, sports, and more at UIT Prayagraj."
        )


# ═══════════════════════════════════════════════════════════════════════
# RAG Pipeline (Main Interface)
# ═══════════════════════════════════════════════════════════════════════
class RAGPipeline:
    """
    Complete RAG pipeline combining document chunking, dual retrieval,
    re-ranking, and answer generation.

    Usage:
        from rag_pipeline import RAGPipeline

        rag = RAGPipeline(knowledge_base)
        answer, confidence = rag.query("What about placements?")
    """

    def __init__(self, knowledge_base: list[dict], use_semantic: bool = True):
        """
        Args:
            knowledge_base:  List of {question, answer, category} dicts.
            use_semantic:    Whether to use semantic embeddings (requires
                            sentence-transformers). Set False for TF-IDF only.
        """
        print("🔧 Building RAG pipeline...")

        # Step 1: Chunk the knowledge base
        self.chunks = chunk_knowledge_base(knowledge_base)
        print(f"   📄 {len(knowledge_base)} KB entries → {len(self.chunks)} chunks")

        # Step 2: Build TF-IDF retriever
        self.tfidf_retriever = TFIDFRetriever(self.chunks)
        print("   📊 TF-IDF retriever ready")

        # Step 3: Build semantic retriever (if available)
        self.semantic_retriever = None
        if use_semantic and SEMANTIC_AVAILABLE:
            self.semantic_retriever = SemanticRetriever(self.chunks)
            if not self.semantic_retriever.available:
                self.semantic_retriever = None

        # Step 4: Answer generator
        self.generator = AnswerGenerator()

        mode = "Hybrid (TF-IDF + Semantic)" if self.semantic_retriever else "TF-IDF only"
        print(f"   ✅ RAG pipeline ready — mode: {mode}")

    @property
    def has_semantic(self) -> bool:
        return self.semantic_retriever is not None and self.semantic_retriever.available

    def query(self, user_input: str, top_k: int = 10) -> tuple[str, float]:
        """
        Run the full RAG pipeline on a user query.

        Args:
            user_input: User's question/query string.
            top_k:      Number of candidates to retrieve.

        Returns:
            (answer_text, confidence_score)
        """
        if not user_input or not user_input.strip():
            return "Please ask a question about UIT Prayagraj.", 0.0

        # Retrieve from TF-IDF
        tfidf_results = self.tfidf_retriever.retrieve(user_input, top_k=top_k)

        if self.has_semantic:
            # Retrieve from semantic search
            semantic_results = self.semantic_retriever.retrieve(user_input, top_k=top_k)

            # Fuse with re-ranking
            fused = HybridReRanker.fuse(
                tfidf_results, semantic_results,
                tfidf_weight=0.4, semantic_weight=0.6,
            )
        else:
            fused = tfidf_results

        # Generate answer
        answer, confidence = self.generator.generate(user_input, fused)

        return answer, confidence

    def query_with_context(self, user_input: str, top_k: int = 5) -> dict:
        """
        Extended query that returns full context for debugging.

        Returns dict with: answer, confidence, top_chunks, retrieval_mode
        """
        tfidf_results = self.tfidf_retriever.retrieve(user_input, top_k=top_k)

        if self.has_semantic:
            semantic_results = self.semantic_retriever.retrieve(user_input, top_k=top_k)
            fused = HybridReRanker.fuse(tfidf_results, semantic_results)
            mode = "hybrid"
        else:
            fused = tfidf_results
            semantic_results = []
            mode = "tfidf_only"

        answer, confidence = self.generator.generate(user_input, fused)

        return {
            "answer": answer,
            "confidence": confidence,
            "retrieval_mode": mode,
            "top_chunks": [
                {
                    "question": c.question,
                    "answer": c.answer,
                    "category": c.category,
                    "score": round(s, 4),
                }
                for c, s in fused[:5]
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    # Load knowledge base
    kb_path = BASE_DIR / "knowledge_base.json"
    if not kb_path.exists():
        print("❌ knowledge_base.json not found. Run parse_txt.py first.")
        sys.exit(1)

    kb = json.loads(kb_path.read_text(encoding="utf-8"))

    print("=" * 60)
    print("  RAG Pipeline — Interactive Test")
    print("=" * 60)

    rag = RAGPipeline(kb)

    print(f"\nLoaded {len(kb)} KB entries.\n")
    print("Type a question (or 'quit' to exit):\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if q.lower() in ("quit", "exit", "q"):
            break

        if q.lower() == "debug":
            q = input("  Query: ").strip()
            result = rag.query_with_context(q)
            print(f"\n  Answer: {result['answer']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Mode: {result['retrieval_mode']}")
            print(f"  Top chunks:")
            for i, chunk in enumerate(result["top_chunks"]):
                print(f"    [{i+1}] score={chunk['score']:.4f} | {chunk['category']}")
                print(f"        Q: {chunk['question']}")
                print(f"        A: {chunk['answer'][:80]}...")
            print()
            continue

        answer, conf = rag.query(q)
        print(f"\nBot: {answer}")
        print(f"  (confidence: {conf:.4f})\n")
