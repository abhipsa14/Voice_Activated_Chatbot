import os
import hashlib
import json
import joblib
from pathlib import Path
from cachetools import TTLCache
import string

BASE_DIR = Path(__file__).resolve().parent
TTS_CACHE_DIR = BASE_DIR / ".tts_cache"
ML_CACHE_DIR = BASE_DIR / ".cache"
KB_FILE = BASE_DIR / "knowledge_base.json"

class CacheManager:
    # Use class variables to persist state across instances in the same process
    # if instantiated multiple times, but ideally we instantiate once.
    _query_cache = TTLCache(maxsize=300, ttl=86400)
    
    def __init__(self):
        # A) QUERY RESPONSE CACHE
        self.stats_file = ML_CACHE_DIR / "stats.json"
        self.query_hits: int = 0
        self.query_misses: int = 0
        self._load_stats()

        # Create cache directories
        TTS_CACHE_DIR.mkdir(exist_ok=True)
        ML_CACHE_DIR.mkdir(exist_ok=True)

        # B) TTS AUDIO CACHE mapping (in-memory dict)
        self.tts_map = {}
        # Load existing tts files into mapping
        self._load_tts_map()

    # --- A) QUERY RESPONSE CACHE ---
    def _load_stats(self):
        if self.stats_file.exists():
            try:
                data = json.loads(self.stats_file.read_text())
                self.query_hits = data.get("hits", 0)
                self.query_misses = data.get("misses", 0)
            except:
                self.query_hits = 0
                self.query_misses = 0
        else:
            self.query_hits = 0
            self.query_misses = 0

    def _save_stats(self):
        if hasattr(self, 'stats_file'):
            data = {
                "hits": self.query_hits,
                "misses": self.query_misses,
                "size": len(self._query_cache)
            }
            try:
                self.stats_file.write_text(json.dumps(data))
            except:
                pass

    def _normalize_query(self, query: str) -> str:
        """Lowercase, strip punctuation, strip whitespace."""
        query = query.lower()
        query = query.translate(str.maketrans('', '', string.punctuation))
        return " ".join(query.split())

    def get_response(self, query: str) -> str | None:
        norm_query = self._normalize_query(query)
        if norm_query in self._query_cache:
            self.query_hits += 1
            self._save_stats()
            return self._query_cache[norm_query]
        self.query_misses += 1
        self._save_stats()
        return None

    def set_response(self, query: str, answer: str):
        norm_query = self._normalize_query(query)
        self._query_cache[norm_query] = answer
        self._save_stats()

    def get_stats(self) -> dict:
        self._load_stats()
        total = self.query_hits + self.query_misses
        hit_rate = (self.query_hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.query_hits,
            "misses": self.query_misses,
            "hit_rate": hit_rate,
            "size": len(self._query_cache)
        }

    # --- B) TTS AUDIO CACHE ---
    def _get_md5(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_tts_map(self):
        # We could load all existing mp3 files in the folder if we wanted,
        # but the mapping needs the original text. We can only hash the text.
        # So we just keep it empty on start, and populate it as we go.
        pass

    def get_audio_path(self, text: str) -> str | None:
        md5 = self._get_md5(text)
        path = TTS_CACHE_DIR / f"{md5}.mp3"
        if md5 in self.tts_map and path.exists():
            return str(path)
        if path.exists():
            self.tts_map[md5] = str(path)
            return str(path)
        return None

    def set_audio_path(self, text: str, path: str):
        md5 = self._get_md5(text)
        self.tts_map[md5] = path

    def pregenerate(self, text_list: list):
        """Batch pre-generate audio for a list of responses."""
        import asyncio
        from tts_module import _get_engine, _clean_text
        
        engine = _get_engine()
        if not engine.use_edge:
            # We only pregenerate for Edge TTS
            return

        async def _batch_generate():
            import edge_tts
            for text in text_list:
                clean = _clean_text(text)
                if not clean:
                    continue
                md5 = self._get_md5(text)
                out_path = TTS_CACHE_DIR / f"{md5}.mp3"
                if not out_path.exists():
                    communicate = edge_tts.Communicate(
                        clean,
                        voice=engine.voice,
                        rate=engine.rate,
                        volume=engine.volume,
                        pitch=engine.pitch,
                    )
                    await communicate.save(str(out_path))
                self.set_audio_path(text, str(out_path))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(_batch_generate())

    # --- C) EMBEDDING / TF-IDF CACHE ---
    def load_tfidf_cache(self):
        vec_path = ML_CACHE_DIR / "tfidf_vectorizer.pkl"
        mat_path = ML_CACHE_DIR / "tfidf_matrix.pkl"
        mtime_path = ML_CACHE_DIR / "kb_mtime.txt"

        if not (vec_path.exists() and mat_path.exists() and mtime_path.exists() and KB_FILE.exists()):
            return None

        # Check mtime
        current_mtime = str(os.path.getmtime(KB_FILE))
        saved_mtime = mtime_path.read_text().strip()

        if current_mtime != saved_mtime:
            # Invalidated
            return None

        vectorizer = joblib.load(vec_path)
        matrix = joblib.load(mat_path)
        return vectorizer, matrix

    def save_tfidf_cache(self, vectorizer, matrix):
        vec_path = ML_CACHE_DIR / "tfidf_vectorizer.pkl"
        mat_path = ML_CACHE_DIR / "tfidf_matrix.pkl"
        mtime_path = ML_CACHE_DIR / "kb_mtime.txt"

        joblib.dump(vectorizer, vec_path)
        joblib.dump(matrix, mat_path)

        if KB_FILE.exists():
            current_mtime = str(os.path.getmtime(KB_FILE))
            mtime_path.write_text(current_mtime)
