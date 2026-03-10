"""
Multi-Level Cache Manager
============================
Three-tier caching for sub-second response latency on Raspberry Pi:

  L1 – In-memory TTLCache (0 ms lookup, lost on restart)
  L2 – Disk JSON cache   (1-5 ms lookup, persists across restarts)
  L3 – Redis             (1-2 ms lookup, shared across processes)

Also handles TTS audio caching:  text → pre-generated WAV path.
"""

import hashlib
import json
import os
import time
from pathlib import Path

try:
    from cachetools import TTLCache
    CACHETOOLS_OK = True
except ImportError:
    CACHETOOLS_OK = False

try:
    import redis as _redis_lib
    REDIS_LIB_OK = True
except ImportError:
    REDIS_LIB_OK = False

from config.config import (
    CACHE_DIR, TTS_CACHE_DIR,
    MEMORY_CACHE_SIZE, MEMORY_CACHE_TTL,
    REDIS_ENABLED, REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL,
    CACHE_ENABLED,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class CacheManager:
    """
    Multi-level cache for query responses **and** TTS audio files.

    Usage:
        cache = CacheManager()
        cache.get("who is the principal")   # check all levels
        cache.put("who is the principal", "Prof. Sanjay Srivastava...")
    """

    # ── Class-level L1 so it survives multiple instantiations ─────────
    _l1: "TTLCache | dict" = (
        TTLCache(maxsize=MEMORY_CACHE_SIZE, ttl=MEMORY_CACHE_TTL)
        if CACHETOOLS_OK else {}
    )

    def __init__(self):
        CACHE_DIR.mkdir(exist_ok=True)
        TTS_CACHE_DIR.mkdir(exist_ok=True)

        # L2 – disk
        self._disk_path = CACHE_DIR / "response_cache.json"
        self._disk: dict[str, dict] = self._load_disk()

        # L3 – Redis
        self._redis = None
        if CACHE_ENABLED and REDIS_ENABLED and REDIS_LIB_OK:
            try:
                self._redis = _redis_lib.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                    decode_responses=True, socket_connect_timeout=2,
                )
                self._redis.ping()
            except Exception:
                self._redis = None

        # Stats
        self._hits = 0
        self._misses = 0

    # ── Public: text response cache ────────────────────────────────────

    def get(self, query: str) -> str | None:
        """Check L1 → L2 → L3.  Returns cached answer or None."""
        if not CACHE_ENABLED:
            return None
        key = _sha256(query)

        # L1
        val = self._l1.get(key)
        if val:
            self._hits += 1
            return val

        # L2
        entry = self._disk.get(key)
        if entry and time.time() - entry.get("ts", 0) < MEMORY_CACHE_TTL:
            self._l1[key] = entry["v"]
            self._hits += 1
            return entry["v"]

        # L3
        if self._redis:
            try:
                val = self._redis.get(f"q:{key}")
                if val:
                    self._l1[key] = val
                    self._hits += 1
                    return val
            except Exception:
                pass

        self._misses += 1
        return None

    def put(self, query: str, answer: str) -> None:
        """Write to all three levels."""
        if not CACHE_ENABLED:
            return
        key = _sha256(query)

        # L1
        self._l1[key] = answer

        # L2
        self._disk[key] = {"v": answer, "ts": time.time()}
        self._flush_disk()

        # L3
        if self._redis:
            try:
                self._redis.setex(f"q:{key}", REDIS_TTL, answer)
            except Exception:
                pass

    # ── TTS audio cache ────────────────────────────────────────────────

    def get_audio(self, text: str) -> str | None:
        """Return cached WAV/MP3 path for this text, or None."""
        md5 = _md5(text)
        for ext in (".wav", ".mp3"):
            p = TTS_CACHE_DIR / f"{md5}{ext}"
            if p.exists():
                return str(p)
        return None

    def audio_cache_path(self, text: str, ext: str = ".wav") -> str:
        """Return the path where TTS audio *should* be saved."""
        return str(TTS_CACHE_DIR / f"{_md5(text)}{ext}")

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total * 100:.1f}%" if total else "n/a",
            "l1_size": len(self._l1),
            "l2_size": len(self._disk),
            "redis": self._redis is not None,
        }

    # ── Disk helpers ───────────────────────────────────────────────────

    def _load_disk(self) -> dict:
        if self._disk_path.exists():
            try:
                return json.loads(self._disk_path.read_text("utf-8"))
            except Exception:
                pass
        return {}

    def _flush_disk(self) -> None:
        try:
            self._disk_path.write_text(
                json.dumps(self._disk, ensure_ascii=False), "utf-8"
            )
        except Exception:
            pass

    # ── Batch warm-up ──────────────────────────────────────────────────

    def warm(self, pairs: list[tuple[str, str]]) -> None:
        """Pre-populate cache with (query, answer) pairs."""
        for q, a in pairs:
            self.put(q, a)
