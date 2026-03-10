import os
import datetime
from pathlib import Path

from cache_manager import CacheManager, TTS_CACHE_DIR, ML_CACHE_DIR, KB_FILE

def main():
    manager = CacheManager()
    
    # 1. Query Cache
    stats = manager.get_stats()
    print(f"Query cache: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']:.1f}% hit rate, {stats['size']} entries")
    
    # 2. TTS Cache
    if TTS_CACHE_DIR.exists():
        files = list(TTS_CACHE_DIR.glob("*.mp3"))
        size_bytes = sum(f.stat().st_size for f in files)
        print(f"TTS cache: {len(files)} audio files in .tts_cache/, {size_bytes / (1024*1024):.2f} MB")
    else:
        print("TTS cache: 0 audio files in .tts_cache/, 0.00 MB")
        
    # 3. TF-IDF Cache
    tfidf_vec_path = ML_CACHE_DIR / "tfidf_vectorizer.pkl"
    tfidf_mat_path = ML_CACHE_DIR / "tfidf_matrix.pkl"
    if tfidf_vec_path.exists() and tfidf_mat_path.exists():
        print("TF-IDF cache: loaded from disk")
    else:
        print("TF-IDF cache: not found")
        
    # 4. KB Update
    if KB_FILE.exists():
        mtime = os.path.getmtime(KB_FILE)
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"Last knowledge base update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Last knowledge base update: Not found")

if __name__ == '__main__':
    main()
