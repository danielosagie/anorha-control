"""
Fast embedding storage using SQLite + NumPy.
No external vector DB needed - simple and fast for <100K embeddings.
"""
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import time


@dataclass
class EmbeddingEntry:
    """An embedding with metadata."""
    id: int
    embedding: np.ndarray
    metadata: Dict[str, Any]
    created_at: float


class EmbeddingStore:
    """
    Fast embedding storage using SQLite + NumPy.
    
    Uses cosine similarity for search.
    Optimized for speed with in-memory caching.
    """
    
    def __init__(self, db_path: Path, embedding_dim: int = 256, cache_size: int = 10000):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.cache_size = cache_size
        
        # In-memory cache for fast search
        self._embeddings_cache: Optional[np.ndarray] = None
        self._ids_cache: Optional[List[int]] = None
        self._cache_dirty = True
        
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> int:
        """Add an embedding to the store."""
        if embedding.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {embedding.shape[-1]}")
        
        # Normalize for cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO embeddings (embedding, metadata) VALUES (?, ?)",
            (embedding.tobytes(), json.dumps(metadata or {}))
        )
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self._cache_dirty = True
        return entry_id
    
    def add_batch(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]] = None) -> List[int]:
        """Add multiple embeddings efficiently."""
        if metadata_list is None:
            metadata_list = [{}] * len(embeddings)
        
        # Normalize all
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        ids = []
        for emb, meta in zip(embeddings, metadata_list):
            cursor.execute(
                "INSERT INTO embeddings (embedding, metadata) VALUES (?, ?)",
                (emb.tobytes(), json.dumps(meta))
            )
            ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        
        self._cache_dirty = True
        return ids
    
    def _load_cache(self):
        """Load embeddings into memory for fast search."""
        if not self._cache_dirty:
            return
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Load most recent embeddings up to cache size
        cursor.execute(
            "SELECT id, embedding FROM embeddings ORDER BY id DESC LIMIT ?",
            (self.cache_size,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            self._ids_cache = [row[0] for row in rows]
            self._embeddings_cache = np.array([
                np.frombuffer(row[1], dtype=np.float32)
                for row in rows
            ])
        else:
            self._ids_cache = []
            self._embeddings_cache = np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        self._cache_dirty = False
    
    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query: Query embedding
            k: Number of results
            
        Returns:
            List of (id, similarity, metadata) tuples
        """
        self._load_cache()
        
        if len(self._ids_cache) == 0:
            return []
        
        # Normalize query
        query = query.astype(np.float32).flatten()
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Compute cosine similarities (embeddings already normalized)
        similarities = self._embeddings_cache @ query
        
        # Get top-k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Fetch metadata for top results
        results = []
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        for idx in top_indices:
            entry_id = self._ids_cache[idx]
            sim = float(similarities[idx])
            
            cursor.execute("SELECT metadata FROM embeddings WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            metadata = json.loads(row[0]) if row else {}
            
            results.append((entry_id, sim, metadata))
        
        conn.close()
        return results
    
    def get(self, entry_id: int) -> Optional[EmbeddingEntry]:
        """Get a specific embedding by ID."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, embedding, metadata, created_at FROM embeddings WHERE id = ?",
            (entry_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return EmbeddingEntry(
                id=row[0],
                embedding=np.frombuffer(row[1], dtype=np.float32),
                metadata=json.loads(row[2]),
                created_at=row[3],
            )
        return None
    
    def count(self) -> int:
        """Get total number of embeddings."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM embeddings")
        row = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_embeddings": total,
            "embedding_dim": self.embedding_dim,
            "cache_size": self.cache_size,
            "cache_loaded": not self._cache_dirty,
            "oldest": row[0] if row[0] else None,
            "newest": row[1] if row[1] else None,
        }


# Benchmark
if __name__ == "__main__":
    import tempfile
    
    print("Benchmarking EmbeddingStore...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddingStore(Path(tmpdir) / "test.db", embedding_dim=256)
        
        # Add embeddings
        n = 1000
        embeddings = np.random.randn(n, 256).astype(np.float32)
        
        start = time.time()
        ids = store.add_batch(embeddings)
        add_time = time.time() - start
        print(f"Added {n} embeddings in {add_time:.3f}s ({n/add_time:.0f}/s)")
        
        # Search
        query = np.random.randn(256).astype(np.float32)
        
        # First search loads cache
        results = store.search(query, k=5)
        
        # Benchmark search
        start = time.time()
        for _ in range(100):
            results = store.search(query, k=5)
        search_time = (time.time() - start) / 100
        print(f"Search time: {search_time*1000:.2f}ms")
        
        print(f"Stats: {store.stats()}")
        print(f"Top 3 results: {[(r[0], f'{r[1]:.3f}') for r in results[:3]]}")
