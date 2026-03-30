"""
Embedding layer: uses sentence-transformers (offline) or OpenAI (if key present).
Falls back to a simple TF-IDF-style bag-of-words for zero-dependency demo mode.
"""
from __future__ import annotations
import os
import math
from collections import Counter
from typing import List


class EmbeddingModel:
    """
    Lightweight embedding model with three backends:
      1. sentence-transformers (preferred for quality)
      2. OpenAI text-embedding-3-small (if OPENAI_API_KEY set)
      3. TF-IDF fallback (always works, no network required)
    """

    def __init__(self, backend: str = "auto"):
        self.backend = self._resolve_backend(backend)
        self._model = None
        self._vocab: List[str] = []
        self._initialize()

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        try:
            import sentence_transformers  # noqa
            return "sentence_transformers"
        except ImportError:
            return "tfidf"

    def _initialize(self):
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        elif self.backend == "openai":
            import openai  # noqa – just verify importable
        # tfidf needs no init

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "sentence_transformers":
            return self._model.encode(texts, convert_to_numpy=True).tolist()
        elif self.backend == "openai":
            return self._openai_embed(texts)
        else:
            return self._tfidf_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    # ── OpenAI ───────────────────────────────────────────────────────────────
    def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in resp.data]

    # ── TF-IDF fallback ──────────────────────────────────────────────────────
    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        """Deterministic bag-of-words vectors for demo/test environments."""
        if not self._vocab:
            all_words = set()
            for t in texts:
                all_words.update(t.lower().split())
            self._vocab = sorted(all_words)

        vectors = []
        for text in texts:
            counts = Counter(text.lower().split())
            vec = [counts.get(w, 0) for w in self._vocab]
            norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors
