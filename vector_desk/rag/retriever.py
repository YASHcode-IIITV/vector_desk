"""
VectorDesk Retriever: query → RetrievedDocument list with scores.
"""
from __future__ import annotations
from typing import List, Optional

from .vector_store import VectorStore
from .embeddings import EmbeddingModel
from environment.state import RetrievedDocument


class Retriever:
    """Wraps VectorStore and converts results to typed RetrievedDocument objects."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self._store = vector_store or VectorStore()

    def retrieve(self, query: str, task_type: Optional[str] = None, top_k: int = 3) -> List[RetrievedDocument]:
        """Embed query, search vector store, return typed results."""
        query_emb = self._store._embedder.embed_query(query)
        results = self._store.search(query_emb, top_k=top_k, filter_type=task_type)
        return [
            RetrievedDocument(
                content=doc["content"],
                source=doc.get("source", "unknown"),
                relevance_score=min(1.0, max(0.0, float(score))),
                metadata={"id": doc.get("id", ""), "type": doc.get("type", "")},
            )
            for doc, score in results
        ]

    def add_to_memory(self, content: str, source: str, task_type: str) -> None:
        """Add a new document to the vector store (agent memory)."""
        self._store.add_documents([{"content": content, "source": source, "type": task_type, "id": f"mem_{hash(content)%10000}"}])
