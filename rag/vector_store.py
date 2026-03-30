"""
VectorDesk Vector Store: FAISS-backed with Chroma fallback.
Stores office documents, emails, policies, and conversation history.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingModel


# ── Built-in Knowledge Base ────────────────────────────────────────────────

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    # Email policies
    {"id": "email_pol_1", "content": "Emails marked URGENT or containing 'production down', 'outage', 'security breach' must be escalated within 15 minutes.", "source": "email_policy", "type": "email"},
    {"id": "email_pol_2", "content": "Billing and invoice emails over $10,000 require CFO approval before payment. Always CC finance@company.com.", "source": "email_policy", "type": "email"},
    {"id": "email_pol_3", "content": "Security vulnerability reports must be forwarded to security@company.com immediately and assigned HIGH priority.", "source": "email_policy", "type": "email"},
    {"id": "email_pol_4", "content": "Social and event emails can be handled within 48 hours. Low priority unless they require RSVP.", "source": "email_policy", "type": "email"},
    # Support policies
    {"id": "sup_pol_1", "content": "Premium tier customers are entitled to full refunds within 30 days, no questions asked. Issue refund via the billing portal.", "source": "premium_refund_policy", "type": "support"},
    {"id": "sup_pol_2", "content": "Enterprise SLA guarantees 99.9% uptime with P0 response in under 1 hour. Always escalate P0s to engineering immediately.", "source": "enterprise_sla_policy", "type": "support"},
    {"id": "sup_pol_3", "content": "For account access issues, verify identity first. Then trigger password reset from admin console. Document in ticket.", "source": "account_access_policy", "type": "support"},
    {"id": "sup_pol_4", "content": "General support tickets should be resolved within 24 hours. Link relevant documentation and ask if issue is resolved.", "source": "general_support_policy", "type": "support"},
    # Calendar policies
    {"id": "cal_pol_1", "content": "Executive meetings require 24-hour advance notice. Prefer morning slots (9am-12pm) for strategy sessions.", "source": "calendar_policy", "type": "calendar"},
    {"id": "cal_pol_2", "content": "Daily standups should be no longer than 30 minutes. Use recurring calendar blocks. Avoid scheduling after 10am.", "source": "calendar_policy", "type": "calendar"},
    {"id": "cal_pol_3", "content": "Client meetings that are marked urgent should preempt internal meetings. Always confirm attendance with external attendees.", "source": "calendar_policy", "type": "calendar"},
    # Historical context
    {"id": "hist_1", "content": "Previous production incident on Feb 1 was resolved by restarting the Redis cache cluster. Root cause: memory leak in v2.3.1.", "source": "incident_log", "type": "email"},
    {"id": "hist_2", "content": "Client BigCorp has escalated twice before for API downtime. They are a Tier-1 enterprise client worth $500K ARR.", "source": "crm_notes", "type": "support"},
    {"id": "hist_3", "content": "CSV export feature is on the roadmap for Q2 2024. Clients can use the API endpoint /export?format=csv in the meantime.", "source": "product_docs", "type": "support"},
]


class VectorStore:
    """
    In-memory vector store backed by FAISS when available,
    with a pure-Python cosine-similarity fallback.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self._embedder = embedding_model or EmbeddingModel()
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []
        self._faiss_index = None
        self._use_faiss = self._try_import_faiss()
        # Seed with built-in knowledge base
        self.add_documents(KNOWLEDGE_BASE)

    def _try_import_faiss(self) -> bool:
        try:
            import faiss  # noqa
            return True
        except ImportError:
            return False

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents and their embeddings to the store."""
        texts = [d["content"] for d in docs]
        embeddings = self._embedder.embed(texts)
        self._documents.extend(docs)
        self._embeddings.extend(embeddings)
        if self._use_faiss:
            self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        import faiss
        import numpy as np
        dim = len(self._embeddings[0])
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        vecs = np.array(self._embeddings, dtype="float32")
        # Normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-8)
        index.add(vecs)
        self._faiss_index = index

    def search(self, query_embedding: List[float], top_k: int = 3, filter_type: Optional[str] = None) -> List[Tuple[Dict, float]]:
        """Return top-k most similar documents with scores."""
        filtered_docs = self._documents
        filtered_embs = self._embeddings
        if filter_type:
            pairs = [(d, e) for d, e in zip(self._documents, self._embeddings) if d.get("type") == filter_type]
            if pairs:
                filtered_docs, filtered_embs = zip(*pairs)
                filtered_docs, filtered_embs = list(filtered_docs), list(filtered_embs)

        if self._use_faiss and not filter_type:
            return self._faiss_search(query_embedding, top_k)
        return self._cosine_search(query_embedding, filtered_embs, filtered_docs, top_k)

    def _cosine_search(self, query_emb, embeddings, docs, top_k) -> List[Tuple[Dict, float]]:
        import math
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x ** 2 for x in a)) or 1e-8
            nb = math.sqrt(sum(x ** 2 for x in b)) or 1e-8
            return dot / (na * nb)

        scores = [(doc, cosine(query_emb, emb)) for doc, emb in zip(docs, embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _faiss_search(self, query_emb, top_k) -> List[Tuple[Dict, float]]:
        import faiss
        import numpy as np
        vec = np.array([query_emb], dtype="float32")
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        distances, indices = self._faiss_index.search(vec, min(top_k, len(self._documents)))
        return [(self._documents[i], float(distances[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
