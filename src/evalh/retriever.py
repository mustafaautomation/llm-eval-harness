"""A small BM25 retriever over the knowledge base (no external dependencies)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from evalh.dataset import Document
from evalh.text import content_tokens


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Bm25Retriever:
    def __init__(self, documents: dict[str, Document], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("retriever needs at least one document")
        self._docs = documents
        self._k1 = k1
        self._b = b
        self._tf = {doc_id: Counter(content_tokens(d.text)) for doc_id, d in documents.items()}
        self._len = {doc_id: sum(tf.values()) for doc_id, tf in self._tf.items()}
        self._avg_len = sum(self._len.values()) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(documents)
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = content_tokens(query)
        scores: list[Hit] = []
        for doc_id, tf in self._tf.items():
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = (
                    f
                    * (self._k1 + 1)
                    / (f + self._k1 * (1 - self._b + self._b * self._len[doc_id] / self._avg_len))
                )
                score += self._idf.get(term, 0.0) * norm
            if score > 0:
                scores.append(Hit(doc_id, score))
        scores.sort(key=lambda h: (-h.score, h.doc_id))
        return scores[:k]

    def context(self, hits: list[Hit]) -> str:
        return "\n\n".join(f"[{h.doc_id}]\n{self._docs[h.doc_id].text}" for h in hits)
