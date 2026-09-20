"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

BM25 is reimplemented here rather than shared with any other system. That
duplication is deliberate -- see r-d/README.md. If three systems independently
grow the same lexical index, that is evidence the library needs one; if they
don't, sharing it now would have been a guess.

RRF (Cormack et al., 2009) fuses by rank rather than score, so the two legs need
no calibration against each other:  score(d) = sum_i 1 / (k + rank_i(d))
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from harness.task import Query, Response, Task

__all__ = ["HybridRRF"]

_TOKEN = re.compile(r"\w+")


def _tok(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class _BM25:
    """Okapi BM25 with Elasticsearch's defaults (k1=1.2, b=0.75)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lens: list[int] = []
        self.avg_len = 0.0
        self.idf: dict[str, float] = {}

    def index(self, ids: list[str], texts: list[str]) -> None:
        self.ids = ids
        df: Counter[str] = Counter()
        for i, text in enumerate(texts):
            tf = Counter(_tok(text))
            self.lens.append(sum(tf.values()))
            for term, n in tf.items():
                self.postings[term].append((i, n))
            df.update(tf.keys())
        n_docs = len(ids)
        self.avg_len = (sum(self.lens) / n_docs) if n_docs else 0.0
        self.idf = {
            t: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in _tok(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = 1 - self.b + self.b * (self.lens[i] / self.avg_len or 1.0)
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.ids[i], s) for i, s in top]


class HybridRRF:
    name = "hybrid_rrf"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        k: int = 10,
        depth: int = 50,
        rrf_k: int = 60,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.k, self.depth, self.rrf_k = k, depth, rrf_k
        self.device = device
        self._model = None
        self._bm25 = _BM25()
        self._ids: list[str] = []
        self._emb: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def index(self, task: Task) -> None:
        self._ids = [d.doc_id for d in task.docs]
        texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
        self._bm25.index(self._ids, texts)
        self._emb = self._load().encode(
            texts, batch_size=128, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def answer(self, query: Query) -> Response:
        if self._emb is None:
            raise RuntimeError("index() first")
        sparse = [d for d, _ in self._bm25.search(query.text, self.depth)]

        q = self._load().encode(
            [query.text], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = self._emb @ q
        dense = [self._ids[i] for i in np.argsort(-sims)[: self.depth]]

        fused: dict[str, float] = defaultdict(float)
        for leg in (sparse, dense):
            for rank, doc_id in enumerate(leg, start=1):
                fused[doc_id] += 1.0 / (self.rrf_k + rank)

        ranked = [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])[: self.k]]
        return Response(
            ranked=ranked,
            confidence=float(fused[ranked[0]]) if ranked else 0.0,
            usage={"legs": 2},
        )
