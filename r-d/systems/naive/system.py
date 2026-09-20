"""Naive dense RAG: embed everything, cosine top-k. The control.

Unglamorous and mandatory. CANDIDATES.md §2.1 found that of 18 RAG papers
shipping artifacts, none fully reproduced, and that only *relative* ordering
survived replication. That makes this file the measuring stick: every later
number means "better than this, by this much, on this data".
"""

from __future__ import annotations

import numpy as np

from harness.task import Query, Response, Task

__all__ = ["NaiveDense"]


class NaiveDense:
    name = "naive"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        k: int = 10,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.k = k
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._ids: list[str] = []
        self._emb: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def index(self, task: Task) -> None:
        model = self._load()
        self._ids = [d.doc_id for d in task.docs]
        texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
        self._emb = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def answer(self, query: Query) -> Response:
        if self._emb is None:
            raise RuntimeError("index() first")
        q = self._load().encode(
            [query.text], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = self._emb @ q
        top = np.argsort(-sims)[: self.k]
        return Response(
            ranked=[self._ids[i] for i in top],
            confidence=float(sims[top[0]]) if len(top) else 0.0,
        )
