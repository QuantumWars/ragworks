"""Dense retrieval with a configurable rerank stage.

The retrieval leg is deliberately identical to `naive` -- same model, same
top-k -- and duplicated here rather than imported, so the only thing varying
between runs is the reranker. That is what makes the comparison attributable.
"""

from __future__ import annotations

import numpy as np
import ragworks

from harness.task import Query, Response, Task

__all__ = ["Reranked"]


class Reranked:
    name = "reranked"

    def __init__(
        self,
        *,
        reranker: str | None = None,
        config: dict | None = None,
        depth: int = 10,
        k: int = 10,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.reranker_name = reranker
        self.config = config
        self.depth = depth
        self.k = k
        self.model_name = model_name
        self._model = None
        self._rr: ragworks.Reranker | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._emb: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, task: Task) -> None:
        self._ids = [d.doc_id for d in task.docs]
        self._texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
        self._emb = self._load().encode(
            self._texts, batch_size=128, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        self._rr = (
            ragworks.Reranker(self.reranker_name, self.config) if self.reranker_name else None
        )

    def answer(self, query: Query) -> Response:
        if self._emb is None:
            raise RuntimeError("index() first")
        q = self._load().encode(
            [query.text], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        top = np.argsort(-(self._emb @ q))[: self.depth]

        if self._rr is None:
            ranked = [self._ids[i] for i in top[: self.k]]
            return Response(ranked=ranked)

        scores = self._rr.rerank(query.text, [self._texts[i] for i in top])
        order = sorted(range(len(top)), key=lambda j: -scores[j])
        return Response(
            ranked=[self._ids[top[j]] for j in order[: self.k]],
            confidence=float(scores[order[0]]) if order else 0.0,
        )
