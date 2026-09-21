"""Dense retrieval with a query transform in front of it.

The retrieval leg is identical to `naive` and duplicated rather than imported,
so the transform is the only thing varying between runs.

A transform returning several queries means retrieve for each and fuse the runs
with reciprocal rank fusion. A transform declaring `uses_feedback` gets a first
retrieval pass to read before it rewrites.
"""

from __future__ import annotations

import numpy as np
import ragworks

from harness.task import Query, Response, Task

__all__ = ["QueryRewrite"]


class QueryRewrite:
    name = "query_rewrite"

    def __init__(
        self,
        *,
        transform: str = "identity",
        config: dict | None = None,
        k: int = 10,
        feedback_docs: int = 3,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.transform_name = transform
        self.config = config
        self.k = k
        self.feedback_docs = feedback_docs
        self.model_name = model_name
        self._model = None
        self._tf: ragworks.QueryTransform | None = None
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
        self._tf = ragworks.QueryTransform(self.transform_name, self.config)

    def _search(self, text: str, k: int) -> list[tuple[int, float]]:
        q = self._load().encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = self._emb @ q
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]

    def answer(self, query: Query) -> Response:
        if self._emb is None or self._tf is None:
            raise RuntimeError("index() first")

        feedback: list[str] = []
        if self._tf.uses_feedback:
            first = self._search(query.text, self.feedback_docs)
            feedback = [self._texts[i] for i, _ in first]

        queries = self._tf.transform(query.text, feedback) or [query.text]

        if len(queries) == 1:
            hits = self._search(queries[0], self.k)
            return Response(
                ranked=[self._ids[i] for i, _ in hits],
                confidence=hits[0][1] if hits else 0.0,
            )

        # Several rewrites: retrieve for each and fuse by rank, so legs with
        # incomparable score scales need no calibration.
        runs = [
            [(i, s) for i, s in self._search(q, self.k * 2)] for q in queries
        ]
        fused = ragworks.rrf(runs, top=self.k)
        return Response(
            ranked=[self._ids[i] for i, _ in fused],
            confidence=float(fused[0][1]) if fused else 0.0,
        )
