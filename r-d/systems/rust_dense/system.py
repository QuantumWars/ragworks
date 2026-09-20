"""Dense retrieval backed entirely by the Rust core.

Embeds with `ragworks.Embedder` and searches with `ragworks.Flat`, so the whole
path from text to ranked ids runs in Rust. Used here to measure whether
character n-grams actually improve the offline hashing embedder on a real
retrieval task, rather than only on hand-picked word pairs.
"""

from __future__ import annotations

import ragworks

from harness.task import Query, Response, Task

__all__ = ["RustDense"]


class RustDense:
    name = "rust_dense"

    def __init__(self, *, embedder: str = "hashing", config: dict | None = None, k: int = 10) -> None:
        self.embedder_name = embedder
        self.config = config or {}
        self.k = k
        self._emb: ragworks.Embedder | None = None
        self._store: ragworks.Flat | None = None
        self._ids: list[str] = []

    def index(self, task: Task) -> None:
        self._emb = ragworks.Embedder(self.embedder_name, self.config)
        self._ids = [d.doc_id for d in task.docs]
        texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
        vectors = self._emb.embed(texts)
        self._store = ragworks.Flat(dim=self._emb.dim)
        self._store.add(list(range(len(texts))), vectors)

    def answer(self, query: Query) -> Response:
        if self._store is None or self._emb is None:
            raise RuntimeError("index() first")
        hits = self._store.search(self._emb.embed([query.text]), self.k)
        return Response(
            ranked=[self._ids[i] for i, _ in hits],
            confidence=float(hits[0][1]) if hits else 0.0,
        )
