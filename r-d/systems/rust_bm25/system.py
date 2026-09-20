"""BM25 retrieval backed by the Rust core.

Deliberately the same algorithm as the BM25 leg inside `hybrid_rrf`, so the two
can be compared for agreement as well as speed. If they rank differently, one of
them is wrong -- and the Python one has been exercised on this task already,
which makes it the reference.
"""

from __future__ import annotations

import ragworks

from harness.task import Query, Response, Task

__all__ = ["RustBm25"]


class RustBm25:
    name = "rust_bm25"

    def __init__(self, *, k: int = 10, config: dict | None = None) -> None:
        self.k = k
        self.config = config
        self._idx: ragworks.Bm25 | None = None
        self._ids: list[str] = []

    def index(self, task: Task) -> None:
        self._idx = ragworks.Bm25(self.config)
        self._ids = [d.doc_id for d in task.docs]
        texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
        # One crossing for the whole corpus.
        self._idx.add_many(list(range(len(texts))), texts)

    def answer(self, query: Query) -> Response:
        if self._idx is None:
            raise RuntimeError("index() first")
        hits = self._idx.search(query.text, self.k)
        return Response(
            ranked=[self._ids[i] for i, _ in hits],
            confidence=float(hits[0][1]) if hits else 0.0,
        )
