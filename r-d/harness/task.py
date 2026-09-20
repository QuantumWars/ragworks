"""The task format and the system contract.

Deliberately minimal. Every field here must be justified by something a wave-1
system or metric actually needs; anything speculative belongs inside a system,
not in the shared contract. Widening this file is how the experiment gets
contaminated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["Doc", "Query", "Response", "System", "Task"]


@dataclass(frozen=True, slots=True)
class Doc:
    """A retrievable unit. Whatever granularity the task defines it at."""

    doc_id: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Query:
    gold_doc_ids: frozenset[str]
    qid: str
    text: str
    answer: str | None = None
    #: False for deliberately unsupported queries -- the abstention condition.
    answerable: bool = True
    #: Restrict retrieval to one document, for per-book tasks.
    scope: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class Response:
    """What a system returns for one query.

    ``ranked`` is required; everything else is optional, so a pure retriever and
    a full generate-and-verify pipeline use the same type.
    """

    ranked: list[str]
    answer: str | None = None
    abstained: bool = False
    #: "supports" | "refutes" | "insufficient" -- SURE-RAG's three-way verdict.
    verdict: str | None = None
    #: Used for risk-coverage curves. Higher means more confident.
    confidence: float | None = None
    #: Tokens and dollars, when the system called a model. Latency is the
    #: runner's job, not the system's -- self-reported timings are not trusted.
    usage: dict = field(default_factory=dict)


@dataclass(slots=True)
class Task:
    name: str
    docs: list[Doc]
    queries: list[Query]
    meta: dict = field(default_factory=dict)

    def qrels(self) -> dict[str, frozenset[str]]:
        return {q.qid: q.gold_doc_ids for q in self.queries}

    def by_id(self) -> dict[str, Doc]:
        return {d.doc_id: d for d in self.docs}

    def scoped_docs(self, scope: str | None) -> list[Doc]:
        if scope is None:
            return self.docs
        return [d for d in self.docs if d.meta.get("scope") == scope]

    def __repr__(self) -> str:
        n_unans = sum(1 for q in self.queries if not q.answerable)
        return (
            f"Task({self.name!r}, docs={len(self.docs)}, queries={len(self.queries)}"
            f"{f', unanswerable={n_unans}' if n_unans else ''})"
        )


@runtime_checkable
class System(Protocol):
    """The entire interface. Two methods."""

    name: str

    def index(self, task: Task) -> None:
        """Prepare for retrieval over this task. Called once."""
        ...

    def answer(self, query: Query) -> Response:
        """Answer one query. Called once per query, possibly resumed."""
        ...
