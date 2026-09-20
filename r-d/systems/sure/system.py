"""SURE-RAG: set-level evidence sufficiency with abstention.

Following arXiv:2605.03534. Two properties of that paper drive this design:

1. **Sufficiency is a set-level property.** "Missing hops and unresolved
   conflicts cannot be detected by independent passage scoring." So the whole
   candidate set goes into one question, not one question per passage.
2. **The verdict is three-way** -- supports / refutes / insufficient -- and the
   system abstains unless support is established.

Jev is the verifier backend. Its answers are sampled from the supplied schema,
so an out-of-schema verdict cannot occur, and its `confidence` field is what the
risk-coverage curve is swept over. Per-passage relevance questions ride along in
the same request because fan-out is close to free.

Retrieval here is deliberately its own copy, shared with nothing.
"""

from __future__ import annotations

import numpy as np

from harness.task import Query, Response, Task

from .jev import choice, noul, system_one

__all__ = ["SureRAG"]

VERDICTS = {
    "supports": "The evidence together establishes an answer to the question",
    "refutes": "The evidence together contradicts the question's premise",
    "insufficient": "The evidence is missing a step, or conflicts, so no answer is established",
}


class SureRAG:
    name = "sure"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        k: int = 10,
        n_evidence: int = 5,
        min_confidence: float = 0.0,
        rerank: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.k, self.n_evidence = k, n_evidence
        self.min_confidence = min_confidence
        self.rerank = rerank
        self.device = device
        self._model = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._emb: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def index(self, task: Task) -> None:
        self._ids = [d.doc_id for d in task.docs]
        self._texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
        self._emb = self._load().encode(
            self._texts, batch_size=128, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def answer(self, query: Query) -> Response:
        if self._emb is None:
            raise RuntimeError("index() first")
        q = self._load().encode(
            [query.text], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        top = np.argsort(-(self._emb @ q))[: self.k]
        ranked = [self._ids[i] for i in top]

        shortlist = list(top[: self.n_evidence])
        state = {
            "question": query.text,
            "evidence": [self._texts[i][:1200] for i in shortlist],
        }

        questions = {
            "verdict": choice(
                "Taken TOGETHER, do the passages in `evidence` establish an answer to "
                "`question`? Judge the set as a whole: a missing reasoning step or a "
                "conflict between passages means the evidence is insufficient.",
                VERDICTS,
            )
        }
        if self.rerank:
            for j in range(len(shortlist)):
                questions[f"e{j}"] = noul(
                    f"Passage `evidence[{j}]` contributes evidence needed to answer `question`.",
                    true="It supplies a fact the answer depends on",
                    false="It is off-topic, or mentions the subject only in passing",
                )

        resp = system_one(state, questions)
        answers = resp.get("answers", {})
        v = answers.get("verdict", {})
        verdict = v.get("choice", "insufficient")
        confidence = float(v.get("confidence", 0.0) or 0.0)

        if self.rerank:
            scored = [
                (float(answers.get(f"e{j}", {}).get("noul", 0.0) or 0.0), i)
                for j, i in enumerate(shortlist)
            ]
            scored.sort(key=lambda t: -t[0])
            promoted = [self._ids[i] for _, i in scored]
            ranked = promoted + [d for d in ranked if d not in set(promoted)]

        abstain = verdict == "insufficient" or confidence < self.min_confidence
        return Response(
            ranked=ranked[: self.k],
            abstained=abstain,
            verdict=verdict,
            confidence=confidence,
            usage={
                "usd": float((resp.get("usage") or {}).get("cost", 0.0) or 0.0),
                "tokens_in": float((resp.get("usage") or {}).get("prompt_tokens", 0.0) or 0.0),
                "calls": 1,
            },
        )
