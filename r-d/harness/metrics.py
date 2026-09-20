"""Metrics: retrieval, abstention, cost, and a paired significance test.

Two commitments from CANDIDATES.md are enforced structurally here.

* **Per-query scores are always retained.** Aggregates are derived from them and
  never stored alone -- a paired significance test needs the pairs, and a
  library that keeps only means can never tell you whether a gap is real.
* **Abstention is an outcome, not a failure.** A system that declines to answer
  is scored on a risk-coverage curve. Marking it wrong would make abstention
  strictly dominated, which is exactly backwards.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "RiskCoverage",
    "RunMetrics",
    "evaluate",
    "ndcg_at_k",
    "randomization_test",
    "recall_at_k",
    "risk_coverage",
]


def recall_at_k(ranked: Sequence[str], gold: frozenset[str], k: int) -> float:
    """Fraction of gold documents present in the top k.

    Multi-gold by default: HotpotQA needs two supporting paragraphs, so the
    single-gold shortcut (``1 if hit else 0``) would silently score a system
    that found half the evidence as perfect.
    """
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def ndcg_at_k(ranked: Sequence[str], gold: frozenset[str], k: int) -> float:
    """Binary-gain nDCG@k."""
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def mrr(ranked: Sequence[str], gold: frozenset[str]) -> float:
    for i, d in enumerate(ranked):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranked: Sequence[str], gold: frozenset[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & gold else 0.0


_RETRIEVAL = {
    "recall@1": lambda r, g: recall_at_k(r, g, 1),
    "recall@5": lambda r, g: recall_at_k(r, g, 5),
    "recall@10": lambda r, g: recall_at_k(r, g, 10),
    "ndcg@10": lambda r, g: ndcg_at_k(r, g, 10),
    "mrr": mrr,
    "hit@1": lambda r, g: hit_at_k(r, g, 1),
    "hit@5": lambda r, g: hit_at_k(r, g, 5),
}
DEFAULT_METRICS = ("recall@1", "recall@5", "ndcg@10", "mrr", "hit@5")


@dataclass(slots=True)
class RiskCoverage:
    """Selective-prediction curve, following SURE-RAG (arXiv:2605.03534).

    ``aurc`` is the area under the risk-coverage curve -- lower is better. A
    system with useful confidence sorts its errors to the low-confidence end,
    so risk stays low while coverage grows.
    """

    coverage: float
    risk: float
    curve: list[tuple[float, float]] = field(default_factory=list)
    aurc: float | None = None

    def as_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "risk": self.risk,
            "aurc": self.aurc,
            "curve": self.curve,
        }


def risk_coverage(
    outcomes: Sequence[tuple[bool, float | None, bool]],
) -> RiskCoverage:
    """From ``(correct, confidence, abstained)`` triples.

    Coverage is the fraction answered; risk is the error rate *among answered*.
    When confidences are present the full curve is swept over them.
    """
    n = len(outcomes)
    if n == 0:
        return RiskCoverage(0.0, 0.0)

    answered = [(c, conf) for c, conf, ab in outcomes if not ab]
    coverage = len(answered) / n
    risk = (sum(1 for c, _ in answered if not c) / len(answered)) if answered else 0.0

    scored = [(c, conf) for c, conf in answered if conf is not None]
    curve: list[tuple[float, float]] = []
    aurc: float | None = None
    if scored:
        scored.sort(key=lambda t: -t[1])
        errors = 0
        for i, (c, _) in enumerate(scored, start=1):
            errors += 0 if c else 1
            curve.append((i / n, errors / i))
        aurc = sum(r for _, r in curve) / len(curve)

    return RiskCoverage(coverage, risk, curve, aurc)


@dataclass(slots=True)
class RunMetrics:
    system: str
    task: str
    n_queries: int
    aggregate: dict[str, float]
    per_query: dict[str, dict[str, float]]
    risk: RiskCoverage | None = None
    cost: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    n_failed: int = 0

    def scores(self, metric: str) -> dict[str, float]:
        return {q: m[metric] for q, m in self.per_query.items() if metric in m}

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "task": self.task,
            "n_queries": self.n_queries,
            "n_failed": self.n_failed,
            "aggregate": self.aggregate,
            "risk": self.risk.as_dict() if self.risk else None,
            "cost": self.cost,
            "latency": self.latency,
            "per_query": self.per_query,
        }


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)

    def p(q: float) -> float:
        return s[min(len(s) - 1, int(q * len(s)))]

    return {"p50": p(0.50), "p95": p(0.95), "p99": p(0.99), "mean": sum(s) / len(s)}


def evaluate(
    records: Sequence[dict],
    qrels: dict[str, frozenset[str]],
    *,
    system: str,
    task: str,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> RunMetrics:
    """Score raw runner records.

    A record is one query's result: ``qid, ranked, abstained, confidence,
    correct, usage, latency_s, error``. Failed queries are counted and excluded
    rather than scored as zero -- an API timeout is not a retrieval failure, and
    conflating them makes a rate limit look like a quality regression.
    """
    per_query: dict[str, dict[str, float]] = {}
    outcomes: list[tuple[bool, float | None, bool]] = []
    lat: list[float] = []
    usd = tokens_in = tokens_out = 0.0
    n_failed = 0

    for rec in records:
        if rec.get("error"):
            n_failed += 1
            continue
        qid = rec["qid"]
        gold = qrels.get(qid, frozenset())
        ranked = rec.get("ranked") or []
        abstained = bool(rec.get("abstained"))

        # Queries with no gold document are the deliberate unanswerable
        # condition. Scoring them for retrieval would be meaningless -- there is
        # nothing to retrieve -- and averaging their forced 0.0 into recall
        # would punish every system for a property of the task. They are scored
        # only on whether they correctly declined.
        if gold:
            per_query[qid] = {m: _RETRIEVAL[m](ranked, gold) for m in metrics}
            correct = rec.get("correct")
            if correct is None:  # no answer-level judgement: fall back to retrieval
                correct = bool(set(ranked[:1]) & gold)
        else:
            correct = abstained

        outcomes.append((bool(correct), rec.get("confidence"), abstained))

        if (t := rec.get("latency_s")) is not None:
            lat.append(float(t))
        u = rec.get("usage") or {}
        usd += float(u.get("usd", 0.0) or 0.0)
        tokens_in += float(u.get("tokens_in", 0.0) or 0.0)
        tokens_out += float(u.get("tokens_out", 0.0) or 0.0)

    n = len(per_query)
    aggregate = {
        m: (sum(v[m] for v in per_query.values()) / n if n else 0.0) for m in metrics
    }
    return RunMetrics(
        system=system,
        task=task,
        n_queries=n,
        aggregate=aggregate,
        per_query=per_query,
        risk=risk_coverage(outcomes) if outcomes else None,
        cost={"usd": usd, "tokens_in": tokens_in, "tokens_out": tokens_out},
        latency=_percentiles(lat),
        n_failed=n_failed,
    )


def randomization_test(
    a: dict[str, float], b: dict[str, float], *, n_samples: int = 10000, seed: int = 42
) -> dict:
    """Paired randomization test (Smucker et al., 2007) over per-query scores.

    Under the null the system label is exchangeable within each query, so the
    sampling distribution of the mean difference comes from flipping each pair
    with probability 1/2. Two-sided.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0, "p_value": 1.0, "diff": 0.0}
    diffs = [b[q] - a[q] for q in shared]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_samples):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(s / len(diffs)) >= abs(observed) - 1e-12:
            hits += 1
    return {
        "n": len(shared),
        "mean_a": sum(a[q] for q in shared) / len(shared),
        "mean_b": sum(b[q] for q in shared) / len(shared),
        "diff": observed,
        "p_value": (hits + 1) / (n_samples + 1),
        "n_samples": n_samples,
    }
