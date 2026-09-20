"""Run a system over a task, resumably.

The runner owns everything a system must not be trusted to report about itself:
wall-clock latency, failure handling, rate limiting and persistence.

Resumability is a hard requirement, not a convenience. CANDIDATES.md measures
our LLM budget at ~20 requests/minute, which puts a 1000-query multi-step
evaluation at over four hours. A rate limit at query 800 must cost 200 queries,
not 1000.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Iterator
from pathlib import Path

from .metrics import RunMetrics, evaluate
from .task import Query, Response, System, Task

__all__ = ["load_records", "run"]


class _RateLimiter:
    """Simple requests-per-minute cap. `rpm <= 0` disables it."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()


def load_records(path: Path) -> dict[str, dict]:
    """Existing results keyed by qid, for resume. Tolerates a truncated tail."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial final line from an interrupted run
        out[rec["qid"]] = rec
    return out


def _queries(task: Task, limit: int | None) -> Iterator[Query]:
    qs = task.queries[:limit] if limit else task.queries
    yield from qs


def run(
    system: System,
    task: Task,
    *,
    out_dir: str | Path = "runs",
    limit: int | None = None,
    rpm: float = 0.0,
    resume: bool = True,
    verbose: bool = True,
) -> RunMetrics:
    """Evaluate ``system`` on ``task``; write records incrementally; return metrics."""
    out = Path(out_dir) / f"{task.name}__{system.name}"
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"

    done = load_records(records_path) if resume else {}
    pending = [q for q in _queries(task, limit) if q.qid not in done]

    if verbose:
        total = len(list(_queries(task, limit)))
        msg = f"[{system.name} @ {task.name}] {total} queries"
        if done:
            msg += f", {len(done)} already done, {len(pending)} to run"
        print(msg, flush=True)

    t_index = time.perf_counter()
    system.index(task)
    index_s = time.perf_counter() - t_index
    if verbose and pending:
        print(f"  indexed in {index_s:.1f}s", flush=True)

    limiter = _RateLimiter(rpm)
    with records_path.open("a") as fh:
        for i, q in enumerate(pending, start=1):
            limiter.wait()
            t0 = time.perf_counter()
            rec: dict = {"qid": q.qid}
            try:
                resp: Response = system.answer(q)
                rec.update(
                    ranked=list(resp.ranked),
                    answer=resp.answer,
                    abstained=bool(resp.abstained),
                    verdict=resp.verdict,
                    confidence=resp.confidence,
                    usage=dict(resp.usage or {}),
                )
            except Exception as exc:  # noqa: BLE001 - deliberate: one bad
                # query must never abort a run. The error is recorded per-query
                # and excluded from scoring, never counted as a wrong answer.
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc(limit=3)
            rec["latency_s"] = time.perf_counter() - t0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if verbose and (i % 25 == 0 or i == len(pending)):
                print(f"  {i}/{len(pending)}", flush=True)

    records = list(load_records(records_path).values())
    m = evaluate(records, task.qrels(), system=system.name, task=task.name)
    m.latency["index_s"] = index_s

    (out / "metrics.json").write_text(json.dumps(m.as_dict(), indent=2))
    (out / "card.json").write_text(
        json.dumps(
            {
                "system": system.name,
                "task": task.name,
                "n_docs": len(task.docs),
                "n_queries": m.n_queries,
                "n_failed": m.n_failed,
                "limit": limit,
                "rpm": rpm,
                "index_s": index_s,
                "aggregate": m.aggregate,
                "cost": m.cost,
                "latency": m.latency,
                "task_meta": task.meta,
            },
            indent=2,
        )
    )
    if verbose:
        agg = "  ".join(f"{k}={v:.3f}" for k, v in m.aggregate.items())
        print(f"  {agg}", flush=True)
        if m.n_failed:
            print(f"  WARNING: {m.n_failed} queries failed (excluded, not scored 0)", flush=True)
    return m
