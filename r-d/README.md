# r-d — research directory

Implement real RAG systems, measure them honestly, and let the **abstraction be
discovered rather than designed**. The library is extracted from what works here;
it is not specified in advance.

## The one rule

> **Share the measurement. Never share the mechanism.**

`systems/<name>/` may import from `harness/` and from nothing else — in
particular, **not from each other**. Each system is free to invent whatever
structures it wants, however badly they duplicate a neighbour. That duplication
is the data: an abstraction is load-bearing only if it shows up independently in
several systems that had no obligation to agree.

Sharing code between systems now would assume the answer and then "discover" it.

## Layout

```
harness/     the ONLY shared code — task format, metrics, runner, report
tasks/       dataset loaders producing a Task
systems/     one directory per RAG system, mutually isolated
findings/    NOTES.md per system: which of R1–R47 it exercised, what it wanted
```

## The contract

A system implements two methods. That is the entire interface.

```python
class System(Protocol):
    name: str
    def index(self, task: Task) -> None: ...
    def answer(self, query: Query) -> Response: ...
```

`Response` carries a ranked list of `doc_id`s, and optionally an answer, an
abstention, a verdict and a confidence. Everything else is the system's business.

## Evaluation rules

These follow from [`CANDIDATES.md`](CANDIDATES.md) §2 and are not negotiable:

1. **No published number is an acceptance test.** Of 18 RAG papers shipping
   artifacts, none fully reproduced. Only *relative ordering against our own
   baselines* is trustworthy — which is why `naive` and `hybrid_rrf` exist.
2. **Per-query scores are always retained.** Aggregates cannot support a paired
   significance test.
3. **Partial results beat lost results.** Runs are resumable and write
   incrementally; a rate limit at query 800 of 1000 must not discard 800 results.
4. **Cost and latency are measured, not estimated.** The runner records them;
   systems do not self-report timing.
5. **Abstention is an outcome, not a failure.** A system that declines is scored
   on a risk-coverage curve, not marked wrong.

## Language

`r-d` is **Python**, deliberately. The Rust core is extracted *after* the
abstraction is known. Optimising an interface we haven't found yet would be the
same mistake as specifying it.

## Status

Wave 1: `harness` · `naive` · `hybrid_rrf` · `sure`.
STAIR is excluded — it lives in its own directory and has already been tested.
