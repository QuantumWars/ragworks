# ragworks

Infrastructure for building, measuring and comparing RAG systems.

A Rust core with Python bindings for the parts that touch every document, and a
research directory where systems are implemented and measured before anything is
generalised into the library.

> `ragworks` is a working name. Renaming is a `sed` across four `Cargo.toml`
> files; nothing depends on it yet.

## Why this exists

Comparing two RAG systems usually means comparing two codebases with two notions
of "a query", two cost accountings and two definitions of recall. This project
separates the two things that get conflated:

- **`lib/`** — the substrate. Every stage a developer swaps (chunker, tokenizer,
  index, embedder, reranker, verifier) is a trait with a registry in front of it,
  so a pipeline is configuration rather than code.
- **`r-d/`** — the evidence. Real systems, implemented in isolation from each
  other and measured through one shared harness. The library is extracted from
  what survives measurement, not specified in advance.

## Layout

| Path | Contents |
|---|---|
| [`lib/`](lib) | Rust workspace: `core`, `net`, `chunk`, `index`, `embed`, `read`, `query`, `judge`, `py` |
| [`r-d/`](r-d) | Research directory: harness, tasks, systems, findings |
| [`CONFORMANCE.md`](CONFORMANCE.md) | Requirements R1–R47, derived from 20 published RAG systems |
| [`r-d/CANDIDATES.md`](r-d/CANDIDATES.md) | Which systems are worth implementing, with cost and code-availability data |
| [`DESIGN.md`](DESIGN.md) | First architecture draft. **Superseded** — see `CONFORMANCE.md` §6 for what it got wrong |

## Quick start

```sh
# Rust core
cd lib && cargo test

# Python bindings
cd lib/crates/py && maturin develop --release

# Reproduce the wave-1 comparison
cd r-d && python run_wave1.py
```

## Results so far

From [`r-d/findings/`](r-d/findings), on HotpotQA with 1,461 paragraphs and 135
scored queries. Every comparison is a paired randomization test over per-query
scores.

| finding | measurement |
|---|---|
| Hybrid BM25+dense fusion beat dense alone | +0.019 recall@5, **p = 0.43** — not at all |
| BM25 alone beat dense alone | −0.007 recall@5, **p = 0.90** — indistinguishable |
| Typed reranking beat dense alone | **+0.074 recall@5, p = 0.0001** |
| Set-level sufficiency with abstention | risk 0.260 → **0.062**, AURC 0.186 → **0.040** |
| Rust BM25 against the Python reference | 100% top-1 agreement, **21.7× faster search** |
| Lexical reranking beat no reranking | +0.019 recall@5, **p = 0.55** — not at all |
| Typed reranking beat the lexical baseline | **+0.104 recall@5, p = 0.0001** |
| Best of five query transforms beat the plain query | +0.004 recall@5, **p = 1.00** — not at all |
| Query decomposition | **−0.052 recall@1, p = 0.0019** — significantly worse |

The pattern, across four independent attempts: everything that manipulates the
query or the ranking *lexically* produced nothing. The only interventions that
moved anything read the text and judged it.

## Principles

**Share the measurement, never the mechanism.** Systems in `r-d/systems/` may
import the harness and nothing else — not even each other. Duplication between
them is the evidence for what belongs in the library.

**No published number is an acceptance test.** Of 18 RAG papers shipping
artifacts, none fully reproduced. Only relative ordering against our own
baselines is trusted, which is why the unglamorous controls exist.

**Per-query scores are always retained.** Aggregates cannot support a paired
significance test.

**Partial results beat lost results.** Runs are resumable and write
incrementally; a rate limit at query 800 must not discard 800 results.

## Licence

MIT ([`LICENSE-MIT`](LICENSE-MIT)) or Apache-2.0
([`LICENSE-APACHE`](LICENSE-APACHE)), at your option.
