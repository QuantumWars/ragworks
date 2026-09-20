# References and prior art

What this project read before building, and what it took from each.

## Frameworks studied

**[agno](https://github.com/agno-agi/agno)** — Python agent platform, Apache-2.0.
Its `knowledge/` layer is a practical RAG pipeline with pluggable chunkers,
embedders, readers and rerankers over 24 vector-store backends.

Taken from it:

- **Deterministic, content-derived chunk ids.** Without them, caches and run
  comparisons break silently between runs.
- **Provider errors normalised into one type that preserves the HTTP status**, so
  a caller can distinguish a rate limit from an auth failure. `error.rs` turns
  that into `ProviderFault` and `Error::is_retryable`.

Deliberately *not* taken: `ChunkingStrategy.chunk()` returns new document objects
owning fresh copies of their text. Chunking 1 GB at 20% overlap allocates over a
gigabyte of duplicated strings. Here a chunk is a 56-byte view.

**[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)** — agent
harness, MIT, built on an everything-is-a-plugin architecture over
[Cordis](https://github.com/cordiverse/cordis). Roughly 50 packages with a small
`core` and separate `mcp`, `skill`, `hooks`, `guard` and `session` packages.

Taken from it: the fine-grained package boundary, and the observation that its
*entire* native component is Linux Landlock confinement and POSIX file locks —
sandboxing, not speed. An agent harness is I/O-bound. That sharpened where a
Rust core earns its place here: indexing, scoring and metrics over many
documents, not the orchestration loop.

## Papers the requirements were derived from

Requirements R1–R47 in [`CONFORMANCE.md`](CONFORMANCE.md) come from reading 20
RAG systems catalogued by [Turing Post](https://www.turingpost.com/p/ragtypes).
The ones that forced changes to the design:

| System | Paper | What it demanded |
|---|---|---|
| MiA-RAG | [2512.17220](https://arxiv.org/abs/2512.17220) | a conditioning channel feeding both retrieval and generation |
| Graph-O1 | [2512.17912](https://arxiv.org/abs/2512.17912) | tree search and a steppable environment |
| HGMem | [2512.23959](https://arxiv.org/abs/2512.23959) | mutable, episode-scoped working memory |
| SURE-RAG | [2605.03534](https://arxiv.org/abs/2605.03534) | set-level sufficiency, three-way verdicts, calibration |
| A-RAG | [2602.03442](https://arxiv.org/abs/2602.03442) | retrievers exposed as tools at several granularities |
| Bidirectional RAG | [2512.22199](https://arxiv.org/abs/2512.22199) | validated corpus write-back and run lineage |
| Predictive Prefetching | [2605.17989](https://arxiv.org/abs/2605.17989) | token-level generation hooks and speculative retrieval |
| TV-RAG | [2512.23483](https://arxiv.org/abs/2512.23483) | modality, locator and derivation as independent axes |

## Evaluation methodology

- **["Stop Comparing LLM Agents Without Disclosing the Harness"](https://arxiv.org/abs/2605.23950)**
  — eight harness dimensions that silently reorder leaderboards. They become
  required run-card fields.
- **["Inside the Scaffold"](https://arxiv.org/abs/2604.03515)** — a source-code
  taxonomy of coding agent architectures. Queued for the agent layer.
- **Smucker et al. (2007)** — the paired randomization test used for every
  comparison in [`r-d/findings/`](r-d/findings).

## Algorithms

- **Okapi BM25** — Robertson and Zaragoza, with Elasticsearch's `k1=1.2`, `b=0.75`.
- **Reciprocal Rank Fusion** — Cormack, Clarke and Buettcher (2009).
- **UAX #29** — Unicode text segmentation, for the `unicode` tokenizer.
