# Working on ragworks

Context and standing decisions for anyone — human or agent — picking this up.
Read [`README.md`](README.md) for what it is; this is how to work on it and
what to do next.

---

## 1. What this project is, and is not

**It is** infrastructure for *building, measuring and comparing* RAG systems:
typed representation, an orchestration substrate, and an evaluation harness that
takes statistical significance seriously.

**It is not** a retrieval engine competing with Tantivy, Qdrant or LanceDB, and
should not try to become one. Scope creep in that direction is the main risk to
this project.

The competitive position, stated plainly:

| Against | Where we stand |
|---|---|
| **Tantivy** | our BM25 is far smaller and trivially embeddable; it has segmented persistence, compressed postings, multithreaded indexing, phrase queries, filtering and mature analysis. Not comparable. |
| **Qdrant / LanceDB** | a flat scan is not in the same scale class. They have ANN, filtering, quantization, persistence and million-vector benchmarks. |
| **Rig** | our contiguous `Vec<f32>` is *probably* more cache-friendly than a document-oriented HashMap — **an inference, not a benchmark**. Rig has LSH and far more integrations. |
| **Swiftide** | behind on streaming ingestion, async parallelism, caching, storage integrations and tracing. |
| **`rag` crate** | no HNSW, IVF, explicit SIMD, int8 quantization or write-ahead log. |
| **LlamaIndex ingestion** | leaner representation; no transformation caching, async ingestion, multiprocessing, dedup or remote cache. |

The only *measured* performance claim this project owns is **3.2x indexing and
21.7x search against its own Python reference**, at 100% top-1 agreement.
Performance against any Rust or production search library is **unmeasured**.
Do not claim otherwise anywhere — README, commit messages, or conversation.

---

## 2. How to work here

These are not style preferences. They are why the findings in `r-d/findings/`
are worth anything.

**Verify before acting on a claim.** Including claims in this file, in a paper,
or in a code review. Two real errors in this project came from a plausible story
adopted without testing: a Hebrew-combining-marks diagnosis that cost a
tokenizer rewrite (the real cause was underscores in 4 documents), and a
`Chunk` size documented at 32 bytes before it was measured at 64.

**A baseline is mandatory, not optional.** Every measured intervention here that
*looked* good and turned out to be nothing was caught by a control:

| intervention | effect |
|---|---|
| hybrid BM25+dense fusion | +0.019 recall@5, p=0.43 |
| lexical reranking | +0.019 recall@5, p=0.55 |
| query reformulation, best of five | +0.004 recall@5, p=1.00 |
| RM3 relevance feedback | **−0.081 recall@1, p=0.0005** |
| query decomposition | **−0.052 recall@1, p=0.0019** |

**No published number is an acceptance test.** Of 18 RAG papers shipping
artifacts, none fully reproduced. Only relative ordering against our own
baselines is trusted.

**Retain per-query scores and use paired randomization tests.** Aggregates
cannot support a significance test, and a difference without a p-value is not a
result.

**In `r-d/`: share the measurement, never the mechanism.** Systems may import
the harness and nothing else — not even each other. Duplication between them is
the evidence for what belongs in the library.

**Extract on the second need, not the first.** `ragworks-net` exists because
embedders and judges independently required the same HTTP, retry and rate-limit
code.

**Partial results beat lost results.** Runs resume; per-item errors are
collected, not propagated.

**Degrade on transient failure, fail loudly on misconfiguration.** A 503 after
retries falls back to the plain query; a 401 propagates. A bad key must never
present itself as poor retrieval quality.

**Keys come from the environment, named in config. Never in config.**

---

## 3. The strategic fact that should drive priorities

Measured p95 latency, from `r-d/runs/`:

| stage | p95 |
|---|---|
| Rust BM25 search | **0.15 ms** |
| dense retrieval, no rerank | 8 ms |
| typed reranking | 614 ms |
| set-level verification | ~1 s |
| LLM query rewriting | ~4.2 s |

**Once any LLM-backed stage is in the pipeline, retrieval is under 0.1% of
total latency.** Wave 4 spent 436 s of wall clock against 13 s for the
identity baseline — a 35x difference entirely outside retrieval.

The consequence: **another BM25 micro-optimization is close to worthless.**
Provider concurrency, caching and request *reduction* are where the time is.
Optimize there first, and be able to justify any retrieval optimization against
this table.

---

## 4. Do this before optimizing anything

**Build the benchmark suite first.** Optimizing without one is guessing, and
this project has no standing to claim performance it has not measured.

Compare against:
- **Tantivy** and the community **`bm25`** crate — lexical
- **Rig** exact and LSH — in-process Rust
- **Qdrant** and **LanceDB** — ANN
- the **Python reference in `r-d/`** — regression baseline

Measure at **50K, 1M and 10M** documents/vectors:
index time, peak RSS, index size, p50/p95/p99, single-query latency *and*
concurrent QPS, exact recall or ANN recall@10, warm and cold start, and
end-to-end latency with provider cost.

Until that exists, the honest summary of this project's performance is section 1.

---

## 5. Optimization order, once benchmarks exist

Highest value first. The first three are worth more than everything below them
combined, because of section 3.

1. **Parallelize `search_many`** — `crates/py/src/lib.rs`. It releases the GIL
   and then loops serially. `rayon` is already a workspace dependency and is
   **used by no crate**; this is its first real job.
2. **Make provider calls concurrent and bounded** — `crates/embed/src/openai.rs`,
   `crates/query/src/chat.rs`, `crates/judge/src/jev.rs`. Batches go one at a
   time through blocking `ureq`, with blocking sleeps for retry and rate limit.
3. **Persistent response caching**, content-addressed on (model, prompt,
   params). Re-running an evaluation after a metric change should cost nothing.
   This is the single largest cost saver available and does not exist yet.
4. **BM25 query accumulation** — replace the per-query `HashMap` with a dense
   score buffer plus a touched-document list (`crates/index/src/bm25.rs`).
5. **Remove the per-token `String` clone** — `bm25.rs:140` clones on every
   occurrence, not just on insert. Use `raw_entry` or a two-step lookup.
6. **Normalize dense vectors on insertion and use dot product** —
   `crates/index/src/flat.rs` computes and stores norms unconditionally, then
   divides per document per query; `Metric::Dot` never reads them.
7. **Benchmark explicit SIMD** before writing any. The current loop
   auto-vectorizes; assume nothing.
8. **mmap-backed corpus and vector matrix.** `DESIGN.md` describes this;
   `corpus.rs` owns a `String` and copies each document into it. Chunks are
   zero-copy *after* ingestion, not from disk.
9. **Integrate an existing ANN backend** (usearch, hnsw_rs, or LanceDB) rather
   than writing another one.

---

## 6. Known gaps, recorded so they are decisions

No persistent index, incremental commit, recovery, deletion/update model,
compressed postings, stage-output cache, or content-addressed artifact store.
BM25 has no stemming, stop-word handling or language-aware analysis, and both
indexing and batch search are single-threaded.

---

## 7. Where the real contribution is

The measurement discipline, not the speed. The harness retains per-query
scores, runs paired randomization tests, and has repeatedly rejected complexity
that did not earn its place. Findings worth keeping:

- character n-grams on the offline embedder: **+0.204 recall@5, p=0.0001**
- typed reranking: **+0.104 over a lexical baseline, p=0.0001**, hit@5 = 1.000
- set-level verification: risk **0.260 → 0.062**, AURC 0.186 → 0.040
- five query transforms: none helped, two significantly hurt

**The evidence base is one HotpotQA slice — 1,461 paragraphs, 135 scored
queries, one embedding model.** That is enough to steer this project's own
decisions and *not* enough for any general claim. Widening it is worth more than
any micro-optimization in section 5.

---

## 8. Anti-goals

- Do not reimplement an ANN index, a segmented inverted index or a vector
  database. Integrate one.
- Do not add a dependency on a native system library where a pure-Rust option
  exists; single-file wheels are a deliberate constraint (see the PDF reader).
- Do not claim performance against libraries that have not been benchmarked.
- Do not remove a baseline because it is uninteresting. The uninteresting ones
  are the ones that caught the null results.
- Do not add `debug = 1` removal to the release profile. It is load-bearing on
  macOS/arm64 and the reason is in `lib/Cargo.toml`.
