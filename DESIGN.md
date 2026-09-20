# Strata — architecture specification

> **SUPERSEDED.** This is the first architecture draft, written before the
> requirements were derived from real systems. Eleven of its claims turned out
> wrong or insufficient and two concepts were missing outright; they are
> tabulated in [`CONFORMANCE.md`](CONFORMANCE.md) §6. Kept because the record of
> what a plausible-sounding design got wrong is more useful than deleting it.
>
> `stair-rag` is a from-scratch reimplementation of STAIR (arXiv:2609.03874) kept in a separate, unpublished repository; references to it below record where a measurement came from, not a path in this repo.
>
> Infrastructure for building, training, evaluating and comparing **any** RAG system.
> Rust core, Python surface. Status: **draft for sign-off — no code written yet.**
>
> `Strata` is a working name (the structure/layers pun is deliberate: this library
> treats document structure as a first-class citizen). Renaming is cheap today and
> expensive after the first release — see §10, Open questions.

| | |
|---|---|
| Version | 0.0.1-draft |
| Date | 2026-09-20 |
| Target platform | macOS arm64 (dev), manylinux + macOS + Windows (release) |
| Toolchain present | rustc 1.97.1, cargo 1.97.1, uv 0.11.21, CPython 3.12.13 |
| Toolchain missing | `maturin` (`uv tool install maturin`) |
| Prior art | `stair-rag` (8.5k LOC) and `jev` — both in separate, unpublished repositories |

---

## 1. Goals and non-goals

### 1.1 What this is

A **substrate**, not a framework. You bring a corpus, a retrieval idea and an LLM;
Strata supplies the typed scaffolding, the execution engine, the measurement
apparatus and the reproducibility guarantees that every RAG experiment re-invents
badly.

Six goals, in priority order. When two conflict, the earlier one wins.

1. **Comparability.** Two RAG systems evaluated under Strata are comparable by
   construction — same corpus fingerprint, same splits, same metric definitions,
   same significance machinery. This is the single most valuable property and
   everything else is subordinate to it.
2. **Ergonomics.** The common path is ten lines in a notebook. The feel to aim for
   is `pandas` / `data.table`: declarative, inspectable, DataFrames out, one-call
   plots, sensible defaults that are always printable.
3. **Type safety end to end.** Rust types are the single source of truth. Python
   receives generated `.pyi` stubs, so an invalid pipeline is a type error in the
   editor, not a `KeyError` forty minutes into a run.
4. **Measurement as a first-class concern.** Hallucination, groundedness, citation
   fidelity, latency and **cost** are not add-ons — the engine records them because
   it owns the dataflow, not because the user remembered to instrument it.
5. **Efficiency.** Memory-mapped corpora (a corpus larger than RAM is normal),
   zero-copy node views, rayon-parallel index build and metric kernels, and a
   disciplined FFI boundary (§4.3).
6. **Extensibility without forking.** STAIR, GraphRAG and agentic RAG are
   *configurations* of the same engine. Adding a strategy means implementing
   traits, never patching the core.

### 1.2 What this is explicitly *not*

Stating these prevents scope rot later.

| Not | Why, and what we do instead |
|---|---|
| A vector database | Qdrant, Milvus and LanceDB are good. Strata defines an `Index` trait and ships a fast local implementation; external stores are adapters. |
| A serving framework | The target is the research and evaluation loop. A validated pipeline exports to a frozen artifact; serving it is someone else's concern. |
| An LLM training framework | We orchestrate `torch` / `peft` / `transformers`. We do not reimplement an optimiser. |
| A general agent framework | The `Policy` trait (§3.3) closes a retrieval loop. It is not LangGraph and should not grow into it. |
| Opinionated about embeddings | Any model, any dimension, any provider. The corpus format does not care. |
| A prompt-management product | Prompts are typed templates versioned inside the run config. Nothing more. |

### 1.3 Design principles

- **The corpus is immutable.** Every derived thing is content-addressed against it.
  Mutation happens by producing a new corpus version, never in place.
- **Nothing expensive is implicit.** `.run()` is the only call that spends money,
  and `.estimate()` tells you the bill first.
- **Partial results beat lost results.** A budget breach, a rate limit or a bad
  query yields an incomplete-but-marked run, never an exception that discards an
  hour of work.
- **Everything is replayable.** The trace log is sufficient to rebuild every
  metric, table and figure without re-running the pipeline.
- **The fast path is the default path.** If a user has to opt in to the efficient
  behaviour, the design is wrong.

---

## 2. The typed data model

### 2.1 Identifiers

Newtypes over `u64`, interned in a per-corpus symbol table. Never bare strings in
hot structures — string comparison in a retrieval inner loop is how systems get
slow without anyone noticing.

```rust
pub struct CorpusId(u64);   pub struct DocId(u64);
pub struct NodeId(u64);     pub struct QueryId(u64);
pub struct RunId(u64);      pub struct StageId(u32);
```

Human-facing labels (`"4.2.3 Electoral Systems"`) live in a side table and are
resolved only at the boundary — display, prompts, and the trie's surface form.

### 2.2 Core entities

```rust
/// A source unit: a book, a PDF, a web page, a ticket.
pub struct Document {
    pub doc_id: DocId,
    pub uri:    Uri,
    pub blob:   ByteRange,        // view into the mmapped corpus blob
    pub meta:   MetaRef,          // Arrow struct: domain, title, url, ...
}

/// THE ATOM OF RETRIEVAL — and the most important type in the library.
pub struct Node {
    pub node_id:  NodeId,
    pub doc_id:   DocId,
    pub parent:   Option<NodeId>, // structure, not flat chunks
    pub children: NodeSlice,      // contiguous range in the node table
    pub depth:    u16,
    pub ordinal:  u32,            // position among siblings; gives reading order
    pub label:    SymbolId,       // "4.2.3 Electoral Systems and Political Parties"
    pub span:     ByteRange,      // zero-copy view into the document blob
    pub kind:     NodeKind,       // Section | Paragraph | Chunk | Table | Figure | Entity
    pub vector:   Option<VecRef>, // slot in the embedding matrix
}

/// Typed relation. Makes GraphRAG first-class with no second data model.
pub struct Edge {
    pub src: NodeId, pub dst: NodeId,
    pub kind: EdgeKind,   // Contains | Precedes | Refers | Similar | Mentions | Derived
    pub weight: f32,
}
```

**Why one `Node` type covers everything.** A naive chunker emits depth-1 nodes with
no parent — flat RAG. A ToC parser emits a tree whose leaves are the retrieval
targets — that is STAIR, exactly. A GraphRAG entity extractor emits `Entity` nodes
plus `Mentions` edges. Three very different systems, one table, one index, one set
of metrics. This is the central bet of the design: **flat chunking is the
degenerate case of structured chunking, so build for the general case and get the
common case free.**

`stair-rag` discovered this the hard way — its `TocTree` and `leaf_docids()` are
this idea in a book-specific form. Strata promotes it to the substrate.

### 2.3 Query, judgment, result

```rust
pub struct Query {
    pub qid:   QueryId,
    pub text:  SymbolId,
    pub scope: Option<DocId>,   // STAIR retrieves within one book; None = whole corpus
    pub meta:  MetaRef,
}

/// Graded relevance. stair-rag's single-gold case is `{node: 1}` — a strict subset.
pub type Qrels = Map<QueryId, Map<NodeId, u8>>;

pub struct Candidate {
    pub node_id:    NodeId,
    pub score:      f32,
    pub rank:       u32,
    pub provenance: StageId,   // which stage produced it — essential for debugging fusion
}

pub struct Run { /* Arrow columnar: qid | node_id | score | rank | provenance */ }
```

`provenance` is not decoration. In a hybrid pipeline, "did this win because of BM25
or the dense leg?" is the first question you ask, and without provenance you cannot
answer it after the fact.

### 2.4 The generation side

Most RAG libraries model retrieval carefully and generation vaguely. That asymmetry
is precisely why hallucination is hard to measure, so Strata types both.

```rust
/// Exactly what entered the prompt — the ground truth for groundedness.
pub struct Context {
    pub items:  Vec<ContextItem>,  // (node_id, span, rendered_tokens, position)
    pub budget: BudgetReport,      // requested / used / evicted, and by which policy
    pub render: TemplateId,
}

pub struct Answer {
    pub text:      String,
    pub citations: Vec<Citation>,  // claim span -> supporting node + node span
    pub usage:     Usage,          // prompt/completion tokens, model, cached?
}

pub struct Verdict {
    pub structural:  Option<StructuralCheck>, // id ∉ valid set — cheap, exact
    pub grounded:    Option<GroundednessScore>,
    pub per_claim:   Vec<ClaimVerdict>,
    pub backend:     VerifierId,
}
```

Because `Context` records the exact spans that entered the prompt, groundedness is
a decidable question about two known texts rather than a vibe. This is the type
that makes §6.2 possible.

### 2.5 Trace and cost

```rust
pub struct TraceEvent {
    pub run_id: RunId, pub stage: StageId, pub qid: Option<QueryId>,
    pub t_start_ns: u64, pub dur_ns: u64,
    pub cache: CacheOutcome,          // Hit | Miss | Bypass
    pub usage: Option<Usage>,
    pub error: Option<ErrorRef>,
}

pub struct CostLedger {
    pub by_model: Map<ModelId, Usage>,
    pub usd: f64,
    pub saved_usd: f64,              // what the cache avoided — the number that justifies the cache
    pub budget: Budget, pub breached: bool,
}
```

The event log is an **append-only Arrow IPC stream**, written as the run proceeds.
Consequences worth having: a killed run keeps its partial trace; the live dashboard
and the tracker exporters are both just readers of this stream; and every metric,
table and figure is recomputable from it without touching the model again.

### 2.6 Memory layout

The decision that determines whether large corpora are pleasant or painful:

```
corpus/<fingerprint>/
├─ blob.bin           one contiguous byte blob, mmapped, never fully read
├─ nodes.arrow        the node table (fixed-width columns, cache-friendly scans)
├─ edges.arrow        optional, for graph strategies
├─ vectors.f32        aligned embedding matrix, mmapped
├─ symbols.arrow      interned strings
└─ manifest.json      fingerprint, schema version, provenance of every document
```

A `Node` is 48 bytes and owns nothing. Text is a `&str` into the mmapped blob,
materialised only when rendered into a prompt. A 50 GB corpus opens instantly at
near-zero RSS, and the OS page cache handles residency better than any policy we
would write.

---

## 3. Stage protocols and the loop form

### 3.1 The nine stages

Each is a Rust trait with a 1:1 Python `Protocol`. A stage is a pure function of
its inputs plus a declared cache key; the engine owns scheduling, batching,
retries, budget enforcement and tracing so implementations never reimplement them.

| Stage | Signature | Examples |
|---|---|---|
| `Ingest` | `Source → Vec<Document>` | PDF, Markdown, HTML, JSONL, Wikipedia |
| `Structure` | `Document → NodeTree + Vec<Edge>` | fixed chunker, semantic chunker, ToC parser, entity graph |
| `Index` | `NodeTree → Index` | BM25, HNSW, trie, hybrid, external adapter |
| `Retrieve` | `(Query, &Index) → Vec<Candidate>` | sparse, dense, generative/constrained, graph walk |
| `Rerank` | `(Query, Vec<Candidate>) → Vec<Candidate>` | cross-encoder, RRF, LLM rerank, MMR |
| `Compose` | `(Query, Vec<Candidate>, Budget) → Context` | top-k, parent-expansion, dedup, packing |
| `Generate` | `(Query, Context) → Answer` | API LLM, local model, constrained decode |
| `Verify` | `(Query, Context, Answer) → Verdict` | structural, NLI, LLM judge, **jev** |
| `Policy` | `LoopState → Action` | continue / refine / decompose / stop |

### 3.2 Why this decomposition and not another

Two rules were applied, and they are what keep the stage list from growing:

1. **A stage boundary exists where the data type changes.** `Retrieve` and `Rerank`
   are separate because reranking is the only stage that is `Vec<Candidate> →
   Vec<Candidate>` — which is exactly why it composes and stacks freely.
2. **A stage boundary exists where the cache key changes.** `Structure` output
   depends only on the document; `Retrieve` output depends on query + index.
   Different invalidation domains must not share a cache entry, or you get the
   subtlest class of experiment bug there is: stale results that look plausible.

`Compose` gets its own stage — often folded into generation elsewhere — because
context packing is where most RAG quality is silently won or lost, and it must be
independently swappable and independently measurable.

### 3.3 The loop form: how one engine covers every RAG family

A RAG system is a DAG of stages plus a **loop form**. Three forms are exhaustive
for everything in the literature today:

```
Linear                  Iterative(policy, max_steps)        Branching(policy, fanout)
──────                  ────────────────────────────        ─────────────────────────
Retrieve                    ┌─────────────────┐              Query
   ↓                        ↓                 │                ↓ decompose
Rerank                  Retrieve              │             ┌──┴──┬─────┐
   ↓                        ↓                 │             ↓     ↓     ↓
Compose                 Compose               │           sub-q  sub-q  sub-q
   ↓                        ↓                 │             ↓     ↓     ↓
Generate                Generate              │          (each: Linear)
                            ↓                 │             └──┬──┴─────┘
naive RAG               Verify ──Continue?────┘                ↓ merge evidence
hybrid RAG                  ↓ Stop                          Generate
STAIR                   answer
                                                            agentic decomposition
                        Self-RAG, FLARE, iterative           multi-query, RAG-fusion
```

The engine provides identical tracing, caching, budgeting and metrics for all
three. **That is the whole value proposition**: today, comparing STAIR against an
agentic system means comparing two codebases with two notions of "a query", two
cost accountings and two definitions of recall. Here it is one `.compare()` call
over one run table.

Loop safety is enforced by the engine, not by trust: `max_steps`, a per-query
budget slice, and cycle detection on the `Action` graph. An agentic strategy that
would spin cannot — it terminates with `Incomplete { reason: StepLimit }` and the
partial evidence it gathered.

### 3.4 Composition

```python
from strata import Pipeline, stages

stair = Pipeline("stair")
    .structure(stages.TocParser(min_depth=2))
    .index(stages.TrieIndex(tokenizer="mistral-7b"))
    .retrieve(stages.ConstrainedGenerate(model=ft, beams=5))
    .compose(stages.TopK(k=3, expand="parent"))
    .generate(stages.Chat(model="claude-haiku-4-5-20251001"))
    .verify(stages.Structural() | stages.Jev(threshold=0.8))
    .build()          # ← type-checks the whole graph here, before anything runs
```

`.build()` validates statically: every stage's output type feeds the next, the
index kind supports the retriever, the tokenizer matches the generator. A mismatch
is an error at build time with a message naming both stages — never a runtime
surprise after an expensive ingest.

---

## 4. The native core boundary

### 4.1 What lives in Rust

Chosen by one test: *does it touch every document, every candidate, or every
token?* If yes, Rust. If no, Python, where iteration speed matters more.

| Component | Crate | Rationale |
|---|---|---|
| Corpus store, mmap, Arrow tables | `strata-core` | Every document. Zero-copy is the point. |
| Chunkers, tokenizer wrapper | `strata-core` | Millions of docs; embarrassingly parallel under rayon. |
| DAG scheduler, cache, trace writer | `strata-core` | Needs real threads. The GIL caps parallel retrieval over many books — a measurable ceiling in `stair-rag` today. |
| BM25 sparse index | `strata-index` | Classic hot loop; postings intersection in Python is untenable. |
| Dense ANN (HNSW) | `strata-index` | SIMD distance kernels; `usearch` via FFI if ours underperforms. |
| Prefix trie + batched allowed-tokens | `strata-index` | **The highest-leverage item** — see §4.4. |
| Metric kernels | `strata-metrics` | Whole-run-table scans, vectorised. |
| Significance, bootstrap | `strata-metrics` | Permutation tests are O(samples × queries); rayon makes this instant instead of a coffee break. |
| Arrow/Parquet IO, RunCard | `strata-io` | Zero-copy handoff to Python. |
| PyO3 bindings, stub generation | `strata-py` | The only crate that knows Python exists. |

### 4.2 What stays in Python

Strategy definitions, LLM adapters and rate limiting, prompt templates, training
loops (`torch`/`peft`), plotting, tracker exporters, the CLI, the notebook API.

The rule: **Python is where you iterate on ideas, Rust is where you iterate on
microseconds.** A researcher must be able to write a new strategy without touching
`cargo`. If a plausible research idea requires editing Rust, the trait design has
failed and the trait is what gets fixed.

### 4.3 FFI rules

Getting this wrong makes a Rust core *slower* than pure Python. Four rules,
non-negotiable:

1. **Nothing crosses per-token or per-candidate.** Batch at query level or coarser.
   A per-candidate FFI hop in a 10k-candidate rerank costs more than the rerank.
2. **Data crosses as Arrow buffers** via the C Data Interface — zero-copy, no
   serialisation. Never Python lists of dicts. A `Run` becomes a `pyarrow.Table`
   sharing the same memory, so `res.table()` is free.
3. **The GIL is released around all native work** (`py.allow_threads`), so an
   8-book parallel retrieval actually uses 8 cores.
4. **Python callbacks into Rust are permitted but marked.** A user's custom
   `Retriever` written in Python works — the engine batches calls and warns once
   in the trace that this stage is GIL-bound, so a slow experiment explains itself.

### 4.4 The constrained-decoding fix

Concrete, measurable, and the clearest proof the native core earns its place.

`stair-rag/src/stair/retrievers/trie.py` builds a nested-dict trie, and HuggingFace
calls `prefix_allowed_tokens_fn` **once per beam per decoding step**. With 5 beams
× ~20 steps × N queries, that is a Python function call and a dict walk in the
innermost loop of generation — pure interpreter overhead on the critical path.

Strata inverts it:

```
Python (once per step):  logits [batch·beams, vocab]  ──Arrow──▶  Rust
Rust:                    walk all beam prefixes in parallel,
                         write a packed boolean mask in place
Python:                  apply mask                   ◀──Arrow──   Rust

one FFI crossing per decoding step, instead of batch × beams
```

The trie itself becomes a flat, cache-friendly array (`Vec<(token, first_child,
n_children)>`) rather than a pointer-chasing dict-of-dicts. Expected: the
allowed-tokens overhead stops being visible in a profile at all. This gets a
criterion benchmark against the Python implementation in CI so the claim stays
honest rather than aspirational.

### 4.5 Workspace layout

```
infrastructure/
├─ Cargo.toml                    # workspace
├─ crates/
│  ├─ strata-core/               # types, corpus, engine, trace, cost, cache
│  ├─ strata-index/              # bm25, dense, trie, fusion
│  ├─ strata-metrics/            # metrics, significance, aggregation
│  ├─ strata-io/                 # arrow, parquet, runcard, mmap
│  └─ strata-py/                 # PyO3 bindings + .pyi generation
├─ python/
│  └─ strata/
│     ├─ __init__.py             # the 10-line API surface
│     ├─ _core.pyi               # GENERATED — never hand-edited
│     ├─ stages/                 # Python-side stage implementations
│     ├─ strategies/             # stair, hybrid, graph, agentic
│     ├─ llm/                    # adapters, fallback chain, rate limits
│     ├─ train/                  # torch/peft orchestration
│     ├─ ops/                    # runcard, exporters, budget
│     ├─ plot/                   # the .plot namespace
│     └─ cli.py
├─ tests/                        # pytest: integration + golden runs
├─ benches/                      # criterion
├─ pyproject.toml                # maturin backend
└─ DESIGN.md                     # this file
```

---

## 5. Artifact store, RunCard, and the tracker exporters

### 5.1 On-disk form

```
runs/2026-09-20T18-04-11Z_a31f9c7e/
├─ card.json          the RunCard — everything needed to judge and reproduce
├─ config.yaml        fully resolved config, defaults inlined (no hidden state)
├─ events.arrow       append-only trace stream
├─ run.parquet        qid | node_id | score | rank | provenance
├─ answers.parquet    qid | text | citations | usage        (generative runs)
├─ verdicts.parquet   qid | structural | grounded | per-claim
├─ metrics.json       aggregated, with per-query scores retained
└─ figures/*.png|svg
```

### 5.2 The RunCard

```jsonc
{
  "schema_version": "1.0",
  "run_id": "a31f9c7e",
  "corpus": { "fingerprint": "blake3:9f2c…", "n_docs": 18, "n_nodes": 12844 },
  "config_hash": "blake3:41ab…",
  "code":   { "git_sha": "…", "dirty": false, "strata": "0.1.0", "crates": {…} },
  "env":    { "os": "macos-27.0", "arch": "arm64", "cpu": "Apple M1 Max",
              "accel": "mps", "python": "3.12.13", "rustc": "1.97.1" },
  "seed": 42,
  "status": "complete",              // complete | incomplete | failed
  "incomplete_reason": null,          // BudgetExceeded | StepLimit | RateLimited | Interrupted
  "metrics": { "recall@1": 0.826, "ndcg@3": 0.871, "hallucination_rate": 0.0005 },
  "cost":    { "usd": 0.41, "saved_usd": 2.18, "tokens_in": 1840221, "tokens_out": 12044 },
  "latency": { "p50_ms": 41, "p95_ms": 220, "p99_ms": 480 }
}
```

**The reproducibility contract.** Same `config_hash` + same corpus `fingerprint` +
same `seed` ⇒ same `run_id`. If `run_id` differs, something in `code` or `env`
changed, and diffing two cards says exactly what. This turns "why did the number
move?" from an archaeology project into a one-line diff.

`status` is load-bearing. An incomplete run is a first-class, publishable object
carrying its own reason — because the realistic failure mode with free-tier models
is a rate limit at query 800 of 1000, and throwing away 800 results is unacceptable.

### 5.3 Exporters

Local artifacts are the source of truth. Trackers **mirror**; they never own.

```python
import strata
from strata.ops import MLflow, WandB

strata.ops.export(MLflow(uri="http://localhost:5000", experiment="rag"))
strata.ops.export(WandB(project="rag-infra", entity="…"))

res = exp.run()        # → local artifacts, always
                       # → mirrored live to every registered exporter
```

The `Exporter` protocol is a pure reader of the trace stream:

```python
class Exporter(Protocol):
    def on_run_start(self, card: RunCard) -> None: ...
    def on_event(self, ev: TraceEvent) -> None: ...
    def on_metric(self, step: int, name: str, value: float) -> None: ...
    def on_artifact(self, path: Path, kind: ArtifactKind) -> None: ...
    def on_run_end(self, card: RunCard) -> None: ...
```

Two consequences of defining it as a replayable reader rather than a live hook:

- **Offline-first works properly.** Train on a plane, then
  `strata ops push runs/a31f9c7e --to wandb` replays the whole stream, training
  curves intact. Given how much of this work happens against rate-limited free
  models with flaky connectivity, this is the difference between exporters being
  useful and being abandoned.
- **A tracker outage cannot fail a run.** Exporter errors are logged into the trace
  and swallowed. Nothing a metrics dashboard does should ever kill a training job.

Both adapters map onto native concepts where possible (W&B tables for run
comparisons, MLflow nested runs for sweeps) and degrade to JSON artifacts where
not — deliberately accepting their schema rather than fighting it.

---

## 6. Metrics, cost, significance

### 6.1 The metric trait

```rust
pub trait Metric {
    fn per_query(&self, run: &Run, qrels: &Qrels, qid: QueryId) -> Option<f64>;
    fn aggregate(&self, scores: &[f64]) -> f64 { mean(scores) }
    fn requires(&self) -> Requires;   // Run | Answers | Verdicts | Context
}
```

**Per-query scores are always retained.** This is non-negotiable and is the lesson
`stair-rag/src/stair/significance.py` already encodes: paired significance tests
need per-query pairs, and a library that only stores aggregates can never tell you
whether a +2.1 point difference is real. Aggregates are derived; pairs are the data.

### 6.2 Families

**Retrieval** — `recall@k`, `ndcg@k`, `mrr`, `map`, `hit_rate`, `coverage`.
Graded-relevance definitions throughout; `stair-rag`'s single-gold case is the
degenerate instance, so its published numbers remain directly reproducible.

**Hallucination**, and the distinction that matters most in this library:

| Kind | Definition | How measured | Cost |
|---|---|---|---|
| *Structural* | emitted identifier ∉ valid id set | exact set membership | free |
| *Semantic* | a claim in the answer is unsupported by `Context` | per-claim entailment | a model call |

Structural hallucination is STAIR §6.1 exactly, and is the rate driven to ~0.05% by
constrained decoding. It generalises beyond STAIR to any system that names its
sources: a citation to a node that does not exist is the same bug. It is exact and
free, so it is **always on**.

Semantic hallucination needs a backend, and the choice is consequential:

| Backend | Trade-off |
|---|---|
| NLI cross-encoder | local, cheap, fast; weaker on long contexts |
| LLM judge | strongest; costly, and *can itself hallucinate* |
| **jev** (`../jev`) | **default** — typed schema output, so an out-of-schema verdict is structurally impossible, and `confidence` gates escalation |

Making jev the default verifier is a genuinely nice fit: a judge that cannot emit
an invalid verdict is the right instrument for measuring invalid output, and its
`confidence` field gives a principled escalation rule — verify cheaply, escalate to
an LLM judge only below threshold, which keeps the evaluation bill bounded.

**Citation fidelity** — precision (cited spans support the claim), recall (claims
carry citations), span-level IoU against gold where available.

**Context quality** — context precision/recall, positional bias (does the gold node
land where the model attends), compression ratio.

**Efficiency** — latency percentiles, tokens, $/query, cache hit rate, index build
time, peak RSS, index bytes/node.

### 6.3 Significance testing

Built in, because "system B beat system A" without it is not a result.

- **Paired randomization test** (Smucker et al. 2007) — exact enumeration for
  n ≤ 20, Monte-Carlo beyond. Ported from `stair-rag/src/stair/significance.py`
  into a rayon-parallel kernel.
- **Paired bootstrap** confidence intervals on the difference.
- **Holm–Bonferroni correction**, applied automatically when a comparison table has
  more than two systems. This is routinely omitted in RAG papers and it is exactly
  how a leaderboard manufactures a winner from noise; here, correction is the
  default and turning it off is explicit.

```python
res.compare("stair", "bm25", metric="recall@1", test="randomization")
# ComparisonResult(mean_a=0.412, mean_b=0.826, diff=+0.414,
#                  ci95=(0.381, 0.447), p=0.0009, corrected_p=0.0027, n=1204)
```

### 6.4 Cost as an enforced budget

A budget that is only reported is not a budget.

```python
exp.run(budget=strata.Budget(usd=5.0, tokens=2_000_000, wall="30m"))
```

The engine tracks spend per stage and, on breach, **stops and returns a partial
run** marked `incomplete_reason: BudgetExceeded`. It never silently continues.
`exp.estimate()` gives a projected cost from a sample of queries before committing.

The LLM adapter layer lifts the machinery `stair-rag/src/stair/llm.py` already
proved necessary against OpenRouter's free tier, and generalises it:

- an ordered **fallback chain** of models, since individual free models rate-limit
  and go out of service unpredictably;
- request **and token** rate caps (~20 rpm default);
- retry with exponential backoff and jitter;
- a content-addressed **response cache** keyed on (model, prompt, params), which is
  the single biggest cost saver in an eval loop — reruns after a metric change cost
  nothing, and `saved_usd` on the RunCard proves it;
- a `strata llm probe` preflight reporting which models actually respond right now,
  before a long run commits to one.

---

## 7. Strategy catalogue

Each is a wiring, not a subsystem. This table is the test of whether the
abstraction holds — if a strategy needs a core change, the design is wrong.

| Strategy | Structure | Index | Retrieve | Loop | Notes |
|---|---|---|---|---|---|
| **Naive** | fixed chunker | dense | top-k | Linear | the baseline everyone skips reporting |
| **Hybrid** | fixed chunker | BM25 + dense | RRF fusion | Linear | provenance shows which leg won |
| **STAIR** | ToC parser | trie | constrained generate | Linear | port of `stair-rag`; validates the whole design |
| **DSI** | ToC parser | trie | unconstrained generate | Linear | STAIR's own baseline; identical wiring, one flag |
| **HyDE** | any | dense | hypothetical-doc embed | Linear | a `Retrieve` impl, nothing more |
| **GraphRAG** | entity extractor + edges | graph + dense | subgraph walk → summarise | Linear | uses `Edge`; no new data model |
| **Self-RAG** | any | any | any | Iterative | `Policy` reads `Verdict`, re-retrieves on low groundedness |
| **Agentic** | any | any | any | Branching | decompose → per-sub-query Linear → merge evidence |
| **MIA-RAG** | any | any | any | Iterative | *see note* |

**Note on MIA-RAG.** I am not confident which paper you mean — the name is used for
more than one multi-agent/iterative retrieval method, and I would rather flag that
than implement a guess. Send me the paper or arXiv id and I will map it onto the
loop forms above; my expectation is that it is an `Iterative` or `Branching`
configuration with a specific `Policy`, requiring no core change. If it turns out
not to fit, that is genuinely useful information about the design and I want to
find out before writing the engine, not after.

### 7.1 Migration path for `stair-rag`

`stair-rag` stays where it is and keeps working. Strata absorbs it by mapping,
and the mapping is the acceptance test for the abstraction:

| `stair-rag` | Strata |
|---|---|
| `TocTree`, `leaf_docids()` | `Node` tree + `NodeKind::Section`, leaves = `children.is_empty()` |
| `Book`, `QueryExample` | `Document` + `Query` (`scope = doc_id`) |
| `Retriever` ABC | `Retrieve` trait |
| `RetrievalResult` (`dict[str, list[str]]`) | `Run` (Arrow, with scores and provenance) |
| `metrics._METRICS` registry | `Metric` trait impls |
| `significance.randomization_test` | `strata-metrics` rayon kernel |
| `TokenTrie` (dict) | flat array trie + batched mask kernel (§4.4) |
| `Config` + `SYSTEM_BUILDERS` | `Pipeline` builder + resolved config |
| `scripts/make_figures.py` | `res.plot.*` |

**The acceptance criterion for milestone 2:** Strata reproduces `stair-rag`'s
published SearchTome numbers within tolerance, from the same corpus, as a golden
test in CI. If it cannot, the abstraction lost something real and gets revised.

---

## 8. Public API and notebook ergonomics

### 8.1 The ten-line path

```python
import strata

corpus = strata.corpus("stair-rag/data/searchtome")     # lazy, mmapped, fingerprinted
exp    = strata.experiment("stair-vs-bm25", corpus=corpus, seed=42)

exp.add(strata.strategies.BM25())
exp.add(strata.strategies.Hybrid(dense="bge-small-en-v1.5"))
exp.add(strata.strategies.STAIR(model="mistral-7b", lora="r16"))

exp.estimate()                                   # → $0.41 est., ~6 min, 1.8M tokens
res = exp.run(budget=strata.Budget(usd=5))

res.table("recall@1", by="domain")               # → DataFrame
res.plot.hallucination()                         # → Figure
res.compare("stair", "bm25", test="randomization")
```

### 8.2 The rules that produce the pandas/R feel

1. **One import.** `import strata`, and everything is reachable from it.
2. **Defaults are good and printable.** `strata.defaults()` prints every default in
   force. No hidden state, ever — a default you cannot see is a bug you cannot find.
3. **Everything has a real `__repr__`,** plus `_repr_html_` in notebooks. A
   `Pipeline` renders as its stage graph; a `Corpus` as its shape and fingerprint.
4. **Results are DataFrames.** `res.table(...)` returns pandas (polars optional).
   Users already know how to slice, pivot and plot those — do not invent a result
   object that has to be learned.
5. **`.plot` is a namespace**, exactly like `df.plot`: `.recall_by_domain()`,
   `.hallucination()`, `.training_curve()`, `.cost_breakdown()`,
   `.recall_by_support()`, `.latency_cdf()`. Each returns a matplotlib `Figure`
   you can keep styling.
6. **Lazy and cached by default.** Building pipelines is free; `.run()` is the only
   call that spends money, and it says so first.
7. **Errors name the stage and the query.**
   `StageError(stage='retrieve/ConstrainedGenerate', qid='q_0412', cause=RateLimited(retry_after=12s))`
   — never a bare `KeyError` from four frames deep.
8. **Progress is honest.** A live bar showing queries done, spend so far, spend
   projected, and current cache hit rate.

### 8.3 Training

```python
tr = strata.train.Retriever(
    base="mistral-7b", method="lora", r=16, alpha=32,
    data=corpus.queries(split="train"),
    select_on="dev/recall@1", early_stop=3,
)
run = tr.fit()                      # streams to events.arrow → live to MLflow/W&B
run.plot.training_curve()           # loss, dev recall, hallucination on one axis
```

Training emits the same trace stream as evaluation, so training curves, eval
metrics and cost all live in one comparable object — and `ops push` mirrors them to
your tracker whether or not the machine was online at the time.

### 8.4 CLI

```
strata ingest <src> --out corpus/        strata run <config.yaml>
strata eval <run> --metric recall@1      strata compare <run-a> <run-b>
strata ops push <run> --to wandb         strata llm probe
strata bench index --corpus <c>
```

The CLI is a thin shell over the same API; anything doable in one is doable in the
other, and the CLI prints the equivalent Python for whatever it just ran.

---

## 9. Build, testing, CI, release

### 9.1 Developer setup

```bash
uv tool install maturin          # the one missing piece today
cd infrastructure
uv sync                          # resolves Python deps
maturin develop --release        # builds the Rust core into the venv
pytest -q && cargo test
```

Pinned: `rustc 1.97.1` via `rust-toolchain.toml`, CPython ≥ 3.10, `uv` for Python.
The existing `.venv` at the repo root is reused so `stair-rag` and Strata coexist
during migration.

### 9.2 Testing strategy

| Layer | Tool | What it guards |
|---|---|---|
| Rust units | `cargo test` | kernels, trie, corpus store |
| Rust properties | `proptest` | invariants: a trie never permits an invalid id; `recall@k` is monotone in `k`; `ndcg ≤ 1` |
| Python units | `pytest` | stages, adapters, config resolution |
| Type surface | `mypy --strict` + generated `.pyi` | the Rust→Python type contract |
| Integration | `pytest` | pipelines end to end on a tiny fixture corpus |
| **Golden runs** | `pytest -m golden` | **STAIR numbers reproduced within tolerance** |
| Benchmarks | `criterion` + `pytest-benchmark` | perf regressions fail CI at >10% |

Golden runs are the conscience of the project. `stair-rag`'s existing
`results/searchtome/table4_summary.json` and `data/paper_reference.yaml` become
CI fixtures, so a refactor that quietly degrades retrieval quality cannot merge.

### 9.3 Type-safety mechanics

`.pyi` stubs are **generated from the Rust types** in `strata-py` and checked into
the repo. CI regenerates and fails on drift, which makes it impossible for the
Python surface to silently diverge from the core. Pipeline composition is checked
at `.build()` against declared stage input/output types, so a wiring error is a
build-time message naming both stages rather than a runtime failure mid-run.

### 9.4 Release

`abi3-py310` wheels — one wheel per platform covers 3.10 through 3.14. Targets:
macOS arm64 + x86_64, manylinux 2_28 x86_64 + aarch64, Windows x86_64. Built by
`maturin` in CI, published on tag. The Rust core and Python package share a
version; the **artifact schema version is separate and explicit**, so a library
upgrade never silently invalidates last month's runs.

---

## 10. Open questions for you

Answers to these shape milestone 2. Marked with where I lean.

1. **Name.** `strata` is a placeholder. It is unclaimed on PyPI as far as I know
   and I would check before committing. Alternatives: `arbor`, `loom`, `rift`,
   `corpus`. *Low stakes now, high stakes after release.*
2. **MIA-RAG** — which paper? (§7) Genuinely blocking for that strategy only;
   nothing else waits on it.
3. **Corpus scale target.** Designing for 50 GB mmapped corpora. If you need
   100 M+ nodes and distributed indexing, the `Index` trait needs a sharding story
   now rather than retrofitted. *I lean single-node; say if not.*
4. **GPU in the core?** Currently: embeddings and generation go through Python
   (`torch`/MPS), Rust stays CPU-only. A Rust GPU path (`candle`) would cut a hop
   but adds real complexity. *I lean no, for now.*
5. **External vector stores in milestone 2, or later?** *I lean later* — a fast
   local index first keeps the `Index` trait honest, and adapters are easy once
   the trait has survived contact with two real implementations.
6. **Multi-tenancy / multi-user.** Assuming single-user local research. If runs
   need to be shared across a team beyond the W&B/MLflow mirroring, the artifact
   store needs a remote backend. *Say the word and I will spec it.*

---

## 11. Proposed build order

Nothing here is started — this is what I would do once you sign off.

| # | Milestone | Contents | Exit criterion |
|---|---|---|---|
| 1 | **Skeleton** | workspace, crates, maturin build, CI, stub generation | `import strata` works from a clean checkout on macOS + linux |
| 2 | **Core + STAIR parity** | corpus store, `Node`/`Edge`, engine, BM25, trie, metrics, significance, RunCard | golden test reproduces `stair-rag` SearchTome numbers |
| 3 | **Ops** | trace stream, budget enforcement, cost ledger, MLflow + W&B exporters, `ops push`, `.plot` | a training run mirrors live to W&B and replays offline |
| 4 | **Generation + verification** | `Compose`, `Generate`, `Verify`, jev backend, citation metrics | semantic hallucination measured end to end |
| 5 | **Loop forms** | `Policy`, Iterative, Branching; Self-RAG + agentic strategies | an agentic system compared against STAIR in one `.compare()` |
| 6 | **Graph + scale** | `Edge` indexing, GraphRAG, external store adapters, sharding if needed | GraphRAG on a real corpus |

Milestone 2 is the one that matters. If Strata cannot reproduce `stair-rag` from
the same corpus, the abstraction is wrong and it is far cheaper to learn that at
milestone 2 than at milestone 5.
