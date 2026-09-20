# ragworks — the library

Rust core for RAG infrastructure. Every stage a developer swaps is a trait with
a registry in front of it, so pipelines are configuration rather than code.

> `ragworks` is a **working name**, free on crates.io and PyPI as of 2026-09-20.
> Renaming is a `sed` across `Cargo.toml` files; do it before the first publish.

## Status

| Crate | State |
|---|---|
| `ragworks-core` | corpus, errors, plugin registry, tokenizers, swap-point traits |
| `ragworks-chunk` | `fixed`, `recursive`, `markdown` |
| `ragworks-index` | `bm25`, `flat` dense, RRF fusion |
| `ragworks-py` | PyO3 bindings, abi3 wheel — `Corpus`, `Chunker`, `Bm25`, `Flat`, `rrf`, `catalogue` |
| `ragworks-read` | not started |
| `ragworks-embed` | not started |

**54 tests, 0 clippy warnings.**

Against the Python BM25 in `r-d`, on 2,964 paragraphs and 300 queries:
**100% top-1 agreement, 1.000 set overlap@10, 21.7× faster search, 3.2× faster
indexing.** See [`r-d/findings/wave1_5_rust.md`](../r-d/findings/wave1_5_rust.md).

```sh
cd crates/py && maturin develop --release --uv
```

```sh
cargo test                                          # everything
cargo run -p ragworks-chunk --example catalogue     # the plugin system
cargo run --release -p ragworks-index --example pipeline
```

Measured on an M1 Max, 50,000 synthetic documents of 40 tokens:

```
index   0.20s  (252,838 docs/s)
search  1.56 ms/query
```

Search is measured on the worst case — single-term queries matching roughly an
eighth of the corpus, so nearly every posting is scored. Real multi-term queries
touch fewer documents each.

## What a plugin looks like

Implement the trait, declare a config, register it. `schemars` derives the JSON
Schema; the registry handles construction, validation and discovery.

```rust
#[derive(Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct MyConfig { pub size: usize }

impl Component for MyChunker {
    type Config = MyConfig;
    const NAME: &'static str = "mine";
    const SUMMARY: &'static str = "One line for the catalogue.";
    fn build(config: Self::Config) -> Result<Self> { /* validate here */ }
}

impl Chunker for MyChunker {
    fn name(&self) -> &'static str { Self::NAME }
    fn chunk(&self, doc: DocView<'_>, out: &mut Vec<Chunk>) -> Result<()> { ... }
}

registry.register::<MyChunker>(|c| Box::new(c))?;
```

`registry.describe()` then emits it with its schema — which is what generates
docs, validates YAML pipelines and will generate the Python stubs.

## Design decisions, and why

**Chunks are views, not copies.** agno's `ChunkingStrategy.chunk()` returns new
`Document` objects owning fresh copies of their text; chunking 1 GB at 20%
overlap allocates over a gigabyte of duplicated strings. A `Chunk` here is 56
bytes — a doc id, two spans, and its place in the tree. The size is asserted by
a test, because at 100M chunks every 8 bytes is 800 MB.

**Output buffers are passed in.** `chunk(doc, &mut out)` appends, so indexing a
million documents reuses one allocation. Same for embedders, which write a
row-major `Vec<f32>` rather than a `Vec<Vec<f32>>` — one allocation per batch,
and the result can go straight to BLAS or be mmapped.

**One structure type covers flat and hierarchical.** `Chunk` carries `parent`,
`depth` and `label_span`. A fixed-size chunker leaves them empty; the markdown
chunker builds a heading tree. Flat chunking is the degenerate case of
structured chunking, so one index holds both — and several granularities at
once.

**Errors classify themselves.** Taken from agno, which normalises every
provider exception into one type preserving `status_code`. `Error::is_retryable`
distinguishes a rate limit from an auth failure, which against a rate-limited
free tier is the difference between a run that finishes and one that burns its
quota retrying a 401.

**Bad config fails at build time.** `deny_unknown_fields` plus validation in
`build()` means a typo is a named error before indexing starts, not a silently
ignored setting discovered an hour in.

**Config is JSON, not generics.** Traits stay object-safe so `Box<dyn Chunker>`
works and plugins can be chosen at runtime from a file. A component embeds
another through `PluginSpec`, which accepts either `"simple"` or
`{"name": "simple", "config": {...}}` — so a BM25 index names its tokenizer
without knowing which one it will get.

**No half-built state.** BM25 maintains collection statistics as documents
arrive and computes IDF per query term at search time, so `finish()` is a no-op
and "indexed but not finished" cannot happen. Removing an invalid state beats
guarding it.

## Layout

```
crates/core/src/
  corpus.rs   Corpus, Doc, DocView, Chunk, Span, DocId
  error.rs    Error, ProviderFault, retry classification
  plugin.rs   Component, Registry, schema generation
  traits.rs   Chunker Reader Embedder VectorStore Reranker Verifier
crates/core/src/
  tokenize.rs Tokenizer trait + `simple`
crates/chunk/src/
  fixed.rs recursive.rs markdown.rs
crates/index/src/
  bm25.rs   Okapi BM25 inverted index
  flat.rs   exact dense search, cosine or dot
  fuse.rs   reciprocal rank fusion
```

`traits.rs` declares every swap point, including ones with no implementation
yet, so the shape of the whole library is visible and checked by the compiler
before it is filled in.
