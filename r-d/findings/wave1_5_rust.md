# Wave 1.5 — the Rust core, measured against the Python one

`ragworks` (Rust, via PyO3) now runs inside `r-d`. This note records what the
integration proved, what it cost, and one bug it exposed.

## Speed, on the same 2,964-paragraph corpus and 300 queries

| | python | rust | |
|---|---|---|---|
| index | 129.8 ms | 41.0 ms | **3.2×** |
| search (300 queries) | 1129.3 ms | 52.0 ms | **21.7×** |
| per query | 3.764 ms | 0.173 ms | |

## Agreement — checked before speed was allowed to matter

A faster implementation that ranks differently is not faster, it is different.

| | |
|---|---|
| identical top-1 | **300/300 (100%)** |
| identical top-10 ordering | 298/300 (99.3%) |
| mean set overlap@10 | **1.000** |

The two remaining order swaps are the `İ` case in §F6, not a scoring difference.

## F5 — a real bug, and a wrong diagnosis on the way to it

The first run showed 100% top-1 agreement but three documents ordered
differently, with score gaps of 0.20 and 0.10 — far too large for f32 rounding.

**The wrong turn.** The disagreeing documents contained Hebrew with niqqud, so I
concluded combining marks were splitting words and rewrote the tokenizer on
UAX #29. Agreement fell from 99% to **55%**. The hypothesis was wrong and had
been adopted without testing it.

**What it actually was**, once measured across all 2,964 documents: exactly
**five** documents tokenize differently, from two causes.

| cause | documents | example |
|---|---|---|
| `_` is a word character in `\w` but not `char::is_alphanumeric` | 4 | `united_states`, `formula_1` |
| Python lowercases *before* tokenizing; `İ` → `i` + combining dot | 1 | `İzmir` |

Five documents out of 2,964 were enough to shift the **document frequency** of
common terms, and therefore the IDF of every query using them, and therefore the
score of documents that never contained the affected text at all. The Hebrew
documents were not special — they merely contained the affected terms.

**The lesson is about method, not Unicode.** An independent reference
implementation localised the fault in one step; a plausible story about the data
sent me the wrong way and cost a rewrite. Differential testing against a
reference is what made this findable.

## F6 — two tokenizers, and the measurement that chose between them

`simple` (alphanumeric runs plus `_`, matching the conventional `\w+`) and
`unicode` (UAX #29: keeps `don't`, combining marks and CJK together).

Rather than argue, both were run through the harness on
`hotpotqa-validation-150-unans`:

| system | recall@1 | recall@5 | MRR |
|---|---|---|---|
| rust_bm25 (`simple`) | 0.389 | 0.741 | 0.863 |
| rust_bm25 (`unicode`) | 0.396 | 0.748 | 0.871 |

**diff +0.007, p = 0.6256.** Inconclusive. The choice cannot be made on this
task's numbers, so it should be made on principle — `unicode` is correct for
multilingual text by construction — while `simple` stays the default because it
keeps results comparable with the rest of the BM25 literature. Both ship.

## F7 — lexical retrieval matches dense on this task

| comparison | diff | p |
|---|---|---|
| naive (dense) → rust_bm25 | −0.007 | **0.8986** |

BM25 alone is statistically indistinguishable from a dense bi-encoder on
HotpotQA, at roughly 20× lower latency and zero model cost. Combined with F1
(hybrid fusion bought nothing, p=0.43), the wave-1 picture is that **on this
task the retrieval mechanism barely matters** — only the reranking and
abstention layers moved anything, and both came from jev.

Same caveat as wave 1: one task, 135 scored queries, `hit@5` already at 0.97–0.99.

## Cost of the integration

- One macOS build problem: a release `cdylib` without debug info is rejected by
  dyld with "mis-aligned LINKEDIT string pool". Bisected — debug builds load,
  every release build without debug info fails, `-ld_classic` does not help.
  Fixed with `debug = 1` in the release profile, recorded in `Cargo.toml`.
- PyO3 0.29 renamed `Python::allow_threads` to `Python::detach`.
- `maturin`'s `strip = true` was an early suspect and is *not* the cause; it was
  left off anyway since the profile now keeps line tables.
