# Fixing the offline hashing embedder

The `hashing` embedder shipped unable to match morphological variants. This
records the defect, the fix, and what measurement said about both.

## The defect

Whole-token hashing sends `writing` and `writes` to unrelated buckets, so their
cosine similarity is exactly **0.0** — not small, zero. Every morphological
variant was invisible.

It surfaced in a worked example rather than in a test. Asked *"why do indexes
make writing slower?"* over a three-section document, the embedder scored the
semantically correct section **0.0000** because the query says "writing" and the
text says "writes".

The existing unit test passed throughout. It compared sentences differing by one
literal token, which is precisely the case where whole-token hashing works. A
test whose premise excludes the failure mode is not evidence of its absence.

## The fix

Hash each token together with its character n-grams, bounded by `<` and `>` as
in FastText. `writing` and `writes` then share `<wr`, `wri` and `rit`.
Widths 3, 4 and 5 by default; `ngrams: []` restores the old behaviour exactly,
so the config is honest about what it does.

Boundary markers matter: they distinguish a prefix n-gram from the same letters
occurring mid-word, which is what keeps `sorted` apart from `resorted`.

## Word-pair check

| pair | tokens only | with n-grams |
|---|---|---|
| `writing` / `writes` | 0.0000 | **0.3441** |
| `index` / `indexes` | 0.0000 | **0.5923** |
| `sorted column` / `sorting columns` | 0.0000 | **0.5183** |
| `volcano` / `database` | 0.0000 | 0.0000 |

The last row is the other half of the fix. Subwords must not make everything
match; unrelated words still score zero.

## Retrieval measurement

Hand-picked pairs are not evidence. Measured through the harness on
`hotpotqa-validation-150-unans`, 1,461 paragraphs and 135 scored queries:

| embedder | recall@1 | recall@5 | nDCG@10 | MRR | hit@5 | risk |
|---|---|---|---|---|---|---|
| hashing, tokens only | 0.233 | 0.419 | 0.446 | 0.562 | 0.681 | 0.580 |
| **hashing, n-grams** | **0.311** | **0.622** | **0.618** | **0.727** | **0.889** | **0.440** |
| learned bi-encoder | 0.411 | 0.748 | 0.779 | 0.885 | 0.985 | 0.260 |

**+0.204 recall@5, p = 0.0001.** The offline hasher now reaches roughly 83% of a
learned bi-encoder's recall@5 at no model cost and about 5x lower latency, where
before it was not usable at all.

It remains significantly worse than the learned embedder (−0.126 recall@5,
p = 0.0001), which is the expected and correct outcome.

## A hypothesis the measurement rejected

Raising the tokenizer's `min_len` to 3 drops most English function words, and on
the three-sentence toy example it improved separation (0.3245 → 0.3415 related,
0.0316 → 0.0214 unrelated).

On the real task it did nothing: **−0.004 recall@5, p = 1.0000.** So it is not
the default. The toy example and the benchmark disagreed, and the benchmark
decided — the same discipline that caught the reranking confound in wave 1.

## What the regression tests now pin

- Morphological variants score above 0.25, with a comment recording the 0.204
  recall cost of regressing it.
- Unrelated words stay below 0.05, so a future change cannot "fix" recall by
  making everything similar.
- `ngrams: []` still yields exactly 0.0 for `writing`/`writes`, so the escape
  hatch is real rather than decorative.
- A token shorter than the n-gram window still contributes its whole-token
  feature instead of vanishing.

## Caveat that survives the fix

Ranking improved but discrimination is narrow. On the worked example the three
sections now score 0.165, 0.159 and 0.146 — the right order, with little
separating them. Common substrings are shared widely and there is no corpus to
derive inverse document frequency from. This is a tool for exercising a pipeline
offline, not for judging retrieval quality.
