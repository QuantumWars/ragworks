# Wave 3 — rerankers and verifiers

Both remaining traits now have a model-backed implementation and an offline
baseline. The baselines are the point: a judge that beats no baseline has not
been shown to do anything.

## Reranking, measured

`hotpotqa-validation-150-unans`, 135 scored queries. The retrieval leg is
identical across all three rows — same model, same top-10 shortlist — so the
only variable is the reranker.

| system | recall@1 | recall@5 | nDCG@10 | MRR | hit@5 | risk | AURC | p95 |
|---|---|---|---|---|---|---|---|---|
| no rerank | 0.411 | 0.748 | 0.779 | 0.885 | 0.985 | 0.260 | — | 8 ms |
| lexical rerank | 0.415 | 0.767 | 0.791 | 0.899 | 0.985 | 0.253 | 0.155 | 8 ms |
| **typed rerank** | **0.456** | **0.870** | **0.861** | **0.950** | **1.000** | **0.180** | **0.075** | 614 ms |

| comparison | diff | p |
|---|---|---|
| none → lexical | +0.019 | **0.5498** |
| none → typed | **+0.122** | **0.0001** |
| lexical → typed | **+0.104** | **0.0001** |

## F8 — the gain is understanding, not term overlap

**Lexical reranking bought nothing measurable** (p=0.55), which is the same
result hybrid fusion gave in wave 1. BM25 rescoring of a shortlist that a dense
retriever already produced adds no information: both are matching roughly the
same signal.

The typed reranker gains +0.104 *over that baseline*, at p=0.0001. Without the
baseline the honest summary would have been "reranking helps by 0.122" with no
way to say whether that was cheap term matching or real semantic judgement. It
is the latter.

`hit@5` reaches **1.000**: for every one of the 135 queries, at least one gold
paragraph was in the top five.

Cost is latency, not money — p95 goes from 8 ms to 614 ms, one round trip per
query regardless of shortlist size, because every question in a request is
scored against the same state in parallel.

## F9 — the coverage baseline is fooled exactly where it matters

A worked example, five passages about database indexes:

| evidence | typed verifier | coverage baseline |
|---|---|---|
| all five passages | `Supports` | `Supports` |
| **answer passage removed** | **`Insufficient`** | **`Supports`** |

With the answering passage deleted, the remaining text still contains every
word of the query — "database", "index", "reads", "writes" — so term coverage
still reports support. The typed verifier detects that no passage establishes
the mechanism.

This is the failure mode SURE-RAG describes, reproduced against our own
baseline: *"missing hops and unresolved conflicts cannot be detected by
independent passage scoring"*. It is also why `coverage` **never returns
`Refutes`**. Term matching cannot see contradiction, and a baseline that
claimed to would misreport what it does.

## F10 — lexical reranking inverts the ranking on a paraphrase

From the live check, query *"Why does a database index make reads faster but
writes slower?"*:

| passage | lexical | rank | typed | rank |
|---|---|---|---|---|
| B-tree explanation (the answer) | 3.296 | 1 | **0.970** | **0** |
| Redis caching, unrelated | **6.121** | **0** | 0.030 | 2 |
| space-time tradeoff, pure paraphrase | 0.134 | 3 | **0.840** | **1** |

The caching passage wins on term overlap because it literally contains "faster
reads". The paraphrase, which explains the mechanism without sharing
vocabulary, ranks third lexically and second under the typed judge. Both
errors are the same error, and it is the one reranking exists to fix.

## Shared plumbing, extracted on the second need

HTTP, retry and rate limiting moved out of the embedder crate into
`ragworks-net` when the judge needed the same three pieces. Two unrelated
components requiring identical code is what justifies a shared crate; the
alternative was a judge crate depending on an embedder crate, which describes
no real relationship.

## Caveats

- One task, 135 scored queries, one shortlist depth (10). A reranker cannot
  reach a document outside the shortlist it is given, and wave 1 measured gold
  documents sitting at rank 88.
- `risk` improves from 0.260 to 0.180, but no system here abstains; that number
  is error rate at full coverage, not selective prediction.
- The typed judge is not a security boundary. Its own documentation states that
  adversarial state is not treated as hostile, and retrieved passages are
  attacker-influenced by definition.
