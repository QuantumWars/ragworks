# Wave 1 findings

Systems: `naive`, `hybrid_rrf`, `sure` (+ `sure_norerank` ablation).
Task: `hotpotqa-validation-150-unans` — 1,461 deduplicated paragraphs, 150 queries,
15 of them made genuinely unanswerable by removing their supporting paragraphs.
Retrieval metrics are computed over the 135 answerable queries.

## Results

| system | recall@1 | recall@5 | nDCG@10 | MRR | coverage | risk | AURC | usd | p95 s |
|---|---|---|---|---|---|---|---|---|---|
| naive | 0.411 | 0.748 | 0.779 | 0.885 | 1.000 | 0.260 | 0.186 | 0 | 0.010 |
| hybrid_rrf | 0.407 | 0.767 | 0.791 | 0.885 | 1.000 | 0.267 | 0.175 | 0 | 0.013 |
| sure_norerank | 0.411 | 0.748 | 0.779 | 0.885 | 0.660 | **0.131** | 0.102 | 0.010 | 0.890 |
| **sure** | **0.467** | **0.822** | **0.855** | **0.961** | 0.647 | **0.062** | **0.040** | 0.013 | 1.016 |

Paired randomization tests on recall@5 (n=135):

| comparison | diff | p |
|---|---|---|
| naive → hybrid_rrf | +0.019 | **0.4254** |
| naive → sure_norerank | +0.000 | 1.0000 |
| sure_norerank → sure | **+0.074** | **0.0001** |

## F1 — Hybrid RRF bought nothing measurable

+0.019 recall@5 at p=0.43, and MRR moved +0.001. On this task, fusing BM25 with
dense retrieval is indistinguishable from dense alone.

This is the control group earning its place on the first run. Reported alone,
`hybrid_rrf`'s 0.767 recall@5 would have read as a result. It isn't one.
Generalise carefully: one task, 135 queries, one embedding model.

## F2 — The two jev mechanisms are independent, and the headline number was confounded

`sure` initially looked like +0.074 recall@5 over `naive` (p=0.0001). The ablation
shows that gain is **entirely** from jev *reranking* — `sure_norerank` scores
identically to `naive` (diff 0.000, p=1.0), because it uses the same retriever and
only adds a verdict.

The decomposition:

- **jev reranking** → all of the retrieval gain: +0.074 recall@5, MRR 0.885 → 0.961.
- **jev sufficiency + abstention** → risk 0.260 → 0.131 at 66% coverage, with
  retrieval quality untouched.
- **Both together** → risk 0.062 (4.2× lower than naive) and AURC 0.040 (4.6× better).

They compose rather than overlap: better reranking produces a better evidence set,
which makes the sufficiency judgement more accurate in turn. Two mechanisms, two
different jobs — one improves *what you find*, the other *whether you answer*.

Had we skipped the ablation we would have attributed a reranking win to the
sufficiency architecture. This is the concrete argument for R39.

## F3 — jev detects missing hops, verified case by case

Four answerable queries, comparing jev's verdict against whether all gold
paragraphs were actually inside the evidence window:

| gold ranks | all in window (8) | verdict | correct |
|---|---|---|---|
| [0, 9] | no | `insufficient` | ✓ |
| [0, 5] | yes | `supports` | ✓ |
| [1, 2] | yes | `supports` | ✓ |
| [0, 88] | no | `insufficient` | ✓ |

4/4. The pattern underneath is the classic multi-hop failure: dense retrieval finds
hop one at rank 0 and misses hop two at rank 9 or 88. jev abstains instead of
answering from half the evidence.

This is SURE-RAG's central claim — *"missing hops and unresolved conflicts cannot
be detected by independent passage scoring"* — reproduced on our own data, and a
direct validation of **R32 (set-level verification)**. A per-passage scorer would
have seen a strong rank-0 hit and proceeded.

Cost: **$0.013 for 150 queries** (~$0.087 per 1,000), p95 1.0 s. jev-as-verifier is
cheap enough to be always-on.

## F4 — A requirement no paper gave us

**A sufficiency verdict is meaningful only relative to the evidence window it
judged.** `insufficient` at window 4 and `insufficient` at window 8 are different
claims, and the first smoke test misread a correct verdict as miscalibration
precisely because the window was invisible.

`n_evidence` is therefore not a tuning knob but part of the verdict's meaning, and
must be recorded alongside it. Nothing in `CONFORMANCE.md` R1–R47 captures this —
it came out of implementation, which is what `r-d` is for.

→ **Candidate R48: verification results carry the evidence window they were
computed over; a verdict without it is uninterpretable.**

## Abstractions that recurred

Evidence for the library, from systems that were forbidden to share code:

| Pattern | Appeared in | Reading |
|---|---|---|
| embed → normalise → matrix → argsort | naive, hybrid_rrf, sure (**3/3**) | a shared dense index is clearly load-bearing |
| shortlist at depth, then rescore | hybrid_rrf (RRF), sure (jev) (**2/3**) | confirms Rerank as `Vec<Candidate> → Vec<Candidate>` (R21) |
| abstention / verdict / confidence | sure only (**1/3**) | correctly optional on `Response`; would have been wrong to force |
| per-query error isolation, resumability | harness | no failures yet — unproven, keep but do not over-build |

The `Response` contract held without modification across a pure retriever, a
fusion system and a generate-and-verify pipeline. No system asked for a field it
didn't have.

## Caveats

- One task, one embedding model, 135 scored queries. Nothing here generalises yet.
- The unanswerable condition is **synthetic** — we removed supporting paragraphs.
  Natural unanswerability may behave differently.
- `hit@5` is 0.985–0.993 across all systems, so this task is near-saturated at k=5;
  recall@1 and MRR are the discriminating metrics here, not hit rate.
