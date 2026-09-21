# Wave 4 — query construction and reformulation

Five transforms measured on `hotpotqa-validation-150-unans`, 135 scored
queries, retrieval leg held identical. **None of them helped. Two of them
significantly hurt.**

## Results

| transform | recall@1 | recall@5 | nDCG@10 | MRR | hit@5 | p95 | wall |
|---|---|---|---|---|---|---|---|
| **identity** | **0.411** | 0.748 | **0.779** | **0.885** | **0.985** | **8 ms** | **13 s** |
| rm3 | 0.330 | 0.733 | 0.729 | 0.790 | 0.963 | 13 ms | 9 s |
| hyde | 0.407 | 0.748 | 0.779 | 0.872 | 0.948 | 4318 ms | 436 s |
| multi_query | 0.389 | 0.752 | 0.765 | 0.861 | 0.970 | 4156 ms | 450 s |
| decompose | 0.359 | 0.733 | 0.740 | 0.810 | 0.948 | 4251 ms | 441 s |

Paired randomization tests against `identity`:

| transform | recall@1 | p | MRR | p | recall@5 | p |
|---|---|---|---|---|---|---|
| rm3 | **−0.081** | **0.0005** | **−0.095** | **0.0001** | −0.015 | 0.46 |
| decompose | **−0.052** | **0.0019** | **−0.074** | **0.0005** | −0.015 | 0.55 |
| multi_query | −0.022 | 0.11 | −0.023 | 0.09 | +0.004 | 1.00 |
| hyde | −0.004 | 1.00 | −0.013 | 0.54 | +0.000 | 1.00 |

## F11 — decomposition degrades ranking precision, as published

`CANDIDATES.md` §2.3 recorded a negative result from the literature: *"query
decomposition yields consistent gains in structured domains but degrades ranking
precision on multi-hop benchmarks."*

Reproduced on our own data: recall@1 **−0.052 (p=0.0019)** and MRR **−0.074
(p=0.0005)**. Recall@5 barely moves, which is the shape the published claim
describes — the right documents are still found, they are ordered worse.

The mechanism is visible in the outputs. Asked to decompose *"What government
position was held by the woman who portrayed Corliss Archer in Kiss and Tell?"*,
the model returned the original question plus **"Shirley Temple"** — an answer,
not a sub-question. Retrieving for a bare entity name pulls in everything about
that entity, and fusing that run with the original dilutes a ranking that was
already correct.

## F12 — relevance feedback hurt the most, and the simplification is why

RM3 was the worst performer: recall@1 **−0.081 (p=0.0005)**. It is also the
cheapest and the only offline one, so this is not a cost-benefit judgement —
it made retrieval worse.

The implementation documents its own simplification: textbook RM3 weights
expansion terms and interpolates them with the original query model, but the
transform interface produces a query *string*, which carries no weights, so ten
terms are appended unweighted. Appending ten unweighted terms to a question
dilutes it — every expansion term counts as much as the entity the question is
actually about.

That is a finding about the **interface**, not about relevance feedback. A
transform that returns `Vec<String>` cannot express a weighted query model. If
feedback expansion is worth pursuing, the trait needs a weighted output type.

## F13 — the headroom is real, but no transform here could reach it

`identity` already retrieves at least one gold paragraph for **98.5%** of
queries, which looks like no room to improve. Looking closer:

| | queries |
|---|---|
| no gold paragraph in top-5 | **2 / 135** |
| **only part of the gold in top-5** | **64 / 135** |
| complete gold in top-5 | 69 / 135 |

Nearly half the queries retrieve one supporting paragraph and miss the other.
That is exactly the rank-88 failure from wave 1, and it is substantial headroom.

**None of these five transforms can address it.** Four rewrite the question in
isolation — paraphrases, a hypothetical answer, sub-questions — and the missing
second hop is not similar to the question. It is similar to something only the
*first hop* reveals. RM3 is the one transform that reads retrieved text, and it
reads it as a bag of words.

What the measurement says to build is a feedback transform that reads the first
hop's content and asks the *next* question. `QueryTransform::uses_feedback`
already carries that shape; nothing implemented here uses it well.

## F14 — reasoning tokens silently broke generation

Before measurement could run at all, `multi_query` failed on every free model
with "empty completion". Cause: reasoning models spend the token budget
deliberating and emit no content. Measured on one model, **414 reasoning tokens
against a 400-token budget, zero content**.

Disabling reasoning fixed it and was strictly better for this task:

| model | reasoning on | reasoning off |
|---|---|---|
| nex-n2.5-mini | 2.1 s, 117 tokens | **0.8 s, 35 tokens** |
| ling-3.0-flash-vl | 4.3 s, **0 content** | **1.0 s, works** |

End to end, HyDE went from 18.6 s to 0.7 s per query. Rewriting a query does not
benefit from deliberation, so `disable_reasoning` defaults to true, and the
error message now reports the reasoning-token count so the next person diagnoses
it in one read.

## The pattern across four waves

| intervention | effect |
|---|---|
| hybrid BM25+dense fusion | +0.019 recall@5, p=0.43 |
| lexical reranking | +0.019 recall@5, p=0.55 |
| query reformulation (best of five) | +0.004 recall@5, p=1.00 |
| **typed reranking** | **+0.104 recall@5, p=0.0001** |
| **typed set-level verification** | **risk 0.260 → 0.062** |

Everything that manipulates the *query or the ranking lexically* has produced
nothing on this task. The only interventions that moved anything read the text
and judged it. That is one task and should not be over-generalised, but it is
four independent attempts pointing the same way.

## Caveats

- One task, 135 scored queries, one embedding model. HotpotQA's distractor pool
  makes most gold paragraphs reachable; a corpus where retrieval genuinely fails
  would give query transforms more room.
- The LLM transforms ran on free-tier models that visibly struggled with the
  instructions — `decompose` returned answers instead of sub-questions. A
  stronger model would likely change these numbers, and this is a measurement of
  *these transforms with these models*, not of the techniques in principle.
- Cost was not the deciding factor but is worth recording: 436-450 s versus 13 s,
  roughly 35x, for a null-to-negative result.
