# Candidate assessment — which RAG systems are worth implementing

> Research on public data: pros, cons, code availability, cost and risk for each
> candidate. Feeds the choice of what `r-d/systems/` implements first.
>
> Companion to [`../CONFORMANCE.md`](../CONFORMANCE.md) (requirements R1–R47).

| | |
|---|---|
| Date | 2026-09-20 |
| Sources | papers, GitHub API (code availability), reproducibility studies, cost analyses |
| Source quality | **A** = paper/peer-reviewed · **B** = vendor blog or secondary — marked inline |

---

## 1. The criterion: "worth implementing" ≠ "best performing"

`r-d/` exists to discover which abstractions are load-bearing, not to crown a winner.
So a candidate's value is:

```
    architectural information gained
    ────────────────────────────────
         build cost + risk
```

A state-of-the-art system that exercises nothing new is **less** valuable here than a
mediocre one that breaks the kernel. Graph-O1 is the clearest case: its numbers are
not the point — the fact that it needs tree search and a steppable environment is.

This reframing matters because three findings below make chasing reported numbers a
bad use of time.

---

## 2. Three findings that dominate the decision

### 2.1 Reproducibility is worse than assumed — `A`

A study of **85 LLM-centric papers** found that of the 18 providing research
artifacts, only five were complete and executable, and **none could be fully
reproduced** — two partially, three not at all.

A [SIGIR 2026 reproducibility study of MetaRAG](https://doi.org/10.1145/3805712.3808551)
found the pattern precisely: **relative improvements over baselines held, absolute
scores did not**, attributed to closed-source LLM updates, missing implementation
details and unreleased prompts.

> **Consequence.** Do not target any paper's absolute numbers as an acceptance test.
> The exception is STAIR, where the reference implementation is ours and the corpus
> is fixed. Everywhere else, treat *relative ordering against our own baselines* as
> the only trustworthy signal. This is also why §4's control group is non-negotiable.

### 2.2 Cost multipliers are large and non-linear — `A` for agentic, `B` for GraphRAG

| Family | Multiplier vs single-pass | Source |
|---|---|---|
| Agentic / multi-step | **3–10× tokens, 2–5× latency**; p99 into tens of seconds | `A` |
| Disco-RAG | **3–4×** (extra LLM calls for RST parsing, graph, planning) | `A` — its own paper |
| GraphRAG indexing | **10–40×** ($50–200 vs $5–20 per corpus) | `B` vendor analysis |
| GraphRAG indexing (original MS) | ~$1,544/M tokens vs ~$1.45/M for vector RAG | `B` — LazyGraphRAG benchmark, via secondary source |

Costs "grow non-linearly, causing serious efficiency bottlenecks." LazyGraphRAG has
since pulled graph indexing back toward vector-RAG parity, so the extreme figure is
historical, not current — but the ordering stands.

### 2.3 There are real negative results — `A`

Two findings worth more than most positive ones, because nobody publicises them:

- **Query decomposition yields consistent gains in structured domains but *degrades
  ranking precision* on multi-hop benchmarks.** It is domain-dependent, not a
  general improvement.
- **Reflection improves citation accuracy at substantial latency cost, for
  inconsistent quality gains.**

Both are mechanisms we would otherwise have assumed were free wins. They are
hypotheses `r-d` should test rather than inherit.

### 2.4 Our own budget constraint — `A`, measured here

LLM calls run on OpenRouter's free tier at roughly **20 requests/minute**. A
1,000-query evaluation of a multi-step system at ~5 calls/query is 5,000 calls —
**over four hours of wall clock**, before retries. This is a hard planning
constraint, not a footnote:

- cap evaluation query counts per system;
- single-pass systems are effectively free to evaluate, multi-step ones are not;
- MCTS-based retrieval is likely **infeasible at full scale here** without a paid
  tier or a local model.

---

## 3. Code availability — measured via GitHub API, 2026-09-20

The single strongest predictor of whether a reimplementation succeeds.

| System | Repository | Stars | Status |
|---|---|---|---|
| **A-RAG** | `Ayanami0730/arag` | **352** | official |
| **HGMem** | `Encyclomen/HGMem` | **131** | official |
| **MegaRAG** | `AI-Application-and-Integration-Lab/MegaRAG` | 69 | official, ACL 2026 |
| **QuCo-RAG** | `ZhishanQ/QuCo-RAG` | 47 | official |
| **Disco-RAG** | `dongqi-me/Disco-RAG` | 14 | official |
| **MG²-RAG** | `Daboolu/MG2-RAG` | 8 | official, ECCV 2026 |
| **SURE-RAG** | `mouwumou/SURE-RAG` | 3 | thin |
| **MiA-RAG** | `ahadkhan9/MiA-RAG` | 0 | **third-party**, not authors' |
| Graph-O1 | — | — | **none found** |
| FT-RAG | — | — | **none found** |
| TV-RAG | — | — | **none found** |
| Bidirectional RAG | — | — | **none found** |
| HiFi-RAG | — | — | **none found** |
| Predictive Prefetching | — | — | **none found** |
| RAGPart / RAGMask | — | — | **none found** |

**Seven of fifteen checked have no public code.** Combined with §2.1, a no-code
system is a research project, not an implementation task — and should only be taken
on when the *architectural* lesson justifies it independently of the paper's numbers.

---

## 4. Per-candidate assessment

### Tier 1 — build first

**Naive + Hybrid RRF** · no paper needed · cost ×1
*Pro:* the control group. Without it every later number is uninterpretable, and §2.1
says relative ordering against our own baselines is the only trustworthy signal.
Exercises the harness end to end before anything complex depends on it.
*Con:* no novelty. Skipping it is the single most common evaluation mistake.
**Teaches:** R21, R36, R38 · **Verdict: build, first.**

**STAIR** · ours · cost ×1 (after training)
*Pro:* ports from an existing 8.5k-LOC reimplementation kept in a separate,
unpublished repository, and it is the **only candidate with numbers we
can legitimately treat as an acceptance test** — same corpus, same code lineage. That
makes it the validator for the harness itself. Unique mechanism (trie-constrained
generative retrieval) not covered by any other candidate.
*Con:* needs a fine-tuned model; single-hop only.
**Teaches:** R4, R7 + harness correctness anchor · **Verdict: build, first.**

**SURE-RAG** · [2605.03534](https://arxiv.org/abs/2605.03534) · code thin (3★) · cost ×1.2
*Pro:* the only candidate testing abstention, set-level sufficiency and calibration
(R31–R34) — a whole region of the requirement space nothing else touches. Mechanism
is simple enough that thin code doesn't matter. **Directly tests the jev bet**: its
three-way supports/refutes/insufficient verdict is a jev `choice` with three options,
and `confidence` feeds the risk-coverage curve. Reported 0.9075 Macro-F1 vs 0.7284
for a GPT-4o judge.
*Con:* needs abstention-aware metrics in the harness from day one — which is a
feature, since retrofitting those later is painful.
**Teaches:** R31, R32, R33, R34 · **Verdict: build, first.**

**A-RAG** · [2602.03442](https://arxiv.org/abs/2602.03442) · **352★ official** · cost ×2–4
*Pro:* best-supported code of any candidate. It is **the RAG↔agent seam** — retrievers
exposed as keyword/semantic/chunk-read tools — so it informs the agent layer while
still being a RAG system. Multi-granularity retrieval (R8) is demanded independently
by MG²-RAG and FT-RAG, so building it once pays three times.
*Con:* agentic, so §2.2's 3–10× cost applies; rate limits bite.
**Teaches:** R8, R20, R44 · **Verdict: build, first.**

### Tier 2 — build after Tier 1 lands

**HGMem** · [2512.23959](https://arxiv.org/abs/2512.23959) · **131★ official** · cost ×3–10
*Pro:* the only test of **working memory** (R6, R9) — a concept `DESIGN.md` v1 was
missing outright, so the kernel cannot be designed without it. Good code.
*Con:* multi-step; expensive to evaluate under our rate limit. Cap query counts.
**Teaches:** R6, R9 · **Verdict: build second.**

**MiA-RAG** · [2512.17220](https://arxiv.org/abs/2512.17220) · third-party code only · cost ×1.5
*Pro:* tests the conditioning side-channel (R16), independently demanded by Disco-RAG
— two papers needing one mechanism makes it a real requirement. Concept
(hierarchical summarisation conditioning both retriever and generator) is
implementable straight from the paper.
*Con:* no official code; a 0-star third-party implementation is not a reference.
Hierarchical summarisation over a long corpus costs real tokens up front.
**Teaches:** R4, R16 · **Verdict: build second, from the paper, not the repo.**

### Tier 3 — architectural probe, not reproduction

**Graph-O1** · [2512.17912](https://arxiv.org/abs/2512.17912) · **no code** · cost ×10+
*Pro:* the sole falsifier of "three loop forms are exhaustive", and the only system
demanding a steppable environment (R12, R13). That lesson is essential to the kernel.
*Con:* no code, MCTS token cost, **plus end-to-end RL training** — very likely
infeasible here at full scale (§2.4).
**Verdict: build a simplified MCTS retrieval loop without RL**, purely to test whether
the loop-form and environment abstractions hold. Do not attempt to reproduce the
paper's numbers, and say so in the findings. The architecture question is answerable
far more cheaply than the performance question.

**Disco-RAG** · [2601.04377](https://arxiv.org/abs/2601.04377) · 14★ official · cost ×3–4
Good code, but its two lessons (R7 dual structure, R16 conditioning) **overlap
MiA-RAG**, at 3–4× cost. **Verdict: defer** unless R7 specifically proves contentious.

**MegaRAG** · 69★ · and **MG²-RAG** · 8★ — both official, both multimodal.
Between them they demand R1, R2, R5, R8, R10, R11, R22, R23. That is the largest
untested block of the requirement set, and you chose full multimodal scope.
**Verdict: defer to a multimodal wave, after the text systems settle the kernel.**
Deferring is safe only because the *requirements* are already recorded.

**QuCo-RAG** · 47★ official · *Pro:* good code, cheap. *Con:* teaches only R24
(external statistics backend) and depends on an Infini-gram service over pre-training
data. Small architectural payoff. **Verdict: defer; cheap to add later.**

### Skip for now

**TV-RAG, Bidirectional RAG, HiFi-RAG, Predictive Prefetching, RAGPart/RAGMask,
FT-RAG** — no public code (§3), and each needs infrastructure that is premature:
video decoding, corpus write-back with lineage, token-level generation hooks, index
partitioning. Their requirements are already captured in `CONFORMANCE.md`; building
them now would be designing the kernel around untested speculation.

---

## 5. Recommended slate

| Wave | Systems | Why | Cost |
|---|---|---|---|
| **1** | naive, hybrid RRF, STAIR, SURE-RAG | control group + harness validator + abstention/verification + the jev bet | ×1–1.2 |
| **2** | A-RAG, HGMem | the agent seam and working memory — the two missing kernel concepts | ×3–10 |
| **3** | MiA-RAG, simplified Graph-O1 | conditioning channel, tree-search loop form | ×1.5 / ×10 |
| later | MegaRAG, MG²-RAG, Disco-RAG, QuCo-RAG | multimodal wave, overlapping lessons | — |

Wave 1 is cheap, fully buildable, and covers the two things needed before anything
else means anything: a control group and a harness validated against known numbers.

**Coverage check.** Waves 1–3 exercise R4, R6, R7, R8, R9, R12, R13, R16, R20, R21,
R31, R32, R33, R34, R36, R38, R44 — **17 of 47**. The untested remainder is dominated
by multimodal (R1–R3, R5, R10, R11, R22, R23) and corpus lifecycle (R26–R30), which is
exactly what waves 4+ are for. No wave-1 decision forecloses them.

---

## 6. What this changes

- **Do not treat published numbers as acceptance tests** (§2.1). Only STAIR gets one.
- **Query decomposition and reflection are hypotheses to test, not wins to inherit**
  (§2.3). Both should be ablations in `r-d`, not assumptions.
- **Rate limits are an architectural constraint** (§2.4), not an operational detail.
  The harness needs per-system query caps and resumable runs from the start.
- **Seven of fifteen systems have no code.** Where we build them, we are testing the
  *architecture*, not replicating the paper — and the findings must say which.
