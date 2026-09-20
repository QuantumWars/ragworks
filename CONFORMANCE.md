# Conformance matrix — requirements, not design

> **Step 1 of 2.** Derives what the architecture must support, from real systems,
> before any design is written. `DESIGN.md` v2 is written against this.
> Nothing here proposes a solution; everything here is a constraint.

| | |
|---|---|
| Date | 2026-09-20 |
| Source taxonomy | [Turing Post, 20 RAG types](https://www.turingpost.com/p/ragtypes) |
| Confidence | **20 of 20 at level A** — every paper located and read |
| Verdicts | 1 `EXPRESSIBLE` · 4 `EXTENDS` · **15 `BREAKS`** |
| Requirements | R1–R47 (§5) |

---

## 1. Method

**Verdicts.** `EXPRESSIBLE` = runs as a wiring of existing stages. `EXTENDS` = needs
a new stage implementation, no kernel change. `BREAKS` = needs a change to kernel
types, scheduler or storage.

**Confidence.** Level A = paper located, abstract and stated mechanism read.
Level B = one-sentence taxonomy description only. **All 20 rows are now level A.**

### 1.1 What the promotion pass cost and bought

The first pass ran at level B on 13 of 20 systems. I estimated a 2-in-7 correction
rate and recommended promoting only the top 4. **That recommendation was wrong, and
the decision to promote all 13 was correct.**

Outcome of reading all 13 papers:

| | |
|---|---|
| Rows whose requirements changed | **13 of 13** |
| Rows whose recorded verdict was wrong | **6** |
| `BREAKS` count before → after | **10 → 15** |

Six requirements were not merely incomplete but *pointing the wrong way* — §1.2.
A design written against the level-B matrix would have been built on them.

### 1.2 The six outright corrections

| System | Level-B requirement (wrong) | Level-A requirement (correct) |
|---|---|---|
| **QuCo-RAG** | statistics over *our* corpus | an **external pre-training-corpus** n-gram service (Infini-gram, 4T tokens) — nothing to do with our index |
| **HiFi-RAG** | multi-stage rerank → `EXPRESSIBLE` | **per-stage model binding** (Gemini Flash filters, Pro generates) → `EXTENDS` |
| **RAGPart/RAGMask** | ingest-time content filter | **retrieval-stage**, operating on the retriever itself: index partitioning + token-masking probes → `BREAKS` |
| **FT-RAG** | table nodes, a new stage | **cell-level locators** — proves the locator set cannot be a closed enum → `BREAKS` |
| **FD-RAG** | federated; deferred wholesale | **dual-path fast/slow routing** is separable from federation and must *not* be deferred |
| **AffordanceRAG** | custom scorer | **corpus constructed by agent exploration** — ingest-as-trajectory → `BREAKS` |

Plus two corrections already found in the first pass: **HGMem** (working memory, not
a corpus hypergraph) and **MiA-RAG** (global conditioning into two stages, not just
derived nodes).

**Eight of twenty summaries materially misled.** That is the empirical case for this
document existing.

---

## 2. The twenty systems

### Content and multimodality

**MiA-RAG** · [2512.17220](https://arxiv.org/abs/2512.17220) — hierarchical mindscape construction + MiA-Emb retriever + MiA-Gen generator; a global semantic representation **conditions both retrieval and generation**. `Retrieve(Query, Index)` and `Generate(Query, Context)` cannot express it. → R4, R16. **`BREAKS`**

**MegaRAG** · [2512.20626](https://arxiv.org/abs/2512.20626) (ACL 2026) — MLLM parallel entity-relation extraction **per page**, merged into a multimodal KG, then a **subgraph-guided refinement round** for cross-modal and cross-page relations. GME gives a **shared vector space** over text, images and structured knowledge. → R2, R5, R10, R11, R23. **`BREAKS`**

**MG²-RAG** · [2604.04969](https://arxiv.org/abs/2604.04969) (ECCV 2026) — textual entities and visual regions fused into **unified multimodal nodes**; retrieval at **chunk, sentence, image and object granularity**; **Personalized PageRank** propagates relevance. A node holds *several* modality components at once. → R1, R8, R22. **`BREAKS`**

**TV-RAG** · [2512.23483](https://arxiv.org/abs/2512.23483) (ACM MM '25) — entropy-weighted keyframe selection across **visual, OCR, ASR and detection streams**; **temporal-decay BM25** folding timestamp proximity into relevance; bi-level draft-and-self-verify. OCR/ASR text is *generated* content that is *also* time-anchored, so derivation and locator are independent. → R1, R3, R17, R21. **`BREAKS`**

**SignRAG** · [2512.12885](https://arxiv.org/abs/2512.12885) — a VLM writes textual descriptions of each reference image; retrieval runs over **those descriptions**, then an LLM reasons over candidates. 303 Ohio MUTCD signs, 95.58% / 82.45%. A **modality bridge via generated text** — architecturally distinct from MegaRAG's shared embedding space, and both must be supported. → R4, R23. **`BREAKS`**

**FT-RAG** · [2605.01495](https://arxiv.org/abs/2605.01495) — tables decomposed to **entry-level semantic units**, structural neighbour expansion over the resulting graph, multi-table integration. Reports table-level *and* **cell-level** hit rates. A table cell is neither a byte range nor a bounding box. → R2, R8, R11, R22, R35. **`BREAKS`**

**AffordanceRAG** · [2512.18987](https://arxiv.org/abs/2512.18987) — **embodied memory built from pre-exploration images**; hierarchical multimodal retrieval reranked by **affordance score**; 85% task success. The corpus is the output of an agent acting in an environment. → R21, R28, R37. **`BREAKS`**

### Structure and memory

**HGMem** · [2512.23959](https://arxiv.org/abs/2512.23959) — memory as a hypergraph whose **hyperedges are memory units**, accumulating higher-order interactions across reasoning steps. Mutable, episode-scoped, distinct from the corpus. → R6, R9. **`BREAKS`**

**Disco-RAG** · [2601.04377](https://arxiv.org/abs/2601.04377) (ACL 2026) — **intra-chunk discourse trees** *and* **inter-chunk rhetorical graphs**, jointly integrated into a **planning blueprint that conditions generation**. Two structures at two scopes simultaneously; second independent demand for a conditioning channel. Costs 3–4× standard RAG in LLM calls. → R7, R16, R42. **`BREAKS`**

### Loop, control and scheduling

**Agentic RAG (SoK)** · [2603.07379](https://arxiv.org/abs/2603.07379) — formalises agentic retrieval-generation loops as **finite-horizon partially observable MDPs**, with explicit control policies and state transitions; taxonomy over planning, retrieval orchestration, memory paradigms and tool coordination. **Observation ≠ state** is a kernel-level distinction my v1 `Policy` sketch did not have. → R12, R46. **`BREAKS`**

**Graph-O1** · [2512.17912](https://arxiv.org/abs/2512.17912) — **MCTS + end-to-end RL** over text-attributed graphs, framed as multi-turn interaction with a **graph environment** under a unified reward. Falsifies "three loop forms are exhaustive". → R12, R13, R41. **`BREAKS`**

**Predictive Prefetching** · [2605.17989](https://arxiv.org/abs/2605.17989) (ICML 2026) — retrieval predictor, context monitor and query generator exploit **semantic precursors several tokens before uncertainty peaks**; async issue with staleness handling. 43.5% latency reduction, 62.4% TTFT. → R14, R15. **`BREAKS`**

**A-RAG** · [2602.03442](https://arxiv.org/abs/2602.03442) · [code](https://github.com/Ayanami0730/arag) — retrieval exposed to the model as **three tools: keyword search, semantic search, chunk read**, agent-selected granularity. **External confirmation that a retriever is a tool** — v1 asserted this; here it is a published system's central design. → R8, R20. **`EXTENDS`**

**FD-RAG** · [2605.27432](https://arxiv.org/abs/2605.27432) — **dual-system**: adaptive hypergraphs over local corpora distilled into compact QA memories; direct memory match for covered queries, LLM reasoning only when needed; anonymised cross-device aggregation. 7.8% accuracy, **8.4× latency reduction**. The fast/slow routing is independent of federation. → R6, R19, R29; federation → R47. **`BREAKS`**

**QuCo-RAG** · [2512.19134](https://arxiv.org/abs/2512.19134) (ACL Findings 2026) — objective uncertainty from **pre-training corpus statistics** rather than model logits: low-frequency entities before generation, **zero entity co-occurrence during generation**, via Infini-gram over 4T tokens. +14 EM on long-tail entities. → R14, R24. **`EXTENDS`**

### Generation, verification and corpus lifecycle

**SURE-RAG** · [2605.03534](https://arxiv.org/abs/2605.03534) — three-way **supports / refutes / insufficient**, abstaining unless support is established. Sufficiency is a **set-level property**: "missing hops and unresolved conflicts cannot be detected by independent passage scoring." Aggregates pair-level relations into coverage, relation strength, disagreement, conflict and retrieval uncertainty. Calibrated 0.9075 Macro-F1 vs GPT-4o judge 0.7284; **risk at 30% coverage 0.2588 → 0.1642**. → R31, R32, R33, R34. **`BREAKS`**

**Bidirectional RAG** · [2512.22199](https://arxiv.org/abs/2512.22199) — **multi-stage acceptance** (NLI grounding, attribution, novelty) gating write-back of generated answers into the corpus. 40.58% vs 20.33% coverage while adding 72% fewer documents. Runs become a DAG: corpus *n+1* derives from run *n*. → R26, R27. **`BREAKS`**

**HiFi-RAG** · [2512.22442](https://arxiv.org/abs/2512.22442) — winner, MMU-RAGent NeurIPS 2025 text-to-text static track. **Gemini 2.5 Flash** for query formulation, hierarchical filtering and citation attribution; **Gemini 2.5 Pro** for final generation. Different models bound to different stages by cost and capability. → R17, R18, R36, R42. **`EXTENDS`**

**RAGPart / RAGMask** · [2512.24268](https://arxiv.org/abs/2512.24268) (UMD) — **retrieval-stage** defences: RAGPart partitions documents exploiting dense-retriever training dynamics; RAGMask flags suspicious tokens by **similarity shift under targeted masking**. Evaluated over 2 benchmarks × 4 poisoning strategies × 4 retrievers. Requires a partitionable index and retriever introspection. → R25, R40. **`BREAKS`**

**Hybrid multilingual RAG** · [2512.12694](https://arxiv.org/abs/2512.12694) — dense+sparse RRF fusion with query expansion over temporal synonyms, historical orthographic variants and paraphrases; **strict grounding with explicit abstention**; explicitly **modular to enable systematic component evaluation**; reports **reduced variance across query formulations**. → R30, R31, R38, R39. **`EXPRESSIBLE`**

---

## 3. Requirement axes

| Axis | Values observed |
|---|---|
| Modality | text · image · video · audio |
| Locator | byte range · bbox · time interval · **table cell** · page · spatial pose · none — **open set** |
| Derivation | source-anchored · generated · **generated-and-anchored** |
| Structure | flat · tree · binary graph · **hypergraph** · multi-stream timeline · **several at once** |
| Structure lifetime | static (corpus) · **dynamic (episode working memory)** |
| Granularity | corpus · document · page · chunk · sentence · object · **cell** |
| Index | sparse · dense · trie · graph-propagation · temporal-decay · multi-granularity · **partitioned** |
| Loop | linear · iterative · branching · **tree search** — all as **finite-horizon POMDPs** |
| Scheduling | synchronous · **async-speculative** · **token-streaming-observable** |
| Conditioning | none · **global signal into Retrieve and Generate** |
| Corpus lifecycle | read-only · **validated write-back + lineage** · **built by agent exploration** · **distilled** |
| Verification | structural · NLI · attribution · novelty · **set-level sufficiency, three-way, calibrated** |
| Model binding | one model · **per-stage tiered binding** |
| Training | none · SFT · **RL with trajectory reward + steppable environment** |

---

## 4. The matrix

| # | System | Verdict | Requirements |
|---|---|---|---|
| 1 | MiA-RAG | `BREAKS` | R4, R16 |
| 2 | HGMem | `BREAKS` | R6, R9 |
| 3 | MegaRAG | `BREAKS` | R2, R5, R10, R11, R23 |
| 4 | Disco-RAG | `BREAKS` | R7, R16, R42 |
| 5 | Agentic RAG (SoK) | `BREAKS` | R12, R46 |
| 6 | A-RAG | `EXTENDS` | R8, R20 |
| 7 | Predictive Prefetching | `BREAKS` | R14, R15 |
| 8 | SURE-RAG | `BREAKS` | R31, R32, R33, R34 |
| 9 | QuCo-RAG | `EXTENDS` | R14, R24 |
| 10 | HiFi-RAG | `EXTENDS` | R17, R18, R36, R42 |
| 11 | Bidirectional RAG | `BREAKS` | R26, R27 |
| 12 | MG²-RAG | `BREAKS` | R1, R8, R22 |
| 13 | FT-RAG | `BREAKS` | R2, R8, R11, R22, R35 |
| 14 | TV-RAG | `BREAKS` | R1, R3, R17, R21 |
| 15 | AffordanceRAG | `BREAKS` | R21, R28, R37 |
| 16 | SignRAG | `BREAKS` | R4, R23 |
| 17 | Hybrid multilingual | `EXPRESSIBLE` | R30, R31, R38, R39 |
| 18 | Graph-O1 | `BREAKS` | R12, R13, R41 |
| 19 | FD-RAG | `BREAKS` | R6, R19, R29, R47 |
| 20 | RAGPart / RAGMask | `BREAKS` | R25, R40 |

---

## 5. Consolidated requirements

### Content and storage
- **R1** — A node's content is a **set of components**, each carrying `(modality, locator, derivation)` as three independent axes. Not one enum; not one component. *(MG²-RAG, TV-RAG)*
- **R2** — Modalities: text, image, video, audio. **Locators are an open, extensible set** — byte range, bounding box, time interval, table cell, page, spatial pose, none. A closed enum is provably insufficient. *(FT-RAG, MegaRAG, TV-RAG)*
- **R3** — A document carries a **shared timeline**; multiple derived streams anchor to it. *(TV-RAG)*
- **R4** — Generated content is storable and retrievable with no source anchor. *(MiA-RAG, SignRAG)*
- **R5** — Document → page → element hierarchy as a first-class granularity. *(MegaRAG)*

### Structure
- **R6** — **N-ary edges**; binary is the common case, not the only case. *(HGMem, FD-RAG)*
- **R7** — **Several structures coexist at different scopes** — a tree within a chunk, a graph across chunks. *(Disco-RAG)*
- **R8** — **Multi-granularity nodes coexist** in one index: chunk, sentence, image, object, cell. *(MG²-RAG, FT-RAG, A-RAG)*
- **R9** — **Working memory**: mutable, episode-scoped, structurally rich, separate from the corpus. *(HGMem)*
- **R10** — Structure construction is **iterative**, with refinement passes — not a pure single-pass function. *(MegaRAG)*
- **R11** — **Cross-document edges** (cross-page, multi-table). *(MegaRAG, FT-RAG)*

### Execution and control
- **R12** — The loop's formal model is a **finite-horizon POMDP**: state, **observation distinct from state**, action, transition, horizon, reward. *(SoK, Graph-O1)*
- **R13** — **Four** loop forms: linear, iterative, branching, **tree search** with backtracking and value estimates. *(Graph-O1)*
- **R14** — `Generate` is a **token-level observable stream with hooks**, not an atomic call. *(Predictive Prefetching, QuCo-RAG)*
- **R15** — **Speculative async execution** with cancellation and staleness detection. *(Predictive Prefetching)*
- **R16** — A **conditioning side-channel** readable by Retrieve and Generate. *(MiA-RAG, Disco-RAG)*
- **R17** — **Two-pass generation**: draft, verify, regenerate. *(HiFi-RAG, TV-RAG)*
- **R18** — **Per-stage model binding** — cheap models for filtering, strong models for generation. *(HiFi-RAG)*
- **R19** — **Dual-path routing**: fast memory match vs slow reasoning, with a router. *(FD-RAG)*

### Retrieval
- **R20** — Retrievers are **tools with schemas**, agent-callable at multiple granularities. *(A-RAG)*
- **R21** — **Pluggable score modifiers** with node-metadata access — temporal decay, affordance. *(TV-RAG, AffordanceRAG)*
- **R22** — **Graph-propagation retrieval**: Personalized PageRank, structural neighbour expansion. *(MG²-RAG, FT-RAG)*
- **R23** — **Two cross-modal strategies, both supported**: a shared multimodal embedding space, and modality bridging via generated text. *(MegaRAG, SignRAG)*
- **R24** — **External statistics backends** — pre-training-corpus n-gram services, not our index. *(QuCo-RAG)*
- **R25** — **Partitioned index build with aggregation**, and retriever introspection for perturbation probes. *(RAGPart/RAGMask)*

### Corpus lifecycle
- **R26** — **Validated write-back** through an acceptance pipeline. *(Bidirectional RAG)*
- **R27** — **Corpus lineage**: versions form a DAG; a run records which version it read. *(Bidirectional RAG)*
- **R28** — **Ingest-as-trajectory**: a corpus built by an agent exploring an environment. *(AffordanceRAG)*
- **R29** — **Distillation**: corpus → compact derived memory as a first-class artifact. *(FD-RAG)*
- **R30** — Text quality (OCR noise, orthographic variation) recorded at ingest. *(Hybrid multilingual)*

### Measurement
- **R31** — **Abstention is first-class**; verdicts are three-way: supports / refutes / insufficient. *(SURE-RAG, Hybrid multilingual)*
- **R32** — **Set-level verification** — sufficiency cannot be computed per-passage. *(SURE-RAG)*
- **R33** — **Calibration and risk-coverage curves** as a metric family. *(SURE-RAG)*
- **R34** — Verification signals: coverage, relation strength, disagreement, **conflict**, retrieval uncertainty. *(SURE-RAG)*
- **R35** — Metrics at **multiple structural granularities** (table-level vs cell-level). *(FT-RAG)*
- **R36** — Generation metrics (ROUGE-L, DeBERTaScore, exact-value recall) beside retrieval metrics. *(HiFi-RAG, FT-RAG)*
- **R37** — **End-task success** metrics. *(AffordanceRAG)*
- **R38** — **Dispersion**, not only central tendency — variance across query formulations. *(Hybrid multilingual)*
- **R39** — **Ablation harness**: swap one stage, hold the rest fixed, measure. *(Hybrid multilingual, explicitly)*
- **R40** — **Adversarial harness**: attack strategies × retrievers, attack success rate plus benign utility. *(RAGPart/RAGMask)*
- **R41** — **Trajectory-level reward** for RL, not only supervised fine-tuning. *(Graph-O1)*
- **R42** — **Cost attributed per stage per model**. *(HiFi-RAG, Disco-RAG at 3–4×)*

### Agent layer
From ["Stop Comparing LLM Agents Without Disclosing the Harness"](https://arxiv.org/pdf/2605.23950): environment setup, tool/action space, observation format, error handling, resource constraints, feedback mechanism, episode length, state management.

- **R43** — Those eight are **required RunCard fields**. They are also exactly the parameters of the POMDP in R12 — the harness paper and the SoK describe the same object from two directions.
- **R44** — `Tool` trait with schema — required by the RAG module for A-RAG, not only by the agent layer.
- **R45** — Resource limits and episode length are **enforced**, not merely recorded.
- **R46** — Taxonomy dimensions to support: planning, retrieval orchestration, memory paradigms, tool coordination. *(SoK)*

### Deferred, recorded
- **R47** — Federated / distributed retrieval across non-centralised shards. **Out of scope**, recorded as a decision. Note FD-RAG's dual-path routing (R19) is *not* deferred with it.

---

## 6. What this changes about `DESIGN.md` v1

| v1 claim | Status |
|---|---|
| "Three loop forms are exhaustive" | **False** — tree search. (R13) |
| Loops modelled as `Policy → Action` | **Insufficient** — POMDP, observation ≠ state. (R12) |
| `Node.span: ByteRange` | **Insufficient** — set of components, three axes. (R1) |
| Four locator kinds would suffice | **False** — open set; table cells break it. (R2) |
| `Edge { src, dst }` | **Insufficient** — n-ary. (R6) |
| One structure per corpus | **False** — several at once, different scopes. (R7) |
| "The corpus is immutable" | **Too strong** — write-back, lineage, exploration, distillation. (R26–R29) |
| Stages are synchronous | **Insufficient** — speculation, token streaming. (R14, R15) |
| `Retrieve(Query, Index)` / `Generate(Query, Context)` | **Insufficient** — conditioning channel. (R16) |
| One model per pipeline | **Insufficient** — per-stage binding. (R18) |
| `Verify` maps over candidates | **False** — set-level property. (R32) |
| Hallucination is structural or semantic | **Insufficient** — three-way, calibrated, risk-coverage. (R31, R33) |
| "A retriever is a tool" (asserted) | **Confirmed externally.** (R20) |
| RAG and agents share a kernel (asserted) | **Confirmed** — Graph-O1 and AffordanceRAG need the environment abstraction inside RAG. (R12, R28) |
| Per-query scores always retained | Unchanged; reinforced by R38. |
| Budget enforced, not reported | Unchanged; extended by R42, R45. |
| No working memory concept | **Missing entirely.** (R9) |
| No environment concept | **Missing entirely.** (R12, R28) |
| No ablation or adversarial harness | **Missing entirely.** (R39, R40) |

Two asserted claims survived and were independently confirmed. Eleven were wrong or
insufficient. Four concepts were missing outright.

---

## 7. Next step

Write `DESIGN.md` v2 plus the compiling type skeleton against R1–R47.

Any design that cannot satisfy a requirement here must say so explicitly and record
why as an ADR — rather than quietly omitting it, which is how v1 reached you looking
more settled than it was.
