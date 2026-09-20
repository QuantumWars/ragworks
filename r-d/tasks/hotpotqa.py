"""HotpotQA as a multi-hop retrieval task.

HotpotQA's distractor setting gives each question ten paragraphs, two of which
are the supporting evidence. Two decisions worth stating, because both change
what the numbers mean:

**Global pool by default.** Paragraphs from every loaded question are merged and
deduplicated by title into one corpus, so retrieval is a real search over
thousands of paragraphs rather than a pick-2-of-10 rerank. ``per_question_scope``
restores the strict distractor setting.

**Unanswerable queries are constructed, not natural.** With
``unanswerable_frac`` we remove a query's supporting paragraphs from the corpus
entirely, so the question genuinely cannot be answered from what remains. Only
queries whose evidence no *other* retained query needs are eligible, so removal
never silently breaks a neighbour. This is a synthetic condition and findings
must say so.
"""

from __future__ import annotations

import random

from harness.task import Doc, Query, Task

__all__ = ["load"]


def _para_id(title: str) -> str:
    return "wiki::" + title.replace(" ", "_")


def load(
    *,
    split: str = "validation",
    limit: int = 500,
    unanswerable_frac: float = 0.0,
    per_question_scope: bool = False,
    seed: int = 42,
) -> Task:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"{split}[:{limit}]")

    docs: dict[str, Doc] = {}
    raw: list[tuple[str, str, str, frozenset[str], dict]] = []

    for ex in ds:
        qid = ex["id"]
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])

        gold_ids = set()
        for title, sents in zip(titles, sentences, strict=False):
            did = _para_id(title) + (f"@{qid}" if per_question_scope else "")
            if did not in docs:
                docs[did] = Doc(
                    doc_id=did,
                    text=" ".join(sents).strip(),
                    title=title,
                    meta={"scope": qid} if per_question_scope else {},
                )
            if title in gold_titles:
                gold_ids.add(did)

        raw.append(
            (
                qid,
                ex["question"],
                ex["answer"],
                frozenset(gold_ids),
                {"type": ex["type"], "level": ex["level"]},
            )
        )

    # Choose unanswerable queries whose evidence nothing else depends on.
    unanswerable: set[str] = set()
    if unanswerable_frac > 0:
        rng = random.Random(seed)
        need = int(len(raw) * unanswerable_frac)
        order = list(range(len(raw)))
        rng.shuffle(order)
        for i in order:
            if len(unanswerable) >= need:
                break
            qid, _, _, gold, _ = raw[i]
            others = {d for j, (q2, _, _, g2, _) in enumerate(raw) if j != i and q2 not in unanswerable for d in g2}
            if gold and not (gold & others):
                unanswerable.add(qid)
                for d in gold:
                    docs.pop(d, None)

    queries = [
        Query(
            gold_doc_ids=frozenset() if qid in unanswerable else gold,
            qid=qid,
            text=text,
            answer=None if qid in unanswerable else answer,
            answerable=qid not in unanswerable,
            scope=qid if per_question_scope else None,
            meta=meta,
        )
        for qid, text, answer, gold, meta in raw
    ]

    return Task(
        name=f"hotpotqa-{split}-{limit}" + ("-unans" if unanswerable else ""),
        docs=list(docs.values()),
        queries=queries,
        meta={
            "source": "hotpotqa/hotpot_qa:distractor",
            "split": split,
            "limit": limit,
            "per_question_scope": per_question_scope,
            "n_unanswerable": len(unanswerable),
            "unanswerable_is_synthetic": True,
        },
    )
