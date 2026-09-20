"""SearchTome / STAIR-format books as a structure-aware retrieval task.

Reads the on-disk layout produced by a STAIR reimplementation kept in a separate
repository (arXiv:2609.03874). Nothing here depends on that project being
present -- point `root` at any directory in the same shape:

    <root>/<book_id>/
        toc.json      the ToC tree, leaves carrying their section text
        test.jsonl    queries, each with the gold leaf it was generated from

Retrieval targets are **leaf sections**, so the corpus is structured rather than
chunked: a human-authored section boundary is the retrieval unit. Each query is
scoped to its own book, matching how STAIR is evaluated.

Nothing ships this corpus: the 18 source textbooks must be fetched and the
questions generated before it exists. `load` fails with an explicit message
rather than silently returning an empty task.
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.task import Doc, Query, Task

__all__ = ["load"]


def _walk(node: dict, book_id: str, out: list[Doc], trail: tuple[str, ...] = ()) -> None:
    """Depth-first over the ToC; emit a Doc for every leaf carrying text."""
    number = str(node.get("number", "") or "")
    title = str(node.get("title", "") or "")
    label = f"{number} {title}".strip()
    path = (*trail, label) if label else trail
    children = node.get("children") or []

    if not children:
        text = (node.get("content") or node.get("text") or "").strip()
        if text:
            out.append(
                Doc(
                    doc_id=f"{book_id}::{label}",
                    text=text,
                    title=label,
                    meta={"scope": book_id, "book_id": book_id, "path": list(path),
                          "depth": len(path)},
                )
            )
    else:
        for child in children:
            _walk(child, book_id, out, path)


def load(
    root: str | Path,
    *,
    split: str = "test",
    books: list[str] | None = None,
    max_queries_per_book: int | None = None,
) -> Task:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. SearchTome is not shipped with this "
            "repository: build it from the source textbooks, or point root at an "
            "existing SearchTome-format directory."
        )

    book_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "toc.json").exists())
    if books:
        keep = set(books)
        book_dirs = [p for p in book_dirs if p.name in keep]
    if not book_dirs:
        raise FileNotFoundError(f"no <book>/toc.json under {root}")

    docs: list[Doc] = []
    queries: list[Query] = []

    for bd in book_dirs:
        book_id = bd.name
        toc = json.loads((bd / "toc.json").read_text())
        before = len(docs)
        _walk(toc.get("root", toc), book_id, docs)
        valid = {d.doc_id for d in docs[before:]}

        qpath = bd / f"{split}.jsonl"
        if not qpath.exists():
            continue
        rows = [json.loads(line) for line in qpath.read_text().splitlines() if line.strip()]
        if max_queries_per_book:
            rows = rows[:max_queries_per_book]
        for r in rows:
            gold = f"{book_id}::{r['gold_docid']}"
            queries.append(
                Query(
                    # A gold leaf missing from the ToC would silently score zero;
                    # dropping it instead keeps the task honest.
                    gold_doc_ids=frozenset({gold}) if gold in valid else frozenset(),
                    qid=r["qid"],
                    text=r["query"],
                    answerable=gold in valid,
                    scope=book_id,
                    meta={"book_id": book_id, "gold_number": r.get("gold_number", "")},
                )
            )

    n_missing = sum(1 for q in queries if not q.answerable)
    return Task(
        name=f"searchtome-{split}-{len(book_dirs)}books",
        docs=docs,
        queries=queries,
        meta={
            "source": str(root),
            "split": split,
            "books": [p.name for p in book_dirs],
            "n_gold_missing_from_toc": n_missing,
            "retrieval_unit": "leaf section",
        },
    )
