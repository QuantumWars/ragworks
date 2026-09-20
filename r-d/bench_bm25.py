#!/usr/bin/env python3
"""Rust BM25 against the Python BM25: agreement first, then speed.

Speed is only interesting if the two agree. A faster implementation that ranks
differently is not faster -- it is different.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import ragworks

from systems.hybrid_rrf.system import _BM25
from tasks.hotpotqa import load

task = load(limit=300)
texts = [f"{d.title}. {d.text}" if d.title else d.text for d in task.docs]
ids = list(range(len(texts)))
queries = [q.text for q in task.queries]
print(f"{len(texts)} docs, {len(queries)} queries\n")

# ---- index -------------------------------------------------------------
t = time.perf_counter()
py = _BM25()
py.index([str(i) for i in ids], texts)
py_index = time.perf_counter() - t

t = time.perf_counter()
rs = ragworks.Bm25()
rs.add_many(ids, texts)
rs_index = time.perf_counter() - t

# ---- search ------------------------------------------------------------
K = 10
t = time.perf_counter()
py_runs = [[int(d) for d, _ in py.search(q, K)] for q in queries]
py_search = time.perf_counter() - t

t = time.perf_counter()
rs_runs = [[i for i, _ in r] for r in rs.search_many(queries, K)]
rs_search = time.perf_counter() - t

# ---- agreement ---------------------------------------------------------
exact = sum(a == b for a, b in zip(py_runs, rs_runs, strict=True))
top1 = sum(
    (a[:1] == b[:1]) for a, b in zip(py_runs, rs_runs, strict=True)
)
overlap = sum(
    len(set(a) & set(b)) / max(len(a), 1) for a, b in zip(py_runs, rs_runs, strict=True)
) / len(queries)

print("agreement")
print(f"  identical top-{K} ordering : {exact}/{len(queries)} ({exact / len(queries):.1%})")
print(f"  identical top-1           : {top1}/{len(queries)} ({top1 / len(queries):.1%})")
print(f"  mean set overlap@{K}       : {overlap:.3f}")

print("\nspeed")
print(f"  index   python {py_index * 1000:8.1f} ms   rust {rs_index * 1000:8.1f} ms"
      f"   {py_index / rs_index:5.1f}x")
print(f"  search  python {py_search * 1000:8.1f} ms   rust {rs_search * 1000:8.1f} ms"
      f"   {py_search / rs_search:5.1f}x")
print(f"  per query      {py_search / len(queries) * 1000:8.3f} ms      "
      f"{rs_search / len(queries) * 1000:8.3f} ms")

if exact < len(queries):
    i = next(i for i, (a, b) in enumerate(zip(py_runs, rs_runs, strict=True)) if a != b)
    print(f"\nfirst disagreement, query {i}: {queries[i][:70]!r}")
    print(f"  python {py_runs[i][:5]}")
    print(f"  rust   {rs_runs[i][:5]}")
