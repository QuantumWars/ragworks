#!/usr/bin/env python3
"""Wave 1: naive, hybrid_rrf, sure on one HotpotQA task.

    ../../.venv/bin/python run_wave1.py [--limit 150] [--no-sure]

Runs are resumable: re-running skips completed queries.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from harness.report import compare, load_metrics, plot_risk_coverage, table
from harness.runner import run
from tasks.hotpotqa import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--unanswerable", type=float, default=0.10)
    ap.add_argument("--evidence", type=int, default=8)
    ap.add_argument("--no-sure", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    task = load(limit=args.limit, unanswerable_frac=args.unanswerable)
    print(task, "\n")

    from systems.hybrid_rrf.system import HybridRRF
    from systems.naive.system import NaiveDense

    systems = [NaiveDense(), HybridRRF()]
    if not args.no_sure:
        from systems.sure.system import SureRAG

        systems.append(SureRAG(n_evidence=args.evidence))

    for s in systems:
        run(s, task, out_dir=args.out, verbose=True)
        print()

    runs = load_metrics(args.out)
    print(table(runs).to_string(float_format=lambda v: f"{v:.3f}"), "\n")

    names = [m.system for m in runs]
    for b in names:
        if b == "naive":
            continue
        r = compare(runs, "naive", b, metric="recall@5")
        print(
            f"recall@5  naive={r['mean_a']:.3f}  {b}={r['mean_b']:.3f}  "
            f"diff={r['diff']:+.3f}  p={r['p_value']:.4f}  n={r['n']}"
        )

    plot_risk_coverage(runs, out_path=f"{args.out}/risk_coverage.png")
    print(f"\nwrote {args.out}/risk_coverage.png")


if __name__ == "__main__":
    main()
