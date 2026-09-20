"""Compare runs: tables and plots.

Results come back as pandas DataFrames on purpose. People already know how to
slice, pivot and plot those; a bespoke result object would have to be learned.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import RunMetrics, randomization_test

__all__ = ["compare", "load_metrics", "plot_risk_coverage", "table"]


def load_metrics(run_dir: str | Path) -> list[RunMetrics]:
    """Load every ``metrics.json`` under ``run_dir``."""
    out = []
    for p in sorted(Path(run_dir).glob("*/metrics.json")):
        d = json.loads(p.read_text())
        from .metrics import RiskCoverage

        risk = None
        if d.get("risk"):
            r = d["risk"]
            risk = RiskCoverage(r["coverage"], r["risk"], r.get("curve", []), r.get("aurc"))
        out.append(
            RunMetrics(
                system=d["system"],
                task=d["task"],
                n_queries=d["n_queries"],
                aggregate=d["aggregate"],
                per_query=d["per_query"],
                risk=risk,
                cost=d.get("cost", {}),
                latency=d.get("latency", {}),
                n_failed=d.get("n_failed", 0),
            )
        )
    return out


def table(runs: list[RunMetrics]):
    """One row per (task, system), with metrics, cost and latency."""
    import pandas as pd

    rows = []
    for m in runs:
        row = {"task": m.task, "system": m.system, "n": m.n_queries, **m.aggregate}
        if m.risk:
            row["coverage"] = m.risk.coverage
            row["risk"] = m.risk.risk
            if m.risk.aurc is not None:
                row["aurc"] = m.risk.aurc
        row["usd"] = m.cost.get("usd", 0.0)
        row["p95_s"] = m.latency.get("p95", 0.0)
        row["failed"] = m.n_failed
        rows.append(row)
    return pd.DataFrame(rows).set_index(["task", "system"]).sort_index()


def compare(
    runs: list[RunMetrics], a: str, b: str, *, metric: str = "recall@5", n_samples: int = 10000
) -> dict:
    """Paired randomization test between two systems on the same task."""
    by = {m.system: m for m in runs}
    if a not in by or b not in by:
        raise KeyError(f"have {sorted(by)}; asked for {a!r} vs {b!r}")
    res = randomization_test(
        by[a].scores(metric), by[b].scores(metric), n_samples=n_samples
    )
    return {"metric": metric, "a": a, "b": b, **res}


def plot_risk_coverage(runs: list[RunMetrics], out_path: str | Path | None = None):
    """Risk against coverage. Lower and flatter is better."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for m in runs:
        if not (m.risk and m.risk.curve):
            continue
        xs = [c for c, _ in m.risk.curve]
        ys = [r for _, r in m.risk.curve]
        label = m.system + (f" (AURC {m.risk.aurc:.3f})" if m.risk.aurc is not None else "")
        ax.plot(xs, ys, label=label, linewidth=1.8)
        plotted += 1
    if not plotted:
        ax.text(0.5, 0.5, "no system reported confidences", ha="center", va="center")
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title("Risk–coverage")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
