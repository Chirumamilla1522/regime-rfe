#!/usr/bin/env python3
"""Generate RFE-Recurrent-Bench tables and figures from the retained pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "recurrent_bench_pilot"
TABLES = ROOT / "paper" / "generated" / "tables"
FIGURES = ROOT / "paper" / "generated" / "figures"
METRICS = ROOT / "paper" / "generated" / "benchmark_metrics.json"

DISPLAY = {
    "pooled": "Pooled",
    "restart_no_reuse": "Restart",
    "cluster_no_quarantine": "Cluster, no Q",
    "recurrence_aware": "DME",
    "oracle_boundary": "Oracle boundary",
    "oracle_mode": "Oracle mode",
    "sliding_window": "Sliding window",
}
METHOD_ORDER = list(DISPLAY)
TASK_ORDER = ("swap_chain", "riverswim", "deepsea", "four_rooms")


def _gap(summary, method):
    for row in summary:
        if row["method"] == method:
            return float(row["worst_reward_value_gap"])
    return float("nan")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    payload = json.loads((RESULTS / "results.json").read_text())
    METRICS.write_text(json.dumps(payload, indent=2, sort_keys=True))

    rows = ["\\begin{tabular}{lrr}", "\\toprule", "Method & Units & Mean worst gap \\\\", "\\midrule"]
    for method in METHOD_ORDER:
        for row in payload["summary"]:
            if row["method"] != method:
                continue
            rows.append(
                f"{DISPLAY[method]} & {int(row['n_units'])} & "
                f"{float(row['worst_reward_value_gap']):.4f} \\\\"
            )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "benchmark_summary.tex").write_text("\n".join(rows) + "\n")

    header = "Task & Pooled & Restart & DME & Oracle mode \\\\"
    task_rows = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for task in TASK_ORDER:
        summary = payload["per_task_summary"][task]
        task_rows.append(
            f"{task.replace('_', '\\_')} & "
            f"{_gap(summary, 'pooled'):.3f} & "
            f"{_gap(summary, 'restart_no_reuse'):.3f} & "
            f"{_gap(summary, 'recurrence_aware'):.3f} & "
            f"{_gap(summary, 'oracle_mode'):.3f} \\\\"
        )
    task_rows.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "benchmark_by_task.tex").write_text("\n".join(task_rows) + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = range(len(TASK_ORDER))
    width = 0.2
    series = [
        ("pooled", "Pooled"),
        ("restart_no_reuse", "Restart"),
        ("recurrence_aware", "DME"),
        ("oracle_mode", "Oracle"),
    ]
    for offset, (method, label) in enumerate(series):
        values = [
            _gap(payload["per_task_summary"][task], method) for task in TASK_ORDER
        ]
        ax.bar([i + (offset - 1.5) * width for i in x], values, width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["swap chain", "RiverSwim", "DeepSea", "Four rooms"])
    ax.set_ylabel("Worst-reward value gap")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_by_task.pdf")
    fig.savefig(FIGURES / "benchmark_by_task.png")
    plt.close(fig)
    print("Wrote benchmark tables and figures")


if __name__ == "__main__":
    main()
