#!/usr/bin/env python3
"""Regenerate paper tables and figures from canonical raw outputs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "canonical"
PAPER = ROOT / "paper"


def read_csv(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    summary = read_csv("summary.csv")
    coverage = read_csv("coverage.csv")

    post = [row for row in summary if row["phase"] == "post"]
    methods = ["fixed_count", "time_count", "fixed_random"]
    drifts = ["goal_shift", "transition_noise", "wall_change"]
    x = np.arange(len(drifts))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for offset, method in enumerate(methods):
        rows = [
            next(
                row
                for row in post
                if row["method"] == method and row["drift_type"] == drift
            )
            for drift in drifts
        ]
        means = np.array([float(row["success_mean"]) for row in rows])
        low = np.array([float(row["success_ci95_low"]) for row in rows])
        high = np.array([float(row["success_ci95_high"]) for row in rows])
        errors = np.vstack((means - low, high - means))
        ax.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=2,
            label=method.replace("_", " "),
        )
    ax.set_xticks(x, [name.replace("_", " ") for name in drifts])
    ax.set_ylabel("Post-drift success rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Ten-seed means; bounded 95% cluster-bootstrap intervals")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "post_drift_success.pdf")
    fig.savefig(PAPER / "figures" / "post_drift_success.png", dpi=200)
    plt.close(fig)

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Drift & Method & Pre success & Post success \\",
        r"\midrule",
    ]
    for drift in drifts:
        for method in methods:
            pre = next(
                row
                for row in summary
                if row["drift_type"] == drift
                and row["method"] == method
                and row["phase"] == "pre"
            )
            post_row = next(
                row
                for row in summary
                if row["drift_type"] == drift
                and row["method"] == method
                and row["phase"] == "post"
            )
            lines.append(
                "{} & {} & {:.2f} & {:.2f} \\\\".format(
                    drift.replace("_", " "),
                    method.replace("_", " "),
                    float(pre["success_mean"]),
                    float(post_row["success_mean"]),
                )
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER / "tables" / "results.tex").write_text("\n".join(lines) + "\n")

    coverage_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Drift & Count & Random \\",
        r"\midrule",
    ]
    for drift in drifts:
        values = {}
        for strategy in ("count", "random"):
            values[strategy] = np.mean(
                [
                    float(row["coverage"])
                    for row in coverage
                    if row["drift_type"] == drift
                    and row["strategy"] == strategy
                ]
            )
        coverage_lines.append(
            "{} & {:.3f} & {:.3f} \\\\".format(
                drift.replace("_", " "), values["count"], values["random"]
            )
        )
    coverage_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER / "tables" / "coverage.tex").write_text(
        "\n".join(coverage_lines) + "\n"
    )


if __name__ == "__main__":
    main()
