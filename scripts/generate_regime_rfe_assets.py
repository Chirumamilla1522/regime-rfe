#!/usr/bin/env python3
"""Generate the recurring-regime manuscript assets from checked pilot CSVs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "recurrent_tabular_pilot"
MINIGRID_RESULTS = ROOT / "results" / "minigrid_recurrent_pilot"
PAPER = ROOT / "paper"
TABLES = PAPER / "generated" / "tables"
FIGURES = PAPER / "generated" / "figures"
METRICS = PAPER / "generated" / "regime_pilot_metrics.json"

METHOD_ORDER = (
    "pooled",
    "restart_no_reuse",
    "cluster_no_quarantine",
    "recurrence_aware",
    "oracle_boundary",
    "oracle_mode",
    "sliding_window",
)
DISPLAY = {
    "pooled": "Pooled",
    "restart_no_reuse": "Restart, no reuse",
    "cluster_no_quarantine": "Cluster, no quarantine",
    "recurrence_aware": "Recurrence-aware",
    "oracle_boundary": "Oracle boundary",
    "oracle_mode": "Oracle mode",
    "sliding_window": "Sliding window",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    payload = json.loads((RESULTS / "results.json").read_text())
    minigrid_payload = json.loads(
        (MINIGRID_RESULTS / "results.json").read_text()
    )
    summary = {row["method"]: row for row in payload["summary"]}
    diagnostics = read_csv("detector_diagnostics.csv")
    diagnoses = read_csv("deployment_diagnosis.csv")
    savings = read_csv("recurrence_savings.csv")

    correct: dict[str, list[int]] = defaultdict(list)
    for row in diagnoses:
        selected = int(row["selected_mode"])
        truth = int(row["true_mode"])
        # Non-oracle restart entries use occurrence IDs as labels, so equality
        # with the true mode is not a meaningful accuracy statistic.
        if row["method"] in {
            "cluster_no_quarantine",
            "recurrence_aware",
            "oracle_boundary",
            "oracle_mode",
        }:
            correct[row["method"]].append(int(selected == truth))

    diag_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in diagnostics:
        diag_by_method[row["method"]].append(row)

    exact = {
        "source_config_hash": payload["config_hash"],
        "n_units": payload["n_units"],
        "runtime_seconds": payload["runtime_seconds"],
        "mean_recurrence_sample_savings": payload[
            "mean_recurrence_sample_savings"
        ],
        "recurrence_gate_passed": payload["recurrence_gate_passed"],
        "gate_checks": payload["gate_checks"],
        "reward_normalization": payload["reward_normalization"],
        "paired_gap_improvement": payload["paired_gap_improvement"],
        "summary": payload["summary"],
        "recurrence_aware_diagnosis_accuracy": mean(
            correct["recurrence_aware"]
        ),
        "recurrence_aware_learned_mode_accuracy": mean(
            int(row["learned_modes"]) == int(row["mode_count"])
            for row in diag_by_method["recurrence_aware"]
        ),
        "recurrence_aware_mean_alarms": mean(
            int(row["alarms"]) for row in diag_by_method["recurrence_aware"]
        ),
        "recurrence_aware_mean_reuse_events": mean(
            int(row["reuse_events"])
            for row in diag_by_method["recurrence_aware"]
        ),
        "recurrence_saving_rows": len(savings),
        "positive_saving_fraction": mean(
            float(row["sample_savings"]) > 0 for row in savings
        ),
        "zero_saving_fraction": mean(
            float(row["sample_savings"]) == 0 for row in savings
        ),
    }
    METRICS.write_text(json.dumps(exact, indent=2, sort_keys=True) + "\n")

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Units & Accept rate & Mean worst-reward gap \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        row = summary[method]
        lines.append(
            f"{DISPLAY[method]} & {row['n_units']} & "
            f"{row['deployment_accept_rate']:.6f} & "
            f"{row['worst_reward_value_gap']:.12f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "pilot_summary.tex").write_text("\n".join(lines) + "\n")

    minigrid_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Seeds & Mean worst-task gap \\",
        r"\midrule",
    ]
    for row in minigrid_payload["summary"]:
        minigrid_lines.append(
            f"{tex_escape(row['method'])} & {row['n_seeds']} & "
            f"{row['worst_task_gap']:.6f} \\\\"
        )
    minigrid_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "minigrid_summary.tex").write_text(
        "\n".join(minigrid_lines) + "\n"
    )

    config = payload["config"]
    config_lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Checked value \\",
        r"\midrule",
        f"Protocol & {tex_escape(payload['protocol'])} \\\\",
        f"Configuration hash & {tex_escape(payload['config_hash'])} \\\\",
        f"Seeds & {len(config['seeds'])} (0--9) \\\\",
        f"States & {config['state_counts'][0]} \\\\",
        f"Modes & {', '.join(map(str, config['mode_counts']))} \\\\",
        f"Separations & {', '.join(map(str, config['separations']))} \\\\",
        f"Dwell (transitions) & {config['dwell']} \\\\",
        f"Cycles & {config['cycles']} \\\\",
        f"Deployment prefix & {config['deployment_prefix']} \\\\",
        f"Finite horizon & {config['horizon']} \\\\",
        "Pathwise reward maximum & "
        f"{payload['reward_normalization']['pathwise_total_maximum']} \\\\",
        f"Value-gap target & {config['value_gap_target']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES / "pilot_config.tex").write_text(
        "\n".join(config_lines) + "\n"
    )

    methods = list(METHOD_ORDER)
    gaps = [float(summary[method]["worst_reward_value_gap"]) for method in methods]
    colors = ["#9ca3af", "#60a5fa", "#a78bfa", "#059669", "#f59e0b", "#d97706", "#ef4444"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(range(len(methods)), gaps, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([DISPLAY[m] for m in methods], rotation=28, ha="right")
    ax.set_ylabel("Mean worst-reward value gap")
    ax.set_title("Checked recurrent tabular pilot (40 units)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_value_gaps.pdf")
    plt.close(fig)

    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in savings:
        grouped[(int(row["mode_count"]), float(row["separation"]))].append(
            float(row["sample_savings"])
        )
    labels = [
        f"M={mode_count}, Δ={separation:g}"
        for mode_count, separation in sorted(grouped)
    ]
    means = [mean(grouped[key]) for key in sorted(grouped)]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(labels, means, color="#059669")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean transition-sample savings")
    ax.set_title("Recurrence reuse versus restart")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_recurrence_savings.pdf")
    plt.close(fig)

    print(f"Wrote {METRICS.relative_to(ROOT)}")
    print(f"Wrote {TABLES.relative_to(ROOT)}")
    print(f"Wrote {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
