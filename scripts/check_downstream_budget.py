#!/usr/bin/env python3
"""Predeclared geometric budget check for downstream undertraining.

This diagnostic is intentionally restricted to goal drift, seeds 0--2,
budgets 500/1500/4000, and both matched count-data representations. It is not
used to select the best reported budget after observing outcomes.
"""

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import (
    ExperimentConfig,
    collect_data,
    evaluate,
    make_env,
    seed_everything,
    train_agent,
    train_encoders,
)


OUTPUT = ROOT / "results" / "budget_check"
BUDGETS = (500, 1500, 4000)
SEEDS = (0, 1, 2)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        collection = ExperimentConfig(
            drift_time=500,
            exploration_steps=1000,
            representation_epochs=8,
            training_steps=1000,
            seeds=(seed,),
            drift_types=("goal_shift",),
        )
        seed_everything(seed)
        records, _ = collect_data(collection, "goal_shift", seed, "count")
        fixed, temporal = train_encoders(records, collection, seed)
        for budget in BUDGETS:
            config = copy.copy(collection)
            config.training_steps = budget
            config.drift_time = budget // 2
            for method, encoder in (
                ("fixed_count", fixed),
                ("time_count", temporal),
            ):
                seed_everything(seed + 2000)
                agent = train_agent(
                    copy.deepcopy(encoder), config, "goal_shift", seed + 2000
                )
                for phase in ("pre", "post"):
                    episodes = evaluate(
                        agent, config, "goal_shift", seed + 2000, phase
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "budget": budget,
                            "method": method,
                            "phase": phase,
                            "success_rate": float(
                                np.mean([row["success"] for row in episodes])
                            ),
                        }
                    )
    with (OUTPUT / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for budget in BUDGETS:
        for method in ("fixed_count", "time_count"):
            for phase in ("pre", "post"):
                values = [
                    row["success_rate"]
                    for row in rows
                    if row["budget"] == budget
                    and row["method"] == method
                    and row["phase"] == phase
                ]
                summary.append(
                    {
                        "budget": budget,
                        "method": method,
                        "phase": phase,
                        "mean_success": float(np.mean(values)),
                    }
                )
    reachability = []
    for seed in SEEDS:
        config = ExperimentConfig(
            drift_time=2000, training_steps=4000, drift_types=("goal_shift",)
        )
        env = make_env(config, "goal_shift", seed + 12_000)
        for phase, step in (("pre", 0), ("post", config.drift_time)):
            env.set_global_step(step)
            reachable = env.reachable_states()
            reachability.append(
                {
                    "seed": seed,
                    "phase": phase,
                    "goal_reachable": all(goal in reachable for goal in env.goals),
                }
            )
    (OUTPUT / "summary.json").write_text(
        json.dumps(
            {
                "design": {
                    "budgets": BUDGETS,
                    "seeds": SEEDS,
                    "drift": "goal_shift",
                    "note": "predeclared diagnostic; not a selection sweep",
                },
                "summary": summary,
                "task_reachability": reachability,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
