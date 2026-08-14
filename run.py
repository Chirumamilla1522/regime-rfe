#!/usr/bin/env python3
"""Canonical reproducible experiment harness.

The implemented explorer is a count-bonus heuristic, not UCRL-RFE.  The
time-conditioned representation receives a normalized public clock; it does
not receive the drift boundary, active goals, walls, or any oracle flag.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict, dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rfe_drift.env import DriftGridWorld, DriftSchedule, DriftType
from rfe_drift.exploration import CountBasedExplorer
from rfe_drift.representations import (
    DriftAwareEncoder,
    FixedEncoder,
    RepresentationTrainer,
)
from rfe_drift.rl import DQNAgent
from rfe_drift.protocol import profile_config, run_regime_experiments
from rfe_drift.recurrent_study import recurrent_profile, run_recurrent_study
from rfe_drift.minigrid_study import minigrid_profile, run_minigrid_study
from rfe_drift.benchmark import benchmark_profile, run_benchmark_study


@dataclass
class ExperimentConfig:
    grid_size: int = 6
    num_walls: int = 2
    drift_strength: float = 0.6
    drift_time: int = 2000
    exploration_steps: int = 4000
    representation_epochs: int = 8
    training_steps: int = 4000
    eval_episodes: int = 20
    max_episode_steps: int = 60
    seeds: Tuple[int, ...] = tuple(range(10))
    drift_types: Tuple[str, ...] = (
        "goal_shift",
        "transition_noise",
        "wall_change",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalized_time(global_step: int, config: ExperimentConfig) -> float:
    return min(1.0, global_step / max(1, config.training_steps))


def make_env(config: ExperimentConfig, drift_type: str, seed: int) -> DriftGridWorld:
    return DriftGridWorld(
        grid_size=config.grid_size,
        drift_type=DriftType(drift_type),
        drift_strength=config.drift_strength,
        drift_schedule=DriftSchedule.SUDDEN,
        drift_time=config.drift_time,
        num_walls=config.num_walls,
        seed=seed,
    )


def collect_data(
    config: ExperimentConfig, drift_type: str, seed: int, strategy: str
) -> Tuple[List[Tuple], float]:
    env = make_env(config, drift_type, seed)
    initial_reachable = env.reachable_states()
    explorer = CountBasedExplorer(
        state_dim=config.grid_size**2, action_dim=4, seed=seed
    )
    rng = np.random.RandomState(seed)
    state, _ = env.reset()
    for _ in range(config.exploration_steps):
        time = normalized_time(env.step_count, config)
        action = (
            explorer.select_action(state)
            if strategy == "count"
            else int(rng.randint(env.action_space.n))
        )
        next_state, _, terminated, truncated, _ = env.step(action)
        next_time = normalized_time(env.step_count, config)
        explorer.update(
            state,
            action,
            next_state,
            done=terminated or truncated,
            time=time,
            next_time=next_time,
        )
        explorer.visited_states.add(tuple(next_state))
        state = env.reset()[0] if terminated or truncated else next_state
    reachable = initial_reachable | env.reachable_states()
    coverage = explorer.get_state_coverage(reachable)
    return explorer.get_replay_buffer(), coverage


def train_encoders(
    records: Sequence[Tuple], config: ExperimentConfig, seed: int
) -> Tuple[FixedEncoder, DriftAwareEncoder]:
    # The representation ablation is strictly matched: both encoders receive
    # the same ordered transitions. The only difference is whether record[4]
    # (the public clock) enters the encoder.
    matched_records = list(records)
    fixed = FixedEncoder(input_dim=2, hidden_dim=32, output_dim=16)
    temporal = DriftAwareEncoder(
        input_dim=2, hidden_dim=32, output_dim=16, context_dim=8
    )
    RepresentationTrainer(fixed, batch_size=32).train_forward_dynamics(
        matched_records, num_epochs=config.representation_epochs, seed=seed
    )
    RepresentationTrainer(temporal, batch_size=32).train_forward_dynamics(
        matched_records, num_epochs=config.representation_epochs, seed=seed
    )
    return fixed, temporal


def train_agent(
    encoder, config: ExperimentConfig, drift_type: str, seed: int
) -> DQNAgent:
    env = make_env(config, drift_type, seed + 10_000)
    agent = DQNAgent(
        action_dim=4,
        encoder=encoder,
        hidden_dim=32,
        batch_size=32,
        epsilon=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.05,
        seed=seed,
    )
    state, _ = env.reset()
    for _ in range(config.training_steps):
        time = normalized_time(env.step_count, config)
        action = agent.select_action(state, time=time, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_time = normalized_time(env.step_count, config)
        episode_done = (
            terminated
            or truncated
            or env.episode_step_count >= config.max_episode_steps
        )
        agent.update(
            state,
            action,
            reward,
            next_state,
            episode_done,
            time=time,
            next_time=next_time,
        )
        state = env.reset()[0] if episode_done else next_state
    return agent


def evaluate(
    agent: DQNAgent,
    config: ExperimentConfig,
    drift_type: str,
    seed: int,
    phase: str,
) -> List[Dict]:
    # Evaluation uses the training task layout/goal and independent stochastic
    # rollouts. A different layout would be an unobservable task-generalization
    # problem because the observation contains only agent coordinates.
    env = make_env(config, drift_type, seed + 10_000)
    phase_step = 0 if phase == "pre" else config.drift_time
    rows = []
    for episode in range(config.eval_episodes):
        # Freeze each trial at the intended side of the boundary. Environment
        # transitions and representation context use the same public clock.
        env.set_global_step(phase_step)
        state, _ = env.reset()
        total_reward = 0.0
        for length in range(1, config.max_episode_steps + 1):
            time = normalized_time(env.step_count, config)
            action = agent.select_action(state, time=time, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rows.append(
            {
                "episode": episode,
                "phase": phase,
                "return": total_reward,
                "success": float(total_reward > 0),
                "length": length,
            }
        )
    return rows


def confidence_interval(
    values: Iterable[float], seed: int = 0, bootstrap_samples: int = 20_000
) -> Tuple[float, Optional[float], Optional[float]]:
    """Bounded percentile cluster bootstrap over independent seed means."""
    values = np.asarray(list(values), dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None, None
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(bootstrap_samples, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean, float(max(0.0, low)), float(min(1.0, high))


def summarize(raw_rows: List[Dict], seed_rows: List[Dict]) -> List[Dict]:
    summaries = []
    keys = sorted(
        {
            (row["drift_type"], row["method"], row["phase"])
            for row in seed_rows
        }
    )
    for drift_type, method, phase in keys:
        selected = [
            row
            for row in seed_rows
            if (row["drift_type"], row["method"], row["phase"])
            == (drift_type, method, phase)
        ]
        ci_seed = sum(ord(char) for char in f"{drift_type}:{method}:{phase}")
        mean, low, high = confidence_interval(
            (row["success_rate"] for row in selected), seed=ci_seed
        )
        # Returns equal binary success in this sparse-reward environment, so
        # the same bounded cluster bootstrap is appropriate.
        ret_mean, ret_low, ret_high = confidence_interval(
            (row["mean_return"] for row in selected), seed=ci_seed
        )
        summaries.append(
            {
                "drift_type": drift_type,
                "method": method,
                "phase": phase,
                "n_seeds": len(selected),
                "success_mean": mean,
                "success_ci95_low": low,
                "success_ci95_high": high,
                "return_mean": ret_mean,
                "return_ci95_low": ret_low,
                "return_ci95_high": ret_high,
            }
        )
    return summaries


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: List[Dict], output: Path) -> None:
    post = [row for row in summary if row["phase"] == "post"]
    labels = [f"{row['drift_type']}\n{row['method']}" for row in post]
    means = np.array([row["success_mean"] for row in post])
    errors = np.array(
        [
            [
                row["success_mean"] - row["success_ci95_low"]
                if row["success_ci95_low"] is not None
                else 0.0
                for row in post
            ],
            [
                row["success_ci95_high"] - row["success_mean"]
                if row["success_ci95_high"] is not None
                else 0.0
                for row in post
            ],
        ]
    )
    fig, ax = plt.subplots(figsize=(max(7, len(post) * 1.1), 4))
    ax.bar(np.arange(len(post)), means, yerr=errors, capsize=3)
    ax.set_xticks(np.arange(len(post)), labels, rotation=25, ha="right")
    ax.set_ylabel("Post-drift success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Mean across seeds (95% cluster-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(config: ExperimentConfig, output: Path) -> Dict:
    output.mkdir(parents=True, exist_ok=True)
    raw_rows: List[Dict] = []
    seed_rows: List[Dict] = []
    coverage_rows: List[Dict] = []
    for drift_type in config.drift_types:
        for seed in config.seeds:
            seed_everything(seed)
            count_data, count_coverage = collect_data(
                config, drift_type, seed, "count"
            )
            random_data, random_coverage = collect_data(
                config, drift_type, seed, "random"
            )
            fixed_count, temporal_count = train_encoders(
                count_data, config, seed
            )
            fixed_random, _ = train_encoders(random_data, config, seed + 1000)
            methods = {
                "fixed_count": fixed_count,
                "time_count": temporal_count,
                "fixed_random": fixed_random,
            }
            coverage_rows.extend(
                [
                    {
                        "drift_type": drift_type,
                        "seed": seed,
                        "strategy": "count",
                        "coverage": count_coverage,
                    },
                    {
                        "drift_type": drift_type,
                        "seed": seed,
                        "strategy": "random",
                        "coverage": random_coverage,
                    },
                ]
            )
            for method, encoder in methods.items():
                # Each method gets identical downstream initialization and env.
                seed_everything(seed + 2000)
                agent = train_agent(
                    copy.deepcopy(encoder), config, drift_type, seed + 2000
                )
                for phase in ("pre", "post"):
                    episodes = evaluate(
                        agent, config, drift_type, seed + 2000, phase
                    )
                    for row in episodes:
                        raw_rows.append(
                            {
                                "drift_type": drift_type,
                                "seed": seed,
                                "method": method,
                                **row,
                            }
                        )
                    seed_rows.append(
                        {
                            "drift_type": drift_type,
                            "seed": seed,
                            "method": method,
                            "phase": phase,
                            "success_rate": float(
                                np.mean([row["success"] for row in episodes])
                            ),
                            "mean_return": float(
                                np.mean([row["return"] for row in episodes])
                            ),
                        }
                    )
    summary = summarize(raw_rows, seed_rows)
    write_csv(output / "episodes.csv", raw_rows)
    write_csv(output / "per_seed.csv", seed_rows)
    write_csv(output / "coverage.csv", coverage_rows)
    write_csv(output / "summary.csv", summary)
    payload = {"config": asdict(config), "summary": summary}
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    plot_summary(summary, output / "post_drift_success.pdf")
    plot_summary(summary, output / "post_drift_success.png")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/canonical"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=("clock", "regime", "recurrent-tabular", "minigrid-recurrent", "recurrent-bench"),
        default="clock",
        help="Select a retained study or the recurring MiniGrid scale study.",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "quick",
            "pilot",
            "full",
            "fourrooms",
            "stress",
            "certified",
            "certified-full",
            "conflict",
            "conflict-full",
        ),
        default="pilot",
        help="Budget profile for --protocol regime.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute completed regime-protocol units.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol == "recurrent-bench":
        config = benchmark_profile("quick" if args.quick else args.profile)
        if args.seeds is not None:
            config = dataclass_replace(config, seeds=tuple(args.seeds))
        payload = run_benchmark_study(
            config, args.output, resume=not args.no_resume
        )
        print(json.dumps(payload["summary"], indent=2))
        print(json.dumps(payload["per_task_summary"], indent=2))
        print(
            "Benchmark gate: "
            f"{'PASS' if payload['benchmark_gate_passed'] else 'FAIL'}"
        )
        print(f"Wrote RFE-Recurrent-Bench outputs to {args.output}")
        return
    if args.protocol == "minigrid-recurrent":
        config = minigrid_profile("quick" if args.quick else args.profile)
        if args.seeds is not None:
            config = dataclass_replace(config, seeds=tuple(args.seeds))
        payload = run_minigrid_study(
            config, args.output, resume=not args.no_resume
        )
        print(json.dumps(payload["summary"], indent=2))
        print(
            f"MiniGrid gate: {'PASS' if payload['minigrid_gate_passed'] else 'FAIL'}"
        )
        print(f"Wrote recurring MiniGrid outputs to {args.output}")
        return
    if args.protocol == "recurrent-tabular":
        config = recurrent_profile("quick" if args.quick else args.profile)
        if args.seeds is not None:
            config = dataclass_replace(config, seeds=tuple(args.seeds))
        payload = run_recurrent_study(
            config, args.output, resume=not args.no_resume
        )
        print(json.dumps(payload["summary"], indent=2))
        print(
            f"Recurrence gate: {'PASS' if payload['recurrence_gate_passed'] else 'FAIL'}"
        )
        print(f"Wrote recurrent-tabular outputs to {args.output}")
        return
    if args.protocol == "regime":
        config = profile_config("quick" if args.quick else args.profile)
        if args.seeds is not None:
            config = dataclass_replace(config, seeds=tuple(args.seeds))
        payload = run_regime_experiments(
            config, args.output, resume=not args.no_resume
        )
        print(json.dumps(payload["summary"], indent=2))
        print(f"Wrote inferred-regime outputs to {args.output}")
        return
    config = ExperimentConfig()
    if args.seeds is not None:
        config.seeds = tuple(args.seeds)
    if args.quick:
        config.exploration_steps = 120
        config.representation_epochs = 2
        config.training_steps = 180
        config.drift_time = 60
        config.eval_episodes = 3
        config.max_episode_steps = 20
        config.drift_types = ("goal_shift",)
    payload = run(config, args.output)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote reproducible outputs to {args.output}")


if __name__ == "__main__":
    main()
