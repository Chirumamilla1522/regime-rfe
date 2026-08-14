"""Experimental protocol for inferred transition regimes.

This module deliberately keeps ground-truth regime labels outside the detector
API.  They are recorded only for evaluation and for the explicitly labelled
oracle upper bound.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rfe_drift.env import DriftGridWorld, DriftSchedule, DriftType
from rfe_drift.exploration import CountBasedExplorer

State = Tuple[int, int]


@dataclass(frozen=True)
class RegimeExperimentConfig:
    profile: str = "pilot"
    grid_sizes: Tuple[int, ...] = (5, 7)
    num_walls: Tuple[int, ...] = (0, 3)
    strengths: Tuple[float, ...] = (0.35, 0.7)
    drift_types: Tuple[str, ...] = ("transition_noise", "wall_change")
    schedules: Tuple[str, ...] = ("sudden", "gradual", "periodic")
    seeds: Tuple[int, ...] = (0, 1, 2)
    exploration_steps: int = 1800
    drift_time: int = 600
    eval_episodes: int = 20
    max_episode_steps: int = 80
    detector_warmup: int = 80
    detector_reference_window: int = 50
    detector_recent_window: int = 16
    detector_threshold: float = 0.18
    detector_cooldown: int = 80
    recovery_budgets: Tuple[int, ...] = (0, 50, 200, 600)
    bootstrap_samples: int = 5000


def profile_config(profile: str) -> RegimeExperimentConfig:
    if profile == "quick":
        return RegimeExperimentConfig(
            profile="quick",
            grid_sizes=(5,),
            num_walls=(2,),
            strengths=(0.7,),
            drift_types=("transition_noise",),
            schedules=("sudden",),
            seeds=(0,),
            exploration_steps=500,
            drift_time=200,
            eval_episodes=6,
            max_episode_steps=50,
            detector_warmup=40,
            detector_reference_window=30,
            detector_recent_window=10,
            detector_cooldown=40,
            recovery_budgets=(0, 40, 150, 300),
            bootstrap_samples=1000,
        )
    if profile == "pilot":
        return RegimeExperimentConfig()
    if profile == "full":
        return RegimeExperimentConfig(
            profile="full",
            grid_sizes=(6, 9),
            num_walls=(2, 6),
            strengths=(0.25, 0.5, 0.75),
            drift_types=("goal_shift", "transition_noise", "wall_change", "combined"),
            schedules=("sudden", "gradual", "periodic"),
            seeds=tuple(range(20)),
            exploration_steps=8000,
            drift_time=2500,
            eval_episodes=50,
            max_episode_steps=120,
            detector_warmup=150,
            detector_reference_window=100,
            detector_recent_window=30,
            detector_cooldown=150,
            recovery_budgets=(0, 100, 500, 1500, 4000),
            bootstrap_samples=20000,
        )
    raise ValueError(f"unknown profile: {profile}")


class ResidualChangeDetector:
    """Deterministic online detector based only on observed transitions.

    A smoothed categorical forward model estimates p(s'|s,a).  Its pre-update
    prediction residual, 1-p(s'|s,a), feeds a two-window mean-shift test.
    Alarms create monotonically increasing inferred regime IDs and reset the
    local forward model.  This is a practical signal, not a theorem or an
    optimal change-point procedure.
    """

    def __init__(
        self,
        warmup: int = 80,
        reference_window: int = 50,
        recent_window: int = 16,
        threshold: float = 0.18,
        cooldown: int = 80,
        alpha: float = 0.5,
        minimum_key_observations: int = 3,
    ):
        self.warmup = warmup
        self.reference_window = reference_window
        self.recent_window = recent_window
        self.threshold = threshold
        self.cooldown = cooldown
        self.alpha = alpha
        self.minimum_key_observations = minimum_key_observations
        self.regime_id = 0
        self.step = 0
        self.last_alarm = -cooldown
        self.residuals: deque[float] = deque(
            maxlen=reference_window + recent_window
        )
        self.counts: Dict[Tuple[State, int], Dict[State, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.totals: Dict[Tuple[State, int], int] = defaultdict(int)

    def update(self, state: State, action: int, next_state: State) -> Tuple[int, float, bool]:
        """Consume one transition; no time, boundary, or environment info."""
        state = tuple(map(int, state))
        next_state = tuple(map(int, next_state))
        key = (state, int(action))
        support = 5.0  # four neighbors plus self-loop
        probability = (
            self.counts[key][next_state] + self.alpha
        ) / (self.totals[key] + self.alpha * support)
        residual = 1.0 - probability
        calibrated = self.totals[key] >= self.minimum_key_observations
        if calibrated:
            self.residuals.append(residual)
        self.counts[key][next_state] += 1
        self.totals[key] += 1
        self.step += 1

        alarm = False
        needed = self.reference_window + self.recent_window
        if (
            self.step >= self.warmup
            and len(self.residuals) == needed
            and self.step - self.last_alarm >= self.cooldown
        ):
            values = np.asarray(self.residuals, dtype=float)
            reference = float(values[: self.reference_window].mean())
            recent = float(values[-self.recent_window :].mean())
            # Requiring elevated absolute surprise avoids alarms from benign
            # downward learning curves.
            alarm = recent - reference > self.threshold and recent > 0.35
        if alarm:
            self.regime_id += 1
            self.last_alarm = self.step
            self.counts.clear()
            self.totals.clear()
            self.residuals.clear()
        return self.regime_id, float(residual), alarm

    def state_dict(self) -> Dict:
        return {
            "regime_id": self.regime_id,
            "step": self.step,
            "last_alarm": self.last_alarm,
            "residuals": list(self.residuals),
            "counts": [
                [list(state), action, list(next_state), count]
                for (state, action), outcomes in sorted(self.counts.items())
                for next_state, count in sorted(outcomes.items())
            ],
        }


@dataclass(frozen=True)
class TransitionRecord:
    step: int
    state_x: int
    state_y: int
    action: int
    next_x: int
    next_y: int
    inferred_regime: int
    true_regime: int
    residual: float
    alarm: int

    @property
    def state(self) -> State:
        return (self.state_x, self.state_y)

    @property
    def next_state(self) -> State:
        return (self.next_x, self.next_y)


def _truth_regime(step: int, schedule: str, drift_time: int) -> int:
    """Evaluation label only; never passed to ResidualChangeDetector."""
    if schedule == "periodic":
        return (step // drift_time) % 2
    if schedule == "gradual":
        return 0 if step < drift_time + drift_time // 2 else 1
    return int(step >= drift_time)


def _make_env(
    config: RegimeExperimentConfig,
    grid_size: int,
    walls: int,
    strength: float,
    drift_type: str,
    schedule: str,
    seed: int,
) -> DriftGridWorld:
    return DriftGridWorld(
        grid_size=grid_size,
        num_walls=walls,
        drift_type=DriftType(drift_type),
        drift_strength=strength,
        drift_schedule=DriftSchedule(schedule),
        drift_time=config.drift_time,
        seed=seed,
    )


def collect_transitions(
    config: RegimeExperimentConfig,
    grid_size: int,
    walls: int,
    strength: float,
    drift_type: str,
    schedule: str,
    seed: int,
) -> Tuple[List[TransitionRecord], Dict]:
    env = _make_env(config, grid_size, walls, strength, drift_type, schedule, seed)
    explorer = CountBasedExplorer(grid_size**2, 4, seed=seed)
    detector = ResidualChangeDetector(
        warmup=config.detector_warmup,
        reference_window=config.detector_reference_window,
        recent_window=config.detector_recent_window,
        threshold=config.detector_threshold,
        cooldown=config.detector_cooldown,
    )
    state, _ = env.reset()
    records: List[TransitionRecord] = []
    visited_states = {tuple(state)}
    visited_transitions = set()
    reachable = env.reachable_states()
    for step in range(1, config.exploration_steps + 1):
        action = explorer.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        regime, residual, alarm = detector.update(
            tuple(state), action, tuple(next_state)
        )
        explorer.update(state, action, next_state, done=terminated or truncated)
        explorer.visited_states.add(tuple(next_state))
        visited_states.add(tuple(next_state))
        visited_transitions.add((tuple(state), action, tuple(next_state)))
        reachable |= env.reachable_states()
        records.append(
            TransitionRecord(
                step=step,
                state_x=int(state[0]),
                state_y=int(state[1]),
                action=int(action),
                next_x=int(next_state[0]),
                next_y=int(next_state[1]),
                inferred_regime=regime,
                true_regime=_truth_regime(step, schedule, config.drift_time),
                residual=residual,
                alarm=int(alarm),
            )
        )
        state = env.reset()[0] if terminated or truncated else next_state
    possible = max(1, len(reachable) * 4)
    return records, {
        "state_coverage": len(visited_states & reachable) / max(1, len(reachable)),
        "state_action_coverage": len({(r.state, r.action) for r in records}) / possible,
        "distinct_transition_coverage": len(visited_transitions) / (possible * 5),
        "detector_state": detector.state_dict(),
    }


class EmpiricalPlanner:
    """Tabular maximum-likelihood model and finite value-iteration planner."""

    def __init__(self, grid_size: int, gamma: float = 0.97):
        self.grid_size = grid_size
        self.gamma = gamma
        self.counts: Dict[Tuple[int, State, int], Dict[State, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, records: Sequence[TransitionRecord], context_field: str) -> None:
        for record in records:
            context = (
                0 if context_field == "stationary" else int(getattr(record, context_field))
            )
            self.counts[(context, record.state, record.action)][record.next_state] += 1

    def policy(self, context: int, goals: Iterable[State]) -> Dict[State, int]:
        goals = {tuple(goal) for goal in goals}
        states = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ]
        values = {state: 0.0 for state in states}
        q_values: Dict[Tuple[State, int], float] = {}
        for _ in range(300):
            updated = {}
            delta = 0.0
            for state in states:
                if state in goals:
                    updated[state] = 0.0
                    continue
                action_values = []
                for action in range(4):
                    outcomes = self.counts.get((context, state, action))
                    if not outcomes:
                        value = -0.01 + self.gamma * values[state]
                    else:
                        total = sum(outcomes.values())
                        value = 0.0
                        for next_state, count in outcomes.items():
                            reward = 1.0 if next_state in goals else -0.01
                            continuation = 0.0 if next_state in goals else values[next_state]
                            value += count / total * (
                                reward + self.gamma * continuation
                            )
                    q_values[(state, action)] = value
                    action_values.append(value)
                updated[state] = max(action_values)
                delta = max(delta, abs(updated[state] - values[state]))
            values = updated
            if delta < 1e-9:
                break
        return {
            state: int(np.argmax([q_values[(state, action)] for action in range(4)]))
            for state in states
            if state not in goals
        }


def evaluate_planner(
    planner: EmpiricalPlanner,
    context: int,
    config: RegimeExperimentConfig,
    grid_size: int,
    walls: int,
    strength: float,
    drift_type: str,
    schedule: str,
    seed: int,
    phase_step: int,
) -> Dict:
    env = _make_env(
        config, grid_size, walls, strength, drift_type, schedule, seed + 100_000
    )
    # Evaluation uses an independently stochastic rollout but the same
    # deterministic geometry generator seed as the model task.
    training_env = _make_env(
        config, grid_size, walls, strength, drift_type, schedule, seed
    )
    env.initial_goals = list(training_env.initial_goals)
    env.drifted_goals = list(training_env.drifted_goals)
    env.initial_walls = set(training_env.initial_walls)
    env.drifted_walls = set(training_env.drifted_walls)
    env.set_global_step(phase_step)
    policy = planner.policy(context, env.goals)
    successes = []
    lengths = []
    for _ in range(config.eval_episodes):
        env.set_global_step(phase_step)
        state, _ = env.reset()
        success = 0.0
        for length in range(1, config.max_episode_steps + 1):
            action = policy.get(tuple(state), 0)
            state, reward, terminated, truncated, _ = env.step(action)
            if reward > 0:
                success = 1.0
            if terminated or truncated:
                break
        successes.append(success)
        lengths.append(length)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def _assignment_accuracy(records: Sequence[TransitionRecord]) -> float:
    """Majority-map inferred IDs to truth labels (label permutation invariant)."""
    by_inferred: Dict[int, List[int]] = defaultdict(list)
    for record in records:
        by_inferred[record.inferred_regime].append(record.true_regime)
    correct = sum(
        max(labels.count(0), labels.count(1)) for labels in by_inferred.values()
    )
    return correct / max(1, len(records))


def detector_metrics(
    records: Sequence[TransitionRecord], schedule: str, drift_time: int
) -> Dict:
    alarms = [record.step for record in records if record.alarm]
    if schedule == "periodic":
        changes = list(range(drift_time, records[-1].step + 1, drift_time))
    else:
        changes = [drift_time]
    matched = []
    used = set()
    for change in changes:
        candidates = [
            alarm for alarm in alarms if alarm >= change and alarm not in used
        ]
        if candidates:
            alarm = min(candidates)
            used.add(alarm)
            matched.append(alarm - change)
    false_alarms = sum(alarm not in used for alarm in alarms)
    return {
        "n_true_changes": len(changes),
        "n_alarms": len(alarms),
        "mean_detection_delay": float(np.mean(matched)) if matched else None,
        "detected_change_fraction": len(matched) / max(1, len(changes)),
        "false_alarms": int(false_alarms),
        "false_alarm_rate_per_1000": 1000.0 * false_alarms / len(records),
        "regime_assignment_accuracy": _assignment_accuracy(records),
    }


def _unit_id(values: Dict) -> str:
    text = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ci(values: Sequence[float], samples: int, seed: int) -> Tuple[float, Optional[float], Optional[float]]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), (samples, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return mean, float(low), float(high)


def _clustered_metric_summary(
    rows: Sequence[Dict],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    bootstrap_samples: int,
) -> List[Dict]:
    """Summarize geometry/strength units after averaging within each seed."""
    groups = sorted({tuple(row[field] for field in group_fields) for row in rows})
    output = []
    for group_index, group in enumerate(groups):
        selected = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == group
        ]
        for metric_index, metric in enumerate(metrics):
            seed_values = []
            for seed in sorted({row["seed"] for row in selected}):
                values = [
                    float(row[metric])
                    for row in selected
                    if row["seed"] == seed and row.get(metric) is not None
                ]
                if values:
                    seed_values.append(float(np.mean(values)))
            if not seed_values:
                continue
            mean, low, high = _ci(
                seed_values,
                bootstrap_samples,
                seed=104729 + 101 * group_index + metric_index,
            )
            output.append(
                {
                    **dict(zip(group_fields, group)),
                    "metric": metric,
                    "n_seeds": len(seed_values),
                    "n_units": len(selected),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def _run_unit(config: RegimeExperimentConfig, spec: Dict) -> Dict:
    started = time.perf_counter()
    records, coverage = collect_transitions(config, **spec)
    metrics = detector_metrics(records, spec["schedule"], config.drift_time)
    phase_step = (
        config.drift_time * 2
        if spec["schedule"] == "gradual"
        else config.drift_time
    )
    final_truth = _truth_regime(phase_step, spec["schedule"], config.drift_time)
    final_inferred = records[-1].inferred_regime
    method_rows = []
    for method, context_field, context in (
        ("stationary_no_context", "stationary", 0),
        ("inferred_regime", "inferred_regime", final_inferred),
        ("oracle_regime_upper_bound", "true_regime", final_truth),
    ):
        planner = EmpiricalPlanner(spec["grid_size"])
        planner.fit(records, context_field)
        result = evaluate_planner(
            planner, context, config, phase_step=phase_step, **spec
        )
        method_rows.append({"method": method, **result})

    recovery_rows = []
    for budget in config.recovery_budgets:
        cutoff = min(len(records), config.drift_time + budget)
        prefix = records[:cutoff]
        for method, context_field in (
            ("stationary_no_context", "stationary"),
            ("inferred_regime", "inferred_regime"),
            ("oracle_regime_upper_bound", "true_regime"),
        ):
            context = 0
            if prefix and context_field != "stationary":
                context = int(getattr(prefix[-1], context_field))
            planner = EmpiricalPlanner(spec["grid_size"])
            planner.fit(prefix, context_field)
            result = evaluate_planner(
                planner, context, config, phase_step=phase_step, **spec
            )
            recovery_rows.append(
                {
                    "method": method,
                    "post_drift_samples": budget,
                    **result,
                }
            )
    return {
        "spec": spec,
        "records": [asdict(record) for record in records],
        "coverage": coverage,
        "detector": metrics,
        "methods": method_rows,
        "recovery": recovery_rows,
        "runtime_seconds": time.perf_counter() - started,
    }


def run_regime_experiments(
    config: RegimeExperimentConfig, output: Path, resume: bool = True
) -> Dict:
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    specs = [
        {
            "grid_size": grid_size,
            "walls": walls,
            "strength": strength,
            "drift_type": drift_type,
            "schedule": schedule,
            "seed": seed,
        }
        for grid_size in config.grid_sizes
        for walls in config.num_walls
        for strength in config.strengths
        for drift_type in config.drift_types
        for schedule in config.schedules
        for seed in config.seeds
    ]
    config_hash = _unit_id(asdict(config))
    units = []
    for spec in specs:
        unit_id = _unit_id({"config": asdict(config), "spec": spec})
        checkpoint = checkpoints / f"{unit_id}.json"
        if resume and checkpoint.exists():
            unit = json.loads(checkpoint.read_text())
            if unit.get("config_hash") == config_hash:
                units.append(unit)
                continue
        unit = _run_unit(config, spec)
        unit["unit_id"] = unit_id
        unit["config_hash"] = config_hash
        checkpoint.write_text(json.dumps(unit, indent=2, sort_keys=True))
        units.append(unit)

    stationary_rows = []
    for grid_size in config.grid_sizes:
        for walls in config.num_walls:
            for seed in config.seeds:
                spec = {
                    "grid_size": grid_size,
                    "walls": walls,
                    "strength": 0.0,
                    "drift_type": "transition_noise",
                    "schedule": "sudden",
                    "seed": seed,
                }
                check_id = _unit_id({"stationary": asdict(config), "spec": spec})
                checkpoint = checkpoints / f"stationary_{check_id}.json"
                if resume and checkpoint.exists():
                    check = json.loads(checkpoint.read_text())
                else:
                    records, coverage = collect_transitions(config, **spec)
                    planner = EmpiricalPlanner(grid_size)
                    planner.fit(records, "stationary")
                    evaluation = evaluate_planner(
                        planner, 0, config, phase_step=0, **spec
                    )
                    check = {
                        **spec,
                        **evaluation,
                        "state_action_coverage": coverage["state_action_coverage"],
                    }
                    checkpoint.write_text(json.dumps(check, indent=2, sort_keys=True))
                stationary_rows.append(check)

    transitions, seed_rows, detection_rows, coverage_rows, recovery_rows = [], [], [], [], []
    for unit in units:
        # JSON checkpoints sort object keys; rebuild a canonical field order so
        # resumed CSVs are byte-identical to uninterrupted CSVs.
        identity = {
            key: unit["spec"][key]
            for key in (
                "grid_size",
                "walls",
                "strength",
                "drift_type",
                "schedule",
                "seed",
            )
        }
        transitions.extend({**identity, **row} for row in unit["records"])
        seed_rows.extend({**identity, **row} for row in unit["methods"])
        detection_rows.append({**identity, **unit["detector"]})
        coverage_rows.append(
            {
                **identity,
                **{k: v for k, v in unit["coverage"].items() if k != "detector_state"},
            }
        )
        recovery_rows.extend({**identity, **row} for row in unit["recovery"])

    summary = []
    keys = sorted(
        {(row["drift_type"], row["schedule"], row["method"]) for row in seed_rows}
    )
    for index, (drift_type, schedule, method) in enumerate(keys):
        selected_rows = [
            row
            for row in seed_rows
            if (row["drift_type"], row["schedule"], row["method"])
            == (drift_type, schedule, method)
        ]
        selected = [
            float(
                np.mean(
                    [
                        row["success_rate"]
                        for row in selected_rows
                        if row["seed"] == seed
                    ]
                )
            )
            for seed in sorted({row["seed"] for row in selected_rows})
        ]
        mean, low, high = _ci(
            selected, config.bootstrap_samples, seed=index + 7919
        )
        summary.append(
            {
                "drift_type": drift_type,
                "schedule": schedule,
                "method": method,
                "n_seeds": len(selected),
                "n_units": len(selected_rows),
                "success_mean": mean,
                "success_ci95_low": low,
                "success_ci95_high": high,
            }
        )
    detection_summary = _clustered_metric_summary(
        detection_rows,
        ("drift_type", "schedule"),
        (
            "mean_detection_delay",
            "detected_change_fraction",
            "false_alarm_rate_per_1000",
            "regime_assignment_accuracy",
        ),
        config.bootstrap_samples,
    )
    coverage_summary = _clustered_metric_summary(
        coverage_rows,
        ("drift_type", "schedule"),
        (
            "state_coverage",
            "state_action_coverage",
            "distinct_transition_coverage",
        ),
        config.bootstrap_samples,
    )
    recovery_summary = _clustered_metric_summary(
        recovery_rows,
        ("drift_type", "schedule", "method", "post_drift_samples"),
        ("success_rate", "mean_episode_length"),
        config.bootstrap_samples,
    )
    stationary_summary = _clustered_metric_summary(
        stationary_rows,
        (),
        ("success_rate", "mean_episode_length", "state_action_coverage"),
        config.bootstrap_samples,
    )
    _write_csv(output / "transitions.csv", transitions)
    _write_csv(output / "per_seed.csv", seed_rows)
    _write_csv(output / "detection.csv", detection_rows)
    _write_csv(output / "coverage.csv", coverage_rows)
    _write_csv(output / "recovery.csv", recovery_rows)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "detection_summary.csv", detection_summary)
    _write_csv(output / "coverage_summary.csv", coverage_summary)
    _write_csv(output / "recovery_summary.csv", recovery_summary)
    _write_csv(output / "stationary_check.csv", stationary_rows)
    _write_csv(output / "stationary_check_summary.csv", stationary_summary)
    payload = {
        "protocol": "inferred_residual_regimes_v1",
        "config": asdict(config),
        "config_hash": config_hash,
        "n_units": len(units),
        "runtime_seconds": sum(unit["runtime_seconds"] for unit in units),
        "summary": summary,
        "detection_summary": detection_summary,
        "stationary_check_summary": stationary_summary,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2))
    return payload


def deterministic_resume_probe(config: RegimeExperimentConfig, output: Path) -> Tuple[str, str]:
    """Run twice and return hashes of derived outputs for a resume test."""
    run_regime_experiments(config, output, resume=True)
    files = ("transitions.csv", "per_seed.csv", "detection.csv", "recovery.csv")
    first = hashlib.sha256(
        b"".join((output / name).read_bytes() for name in files)
    ).hexdigest()
    run_regime_experiments(config, output, resume=True)
    second = hashlib.sha256(
        b"".join((output / name).read_bytes() for name in files)
    ).hexdigest()
    return first, second
