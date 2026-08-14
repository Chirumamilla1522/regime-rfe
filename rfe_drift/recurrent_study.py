"""Runnable recurrent tabular Detect--Match--Explore study.

Ground-truth mode IDs and boundaries are confined to synthetic generation,
evaluation, and methods whose names start with ``oracle_``.  All other mode
assignment and deployment diagnosis consumes transition triples only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from rfe_drift.synthetic import (
    RecurringMDPConfig,
    RecurringTabularMDP,
    exact_finite_horizon_value,
)
from rfe_drift.tabular import (
    ConditionalWindowDetector,
    CoveringThenPlanCollector,
    LikelihoodModeLearner,
    ModeTransitionModel,
    ProbeWindowDetector,
    RecurringModeLearner,
    RecurringModeMatcher,
    Transition,
    finite_horizon_plan,
    identify_deployment_mode,
    probe_window_length,
)


METHODS = (
    "pooled",
    "restart_no_reuse",
    "cluster_no_quarantine",
    "recurrence_aware",
    "mbcd_like",
    "oracle_boundary",
    "oracle_mode",
    "sliding_window",
)
PROTOCOL_VERSION = "recurrent_tabular_finite_horizon_v3"


@dataclass(frozen=True)
class RecurrentStudyConfig:
    profile: str = "pilot"
    seeds: Tuple[int, ...] = tuple(range(10))
    state_counts: Tuple[int, ...] = (9,)
    mode_counts: Tuple[int, ...] = (2, 3)
    separations: Tuple[float, ...] = (0.75, 0.9)
    dwell: int = 500
    cycles: int = 3
    rare_occupancy: float = 0.03
    deployment_prefix: int = 96
    quarantine_steps: int = 20
    confirmation_steps: int = 48
    horizon: int = 20
    value_gap_target: float = 0.05
    gate_gap_margin: float = 0.01
    gate_bootstrap_samples: int = 20_000
    detector_recent: int | None = None
    detector_reference: int | None = None
    detector_cooldown: int | None = None
    occupancy_weights: Tuple[float, ...] | None = None
    certified: bool = False
    declared_separation: float = 0.9
    probe_policy: str = "covering"
    use_residual: bool = True


def recurrent_profile(profile: str) -> RecurrentStudyConfig:
    if profile == "quick":
        return RecurrentStudyConfig(
            profile="quick",
            seeds=(0, 1),
            mode_counts=(2,),
            separations=(0.9,),
            dwell=180,
            cycles=2,
            deployment_prefix=48,
            quarantine_steps=8,
            confirmation_steps=24,
        )
    if profile == "pilot":
        return RecurrentStudyConfig()
    if profile == "full":
        return RecurrentStudyConfig(
            profile="full",
            seeds=tuple(range(30)),
            state_counts=(9, 13),
            mode_counts=(2, 3, 4),
            separations=(0.55, 0.75, 0.9),
            dwell=900,
            cycles=4,
            deployment_prefix=160,
            quarantine_steps=30,
            confirmation_steps=80,
        )
    if profile == "stress":
        return RecurrentStudyConfig(
            profile="stress",
            seeds=tuple(range(10)),
            mode_counts=(2, 3, 5),
            separations=(0.75, 0.9),
            dwell=140,
            cycles=4,
            rare_occupancy=0.12,
            occupancy_weights=(3.0, 1.0),
            quarantine_steps=10,
            confirmation_steps=24,
            detector_recent=12,
            detector_reference=24,
            detector_cooldown=16,
        )
    if profile in ("certified", "certified-full"):
        window = probe_window_length(
            separation=0.9,
            alphabet=9,
            split_candidates=48,
            failure_probability=0.05,
        )
        seeds = tuple(range(30)) if profile == "certified-full" else tuple(range(10))
        return RecurrentStudyConfig(
            profile=profile,
            seeds=seeds,
            state_counts=(9,),
            mode_counts=(2,),
            separations=(0.9,),
            dwell=2 * window + 120,
            cycles=4,
            rare_occupancy=0.03,
            quarantine_steps=window,
            confirmation_steps=window,
            detector_recent=window,
            detector_reference=window,
            detector_cooldown=window,
            certified=True,
            declared_separation=0.9,
            probe_policy="uniform",
            use_residual=False,
        )
    raise ValueError(f"unknown recurrent profile: {profile}")


def _detector(config: RecurrentStudyConfig):
    if config.certified:
        window = int(config.detector_recent or config.confirmation_steps)
        return ProbeWindowDetector(
            window=window,
            separation=config.declared_separation,
            false_alarm_probability=0.05,
        )
    recent = int(config.detector_recent or max(8, config.confirmation_steps // 2))
    reference = int(config.detector_reference or max(24, recent * 2))
    cooldown = int(config.detector_cooldown or max(20, config.confirmation_steps))
    return ConditionalWindowDetector(
        reference_window=reference,
        recent_window=recent,
        false_alarm_probability=0.05,
        cooldown=cooldown,
        minimum_overlap_keys=1,
    )


def _matcher(config: RecurrentStudyConfig, reuse: bool = True) -> RecurringModeMatcher:
    return RecurringModeMatcher(
        acceptance_radius=0.35 if reuse else -1.0,
        minimum_evidence_keys=2,
        minimum_key_samples=2,
        ambiguity_margin=0.05,
        minimum_support_fraction=0.4,
        strict=bool(config.certified),
    )


def _collect(
    environment: RecurringTabularMDP, config: RecurrentStudyConfig
) -> Tuple[List[Transition], List[int], List[int]]:
    """Reward-free trajectory using the greedy covering collector."""
    collector = CoveringThenPlanCollector(
        environment.actions, seed=environment.config.seed + 17
    )
    rng = np.random.RandomState(environment.config.seed + 19)
    transitions: List[Transition] = []
    modes: List[int] = []
    occurrences: List[int] = []
    state = environment.reset()
    if hasattr(environment, "block_lengths"):
        lengths = environment.block_lengths
    else:
        total_blocks = len(getattr(environment, "mode_sequence", ()))
        lengths = tuple(config.dwell for _ in range(max(1, total_blocks)))
    ends = np.cumsum(lengths)
    total = int(ends[-1]) if len(ends) else 0
    for step in range(total):
        if config.probe_policy == "uniform":
            action = int(rng.choice(environment.actions))
        else:
            action = collector.select_action(state)
        mode = environment.mode_at(step)
        next_state = environment.sample(mode, state, action)
        collector.update(state, action, next_state)
        transitions.append((state, action, next_state))
        modes.append(mode)
        occurrences.append(int(np.searchsorted(ends, step, side="right")))
        state = next_state
    return transitions, modes, occurrences


def _fit_online(
    transitions: Sequence[Transition],
    config: RecurrentStudyConfig,
    quarantine: int,
    reuse: bool,
) -> Tuple[ModeTransitionModel, Dict]:
    learner = RecurringModeLearner(
        detector=_detector(config),
        matcher=_matcher(config, reuse=reuse),
        quarantine_steps=quarantine,
        confirmation_steps=config.confirmation_steps,
        rollback_steps=quarantine,
        use_residual=config.use_residual,
    )
    alarms = confirmations = quarantined = 0
    for transition in transitions:
        assignment = learner.update(*transition)
        alarms += int(assignment.alarm)
        confirmations += int(assignment.status == "confirmed")
        quarantined += int(assignment.status == "quarantine")
    return learner.models, {
        "alarms": alarms,
        "confirmations": confirmations,
        "quarantined": quarantined,
        "reuse_events": learner.reuse_events,
        "learned_modes": len(learner.models.modes),
    }


def _fit_mbcd(
    transitions: Sequence[Transition],
    config: RecurrentStudyConfig,
) -> Tuple[ModeTransitionModel, Dict]:
    learner = LikelihoodModeLearner(
        window=max(8, int(config.confirmation_steps // 2)),
    )
    alarms = switches = 0
    for transition in transitions:
        assignment = learner.update(*transition)
        alarms += int(assignment.alarm)
        switches += int(assignment.reused)
    return learner.models, {
        "alarms": alarms,
        "confirmations": learner.switch_events,
        "quarantined": 0,
        "reuse_events": learner.reuse_events,
        "learned_modes": len(learner.models.modes),
    }


def _fit_oracle_boundary(
    transitions: Sequence[Transition],
    modes: Sequence[int],
    config: RecurrentStudyConfig,
) -> Tuple[ModeTransitionModel, Dict]:
    """Use oracle boundaries but transition-only recurrence matching."""
    model = ModeTransitionModel()
    matcher = _matcher(config)
    next_mode = 0
    reused = rejected = 0
    for start in range(0, len(transitions), config.dwell):
        segment = list(transitions[start : start + config.dwell])
        diagnostic = segment[
            config.quarantine_steps : config.quarantine_steps
            + config.confirmation_steps
        ]
        result = matcher.match(diagnostic, model)
        if result.accepted:
            mode_id = int(result.mode)
            reused += 1
        else:
            mode_id = next_mode
            next_mode += 1
            rejected += 1
        model.update_many(mode_id, segment[config.quarantine_steps :])
    return model, {
        "alarms": max(0, len(set(range(0, len(transitions), config.dwell))) - 1),
        "confirmations": len(transitions) // config.dwell,
        "quarantined": config.quarantine_steps * (len(transitions) // config.dwell),
        "reuse_events": reused,
        "learned_modes": len(model.modes),
        "new_mode_events": rejected,
    }


def _fit_models(
    transitions: Sequence[Transition],
    true_modes: Sequence[int],
    config: RecurrentStudyConfig,
) -> Tuple[Dict[str, ModeTransitionModel], Dict[str, Dict]]:
    models: Dict[str, ModeTransitionModel] = {}
    diagnostics: Dict[str, Dict] = {}
    pooled = ModeTransitionModel()
    pooled.update_many(0, transitions)
    models["pooled"] = pooled
    diagnostics["pooled"] = {"learned_modes": 1}

    models["restart_no_reuse"], diagnostics["restart_no_reuse"] = _fit_online(
        transitions, config, config.quarantine_steps, reuse=False
    )
    models["cluster_no_quarantine"], diagnostics["cluster_no_quarantine"] = _fit_online(
        transitions, config, 0, reuse=True
    )
    models["recurrence_aware"], diagnostics["recurrence_aware"] = _fit_online(
        transitions, config, config.quarantine_steps, reuse=True
    )
    models["mbcd_like"], diagnostics["mbcd_like"] = _fit_mbcd(transitions, config)
    models["oracle_boundary"], diagnostics["oracle_boundary"] = _fit_oracle_boundary(
        transitions, true_modes, config
    )
    oracle = ModeTransitionModel()
    for transition, mode in zip(transitions, true_modes):
        oracle.update(mode, *transition)
    models["oracle_mode"] = oracle
    diagnostics["oracle_mode"] = {"learned_modes": len(oracle.modes)}
    window = ModeTransitionModel()
    window.update_many(0, transitions[-config.dwell :])
    models["sliding_window"] = window
    diagnostics["sliding_window"] = {"learned_modes": 1}
    return models, diagnostics


def _balanced_prefix(
    environment: RecurringTabularMDP, mode: int, length: int, seed: int
) -> List[Transition]:
    """Deployment-only prefix; mode is used by the simulator, not diagnosis."""
    rng = np.random.RandomState(seed)
    pairs = [
        (state, action)
        for state in environment.states
        for action in environment.actions
    ]
    output = []
    for index in range(length):
        state, action = pairs[index % len(pairs)]
        distribution = environment.distribution(mode, state, action)
        outcomes = sorted(distribution)
        next_state = int(
            rng.choice(outcomes, p=[distribution[outcome] for outcome in outcomes])
        )
        output.append((state, action, next_state))
    return output


def _optimal_values(
    environment: RecurringTabularMDP,
    rewards: Mapping[str, Mapping[Tuple[int, int, int], float]],
    horizon: int,
) -> Dict[Tuple[int, str], float]:
    truth = environment.true_model()
    output = {}
    for mode in range(environment.config.mode_count):
        for name, reward in rewards.items():
            policy, _ = finite_horizon_plan(
                truth,
                mode,
                environment.states,
                environment.actions,
                reward,
                horizon,
            )
            output[(mode, name)] = exact_finite_horizon_value(
                environment, mode, policy, reward, horizon
            )
    return output


def _evaluate(
    environment: RecurringTabularMDP,
    models: Mapping[str, ModeTransitionModel],
    config: RecurrentStudyConfig,
) -> Tuple[List[Dict], List[Dict]]:
    rewards = environment.reward_family(config.horizon)
    optimal = _optimal_values(environment, rewards, config.horizon)
    rows: List[Dict] = []
    diagnosis: List[Dict] = []
    for true_mode in range(environment.config.mode_count):
        prefix = _balanced_prefix(
            environment,
            true_mode,
            config.deployment_prefix,
            environment.config.seed + 50_000 + true_mode,
        )
        for method, model in models.items():
            if method == "oracle_mode":
                selected, accepted, distance = true_mode, True, 0.0
            elif method in {"pooled", "sliding_window"}:
                selected, accepted, distance = 0, True, 0.0
            else:
                match = identify_deployment_mode(
                    prefix, model, acceptance_radius=0.55, minimum_evidence_keys=2
                )
                selected, accepted, distance = match.mode, match.accepted, match.distance
            diagnosis.append(
                {
                    "method": method,
                    "true_mode": true_mode,
                    "selected_mode": selected,
                    "accepted": int(accepted),
                    "distance": distance,
                    "prefix_samples": len(prefix),
                }
            )
            gaps = []
            for reward_name, reward in rewards.items():
                oracle_value = optimal[(true_mode, reward_name)]
                if accepted and selected is not None:
                    policy, _ = finite_horizon_plan(
                        model,
                        int(selected),
                        environment.states,
                        environment.actions,
                        reward,
                        config.horizon,
                    )
                    learned_value = exact_finite_horizon_value(
                        environment,
                        true_mode,
                        policy,
                        reward,
                        config.horizon,
                    )
                    gap = min(1.0, max(0.0, oracle_value - learned_value))
                else:
                    learned_value = None
                    gap = max(0.0, oracle_value)
                gaps.append(gap)
                rows.append(
                    {
                        "method": method,
                        "true_mode": true_mode,
                        "reward": reward_name,
                        "oracle_value": oracle_value,
                        "learned_value": learned_value,
                        "value_gap": gap,
                    }
                )
            rows.append(
                {
                    "method": method,
                    "true_mode": true_mode,
                    "reward": "__worst__",
                    "oracle_value": None,
                    "learned_value": None,
                    "value_gap": max(gaps),
                }
            )
    return rows, diagnosis


def _recurrence_savings(
    environment: RecurringTabularMDP,
    transitions: Sequence[Transition],
    modes: Sequence[int],
    config: RecurrentStudyConfig,
) -> List[Dict]:
    """Oracle-evaluated samples needed after a recurrence; rewards stay post-hoc."""
    rewards = environment.reward_family(config.horizon)
    optimal = _optimal_values(environment, rewards, config.horizon)
    output = []
    seen: Dict[int, List[Transition]] = {}
    cursor = 0
    lengths = environment.block_lengths
    for occurrence, length in enumerate(lengths):
        mode = modes[cursor] if cursor < len(modes) else 0
        segment = list(transitions[cursor : cursor + length])
        cursor += length
        if mode in seen:
            budgets = sorted(
                set(
                    [8, 16, 32, 64, 128, 256, config.dwell]
                    + [config.confirmation_steps]
                )
            )
            needed = {}
            for method in ("restart_no_reuse", "recurrence_aware"):
                for budget in budgets:
                    model = ModeTransitionModel()
                    data = (
                        seen[mode] + segment[:budget]
                        if method == "recurrence_aware"
                        else segment[:budget]
                    )
                    model.update_many(0, data)
                    worst = 0.0
                    for reward_name, reward in rewards.items():
                        policy, _ = finite_horizon_plan(
                            model,
                            0,
                            environment.states,
                            environment.actions,
                            reward,
                            config.horizon,
                        )
                        value = exact_finite_horizon_value(
                            environment,
                            mode,
                            policy,
                            reward,
                            config.horizon,
                        )
                        worst = max(worst, optimal[(mode, reward_name)] - value)
                    if worst <= config.value_gap_target:
                        needed[method] = min(budget, config.dwell)
                        break
                needed.setdefault(method, config.dwell + 1)
            output.append(
                {
                    "occurrence": occurrence,
                    "true_mode": mode,
                    "restart_samples": needed["restart_no_reuse"],
                    "recurrence_samples": needed["recurrence_aware"],
                    "sample_savings": needed["restart_no_reuse"]
                    - needed["recurrence_aware"],
                    "target_gap": config.value_gap_target,
                }
            )
        seen.setdefault(mode, []).extend(segment)
    return output


def _occupancy_weights(
    config: RecurrentStudyConfig, mode_count: int
) -> Tuple[float, ...] | None:
    if config.occupancy_weights is None:
        return None
    weights = list(config.occupancy_weights)
    if len(weights) >= mode_count:
        return tuple(weights[:mode_count])
    rare = min(weights)
    return tuple(weights + [rare] * (mode_count - len(weights)))


def _unit(config: RecurrentStudyConfig, spec: Dict) -> Dict:
    started = time.perf_counter()
    environment = RecurringTabularMDP(
        RecurringMDPConfig(
            state_count=spec["state_count"],
            mode_count=spec["mode_count"],
            separation=spec["separation"],
            dwell=config.dwell,
            cycles=config.cycles,
            rare_occupancy=config.rare_occupancy,
            occupancy_weights=_occupancy_weights(config, spec["mode_count"]),
            seed=spec["seed"],
        )
    )
    transitions, modes, occurrences = _collect(environment, config)
    models, diagnostics = _fit_models(transitions, modes, config)
    values, diagnosis = _evaluate(environment, models, config)
    savings = _recurrence_savings(environment, transitions, modes, config)
    return {
        "spec": spec,
        "transitions": [
            {"step": i, "state": s, "action": a, "next_state": n, "true_mode": m,
             "occurrence": o}
            for i, ((s, a, n), m, o) in enumerate(
                zip(transitions, modes, occurrences), start=1
            )
        ],
        "values": values,
        "diagnosis": diagnosis,
        "savings": savings,
        "diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started,
    }


def _id(payload: Dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Sequence[Dict], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else float("nan")


def _paired_seed_improvements(
    values: Sequence[Dict], bootstrap_samples: int
) -> Tuple[List[Dict], float, float | None, float | None]:
    rows = []
    for seed in sorted({int(row["seed"]) for row in values}):
        method_means = {}
        for method in ("restart_no_reuse", "recurrence_aware"):
            selected = [
                float(row["value_gap"])
                for row in values
                if int(row["seed"]) == seed
                and row["method"] == method
                and row["reward"] == "__worst__"
            ]
            method_means[method] = float(np.mean(selected))
        rows.append(
            {
                "seed": seed,
                "restart_worst_gap": method_means["restart_no_reuse"],
                "recurrence_worst_gap": method_means["recurrence_aware"],
                "paired_gap_improvement": method_means["restart_no_reuse"]
                - method_means["recurrence_aware"],
            }
        )
    improvements = np.asarray(
        [row["paired_gap_improvement"] for row in rows], dtype=float
    )
    mean = float(improvements.mean())
    if len(improvements) < 2:
        return rows, mean, None, None
    rng = np.random.RandomState(20260813)
    indices = rng.randint(0, len(improvements), (bootstrap_samples, len(improvements)))
    low, high = np.quantile(improvements[indices].mean(axis=1), [0.025, 0.975])
    return rows, mean, float(low), float(high)


def _recurrence_gate_checks(
    config: RecurrentStudyConfig,
    values_normalized: bool,
    paired_mean: float,
    paired_ci_low: float | None,
    mean_savings: float,
    deployment_acceptance: float,
) -> Dict[str, bool]:
    """Predeclared gate on the normalized finite-horizon value scale."""
    return {
        "ten_or_more_seeds": len(config.seeds) >= 10,
        "all_values_and_gaps_in_unit_interval": values_normalized,
        "paired_improvement_at_least_margin": paired_mean
        >= config.gate_gap_margin,
        "paired_bootstrap_ci_excludes_zero": paired_ci_low is not None
        and paired_ci_low > 0.0,
        "positive_recurrence_sample_savings": mean_savings > 0,
        "deployment_acceptance_at_least_half": deployment_acceptance >= 0.5,
    }


def run_recurrent_study(
    config: RecurrentStudyConfig, output: Path, resume: bool = True
) -> Dict:
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config_dict = asdict(config)
    config_hash = _id({"protocol": PROTOCOL_VERSION, "config": config_dict})
    specs = [
        {
            "state_count": state_count,
            "mode_count": mode_count,
            "separation": separation,
            "seed": seed,
        }
        for state_count in config.state_counts
        for mode_count in config.mode_counts
        for separation in config.separations
        for seed in config.seeds
    ]
    units = []
    for spec in specs:
        unit_id = _id(
            {"protocol": PROTOCOL_VERSION, "config": config_dict, "spec": spec}
        )
        path = checkpoints / f"{unit_id}.json"
        if resume and path.exists():
            candidate = json.loads(path.read_text())
            if candidate.get("config_hash") == config_hash:
                units.append(candidate)
                continue
        candidate = _unit(config, spec)
        candidate.update({"unit_id": unit_id, "config_hash": config_hash})
        path.write_text(json.dumps(candidate, indent=2, sort_keys=True))
        units.append(candidate)

    transitions: List[Dict] = []
    values: List[Dict] = []
    diagnoses: List[Dict] = []
    savings: List[Dict] = []
    detector_rows: List[Dict] = []
    for unit in units:
        identity = dict(unit["spec"])
        transitions.extend({**identity, **row} for row in unit["transitions"])
        values.extend({**identity, **row} for row in unit["values"])
        diagnoses.extend({**identity, **row} for row in unit["diagnosis"])
        savings.extend({**identity, **row} for row in unit["savings"])
        detector_rows.extend(
            {**identity, "method": method, **row}
            for method, row in unit["diagnostics"].items()
        )
    summary = []
    for method in METHODS:
        selected = [
            row for row in values
            if row["method"] == method and row["reward"] == "__worst__"
        ]
        diagnosed = [row for row in diagnoses if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n_units": len(units),
                "worst_reward_value_gap": _mean(selected, "value_gap"),
                "deployment_accept_rate": _mean(diagnosed, "accepted"),
            }
        )
    mean_savings = _mean(savings, "sample_savings")
    paired_rows, paired_mean, paired_low, paired_high = _paired_seed_improvements(
        values, config.gate_bootstrap_samples
    )
    all_values_normalized = all(
        -1e-12 <= float(row[field]) <= 1.0 + 1e-12
        for row in values
        for field in ("oracle_value", "learned_value", "value_gap")
        if row.get(field) is not None
    )
    # Predeclared controlled-domain recurrence gate: an absolute one-percent
    # improvement on the normalized [0,1] scale plus a positive paired CI.
    gate_checks = _recurrence_gate_checks(
        config,
        all_values_normalized,
        paired_mean,
        paired_low,
        mean_savings,
        next(
            row["deployment_accept_rate"]
            for row in summary if row["method"] == "recurrence_aware"
        ),
    )
    gate = all(gate_checks.values())
    _write_csv(output / "raw_transitions.csv", transitions)
    _write_csv(output / "value_gaps.csv", values)
    _write_csv(output / "deployment_diagnosis.csv", diagnoses)
    _write_csv(output / "recurrence_savings.csv", savings)
    _write_csv(output / "detector_diagnostics.csv", detector_rows)
    _write_csv(output / "paired_seed_gaps.csv", paired_rows)
    _write_csv(output / "summary.csv", summary)
    payload = {
        "protocol": PROTOCOL_VERSION,
        "config": config_dict,
        "config_hash": config_hash,
        "n_units": len(units),
        "reward_normalization": {
            "stages": config.horizon,
            "per_stage_maximum": 1.0 / config.horizon,
            "pathwise_total_maximum": 1.0,
        },
        "runtime_seconds": sum(float(unit["runtime_seconds"]) for unit in units),
        "summary": summary,
        "mean_recurrence_sample_savings": mean_savings,
        "paired_gap_improvement": {
            "mean": paired_mean,
            "ci95_low": paired_low,
            "ci95_high": paired_high,
            "required_margin": config.gate_gap_margin,
        },
        "gate_checks": gate_checks,
        "recurrence_gate_passed": gate,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def recurrent_resume_hash(
    config: RecurrentStudyConfig, output: Path
) -> Tuple[str, str]:
    files = (
        "raw_transitions.csv",
        "value_gaps.csv",
        "deployment_diagnosis.csv",
        "recurrence_savings.csv",
        "paired_seed_gaps.csv",
        "summary.csv",
    )
    run_recurrent_study(config, output, resume=True)
    first = hashlib.sha256(
        b"".join((output / filename).read_bytes() for filename in files)
    ).hexdigest()
    run_recurrent_study(config, output, resume=True)
    second = hashlib.sha256(
        b"".join((output / filename).read_bytes() for filename in files)
    ).hexdigest()
    return first, second
