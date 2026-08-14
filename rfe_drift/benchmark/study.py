"""Harness for RFE-Recurrent-Bench.

Reuses the Detect--Match--Explore implementation. Per-task dwell is short so
the last block alone is an under-sampled restart baseline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from rfe_drift.benchmark.envs import finite_horizon_value, make_benchmark_env
from rfe_drift.benchmark.suite import TASK_SPECS, BenchmarkStudyConfig
from rfe_drift.recurrent_study import (
    METHODS,
    RecurrentStudyConfig,
    _collect,
    _fit_models,
    _mean,
    _paired_seed_improvements,
    _write_csv,
)
from rfe_drift.tabular import (
    ModeTransitionModel,
    finite_horizon_plan,
    identify_deployment_mode,
)

PROTOCOL_VERSION = "rfe_recurrent_bench_v2"


def _id(payload: Dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _certified_spec(study: BenchmarkStudyConfig, spec: Mapping) -> Dict:
    payload = dict(spec)
    if not study.certified:
        return payload
    window = int(study.detector_window or 160)
    payload["dwell"] = 2 * window + 80
    payload["quarantine_steps"] = window
    payload["confirmation_steps"] = window
    return payload


def _runtime_config(
    study: BenchmarkStudyConfig, spec: Mapping
) -> RecurrentStudyConfig:
    spec = _certified_spec(study, spec)
    dwell = int(spec["dwell"])
    confirmation = int(spec["confirmation_steps"])
    if study.certified:
        window = int(study.detector_window or confirmation)
        return RecurrentStudyConfig(
            profile=study.profile,
            seeds=study.seeds,
            dwell=dwell,
            cycles=int(spec["cycles"]),
            deployment_prefix=int(spec["deployment_prefix"]),
            quarantine_steps=int(spec["quarantine_steps"]),
            confirmation_steps=window,
            horizon=int(spec["horizon"]),
            value_gap_target=study.value_gap_target,
            gate_gap_margin=study.gate_gap_margin,
            gate_bootstrap_samples=study.gate_bootstrap_samples,
            detector_recent=window,
            detector_reference=window,
            detector_cooldown=window,
            certified=True,
            declared_separation=0.9,
            probe_policy="uniform",
            use_residual=False,
        )
    recent = max(6, confirmation // 2)
    reference = max(12, min(max(16, dwell // 3), recent * 2))
    cooldown = max(recent, confirmation)
    return RecurrentStudyConfig(
        profile=study.profile,
        seeds=study.seeds,
        dwell=dwell,
        cycles=int(spec["cycles"]),
        deployment_prefix=int(spec["deployment_prefix"]),
        quarantine_steps=int(spec["quarantine_steps"]),
        confirmation_steps=confirmation,
        horizon=int(spec["horizon"]),
        value_gap_target=study.value_gap_target,
        gate_gap_margin=study.gate_gap_margin,
        gate_bootstrap_samples=study.gate_bootstrap_samples,
        detector_recent=recent,
        detector_reference=reference,
        detector_cooldown=cooldown,
    )


def _optimal_values(environment, rewards, horizon):
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
            output[(mode, name)] = finite_horizon_value(
                environment, mode, policy, reward, horizon
            )
    return output


def _balanced_prefix(environment, mode: int, length: int, seed: int):
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
            rng.choice(outcomes, p=[distribution[o] for o in outcomes])
        )
        output.append((state, action, next_state))
    return output


def _evaluate(environment, models, config: RecurrentStudyConfig):
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
                    prefix,
                    model,
                    acceptance_radius=0.55,
                    minimum_evidence_keys=2,
                )
                selected, accepted, distance = (
                    match.mode,
                    match.accepted,
                    match.distance,
                )
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
                    learned_value = finite_horizon_value(
                        environment, true_mode, policy, reward, config.horizon
                    )
                    gap = min(1.0, max(0.0, oracle_value - learned_value))
                else:
                    learned_value = None
                    gap = min(1.0, max(0.0, oracle_value))
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
                    "value_gap": max(gaps) if gaps else 0.0,
                }
            )
    return rows, diagnosis


def _recurrence_savings(environment, transitions, modes, config: RecurrentStudyConfig):
    rewards = environment.reward_family(config.horizon)
    optimal = _optimal_values(environment, rewards, config.horizon)
    output = []
    seen: Dict[int, list] = {}
    for occurrence, start in enumerate(range(0, len(transitions), config.dwell)):
        mode = modes[start]
        segment = list(transitions[start : start + config.dwell])
        if mode in seen:
            budgets = sorted(
                set(
                    [8, 16, 32, 64, 128, config.dwell]
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
                        value = finite_horizon_value(
                            environment, mode, policy, reward, config.horizon
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


def _unit(study: BenchmarkStudyConfig, spec: Dict) -> Dict:
    started = time.perf_counter()
    runtime = _runtime_config(study, spec)
    env_spec = _certified_spec(study, spec)
    environment = make_benchmark_env(spec["task"], spec["seed"], env_spec)
    transitions, modes, occurrences = _collect(environment, runtime)
    models, diagnostics = _fit_models(transitions, modes, runtime)
    values, diagnosis = _evaluate(environment, models, runtime)
    savings = _recurrence_savings(environment, transitions, modes, runtime)
    return {
        "spec": spec,
        "transitions": [
            {
                "step": i,
                "state": s,
                "action": a,
                "next_state": n,
                "true_mode": m,
                "occurrence": o,
            }
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


def _gate_checks(
    study: BenchmarkStudyConfig,
    values_normalized: bool,
    paired_mean: float,
    paired_ci_low: float | None,
    mean_savings: float,
    deployment_acceptance: float,
    pooled_gap: float,
    recurrence_gap: float,
) -> Dict[str, bool]:
    checks = {
        "ten_or_more_seeds": len(study.seeds) >= 10,
        "all_values_and_gaps_in_unit_interval": values_normalized,
        "paired_improvement_at_least_margin": paired_mean >= study.gate_gap_margin,
        "paired_bootstrap_ci_excludes_zero": paired_ci_low is not None
        and paired_ci_low > 0.0,
        "positive_recurrence_sample_savings": mean_savings > 0,
        "deployment_acceptance_at_least_half": deployment_acceptance >= 0.5,
    }
    if study.require_pooling_loss:
        checks["recurrence_beats_pooling"] = recurrence_gap < pooled_gap - 1e-12
    return checks


def run_benchmark_study(
    study: BenchmarkStudyConfig, output: Path, resume: bool = True
) -> Dict:
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config_dict = asdict(study)
    config_hash = _id(
        {"protocol": PROTOCOL_VERSION, "config": config_dict, "tasks": TASK_SPECS}
    )
    specs = [
        {**TASK_SPECS[task], "seed": seed}
        for task in study.tasks
        for seed in study.seeds
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
        candidate = _unit(study, spec)
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

    def _summarize(selected_values, selected_diag, n_units):
        rows = []
        for method in METHODS:
            chosen = [
                row
                for row in selected_values
                if row["method"] == method and row["reward"] == "__worst__"
            ]
            diagnosed = [row for row in selected_diag if row["method"] == method]
            rows.append(
                {
                    "method": method,
                    "n_units": n_units,
                    "worst_reward_value_gap": _mean(chosen, "value_gap"),
                    "deployment_accept_rate": _mean(diagnosed, "accepted"),
                }
            )
        return rows

    summary = _summarize(values, diagnoses, len(units))
    per_task = {
        task: _summarize(
            [row for row in values if row["task"] == task],
            [row for row in diagnoses if row["task"] == task],
            sum(1 for spec in specs if spec["task"] == task),
        )
        for task in study.tasks
    }
    mean_savings = _mean(savings, "sample_savings")
    paired_rows, paired_mean, paired_low, paired_high = _paired_seed_improvements(
        values, study.gate_bootstrap_samples
    )
    all_values_normalized = all(
        -1e-12 <= float(row[field]) <= 1.0 + 1e-12
        for row in values
        for field in ("oracle_value", "learned_value", "value_gap")
        if row.get(field) is not None
    )
    summary_map = {row["method"]: row for row in summary}
    gate_checks = _gate_checks(
        study,
        all_values_normalized,
        paired_mean,
        paired_low,
        mean_savings,
        summary_map["recurrence_aware"]["deployment_accept_rate"],
        summary_map["pooled"]["worst_reward_value_gap"],
        summary_map["recurrence_aware"]["worst_reward_value_gap"],
    )
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
        "tasks": list(study.tasks),
        "task_specs": {name: TASK_SPECS[name] for name in study.tasks},
        "runtime_seconds": sum(float(unit["runtime_seconds"]) for unit in units),
        "summary": summary,
        "per_task_summary": per_task,
        "mean_recurrence_sample_savings": mean_savings,
        "paired_gap_improvement": {
            "mean": paired_mean,
            "ci95_low": paired_low,
            "ci95_high": paired_high,
            "required_margin": study.gate_gap_margin,
        },
        "gate_checks": gate_checks,
        "benchmark_gate_passed": all(gate_checks.values()),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
