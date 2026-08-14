import inspect

from rfe_drift.protocol import (
    EmpiricalPlanner,
    RegimeExperimentConfig,
    ResidualChangeDetector,
    TransitionRecord,
    deterministic_resume_probe,
    evaluate_planner,
)


def _record(step, state, action, next_state, inferred=0, truth=0):
    return TransitionRecord(
        step=step,
        state_x=state[0],
        state_y=state[1],
        action=action,
        next_x=next_state[0],
        next_y=next_state[1],
        inferred_regime=inferred,
        true_regime=truth,
        residual=0.0,
        alarm=0,
    )


def test_detector_api_has_no_clock_or_oracle_inputs():
    parameters = set(inspect.signature(ResidualChangeDetector.update).parameters)
    assert parameters == {"self", "state", "action", "next_state"}
    source = inspect.getsource(ResidualChangeDetector)
    assert "true_regime" not in source
    assert "drift_time" not in source
    assert "global_step" not in source


def test_residual_detector_fires_on_synthetic_transition_shift():
    detector = ResidualChangeDetector(
        warmup=20,
        reference_window=10,
        recent_window=5,
        threshold=0.2,
        cooldown=10,
    )
    alarms = []
    for step in range(60):
        next_state = (0, 0) if step < 35 else (1, 0)
        _, _, alarm = detector.update((0, 0), 1, next_state)
        if alarm:
            alarms.append(step + 1)
    assert alarms
    assert 36 <= alarms[0] <= 45


def test_regime_controls_use_matched_records():
    records = [
        _record(1, (0, 0), 1, (1, 0), inferred=0, truth=0),
        _record(2, (0, 0), 1, (0, 0), inferred=1, truth=1),
    ]
    counts = []
    for field in ("stationary", "inferred_regime", "true_regime"):
        planner = EmpiricalPlanner(2)
        planner.fit(records, field)
        counts.append(sum(sum(outcomes.values()) for outcomes in planner.counts.values()))
    assert counts == [len(records)] * 3


def test_deterministic_resume(tmp_path):
    config = RegimeExperimentConfig(
        profile="test",
        grid_sizes=(4,),
        num_walls=(0,),
        strengths=(0.7,),
        drift_types=("transition_noise",),
        schedules=("sudden",),
        seeds=(3,),
        exploration_steps=100,
        drift_time=40,
        eval_episodes=2,
        max_episode_steps=25,
        detector_warmup=20,
        detector_reference_window=10,
        detector_recent_window=5,
        detector_cooldown=10,
        recovery_budgets=(0, 20, 50),
        bootstrap_samples=100,
    )
    first, second = deterministic_resume_probe(config, tmp_path)
    assert first == second


def test_tabular_planner_solves_stationary_grid():
    config = RegimeExperimentConfig(
        profile="test",
        grid_sizes=(4,),
        num_walls=(0,),
        strengths=(0.0,),
        drift_types=("transition_noise",),
        schedules=("sudden",),
        seeds=(0,),
        drift_time=1000,
        eval_episodes=10,
        max_episode_steps=20,
    )
    records = []
    step = 0
    for x in range(4):
        for y in range(4):
            for action, delta in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
                step += 1
                next_state = (
                    min(3, max(0, x + delta[0])),
                    min(3, max(0, y + delta[1])),
                )
                records.append(_record(step, (x, y), action, next_state))
    planner = EmpiricalPlanner(4)
    planner.fit(records, "stationary")
    result = evaluate_planner(
        planner,
        context=0,
        config=config,
        grid_size=4,
        walls=0,
        strength=0.0,
        drift_type="transition_noise",
        schedule="sudden",
        seed=0,
        phase_step=0,
    )
    assert result["success_rate"] == 1.0
