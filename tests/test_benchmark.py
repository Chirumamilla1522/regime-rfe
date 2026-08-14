from rfe_drift.benchmark.envs import make_benchmark_env
from rfe_drift.benchmark.suite import TASK_SPECS, FOUR_ROOMS_STATES, benchmark_profile
from rfe_drift.benchmark.study import _runtime_config, run_benchmark_study


def _tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def test_four_rooms_state_count_matches_layout():
    spec = TASK_SPECS["four_rooms"]
    env = make_benchmark_env("four_rooms", 0, spec)
    assert env.config.state_count == FOUR_ROOMS_STATES
    assert len(env.states) == FOUR_ROOMS_STATES
    assert len(env.cells) == FOUR_ROOMS_STATES


def test_each_task_has_separated_modes_and_unit_rewards():
    for task, spec in TASK_SPECS.items():
        env = make_benchmark_env(task, 0, spec)
        distances = []
        for state in env.states:
            for action in env.actions:
                distances.append(
                    _tv(
                        env.distribution(0, state, action),
                        env.distribution(1, state, action),
                    )
                )
        assert max(distances) > 0.2, task
        rewards = env.reward_family(spec["horizon"])
        assert rewards
        for table in rewards.values():
            for (stage, state, action), value in table.items():
                assert 0.0 <= value <= 1.0 / spec["horizon"] + 1e-12


def test_certified_benchmark_uses_pooling_hard_tasks_and_long_windows():
    study = benchmark_profile("certified")
    assert study.tasks == ("swap_chain", "deepsea")
    assert study.certified
    assert len(benchmark_profile("certified-full").seeds) == 30
    for task in study.tasks:
        runtime = _runtime_config(study, TASK_SPECS[task])
        assert runtime.certified
        assert runtime.detector_reference + runtime.detector_recent <= runtime.dwell
        assert not runtime.use_residual

    study = benchmark_profile("quick")
    for task in study.tasks:
        runtime = _runtime_config(study, TASK_SPECS[task])
        assert runtime.detector_reference + runtime.detector_recent <= runtime.dwell
        assert runtime.quarantine_steps < runtime.dwell


def test_quick_benchmark_unit_runs(tmp_path):
    study = benchmark_profile("quick")
    payload = run_benchmark_study(study, tmp_path, resume=False)
    assert payload["n_units"] == 4
    assert payload["protocol"] == "rfe_recurrent_bench_v2"
    methods = {row["method"] for row in payload["summary"]}
    assert "recurrence_aware" in methods
    assert "pooled" in methods
