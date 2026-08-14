import inspect

from rfe_drift.recurrent_study import (
    RecurrentStudyConfig,
    _fit_online,
    _recurrence_gate_checks,
    recurrent_profile,
    recurrent_resume_hash,
)
from rfe_drift.synthetic import (
    RecurringMDPConfig,
    RecurringTabularMDP,
    exact_finite_horizon_value,
)
from rfe_drift.tabular import (
    Assignment,
    Detection,
    ModeTransitionModel,
    RecurringModeLearner,
    RecurringModeMatcher,
    finite_horizon_plan,
    identify_deployment_mode,
)


def _deterministic_mode(outcome, repeats=20):
    return [(state, action, outcome) for state in (0, 1) for action in (0, 1)
            for _ in range(repeats)]


def test_nonoracle_online_path_has_no_oracle_inputs():
    parameters = set(inspect.signature(_fit_online).parameters)
    assert parameters == {"transitions", "config", "quarantine", "reuse"}
    parameters = set(inspect.signature(RecurringModeLearner.update).parameters)
    assert parameters == {"self", "state", "action", "next_state"}
    source = inspect.getsource(_fit_online)
    assert "true_mode" not in source
    assert "boundary" not in source


def test_recurring_mode_is_matched_and_unseen_mode_rejected():
    models = ModeTransitionModel()
    models.update_many(7, _deterministic_mode(2))
    matcher = RecurringModeMatcher(
        acceptance_radius=0.1,
        minimum_evidence_keys=2,
        minimum_key_samples=2,
    )
    recurring = matcher.match(_deterministic_mode(2, repeats=3), models)
    unseen = matcher.match(_deterministic_mode(3, repeats=3), models)
    assert recurring.accepted and recurring.mode == 7
    assert not unseen.accepted and unseen.mode is None


class _AlarmSchedule:
    def __init__(self, alarms):
        self.alarms = set(alarms)
        self.step = 0

    def update(self, state, action, next_state):
        self.step += 1
        alarm = self.step in self.alarms
        return Detection(alarm, float(alarm), 0.5, True)


def test_quarantine_is_excluded_and_resets_at_every_boundary():
    learner = RecurringModeLearner(
        detector=_AlarmSchedule({1, 5}),
        matcher=RecurringModeMatcher(
            acceptance_radius=-1.0,
            minimum_evidence_keys=1,
            minimum_key_samples=1,
        ),
        quarantine_steps=2,
        confirmation_steps=2,
    )
    assignments = [learner.update(0, 0, index) for index in range(7)]
    assert [item.status for item in assignments[:4]] == [
        "quarantine", "quarantine", "candidate", "confirmed"
    ]
    assert [item.status for item in assignments[4:6]] == [
        "quarantine", "quarantine"
    ]
    learned_outcomes = {
        outcome
        for outcomes in learner.models.counts.values()
        for outcome in outcomes
    }
    assert 0 not in learned_outcomes and 1 not in learned_outcomes
    assert 4 not in learned_outcomes and 5 not in learned_outcomes


def test_deployment_identification_uses_prefix_only():
    assert set(inspect.signature(identify_deployment_mode).parameters) == {
        "prefix", "models", "acceptance_radius", "minimum_evidence_keys"
    }
    models = ModeTransitionModel()
    models.update_many(0, _deterministic_mode(2))
    match = identify_deployment_mode(
        _deterministic_mode(2, repeats=3), models, minimum_evidence_keys=2
    )
    assert match.accepted and match.mode == 0


def test_same_model_supports_multiple_posthoc_rewards_without_mutation():
    model = ModeTransitionModel()
    model.update_many(0, [(0, 0, 1)] * 10 + [(0, 1, 0)] * 10
                      + [(1, 0, 1)] * 10 + [(1, 1, 0)] * 10)
    before = {
        key: dict(value) for key, value in model.counts.items()
    }
    reward_left = {(stage, state, 1): 0.5
                   for stage in range(2) for state in (0, 1)}
    reward_right = {(stage, state, 0): 0.5
                    for stage in range(2) for state in (0, 1)}
    left_policy, _ = finite_horizon_plan(
        model, 0, (0, 1), (0, 1), reward_left, horizon=2
    )
    right_policy, _ = finite_horizon_plan(
        model, 0, (0, 1), (0, 1), reward_right, horizon=2
    )
    assert left_policy != right_policy
    assert before == {key: dict(value) for key, value in model.counts.items()}


def test_finite_horizon_planner_avoids_walls_and_is_stage_indexed():
    model = ModeTransitionModel()
    model.update_many(0, [(0, 0, 99)] * 5 + [(0, 1, 1)] * 5
                      + [(1, 0, 1)] * 5 + [(1, 1, 0)] * 5)
    reward = {(stage, 0, 1): 0.5 for stage in range(2)}
    policy, values = finite_horizon_plan(
        model, 0, valid_states=(0, 1), actions=(0, 1),
        reward=reward, horizon=2
    )
    assert set(policy) == {
        (stage, state) for stage in range(2) for state in (0, 1)
    }
    assert all(state != 99 for _, state in policy)
    assert all(state != 99 for _, state in values)
    # At stage zero, waiting via the folded wall self-loop ties collecting now;
    # deterministic tie-breaking waits. At the final stage it collects.
    assert policy[(0, 0)] == 0 and policy[(1, 0)] == 1


def test_normalized_reward_and_exact_finite_horizon_values_are_bounded():
    environment = RecurringTabularMDP(
        RecurringMDPConfig(state_count=5, mode_count=2, seed=4)
    )
    horizon = 7
    rewards = environment.reward_family(horizon)
    assert max(max(reward.values()) for reward in rewards.values()) == 1 / horizon
    truth = environment.true_model()
    for reward in rewards.values():
        policy, _ = finite_horizon_plan(
            truth, 0, environment.states, environment.actions, reward, horizon
        )
        value = exact_finite_horizon_value(
            environment, 0, policy, reward, horizon
        )
        assert 0.0 <= value <= 1.0


def test_indistinguishable_modes_are_rejected_as_ambiguous():
    models = ModeTransitionModel()
    data = _deterministic_mode(2)
    models.update_many(0, data)
    models.update_many(1, data)
    result = RecurringModeMatcher(
        acceptance_radius=0.1,
        ambiguity_margin=0.01,
        minimum_evidence_keys=2,
        minimum_key_samples=2,
    ).match(_deterministic_mode(2, repeats=3), models)
    assert not result.accepted


def test_learner_detects_and_reuses_swapped_action_semantics():
    """A full action swap must produce more than one mode and later reuse."""
    from rfe_drift.recurrent_study import RecurrentStudyConfig, _fit_online
    from rfe_drift.synthetic import RecurringMDPConfig, RecurringTabularMDP

    config = RecurrentStudyConfig(
        dwell=80,
        cycles=4,
        quarantine_steps=4,
        confirmation_steps=12,
        detector_recent=8,
        detector_reference=16,
        detector_cooldown=12,
        horizon=8,
    )
    environment = RecurringTabularMDP(
        RecurringMDPConfig(
            state_count=7,
            mode_count=2,
            separation=0.95,
            dwell=80,
            cycles=4,
            seed=1,
        )
    )
    rng = __import__("numpy").random.RandomState(0)
    transitions = []
    state = environment.reset()
    counts = {}
    for step in range(len(environment.mode_sequence) * 80):
        action_counts = [counts.get((state, a), 0) for a in environment.actions]
        action = int(min(range(len(action_counts)), key=lambda i: action_counts[i]))
        mode = environment.mode_at(step)
        nxt = environment.sample(mode, state, action)
        transitions.append((state, action, nxt))
        counts[(state, action)] = counts.get((state, action), 0) + 1
        state = nxt
    models, diagnostics = _fit_online(transitions, config, quarantine=4, reuse=True)
    assert diagnostics["alarms"] >= 3
    assert diagnostics["learned_modes"] >= 2
    assert diagnostics["reuse_events"] >= 1
    assert len(models.modes) >= 2

    models = ModeTransitionModel()
    data = _deterministic_mode(2)
    models.update_many(0, data)
    models.update_many(1, data)
    result = RecurringModeMatcher(
        acceptance_radius=0.1,
        ambiguity_margin=0.01,
        minimum_evidence_keys=2,
        minimum_key_samples=2,
    ).match(_deterministic_mode(2, repeats=3), models)
    assert not result.accepted


def test_gate_rejects_tiny_but_confident_strict_improvement():
    config = RecurrentStudyConfig(seeds=tuple(range(10)), gate_gap_margin=0.01)
    checks = _recurrence_gate_checks(
        config,
        values_normalized=True,
        paired_mean=0.0005,
        paired_ci_low=0.0001,
        mean_savings=10.0,
        deployment_acceptance=1.0,
    )
    assert checks["paired_bootstrap_ci_excludes_zero"]
    assert not checks["paired_improvement_at_least_margin"]
    assert not all(checks.values())


def test_recurrent_protocol_resume_is_byte_deterministic(tmp_path):
    config = RecurrentStudyConfig(
        profile="test",
        seeds=(0,),
        state_counts=(5,),
        mode_counts=(2,),
        separations=(0.9,),
        dwell=32,
        cycles=2,
        deployment_prefix=24,
        quarantine_steps=2,
        confirmation_steps=4,
        horizon=5,
        value_gap_target=0.2,
        gate_bootstrap_samples=100,
    )
    first, second = recurrent_resume_hash(config, tmp_path)
    assert first == second


def test_rollback_removes_pre_alarm_counts_from_the_active_mode():
    learner = RecurringModeLearner(
        detector=_AlarmSchedule({4}),
        matcher=RecurringModeMatcher(
            acceptance_radius=-1.0,
            minimum_evidence_keys=1,
            minimum_key_samples=1,
        ),
        quarantine_steps=1,
        confirmation_steps=2,
        rollback_steps=2,
    )
    for index in range(8):
        learner.update(0, 0, index)
    learned = {
        outcome
        for outcomes in learner.models.counts.values()
        for outcome in outcomes
    }
    assert 2 not in learned and 3 not in learned
    assert 1 not in learned
    assert 0 in learned
    assert 4 in learned or 5 in learned


def test_probe_window_length_matches_two_window_lemma():
    from rfe_drift.tabular import probe_window_length

    length = probe_window_length(
        separation=0.9, alphabet=4, split_candidates=20, failure_probability=0.05
    )
    assert length >= 1
    shorter = probe_window_length(
        separation=0.9, alphabet=4, split_candidates=5, failure_probability=0.05
    )
    assert length >= shorter


def test_probe_window_detector_thresholds_at_half_separation():
    from rfe_drift.tabular import ProbeWindowDetector

    detector = ProbeWindowDetector(window=4, separation=0.8)
    for index in range(20):
        result = detector.update(0, 0, index % 2)
    assert detector.separation / 2 == 0.4


def test_certified_profile_windows_meet_the_lemma_length():
    from rfe_drift.tabular import probe_window_length

    config = recurrent_profile("certified")
    needed = probe_window_length(0.9, 9, 48, 0.05)
    assert config.certified
    assert config.probe_policy == "uniform"
    assert not config.use_residual
    assert config.detector_recent >= needed
    assert config.dwell >= 2 * needed
    assert len(recurrent_profile("certified-full").seeds) == 30


def test_imbalanced_blocks_make_one_mode_shorter():
    environment = RecurringTabularMDP(
        RecurringMDPConfig(
            state_count=5,
            mode_count=2,
            dwell=40,
            cycles=2,
            occupancy_weights=(3.0, 1.0),
            seed=0,
        )
    )
    lengths = environment.block_lengths
    assert lengths[0] > lengths[1]
    assert environment.mode_at(0) == 0
    assert environment.mode_at(lengths[0]) == 1
