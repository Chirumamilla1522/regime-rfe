import csv
import inspect
import json

import numpy as np

from rfe_drift.minigrid_study import (
    ACTIONS,
    METHODS,
    MiniGridStudyConfig,
    RecurringMiniGridFamily,
    _collect_episode,
    _infer_modes,
    _match_signature,
    _reward_table,
    minigrid_profile,
    minigrid_resume_hash,
    run_minigrid_study,
)


def test_profiles_and_required_baselines_are_predeclared():
    assert len(minigrid_profile("quick").seeds) == 5
    assert len(minigrid_profile("pilot").seeds) == 10
    assert minigrid_profile("pilot").env_id == "MiniGrid-Empty-6x6-v0"
    assert minigrid_profile("fourrooms").env_id == "MiniGrid-FourRooms-v0"
    assert minigrid_profile("conflict").mirror_observations is False
    assert len(minigrid_profile("conflict-full").seeds) == 30
    assert set(METHODS) == {
        "pooled",
        "restart",
        "recurrence_aware",
        "oracle_upper_bound",
    }


def test_wrapper_exposes_only_partial_observations_and_zero_reward():
    family = RecurringMiniGridFamily(seed=3, max_steps=12)
    native, reset_info = family.reset(regime=0, episode_seed=11)
    shifted, shifted_info = family.reset(regime=1, episode_seed=11)
    assert native.shape == shifted.shape == (7, 7, 3)
    assert native.dtype == shifted.dtype == np.uint8
    assert reset_info == shifted_info == {}
    assert not np.array_equal(native, shifted)
    _, reward, _, _, step_info = family.step(ACTIONS[0])
    family.close()
    assert reward == 0.0
    assert step_info == {}


def test_fourrooms_wrapper_uses_walled_layout():
    family = RecurringMiniGridFamily(
        seed=3, max_steps=12, env_id="MiniGrid-FourRooms-v0"
    )
    image, info = family.reset(regime=0, episode_seed=11)
    family.close()
    assert image.shape == (7, 7, 3)
    assert info == {}
    assert family.env_id == "MiniGrid-FourRooms-v0"


def test_nonoracle_diagnosis_has_no_label_or_reward_input():
    assert set(inspect.signature(_infer_modes).parameters) == {"episodes", "config"}
    assert set(inspect.signature(_match_signature).parameters) == {
        "records",
        "prototypes",
        "scale",
        "config",
    }
    for function in (_infer_modes, _match_signature):
        source = inspect.getsource(function)
        assert "true_regime" not in source
        assert "reward" not in source


def test_collection_is_reward_free_and_posthoc_reward_is_arbitrary():
    family = RecurringMiniGridFamily(seed=5, max_steps=10)
    rows = _collect_episode(
        family, np.random.RandomState(9), 8, regime=0, episode_seed=13
    )
    family.close()
    assert all("reward" not in row for row in rows)
    decoded = [
        {
            **row,
            "state": index % 2,
            "next_state": (index + 1) % 2,
            "observation_array": np.asarray(row["observation"], dtype=np.uint8),
            "next_observation_array": np.asarray(
                row["next_observation"], dtype=np.uint8
            ),
        }
        for index, row in enumerate(rows)
    ]
    zeros = _reward_table(decoded, lambda _o, _a, _n: 0.0)
    custom = _reward_table(
        decoded, lambda _o, action, _n: float(action == ACTIONS[0])
    )
    assert zeros != custom
    assert all(value == 0.0 for value in zeros.values())


def _tiny_config():
    return MiniGridStudyConfig(
        profile="test",
        seeds=(0,),
        collection_episodes=6,
        collection_steps=12,
        diagnostic_steps=8,
        evaluation_episodes=1,
        evaluation_steps=8,
        latent_dimensions=2,
        latent_states=6,
        match_radius=2.0,
    )


def test_raw_outputs_retain_observations_and_resume_deterministically(tmp_path):
    output = tmp_path / "run"
    payload = run_minigrid_study(_tiny_config(), output)
    assert payload["n_units"] == 1
    with (output / "raw_transitions.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    observation = np.asarray(json.loads(row["observation_json"]))
    next_observation = np.asarray(json.loads(row["next_observation_json"]))
    assert observation.shape == next_observation.shape == (7, 7, 3)
    assert float(row["reward"]) == 0.0
    first, second = minigrid_resume_hash(_tiny_config(), output)
    assert first == second
import json

import numpy as np

from rfe_drift.minigrid_study import (
    MiniGridStudyConfig,
    RecurringMiniGridFamily,
    _collect_episode,
    minigrid_resume_hash,
)


def _tiny_config():
    return MiniGridStudyConfig(
        profile="test",
        seeds=(0,),
        collection_episodes=4,
        collection_steps=12,
        diagnostic_steps=8,
        evaluation_episodes=1,
        evaluation_steps=8,
        latent_dimensions=2,
        latent_states=6,
        match_radius=2.0,
        recurrence_gap_target=1.0,
    )


def test_wrapper_switches_only_between_episodes_and_changes_process():
    family = RecurringMiniGridFamily(4, schedule=(0, 1), max_steps=20)
    first, _ = family.reset(episode_seed=99)
    first_next, _, _, _, _ = family.step(0)
    second, _ = family.reset(episode_seed=99)
    second_next, _, _, _, _ = family.step(0)
    family.close()
    assert not np.array_equal(first, second)
    assert not np.array_equal(first_next, second_next)


def test_partial_observation_has_no_regime_or_mission_leakage():
    family = RecurringMiniGridFamily(0, schedule=(0,), max_steps=10)
    observation, info = family.reset()
    next_observation, reward, _, _, step_info = family.step(2)
    family.close()
    assert isinstance(observation, np.ndarray)
    assert observation.shape == family.observation_shape
    assert next_observation.shape == family.observation_shape
    assert info == {} and step_info == {}
    assert reward == 0.0


def test_collection_is_reward_free():
    family = RecurringMiniGridFamily(2, max_steps=10)
    records = _collect_episode(family, np.random.RandomState(3), 10, regime=0)
    family.close()
    assert records
    assert all("reward" not in row for row in records)
    assert all(set(row) == {
        "step", "action", "observation", "next_observation", "terminated", "truncated"
    } for row in records)


def test_collection_execution_is_deterministic():
    left = RecurringMiniGridFamily(7, max_steps=20)
    right = RecurringMiniGridFamily(7, max_steps=20)
    first = _collect_episode(left, np.random.RandomState(8), 16, regime=1)
    second = _collect_episode(right, np.random.RandomState(8), 16, regime=1)
    left.close()
    right.close()
    assert first == second


def test_minigrid_output_resume_is_byte_deterministic(tmp_path):
    first, second = minigrid_resume_hash(_tiny_config(), tmp_path)
    assert first == second
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["config_hash"]
    assert payload["n_units"] == 1
