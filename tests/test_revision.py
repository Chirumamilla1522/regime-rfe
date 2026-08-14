import json

import numpy as np
import torch

from rfe_drift.env import DriftGridWorld, DriftSchedule, DriftType
from rfe_drift.exploration import CountBasedExplorer
from rfe_drift.representations import DriftAwareEncoder, RepresentationTrainer
import run as experiment
from run import ExperimentConfig, run, train_encoders


def test_sudden_drift_uses_global_clock_across_reset():
    env = DriftGridWorld(
        grid_size=5,
        num_walls=0,
        drift_type=DriftType.GOAL_SHIFT,
        drift_strength=0.6,
        drift_schedule=DriftSchedule.SUDDEN,
        drift_time=2,
        seed=3,
    )
    initial = list(env.goals)
    env.reset()
    env.step(1)
    assert env.goals == initial
    _, _, _, _, info = env.step(1)
    assert info["drift_applied"]
    assert env.goals == env.drifted_goals
    env.reset()
    assert env.step_count == 2
    assert env.goals == env.drifted_goals


def test_coverage_denominator_is_reachable_states():
    env = DriftGridWorld(grid_size=3, num_walls=0, seed=0)
    env.walls = {(1, 0), (0, 1)}
    assert env.reachable_states() == {(0, 0)}
    assert env.get_state_coverage({(0, 0), (2, 2)}) == 1.0
    explorer = CountBasedExplorer(state_dim=9, action_dim=4, seed=0)
    explorer.visited_states = {(0, 0), (2, 2)}
    assert explorer.get_state_coverage(env.reachable_states()) == 1.0


def test_temporal_encoder_is_trained_with_time_metadata():
    torch.manual_seed(0)
    records = []
    for index in range(64):
        state = (index % 4, (index // 4) % 4)
        next_state = ((state[0] + (index >= 32)) % 4, state[1])
        time = index / 63
        records.append((state, index % 4, next_state, False, time, time + 0.01))
    encoder = DriftAwareEncoder(
        input_dim=2, hidden_dim=16, output_dim=8, context_dim=4
    )
    before = encoder.time_embedding[0].weight.detach().clone()
    losses = RepresentationTrainer(encoder, batch_size=16).train_forward_dynamics(
        records, num_epochs=2, seed=0
    )
    assert losses and not torch.equal(before, encoder.time_embedding[0].weight)
    state = np.array([1, 1])
    assert not np.allclose(encoder.encode(state, 0.0), encoder.encode(state, 1.0))


def test_representation_ablation_uses_identical_records(monkeypatch):
    records = [
        ((0, 0), 1, (1, 0), False, 0.0, 0.1),
        ((1, 0), 2, (1, 1), False, 0.6, 0.7),
    ] * 20
    seen = []

    def capture(self, replay_buffer, **kwargs):
        seen.append(replay_buffer)
        return [0.0]

    monkeypatch.setattr(
        experiment.RepresentationTrainer, "train_forward_dynamics", capture
    )
    train_encoders(records, ExperimentConfig(representation_epochs=1), seed=0)
    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert seen[0] == records


def test_end_to_end_quick_smoke(tmp_path):
    config = ExperimentConfig(
        grid_size=4,
        num_walls=1,
        drift_time=20,
        exploration_steps=40,
        representation_epochs=1,
        training_steps=40,
        eval_episodes=1,
        max_episode_steps=8,
        seeds=(0,),
        drift_types=("goal_shift",),
    )
    result = run(config, tmp_path)
    assert len(result["summary"]) == 6
    assert (tmp_path / "episodes.csv").exists()
    assert json.loads((tmp_path / "results.json").read_text())["config"]["seeds"] == [0]
