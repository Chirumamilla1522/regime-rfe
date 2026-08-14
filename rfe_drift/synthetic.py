"""Theory-aligned recurrent tabular MDP stress tests.

The family independently controls mode count, transition separation, dwell
time, recurrence, rare-state occupancy, and state-actions reserved for
deployment diagnosis.  No learner-specific statistic is used to construct it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from rfe_drift.tabular import ModeTransitionModel, Transition


@dataclass(frozen=True)
class RecurringMDPConfig:
    state_count: int = 9
    action_count: int = 2
    mode_count: int = 2
    separation: float = 0.9
    dwell: int = 500
    cycles: int = 3
    recurrence: bool = True
    rare_occupancy: float = 0.03
    diagnostic_only_pairs: int = 1
    seed: int = 0
    occupancy_weights: Tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.state_count < 5 or self.action_count < 2 or self.mode_count < 2:
            raise ValueError("suite needs >=5 states, >=2 actions, and >=2 modes")
        if not 0.0 <= self.separation <= 1.0:
            raise ValueError("separation must be in [0, 1]")
        if self.dwell < 1 or self.cycles < 1:
            raise ValueError("dwell and cycles must be positive")
        if self.occupancy_weights is not None:
            if len(self.occupancy_weights) != self.mode_count:
                raise ValueError("occupancy_weights must have one entry per mode")
            if any(weight <= 0 for weight in self.occupancy_weights):
                raise ValueError("occupancy_weights must be positive")


class RecurringTabularMDP:
    """Finite continuing MDP with conflicting action semantics across modes."""

    def __init__(self, config: RecurringMDPConfig):
        self.config = config
        self.states = tuple(range(config.state_count))
        self.actions = tuple(range(config.action_count))
        self.rng = np.random.RandomState(config.seed)
        self.state = 0
        self.step = 0

    @property
    def mode_sequence(self) -> Tuple[int, ...]:
        if self.config.recurrence:
            return tuple(
                mode
                for _ in range(self.config.cycles)
                for mode in range(self.config.mode_count)
            )
        return tuple(range(self.config.mode_count))

    @property
    def block_lengths(self) -> Tuple[int, ...]:
        if self.config.occupancy_weights is None:
            return tuple(
                self.config.dwell
                for _ in range(self.config.cycles * self.config.mode_count)
            )
        mean_weight = sum(self.config.occupancy_weights) / len(
            self.config.occupancy_weights
        )
        one_cycle = tuple(
            max(1, int(round(self.config.dwell * weight / mean_weight)))
            for weight in self.config.occupancy_weights
        )
        return one_cycle * self.config.cycles

    def mode_at(self, step: int) -> int:
        remaining = int(step)
        for length, mode in zip(self.block_lengths, self.mode_sequence):
            if remaining < length:
                return mode
            remaining -= length
        return self.mode_sequence[-1]

    def _mode_successor(self, mode: int, state: int, action: int) -> int:
        ordinary = self.config.state_count - 1
        diagnostic = ordinary
        if state == diagnostic:
            return (mode + action + 1) % ordinary
        # Different mode-specific offsets make pooled action effects conflict.
        offset = 1 + (mode % max(1, ordinary - 2))
        direction = 1 if action % 2 == mode % 2 else -1
        return (state + direction * offset) % ordinary

    def distribution(self, mode: int, state: int, action: int) -> Dict[int, float]:
        ordinary = self.config.state_count - 1
        target = self._mode_successor(mode, state, action)
        common = (state + (1 if action % 2 == 0 else -1)) % ordinary
        if state == ordinary:
            common = 0
        probability = self.config.separation
        distribution = {target: probability, common: 1.0 - probability}
        if target == common:
            distribution = {target: 1.0}
        # Rare entry to the diagnostic state is mode-independent.
        if state != ordinary and self.config.rare_occupancy > 0:
            distribution = {
                outcome: value * (1.0 - self.config.rare_occupancy)
                for outcome, value in distribution.items()
            }
            distribution[ordinary] = self.config.rare_occupancy
        return {outcome: value for outcome, value in distribution.items() if value > 0}

    def reset(self, state: int = 0) -> int:
        self.state = int(state)
        return self.state

    def sample(self, mode: int, state: int, action: int) -> int:
        distribution = self.distribution(mode, state, action)
        outcomes = sorted(distribution)
        probabilities = [distribution[outcome] for outcome in outcomes]
        return int(self.rng.choice(outcomes, p=probabilities))

    def step_transition(self, action: int) -> Tuple[int, int]:
        mode = self.mode_at(self.step)
        self.state = self.sample(mode, self.state, int(action))
        self.step += 1
        return self.state, mode

    def true_model(self) -> ModeTransitionModel:
        """Exact model encoded with fixed integer weights for oracle evaluation."""
        model = ModeTransitionModel()
        scale = 100_000
        for mode in range(self.config.mode_count):
            for state in self.states:
                for action in self.actions:
                    for next_state, probability in self.distribution(
                        mode, state, action
                    ).items():
                        model.counts[(mode, state, action)][next_state] = max(
                            1, int(round(scale * probability))
                        )
                    model.mode_samples[mode] += scale
        return model

    def reward_family(
        self, horizon: int
    ) -> Dict[str, Dict[Tuple[int, int, int], float]]:
        """Stage-indexed post-hoc rewards with pathwise total at most one."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        rewards: Dict[str, Dict[Tuple[int, int, int], float]] = {}
        ordinary = self.config.state_count - 1
        targets = sorted({1, ordinary // 2, ordinary - 1})
        scale = 1.0 / horizon
        for target in targets:
            rewards[f"target_{target}"] = {
                (stage, state, action): scale * float(state == target)
                for stage in range(horizon)
                for state in self.states
                for action in self.actions
            }
        for mode in range(self.config.mode_count):
            # The rewarding successor is chosen from a mode conflict, but the
            # learner never sees this reward during collection or ID.
            state = mode % ordinary
            rewards[f"adversarial_mode_{mode}"] = {
                (stage, s, a): scale
                * float(s == state and a == mode % 2)
                for stage in range(horizon)
                for s in self.states
                for a in self.actions
            }
        diagnostic = ordinary
        rewards["diagnostic_state"] = {
            (stage, state, action): scale * float(state == diagnostic)
            for stage in range(horizon)
            for state in self.states
            for action in self.actions
        }
        return rewards


def exact_finite_horizon_value(
    environment: RecurringTabularMDP,
    mode: int,
    policy: Mapping[Tuple[int, int], int],
    reward: Mapping[Tuple[int, int, int], float],
    horizon: int,
    start_distribution: Sequence[float] | None = None,
) -> float:
    """Evaluate a stage-indexed policy by exact finite-horizon recursion."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    count = environment.config.state_count
    continuation = np.zeros(count)
    for stage in reversed(range(horizon)):
        current = np.zeros(count)
        for state in environment.states:
            action = int(policy[(stage, state)])
            current[state] += reward.get((stage, state, action), 0.0)
            for next_state, probability in environment.distribution(
                mode, state, action
            ).items():
                current[state] += probability * continuation[next_state]
        continuation = current
    if start_distribution is None:
        start_distribution = np.full(count, 1.0 / count)
    return float(np.dot(np.asarray(start_distribution), continuation))


def transition_separation(environment: RecurringTabularMDP) -> float:
    """Minimum maximum state-action TV separation between each mode pair."""
    values = []
    for left in range(environment.config.mode_count):
        for right in range(left + 1, environment.config.mode_count):
            distances = []
            for state in environment.states:
                for action in environment.actions:
                    p = environment.distribution(left, state, action)
                    q = environment.distribution(right, state, action)
                    distances.append(
                        0.5
                        * sum(
                            abs(p.get(x, 0.0) - q.get(x, 0.0))
                            for x in set(p) | set(q)
                        )
                    )
            values.append(max(distances))
    return min(values)
