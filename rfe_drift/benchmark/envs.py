"""Classic tabular MDPs with recurring transition regimes and post-hoc rewards.

These are generators, not frozen trajectory files. Collection is reward-free.
Mode labels are used only by the simulator and oracle evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from rfe_drift.synthetic import RecurringMDPConfig, RecurringTabularMDP
from rfe_drift.tabular import ModeTransitionModel


@dataclass(frozen=True)
class EnvConfig:
    task: str
    state_count: int
    action_count: int
    mode_count: int = 2
    dwell: int = 48
    cycles: int = 4
    seed: int = 0
    slip: float = 0.1


def _encode_true_model(environment) -> ModeTransitionModel:
    model = ModeTransitionModel()
    scale = 100_000
    for mode in range(environment.config.mode_count):
        for state in environment.states:
            for action in environment.actions:
                for next_state, probability in environment.distribution(
                    mode, state, action
                ).items():
                    model.counts[(mode, state, action)][next_state] = max(
                        1, int(round(scale * probability))
                    )
                model.mode_samples[mode] += scale
    return model


class _RecurringMixin:
    @property
    def mode_sequence(self) -> Tuple[int, ...]:
        return tuple(
            mode
            for _ in range(self.config.cycles)
            for mode in range(self.config.mode_count)
        )

    def mode_at(self, step: int) -> int:
        index = min(step // self.config.dwell, len(self.mode_sequence) - 1)
        return self.mode_sequence[index]

    @property
    def block_lengths(self) -> Tuple[int, ...]:
        return tuple(self.config.dwell for _ in self.mode_sequence)

    def reset(self, state: int | None = None) -> int:
        self.state = int(self.start_state if state is None else state)
        self.step = 0
        return self.state

    def sample(self, mode: int, state: int, action: int) -> int:
        distribution = self.distribution(mode, state, action)
        outcomes = sorted(distribution)
        return int(
            self.rng.choice(outcomes, p=[distribution[o] for o in outcomes])
        )

    def step_transition(self, action: int) -> Tuple[int, int]:
        mode = self.mode_at(self.step)
        self.state = self.sample(mode, self.state, int(action))
        self.step += 1
        return self.state, mode

    def true_model(self) -> ModeTransitionModel:
        return _encode_true_model(self)

    @property
    def start_distribution(self) -> np.ndarray:
        weights = np.zeros(self.config.state_count)
        weights[self.start_state] = 1.0
        return weights


class SwapChainMDP(RecurringTabularMDP):
    """Existing conflicting-action chain, used as one named benchmark task."""

    def __init__(self, config: RecurringMDPConfig):
        super().__init__(config)
        self.task = "swap_chain"
        self.start_state = 0

    @property
    def start_distribution(self) -> np.ndarray:
        return np.full(self.config.state_count, 1.0 / self.config.state_count)


class RiverSwimMDP(_RecurringMixin):
    """Strehl–Littman RiverSwim with a current that reverses across modes."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.task = "riverswim"
        self.states = tuple(range(config.state_count))
        self.actions = (0, 1)  # left, right
        self.rng = np.random.RandomState(config.seed)
        self.start_state = 0
        self.state = 0
        self.step = 0

    def distribution(self, mode: int, state: int, action: int) -> Dict[int, float]:
        n = self.config.state_count
        left, right = (0, 1) if mode == 0 else (1, 0)
        if action == left:
            nxt = max(0, state - 1)
            return {nxt: 1.0}
        # Right/upstream is stochastic, harder toward the far end.
        if state == 0:
            return {0: 0.4, 1: 0.6}
        if state == n - 1:
            return {n - 2: 0.4, n - 1: 0.6}
        return {state - 1: 0.05, state: 0.6, state + 1: 0.35}

    def reward_family(self, horizon: int) -> Dict[str, Dict[Tuple[int, int, int], float]]:
        n = self.config.state_count
        scale = 1.0 / horizon
        family = {}
        for name, target in (("upstream", n - 1), ("downstream", 0)):
            family[name] = {
                (h, s, a): scale * float(s == target)
                for h in range(horizon)
                for s in self.states
                for a in self.actions
            }
        family["hold_mid"] = {
            (h, s, a): scale * float(s == n // 2)
            for h in range(horizon)
            for s in self.states
            for a in self.actions
        }
        return family


class DeepSeaMDP(_RecurringMixin):
    """Osband-style chain in which the diving action is swapped across modes."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.task = "deepsea"
        self.states = tuple(range(config.state_count))
        self.actions = (0, 1)
        self.rng = np.random.RandomState(config.seed)
        self.start_state = 0
        self.state = 0
        self.step = 0

    def distribution(self, mode: int, state: int, action: int) -> Dict[int, float]:
        n = self.config.state_count
        dive = action if mode == 0 else 1 - action
        slip = self.config.slip
        if dive == 1:
            success = min(n - 1, state + 1)
            fail = max(0, state - 1)
            return {success: 1.0 - slip, fail: slip} if success != fail else {success: 1.0}
        stay = max(0, state - 1)
        return {stay: 1.0}

    def reward_family(self, horizon: int) -> Dict[str, Dict[Tuple[int, int, int], float]]:
        n = self.config.state_count
        scale = 1.0 / horizon
        family = {
            "seafloor": {
                (h, s, a): scale * float(s == n - 1)
                for h in range(horizon)
                for s in self.states
                for a in self.actions
            },
            "surface": {
                (h, s, a): scale * float(s == 0)
                for h in range(horizon)
                for s in self.states
                for a in self.actions
            },
            "dive_action": {
                (h, s, a): scale * float(a == 1)
                for h in range(horizon)
                for s in self.states
                for a in self.actions
            },
        }
        return family


class FourRoomsMDP(_RecurringMixin):
    """Four rooms on a 5x5 grid; door locations move with the hidden mode."""

    SIZE = 5
    DELTAS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # N S E W
    POSSIBLE_DOORS = ((2, 0), (2, 4), (0, 2), (4, 2))

    def __init__(self, config: EnvConfig):
        cells = []
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if (row == 2 or col == 2) and (row, col) not in self.POSSIBLE_DOORS:
                    continue
                cells.append((row, col))
        if config.state_count != len(cells):
            raise ValueError(f"four_rooms expects {len(cells)} states")
        self.config = config
        self.task = "four_rooms"
        self.cells = tuple(cells)
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.states = tuple(range(len(cells)))
        self.actions = (0, 1, 2, 3)
        self.rng = np.random.RandomState(config.seed)
        self.start_state = self.index[(0, 0)]
        self.state = self.start_state
        self.step = 0

    def _open_doors(self, mode: int) -> set:
        if mode == 0:
            return {(2, 0), (0, 2)}
        return {(2, 4), (4, 2)}

    def _blocked(self, mode: int, row: int, col: int) -> bool:
        if not (0 <= row < self.SIZE and 0 <= col < self.SIZE):
            return True
        if (row, col) in self._open_doors(mode):
            return False
        return row == 2 or col == 2

    def _move(self, mode: int, state: int, action: int) -> int:
        row, col = self.cells[state]
        drow, dcol = self.DELTAS[action % 4]
        nxt = (row + drow, col + dcol)
        if self._blocked(mode, *nxt) or nxt not in self.index:
            return state
        return self.index[nxt]

    def distribution(self, mode: int, state: int, action: int) -> Dict[int, float]:
        slip = self.config.slip
        intended = self._move(mode, state, action)
        mass = {intended: 1.0 - slip}
        side_mass = slip / 2.0
        for extra in ((action + 2) % 4, (action + 3) % 4):
            nxt = self._move(mode, state, extra)
            mass[nxt] = mass.get(nxt, 0.0) + side_mass
        return {key: value for key, value in mass.items() if value > 0}

    def reward_family(self, horizon: int) -> Dict[str, Dict[Tuple[int, int, int], float]]:
        scale = 1.0 / horizon
        goals = {
            "nw": (0, 0),
            "ne": (0, 4),
            "sw": (4, 0),
            "se": (4, 4),
        }
        family = {}
        for name, cell in goals.items():
            target = self.index[cell]
            family[f"goal_{name}"] = {
                (h, s, a): scale * float(s == target)
                for h in range(horizon)
                for s in self.states
                for a in self.actions
            }
        family["cross_east"] = {
            (h, s, a): scale * float(a == 2)
            for h in range(horizon)
            for s in self.states
            for a in self.actions
        }
        return family


def finite_horizon_value(
    environment,
    mode: int,
    policy: Mapping[Tuple[int, int], int],
    reward: Mapping[Tuple[int, int, int], float],
    horizon: int,
) -> float:
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
                current[state] += probability * continuation[int(next_state)]
        continuation = current
    start = getattr(environment, "start_distribution", None)
    if start is None:
        start = np.full(count, 1.0 / count)
    return float(np.dot(np.asarray(start, dtype=float), continuation))


def make_benchmark_env(task: str, seed: int, spec: Mapping) -> object:
    dwell = int(spec["dwell"])
    cycles = int(spec["cycles"])
    if task == "swap_chain":
        return SwapChainMDP(
            RecurringMDPConfig(
                state_count=int(spec.get("state_count", 9)),
                action_count=2,
                mode_count=int(spec.get("mode_count", 2)),
                separation=float(spec.get("separation", 0.9)),
                dwell=dwell,
                cycles=cycles,
                rare_occupancy=0.03,
                seed=seed,
            )
        )
    config = EnvConfig(
        task=task,
        state_count=int(spec["state_count"]),
        action_count=int(spec["action_count"]),
        mode_count=int(spec.get("mode_count", 2)),
        dwell=dwell,
        cycles=cycles,
        seed=seed,
        slip=float(spec.get("slip", 0.1)),
    )
    if task == "riverswim":
        return RiverSwimMDP(config)
    if task == "deepsea":
        return DeepSeaMDP(config)
    if task == "four_rooms":
        return FourRoomsMDP(config)
    raise ValueError(f"unknown benchmark task: {task}")
