"""Controlled recurring-regime MiniGrid scale study.

The deployable methods receive only partial image observations and actions.
Rewards are attached after reward-free collection. Regime labels are retained
only for evaluation and the explicitly named oracle upper bound.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    import minigrid  # noqa: F401 - registers MiniGrid environments
except ImportError as error:  # pragma: no cover - exercised without optional extra
    gym = None
    _MINIGRID_IMPORT_ERROR = error
else:
    _MINIGRID_IMPORT_ERROR = None

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
except ImportError as error:  # pragma: no cover - base requirements include sklearn
    MiniBatchKMeans = PCA = None
    _SKLEARN_IMPORT_ERROR = error
else:
    _SKLEARN_IMPORT_ERROR = None


METHODS = ("pooled", "restart", "recurrence_aware", "oracle_upper_bound")
ACTIONS = (0, 1, 2)  # left, right, forward
REGIME_SCHEDULE = (0, 0, 1, 1, 0, 0, 1, 1)


def require_minigrid() -> None:
    """Raise an actionable error instead of substituting another environment."""
    if _MINIGRID_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MiniGrid scale study requires the maintained `minigrid==3.1.0` "
            "package. Install it with `python3 -m pip install minigrid==3.1.0`."
        ) from _MINIGRID_IMPORT_ERROR
    if _SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "MiniGrid scale study requires scikit-learn for its learned PCA/"
            "k-means observation representation."
        ) from _SKLEARN_IMPORT_ERROR


@dataclass(frozen=True)
class MiniGridStudyConfig:
    profile: str = "pilot"
    seeds: Tuple[int, ...] = tuple(range(10))
    collection_episodes: int = 8
    collection_steps: int = 96
    diagnostic_steps: int = 48
    evaluation_episodes: int = 5
    evaluation_steps: int = 64
    latent_dimensions: int = 6
    latent_states: int = 36
    match_radius: float = 1.1
    gamma: float = 0.95
    recurrence_gap_target: float = 0.15
    env_id: str = "MiniGrid-Empty-6x6-v0"
    mirror_observations: bool = True


def minigrid_profile(profile: str) -> MiniGridStudyConfig:
    if profile == "quick":
        return MiniGridStudyConfig(
            profile="quick",
            seeds=tuple(range(5)),
            collection_episodes=6,
            collection_steps=64,
            diagnostic_steps=32,
            evaluation_episodes=3,
            evaluation_steps=48,
            latent_dimensions=4,
            latent_states=24,
        )
    if profile == "pilot":
        return MiniGridStudyConfig()
    if profile == "fourrooms":
        return MiniGridStudyConfig(
            profile="fourrooms",
            env_id="MiniGrid-FourRooms-v0",
            collection_episodes=12,
            collection_steps=128,
            diagnostic_steps=64,
            evaluation_episodes=5,
            evaluation_steps=80,
            latent_dimensions=8,
            latent_states=48,
        )
    if profile == "full":
        return MiniGridStudyConfig(
            profile="full",
            seeds=tuple(range(30)),
            collection_episodes=12,
            collection_steps=160,
            diagnostic_steps=64,
            evaluation_episodes=10,
            evaluation_steps=96,
            latent_dimensions=8,
            latent_states=64,
        )
    if profile == "conflict":
        return MiniGridStudyConfig(
            profile="conflict",
            seeds=tuple(range(10)),
            collection_episodes=10,
            collection_steps=96,
            diagnostic_steps=48,
            evaluation_episodes=5,
            evaluation_steps=64,
            mirror_observations=False,
        )
    if profile == "conflict-full":
        return MiniGridStudyConfig(
            profile="conflict-full",
            seeds=tuple(range(30)),
            collection_episodes=12,
            collection_steps=128,
            diagnostic_steps=64,
            evaluation_episodes=8,
            evaluation_steps=80,
            mirror_observations=False,
        )
    raise ValueError(f"unknown MiniGrid profile: {profile}")


class RecurringMiniGridFamily:
    """MiniGrid family with hidden, reset-only recurring dynamics/observation shifts.

    Regime 0 uses native left/right dynamics and native partial observations.
    Regime 1 swaps left/right controls and mirrors the egocentric observation,
    so both the transition kernel and observation kernel change. The wrapper
    deliberately returns only the 7x7 partial image and strips mission, reward,
    and all regime metadata.
    """

    def __init__(
        self,
        seed: int,
        schedule: Sequence[int] = REGIME_SCHEDULE,
        max_steps: int = 96,
        env_id: str = "MiniGrid-Empty-6x6-v0",
        mirror_observations: bool = True,
    ):
        require_minigrid()
        if not schedule or any(int(item) not in (0, 1) for item in schedule):
            raise ValueError("schedule must contain only regimes 0 and 1")
        self.seed = int(seed)
        self.schedule = tuple(int(item) for item in schedule)
        self.max_steps = int(max_steps)
        self.env_id = str(env_id)
        self.mirror_observations = bool(mirror_observations)
        self.episode_index = 0
        self._active_regime: Optional[int] = None
        self._env = None
        self.last_success = False
        require_minigrid()
        if not schedule or any(int(item) not in (0, 1) for item in schedule):
            raise ValueError("schedule must contain only regimes 0 and 1")
        self.seed = int(seed)
        self.schedule = tuple(int(item) for item in schedule)
        self.max_steps = int(max_steps)
        self.episode_index = 0
        self._active_regime: Optional[int] = None
        self._env = None

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return (7, 7, 3)

    def _transform(self, image: np.ndarray) -> np.ndarray:
        output = np.asarray(image, dtype=np.uint8).copy()
        if self._active_regime == 1 and self.mirror_observations:
            output = np.flip(output, axis=1).copy()
            # A bijective color-channel remapping changes observations without
            # injecting a regime bit or altering object semantics.
            colors = output[..., 1].copy()
            output[..., 1] = np.where(
                colors == 0, 2, np.where(colors == 2, 0, colors)
            )
        return output

    def reset(
        self, *, regime: Optional[int] = None, episode_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        selected = (
            self.schedule[self.episode_index % len(self.schedule)]
            if regime is None
            else int(regime)
        )
        if selected not in (0, 1):
            raise ValueError("regime must be 0 or 1")
        if self._env is not None:
            self._env.close()
        self._active_regime = selected
        self._env = gym.make(self.env_id, max_steps=self.max_steps)
        actual_seed = (
            self.seed + self.episode_index
            if episode_seed is None
            else int(episode_seed)
        )
        observation, _ = self._env.reset(seed=actual_seed)
        self.episode_index += 1
        self.last_success = False
        return self._transform(observation["image"]), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._env is None or self._active_regime is None:
            raise RuntimeError("reset must be called before step")
        action = int(action)
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        mapped = {0: 1, 1: 0}.get(action, action) if self._active_regime else action
        observation, _reward, terminated, truncated, _info = self._env.step(mapped)
        self.last_success = bool(terminated)
        # Exact RFE interface: no environment reward or hidden metadata escapes.
        return self._transform(observation["image"]), 0.0, terminated, truncated, {}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def _hash_observation(observation: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(observation).tobytes()).hexdigest()[:16]


def _collect_episode(
    family: RecurringMiniGridFamily,
    rng: np.random.RandomState,
    steps: int,
    *,
    regime: Optional[int] = None,
    episode_seed: Optional[int] = None,
) -> List[Dict]:
    observation, info = family.reset(regime=regime, episode_seed=episode_seed)
    if info:
        raise AssertionError("wrapper leaked reset metadata")
    records = []
    for index in range(int(steps)):
        action = int(rng.choice(ACTIONS))
        next_observation, reward, terminated, truncated, info = family.step(action)
        if reward != 0.0 or info:
            raise AssertionError("collection must be reward-free and metadata-free")
        records.append(
            {
                "step": index,
                "action": action,
                "observation": observation.tolist(),
                "next_observation": next_observation.tolist(),
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
        )
        observation = next_observation
        if terminated or truncated:
            observation, info = family.reset(
                regime=regime, episode_seed=None if episode_seed is None else episode_seed + index + 1
            )
            if info:
                raise AssertionError("wrapper leaked reset metadata")
    return records


class LearnedObservationEncoder:
    """Reward-free PCA features plus k-means latent states."""

    def __init__(self, dimensions: int, states: int, seed: int):
        self.dimensions = int(dimensions)
        self.states = int(states)
        self.seed = int(seed)
        self.pca = None
        self.kmeans = None

    @staticmethod
    def _matrix(observations: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [np.asarray(item, dtype=np.float32).reshape(-1) / 10.0 for item in observations]
        )

    def fit(self, observations: Sequence[np.ndarray]) -> None:
        matrix = self._matrix(observations)
        dimensions = min(self.dimensions, len(matrix), matrix.shape[1])
        self.pca = PCA(n_components=dimensions, random_state=self.seed)
        features = self.pca.fit_transform(matrix)
        clusters = min(self.states, len(features))
        self.kmeans = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=self.seed,
            n_init=5,
            batch_size=min(256, len(features)),
        )
        self.kmeans.fit(features)

    def features(self, observation: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("encoder has not been fit")
        return self.pca.transform(self._matrix([observation]))[0]

    def state(self, observation: np.ndarray) -> int:
        if self.kmeans is None:
            raise RuntimeError("encoder has not been fit")
        return int(self.kmeans.predict(self.features(observation)[None, :])[0])


def _decode_records(records: Sequence[Mapping], encoder: LearnedObservationEncoder) -> List[Dict]:
    output = []
    for record in records:
        observation = np.asarray(record["observation"], dtype=np.uint8)
        next_observation = np.asarray(record["next_observation"], dtype=np.uint8)
        output.append(
            {
                **record,
                "observation_array": observation,
                "next_observation_array": next_observation,
                "state": encoder.state(observation),
                "next_state": encoder.state(next_observation),
                "feature": encoder.features(observation),
                "next_feature": encoder.features(next_observation),
            }
        )
    return output


def _signature(records: Sequence[Mapping], feature_dim: int) -> np.ndarray:
    """Action-conditional learned-feature transition signature."""
    parts = []
    for action in ACTIONS:
        selected = [row for row in records if int(row["action"]) == action]
        if selected:
            deltas = np.asarray(
                [row["next_feature"] - row["feature"] for row in selected]
            )
            parts.extend([deltas.mean(axis=0), deltas.std(axis=0)])
        else:
            parts.extend([np.zeros(feature_dim), np.zeros(feature_dim)])
    return np.concatenate(parts)


def _signature_distance(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


def _infer_modes(
    episodes: Sequence[Sequence[Mapping]], config: MiniGridStudyConfig
) -> Tuple[List[int], List[np.ndarray], np.ndarray, List[Dict]]:
    signatures = np.asarray(
        [_signature(episode[: config.diagnostic_steps], config.latent_dimensions) for episode in episodes]
    )
    scale = np.std(signatures, axis=0)
    scale = np.maximum(scale, 0.05)
    prototypes: List[np.ndarray] = []
    counts: List[int] = []
    assignments: List[int] = []
    diagnostics = []
    active = None
    for episode_index, signature in enumerate(signatures):
        distances = [
            _signature_distance(signature, prototype, scale) for prototype in prototypes
        ]
        detected = active is not None and distances[active] > config.match_radius
        if distances and min(distances) <= config.match_radius:
            selected = int(np.argmin(distances))
            reused = selected != active and counts[selected] > 0
        else:
            selected = len(prototypes)
            prototypes.append(signature.copy())
            counts.append(0)
            distances.append(float("inf"))
            reused = False
        counts[selected] += 1
        prototypes[selected] += (signature - prototypes[selected]) / counts[selected]
        assignments.append(selected)
        diagnostics.append(
            {
                "episode": episode_index,
                "selected_mode": selected,
                "detected_change": int(detected),
                "reused": int(reused),
                "match_distance": min(distances) if distances else float("inf"),
            }
        )
        active = selected
    return assignments, prototypes, scale, diagnostics


def _counts(records: Sequence[Mapping]) -> Dict[Tuple[int, int], Dict[int, int]]:
    output: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in records:
        output[(int(row["state"]), int(row["action"]))][int(row["next_state"])] += 1
    return output


RewardFunction = Callable[[np.ndarray, int, np.ndarray], float]


def _tasks() -> Dict[str, RewardFunction]:
    # MiniGrid object index 8 is goal. These are arbitrary observation-based
    # rewards selected only after the reward-free records are frozen.
    def reach_goal(observation, action, next_observation):
        return float(next_observation[5, 3, 0] == 8 and int(action) == 2)

    reach_goal.native_success = True
    return {
        "goal_visible": lambda _o, _a, n: float(np.any(n[..., 0] == 8)),
        "goal_centered": lambda _o, _a, n: float(n[5, 3, 0] == 8),
        "visual_change": lambda o, _a, n: float(np.mean(o != n) >= 0.04),
        "reach_goal": reach_goal,
    }


def _reward_table(
    records: Sequence[Mapping], reward_function: RewardFunction
) -> Dict[Tuple[int, int, int], float]:
    totals: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    for row in records:
        key = (int(row["state"]), int(row["action"]), int(row["next_state"]))
        totals[key].append(
            reward_function(
                row["observation_array"], int(row["action"]), row["next_observation_array"]
            )
        )
    return {key: float(np.mean(values)) for key, values in totals.items()}


def _plan(
    records: Sequence[Mapping],
    reward_function: RewardFunction,
    state_count: int,
    gamma: float,
) -> Dict[int, int]:
    transitions = _counts(records)
    rewards = _reward_table(records, reward_function)
    values = np.zeros(state_count, dtype=float)
    q_values = np.zeros((state_count, len(ACTIONS)), dtype=float)
    for _ in range(500):
        updated = np.zeros_like(values)
        for state in range(state_count):
            for action in ACTIONS:
                outcomes = transitions.get((state, action))
                if outcomes:
                    total = sum(outcomes.values())
                    q_values[state, action] = sum(
                        count
                        / total
                        * (rewards.get((state, action, nxt), 0.0) + gamma * values[nxt])
                        for nxt, count in outcomes.items()
                    )
                else:
                    q_values[state, action] = gamma * values[state]
            updated[state] = np.max(q_values[state])
        if float(np.max(np.abs(updated - values))) < 1e-8:
            values = updated
            break
        values = updated
    return {state: int(np.argmax(q_values[state])) for state in range(state_count)}


def _evaluate_policy(
    seed: int,
    regime: int,
    encoder: LearnedObservationEncoder,
    policy: Mapping[int, int],
    reward_function: RewardFunction,
    config: MiniGridStudyConfig,
) -> float:
    returns = []
    for episode in range(config.evaluation_episodes):
        family = RecurringMiniGridFamily(
            seed + 70_000,
            max_steps=config.evaluation_steps,
            env_id=config.env_id,
            mirror_observations=config.mirror_observations,
        )
        observation, _ = family.reset(
            regime=regime, episode_seed=seed + 80_000 + episode
        )
        total = 0.0
        for _step in range(config.evaluation_steps):
            action = int(policy.get(encoder.state(observation), 0))
            next_observation, _reward, terminated, truncated, _ = family.step(action)
            total += reward_function(observation, action, next_observation)
            if getattr(reward_function, "native_success", False):
                total += float(family.last_success)
            observation = next_observation
            if terminated or truncated:
                break
        family.close()
        returns.append(total / config.evaluation_steps)
    return float(np.mean(returns))


def _match_signature(
    records: Sequence[Mapping],
    prototypes: Sequence[np.ndarray],
    scale: np.ndarray,
    config: MiniGridStudyConfig,
) -> Tuple[Optional[int], float]:
    signature = _signature(records, config.latent_dimensions)
    distances = [_signature_distance(signature, item, scale) for item in prototypes]
    if not distances:
        return None, float("inf")
    selected = int(np.argmin(distances))
    return (
        selected if distances[selected] <= config.match_radius else None,
        float(distances[selected]),
    )


def _unit(config: MiniGridStudyConfig, seed: int) -> Dict:
    started = time.perf_counter()
    rng = np.random.RandomState(seed)
    family = RecurringMiniGridFamily(
        seed,
        max_steps=config.collection_steps,
        env_id=config.env_id,
        mirror_observations=config.mirror_observations,
    )
    raw_episodes = []
    truth = []
    for episode in range(config.collection_episodes):
        regime = REGIME_SCHEDULE[episode % len(REGIME_SCHEDULE)]
        raw_episodes.append(
            _collect_episode(family, rng, config.collection_steps, regime=regime)
        )
        truth.append(regime)
    family.close()

    observations = [
        np.asarray(value, dtype=np.uint8)
        for episode in raw_episodes
        for row in episode
        for value in (row["observation"], row["next_observation"])
    ]
    encoder = LearnedObservationEncoder(
        config.latent_dimensions, config.latent_states, seed
    )
    encoder.fit(observations)
    episodes = [_decode_records(episode, encoder) for episode in raw_episodes]
    assignments, prototypes, scale, diagnostics = _infer_modes(episodes, config)
    archives = {
        mode: [row for index, episode in enumerate(episodes) if assignments[index] == mode for row in episode]
        for mode in sorted(set(assignments))
    }
    oracle_archives = {
        regime: [row for index, episode in enumerate(episodes) if truth[index] == regime for row in episode]
        for regime in (0, 1)
    }
    pooled = [row for episode in episodes for row in episode]
    majority = {
        mode: max((0, 1), key=lambda regime: sum(
            assignments[index] == mode and truth[index] == regime
            for index in range(len(episodes))
        ))
        for mode in archives
    }

    values = []
    diagnoses = []
    savings = []
    for regime in (0, 1):
        deployment_family = RecurringMiniGridFamily(
            seed + 30_000,
            max_steps=config.diagnostic_steps,
            env_id=config.env_id,
            mirror_observations=config.mirror_observations,
        )
        deployment_raw = _collect_episode(
            deployment_family,
            np.random.RandomState(seed + 40_000 + regime),
            config.diagnostic_steps,
            regime=regime,
            episode_seed=seed + 50_000 + regime,
        )
        deployment_family.close()
        deployment = _decode_records(deployment_raw, encoder)
        selected, distance = _match_signature(
            deployment, prototypes, scale, config
        )
        diagnoses.append(
            {
                "true_regime": regime,
                "selected_mode": selected,
                "match_distance": distance,
                "accepted": int(selected is not None),
                "deployment_id_error": int(
                    selected is None or majority.get(selected) != regime
                ),
            }
        )
        method_records = {
            "pooled": pooled + deployment,
            "restart": deployment,
            "recurrence_aware": (
                archives.get(selected, []) + deployment if selected is not None else deployment
            ),
            "oracle_upper_bound": oracle_archives[regime] + deployment,
        }
        task_returns: Dict[Tuple[str, str], float] = {}
        for task_name, reward_function in _tasks().items():
            for method in METHODS:
                policy = _plan(
                    method_records[method],
                    reward_function,
                    len(encoder.kmeans.cluster_centers_),
                    config.gamma,
                )
                task_returns[(method, task_name)] = _evaluate_policy(
                    seed, regime, encoder, policy, reward_function, config
                )
            oracle_return = task_returns[("oracle_upper_bound", task_name)]
            for method in METHODS:
                value = task_returns[(method, task_name)]
                values.append(
                    {
                        "true_regime": regime,
                        "method": method,
                        "task": task_name,
                        "return": value,
                        "oracle_return": oracle_return,
                        "task_gap": max(0.0, oracle_return - value),
                    }
                )
        for method in METHODS:
            selected_rows = [
                row for row in values
                if row["true_regime"] == regime and row["method"] == method
            ]
            values.append(
                {
                    "true_regime": regime,
                    "method": method,
                    "task": "__worst__",
                    "return": None,
                    "oracle_return": None,
                    "task_gap": max(row["task_gap"] for row in selected_rows),
                }
            )

        budgets = sorted(set((8, 16, config.diagnostic_steps)))
        needed = {}
        for method in ("restart", "recurrence_aware"):
            needed[method] = config.diagnostic_steps + 1
            for budget in budgets:
                prefix = deployment[:budget]
                if method == "recurrence_aware":
                    matched, _ = _match_signature(prefix, prototypes, scale, config)
                    records = archives.get(matched, []) + prefix if matched is not None else prefix
                else:
                    records = prefix
                worst = 0.0
                for task_name, reward_function in _tasks().items():
                    policy = _plan(
                        records,
                        reward_function,
                        len(encoder.kmeans.cluster_centers_),
                        config.gamma,
                    )
                    value = _evaluate_policy(
                        seed + 100_000, regime, encoder, policy, reward_function, config
                    )
                    oracle_value = task_returns[("oracle_upper_bound", task_name)]
                    worst = max(worst, max(0.0, oracle_value - value))
                if worst <= config.recurrence_gap_target:
                    needed[method] = budget
                    break
        savings.append(
            {
                "true_regime": regime,
                "restart_samples": needed["restart"],
                "recurrence_samples": needed["recurrence_aware"],
                "sample_savings": needed["restart"] - needed["recurrence_aware"],
                "target_gap": config.recurrence_gap_target,
            }
        )

    raw_rows = []
    for episode_index, episode in enumerate(raw_episodes):
        for row in episode:
            raw_rows.append(
                {
                    "episode": episode_index,
                    "step": row["step"],
                    "action": row["action"],
                    "observation_hash": _hash_observation(np.asarray(row["observation"])),
                    "next_observation_hash": _hash_observation(np.asarray(row["next_observation"])),
                    "observation_json": json.dumps(
                        row["observation"], separators=(",", ":")
                    ),
                    "next_observation_json": json.dumps(
                        row["next_observation"], separators=(",", ":")
                    ),
                    "true_regime_evaluation_only": truth[episode_index],
                    "inferred_mode": assignments[episode_index],
                    "reward": 0.0,
                }
            )
    for row, regime in zip(diagnostics, truth):
        row["true_regime_evaluation_only"] = regime
    return {
        "seed": seed,
        "raw_transitions": raw_rows,
        "values": values,
        "diagnosis": diagnoses,
        "savings": savings,
        "diagnostics": diagnostics,
        "runtime_seconds": time.perf_counter() - started,
    }


def _id(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace a result/checkpoint only after its complete payload is written."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _mean(rows: Sequence[Mapping], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else float("nan")


def run_minigrid_study(
    config: MiniGridStudyConfig, output: Path, resume: bool = True
) -> Dict:
    require_minigrid()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config_dict = asdict(config)
    config_hash = _id(config_dict)
    units = []
    for seed in config.seeds:
        unit_id = _id({"config": config_dict, "seed": seed})
        path = checkpoints / f"{unit_id}.json"
        if resume and path.exists():
            candidate = json.loads(path.read_text())
            if candidate.get("config_hash") == config_hash:
                units.append(candidate)
                continue
        candidate = _unit(config, int(seed))
        candidate.update({"unit_id": unit_id, "config_hash": config_hash})
        _atomic_write_text(path, json.dumps(candidate, indent=2, sort_keys=True))
        units.append(candidate)

    raw = [{**{"seed": unit["seed"]}, **row} for unit in units for row in unit["raw_transitions"]]
    values = [{**{"seed": unit["seed"]}, **row} for unit in units for row in unit["values"]]
    diagnoses = [{**{"seed": unit["seed"]}, **row} for unit in units for row in unit["diagnosis"]]
    savings = [{**{"seed": unit["seed"]}, **row} for unit in units for row in unit["savings"]]
    diagnostics = [{**{"seed": unit["seed"]}, **row} for unit in units for row in unit["diagnostics"]]
    summary = []
    for method in METHODS:
        selected = [row for row in values if row["method"] == method and row["task"] == "__worst__"]
        summary.append(
            {
                "method": method,
                "n_seeds": len(config.seeds),
                "worst_task_gap": _mean(selected, "task_gap"),
            }
        )
    recurrence_gap = next(row["worst_task_gap"] for row in summary if row["method"] == "recurrence_aware")
    restart_gap = next(row["worst_task_gap"] for row in summary if row["method"] == "restart")
    pooled_gap = next(row["worst_task_gap"] for row in summary if row["method"] == "pooled")
    id_error = _mean(diagnoses, "deployment_id_error")
    mean_savings = _mean(savings, "sample_savings")
    gate_checks = {
        "ten_or_more_seeds": len(config.seeds) >= 10,
        "recurrence_beats_restart_worst_task_gap": recurrence_gap < restart_gap,
        "recurrence_no_worse_than_pooled": recurrence_gap <= pooled_gap,
        "deployment_id_error_at_most_quarter": id_error <= 0.25,
        "positive_recurrence_sample_savings": mean_savings > 0,
    }
    payload = {
        "protocol": "minigrid_recurring_partial_observation_v1",
        "dependency": "minigrid==3.1.0",
        "config": config_dict,
        "config_hash": config_hash,
        "n_units": len(units),
        "runtime_seconds": sum(float(unit["runtime_seconds"]) for unit in units),
        "summary": summary,
        "deployment_id_error": id_error,
        "mean_recurrence_sample_savings": mean_savings,
        "gate_checks": gate_checks,
        "minigrid_gate_passed": all(gate_checks.values()),
        "limitations": [
            "Symbolic 7x7 partial images, not RGB pixels.",
            "Two deterministic recurring regimes in MiniGrid Empty-6x6.",
            "Memoryless latent-state planning under partial observability.",
            "Oracle is an empirical label-aware upper control, not exact optimal value.",
        ],
    }
    _write_csv(output / "raw_transitions.csv", raw)
    _write_csv(output / "value_gaps.csv", values)
    _write_csv(output / "deployment_diagnosis.csv", diagnoses)
    _write_csv(output / "recurrence_savings.csv", savings)
    _write_csv(output / "detector_diagnostics.csv", diagnostics)
    _write_csv(output / "summary.csv", summary)
    _atomic_write_text(
        output / "results.json", json.dumps(payload, indent=2, sort_keys=True)
    )
    return payload


def minigrid_resume_hash(
    config: MiniGridStudyConfig, output: Path
) -> Tuple[str, str]:
    files = (
        "raw_transitions.csv",
        "value_gaps.csv",
        "deployment_diagnosis.csv",
        "recurrence_savings.csv",
        "detector_diagnostics.csv",
        "summary.csv",
    )
    run_minigrid_study(config, output, resume=True)
    first = hashlib.sha256(
        b"".join((output / filename).read_bytes() for filename in files)
    ).hexdigest()
    run_minigrid_study(config, output, resume=True)
    second = hashlib.sha256(
        b"".join((output / filename).read_bytes() for filename in files)
    ).hexdigest()
    return first, second
