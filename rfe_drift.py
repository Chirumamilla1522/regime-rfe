#!/usr/bin/env python

"""DEPRECATED historical single-file prototype.

This file intentionally remains in the repository to preserve useful history,
but its environment, timing, and evaluation semantics diverge from the
maintained implementation.  New work and reported results must use ``run.py``
and the ``rfe_drift`` package.  In particular, this prototype is not a
faithful implementation of UCRL-RFE and must not be used for paper claims.
"""

#region imports
from __future__ import annotations

import os
from enum import Enum, auto
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
#endregion

if __name__ == "__main__":
    import warnings

    warnings.warn(
        "rfe_drift.py is a deprecated historical prototype; use run.py",
        FutureWarning,
        stacklevel=1,
    )

# ================================================================
# 1. ENVIRONMENT WITH DRIFT
# ================================================================

class DriftType(Enum):
    GOAL_SHIFT = auto()
    TRANSITION_NOISE = auto()


class DriftSchedule(Enum):
    SUDDEN = auto()
    GRADUAL = auto()


class DriftGridWorld(gym.Env):
    """
    Simple gridworld with distributional drift.

    - State: (x, y) in [0, 1]^2
    - Actions: 0=up, 1=right, 2=down, 3=left
    - Drift: either goals move or transition noise increases around drift_time
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 10,
        drift_type: DriftType = DriftType.GOAL_SHIFT,
        drift_strength: float = 0.9,
        drift_schedule: DriftSchedule = DriftSchedule.SUDDEN,
        drift_time: int = 200,
        seed: int = 0,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.drift_type = drift_type
        self.drift_strength = float(drift_strength)
        self.drift_schedule = drift_schedule
        self.drift_time = int(drift_time)

        self.rng = np.random.RandomState(seed)

        # Agent position (integer coords)
        self.agent_pos = np.array([0, 0], dtype=np.int64)

        # Goals before and after drift
        self.goals_before = [np.array([grid_size - 1, grid_size - 1], dtype=np.int64)]
        self.goals_after = [np.array([0, grid_size - 1], dtype=np.int64)]

        # Transition noise parameters
        self.base_noise = 0.0
        self.noise_after = self.drift_strength

        # Gym spaces: observation is normalized position, action is discrete
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.step_count: int = 0
        self.episode_step_count: int = 0

    # ----- Drift-dependent properties --------------------------------------

    @property
    def current_goals(self) -> List[np.ndarray]:
        if self.step_count < self.drift_time:
            return self.goals_before
        return self.goals_after

    @property
    def current_noise(self) -> float:
        if self.drift_type != DriftType.TRANSITION_NOISE:
            return self.base_noise

        if self.step_count < self.drift_time:
            return self.base_noise

        if self.drift_schedule == DriftSchedule.SUDDEN:
            return self.noise_after

        # GRADUAL schedule: ramp up after drift_time
        t = (self.step_count - self.drift_time) / max(1, self.drift_time)
        t = np.clip(t, 0.0, 1.0)
        return (1 - t) * self.base_noise + t * self.noise_after

    # ----- Gym API ---------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng.seed(seed)
        self.episode_step_count = 0
        # Random start
        self.agent_pos = self.rng.randint(
            0, self.grid_size, size=(2,), dtype=np.int64
        )
        obs = self._obs()
        info = {"goals": self._goals_float()}
        return obs, info

    def step(self, action: int):
        self.step_count += 1
        self.episode_step_count += 1

        # Transition noise: sometimes replace action with random one
        if self.rng.rand() < self.current_noise:
            action = self.action_space.sample()

        # Apply action
        if action == 0:        # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:      # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:      # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 3:      # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

        obs = self._obs()

        # Reward-free: env itself always returns 0 reward
        reward = 0.0

        done = any(np.array_equal(self.agent_pos, g) for g in self.current_goals)
        truncated = self.episode_step_count >= (self.grid_size * 4)

        info: Dict[str, Any] = {"goals": self._goals_float()}
        return obs, reward, bool(done), bool(truncated), info

    # ----- Internal helpers ------------------------------------------------

    def _obs(self) -> np.ndarray:
        return (self.agent_pos / (self.grid_size - 1)).astype(np.float32)

    def _goals_float(self) -> np.ndarray:
        return np.stack(
            [g / (self.grid_size - 1) for g in self.current_goals], axis=0
        ).astype(np.float32)


# ================================================================
# 2. REWARD-FREE EXPLORATION
# ================================================================


@dataclass
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    done: bool
    time: float


class UCRLRFE:
    """
    Simple reward-free exploration:
    - Run for num_steps
    - Collect (state, action, next_state, done, time) into a dataset
    """

    def __init__(
        self,
        env: DriftGridWorld,
        num_steps: int = 10_000,
        seed: int = 0,
        epsilon: float = 0.1,
    ):
        self.env = env
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)

    def collect_data(self) -> TensorDataset:
        states: List[np.ndarray] = []
        actions: List[int] = []
        next_states: List[np.ndarray] = []
        dones: List[bool] = []
        times: List[float] = []

        state, _ = self.env.reset()
        for _ in range(self.num_steps):
            time = self.env.step_count / max(1, self.env.drift_time * 2)

            if self.rng.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                # Could be more clever (e.g. UCRL); random is fine for a toy
                action = self.env.action_space.sample()

            next_state, _, terminated, truncated, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            dones.append(terminated or truncated)
            times.append(time)

            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.bool)
        times_t = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)

        return TensorDataset(states_t, actions_t, next_states_t, dones_t, times_t)


# ================================================================
# 3. REPRESENTATIONS
# ================================================================


class FixedEncoder(nn.Module):
    """Encoder that only sees state (x, y)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DriftAwareEncoder(nn.Module):
    """Encoder that also sees time (state + time → embedding)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, emb_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, time], dim=-1)
        return self.net(inp)


@dataclass
class RepresentationTrainerConfig:
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10
    loss_type: Literal["forward_predict"] = "forward_predict"


class RepresentationTrainer:
    """
    Supervised representation learning:
    - Input: (state[, time])
    - Target: next_state
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: RepresentationTrainerConfig,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.config = config
        self.device = device

        # Determine embedding dim (assumes final layer is Linear)
        emb_dim = encoder.net[-1].out_features

        self.predict_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # predict next state (x, y)
        ).to(device)

        self.optim = optim.Adam(
            list(self.encoder.parameters()) + list(self.predict_head.parameters()),
            lr=config.lr,
        )
        self.criterion = nn.MSELoss()

    def train(self, dataset: TensorDataset) -> nn.Module:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.encoder.train()
        self.predict_head.train()

        for epoch in range(self.config.num_epochs):
            total_loss = 0.0
            total_batches = 0

            for states, actions, next_states, dones, times in loader:
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                times = times.to(self.device)

                self.optim.zero_grad()

                if isinstance(self.encoder, DriftAwareEncoder):
                    emb = self.encoder(states, times)
                else:
                    emb = self.encoder(states)

                pred_next = self.predict_head(emb)
                loss = self.criterion(pred_next, next_states)

                loss.backward()
                self.optim.step()

                total_loss += loss.item()
                total_batches += 1

            avg_loss = total_loss / max(1, total_batches)
            print(
                f"[RepTrain] Epoch {epoch+1}/{self.config.num_epochs} "
                f"Loss: {avg_loss:.4f}"
            )

        return self.encoder


# ================================================================
# 4. RL: DQN AGENT USING ENCODERS
# ================================================================


class QNetwork(nn.Module):
    def __init__(self, emb_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 64
    min_buffer_size: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000
    target_update_freq: int = 1_000


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, s, a, r, s2, done):
        self.states[self.ptr] = s
        self.next_states[self.ptr] = s2
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self), size=batch_size)
        return dict(
            states=self.states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_states=self.next_states[idxs],
            dones=self.dones[idxs],
        )


class DQNAgent:
    """
    DQN agent that uses a pretrained encoder
    (either FixedEncoder or DriftAwareEncoder).
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_actions: int,
        config: DQNConfig,
        device: str = "cpu",
        drift_aware: bool = False,
    ):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.drift_aware = drift_aware
        self.device = device

        # Infer emb_dim
        with torch.no_grad():
            dummy_state = torch.zeros(1, 2, device=device)
            dummy_time = torch.zeros(1, 1, device=device)
            if self.drift_aware:
                emb = self.encoder(dummy_state, dummy_time)
            else:
                emb = self.encoder(dummy_state)
            emb_dim = emb.shape[-1]

        self.q_net = QNetwork(emb_dim, num_actions).to(device)
        self.target_q_net = QNetwork(emb_dim, num_actions).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.config = config
        self.optim = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, state_dim=2)
        self.total_steps = 0

    def _epsilon(self) -> float:
        c = self.config
        frac = min(1.0, self.total_steps / max(1, c.epsilon_decay))
        return c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)

    def select_action(self, state, time: float, training: bool = True) -> int:
        eps = self._epsilon()
        if training and np.random.rand() < eps:
            return np.random.randint(0, self.q_net.net[-1].out_features)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        time_t = torch.tensor([[time]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.drift_aware:
                emb = self.encoder(state_t, time_t)
            else:
                emb = self.encoder(state_t)
            q_vals = self.q_net(emb)
        return int(q_vals.argmax(dim=-1).item())

    def update(self, state, action, reward, next_state, done, time: float):
        c = self.config
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.buffer) < c.min_buffer_size:
            return

        batch = self.buffer.sample(c.batch_size)

        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

        times = torch.zeros(len(states), 1, dtype=torch.float32, device=self.device) + time

        with torch.no_grad():
            if self.drift_aware:
                next_emb = self.encoder(next_states, times)
            else:
                next_emb = self.encoder(next_states)
            next_q_vals = self.target_q_net(next_emb)
            max_next_q = next_q_vals.max(dim=-1).values
            target = rewards + c.gamma * (1.0 - dones) * max_next_q

        if self.drift_aware:
            emb = self.encoder(states, times)
        else:
            emb = self.encoder(states)

        q_vals = self.q_net(emb)
        q_taken = q_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = nn.functional.mse_loss(q_taken, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.total_steps % c.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


# ================================================================
# 5. UTILS: METRICS + REWARD FUNCTION
# ================================================================


@dataclass
class MetricsTracker:
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def log(self, key: str, value: float):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(float(value))

    def get(self, key: str) -> List[float]:
        return self.metrics.get(key, [])

    def mean(self, key: str) -> float:
        vals = self.get(key)
        return float(np.mean(vals)) if len(vals) > 0 else 0.0


class GoalRewardFunction:
    """
    Reward:
    - +1 if next_state close to any goal
    - step_penalty otherwise
    """

    def __init__(self, goals: np.ndarray, step_penalty: float = -0.01):
        self.goals = goals  # [G, 2]
        self.step_penalty = step_penalty

    def __call__(self, state, action, next_state) -> float:
        dist = np.linalg.norm(next_state[None, :] - self.goals, axis=-1)
        hit = np.any(dist < 0.05)
        if hit:
            return 1.0
        return self.step_penalty


# ================================================================
# 6. MAIN PIPELINE
# ================================================================


CONFIG = {
    "grid_size": 10,
    "drift_type": DriftType.GOAL_SHIFT,
    "drift_strength": 0.7,
    "drift_schedule": DriftSchedule.SUDDEN,
    "drift_time": 200,
    "num_exploration_steps": 10_000,
    "num_train_episodes": 300,
    "num_eval_episodes": 50,
    "seed": 42,
}


def make_env() -> DriftGridWorld:
    return DriftGridWorld(
        grid_size=CONFIG["grid_size"],
        drift_type=CONFIG["drift_type"],
        drift_strength=CONFIG["drift_strength"],
        drift_schedule=CONFIG["drift_schedule"],
        drift_time=CONFIG["drift_time"],
        seed=CONFIG["seed"],
    )


def train_dqn(
    env: DriftGridWorld,
    encoder: nn.Module,
    drift_aware: bool,
    device: str = "cpu",
) -> Tuple[DQNAgent, MetricsTracker]:
    config = DQNConfig()
    agent = DQNAgent(
        encoder=encoder,
        num_actions=env.action_space.n,
        config=config,
        device=device,
        drift_aware=drift_aware,
    )

    metrics = MetricsTracker()

    for episode in range(CONFIG["num_train_episodes"]):
        state, info = env.reset()
        episode_reward = 0.0

        while True:
            time = env.step_count / (CONFIG["drift_time"] * 2)
            action = agent.select_action(state, time=time, training=True)
            next_state, _, terminated, truncated, info = env.step(action)

            reward_fn = GoalRewardFunction(goals=info["goals"])
            reward = reward_fn(state, action, next_state)

            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated or truncated,
                time=time,
            )

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                metrics.log("train_reward", episode_reward)
                break

        if (episode + 1) % 50 == 0:
            print(
                f"[DQN][{'DriftAware' if drift_aware else 'Fixed'}] "
                f"Episode {episode+1}/{CONFIG['num_train_episodes']} "
                f"Mean reward: {metrics.mean('train_reward'):.3f}"
            )

    return agent, metrics


def eval_policy(
    env: DriftGridWorld,
    agent: DQNAgent,
    episodes: int,
    before_drift: bool,
    device: str = "cpu",
) -> Tuple[float, float]:
    rewards = []

    for _ in range(episodes):
        state, info = env.reset()
        env.step_count = 0 if before_drift else CONFIG["drift_time"] + 1

        episode_reward = 0.0

        while True:
            time = env.step_count / (CONFIG["drift_time"] * 2)
            action = agent.select_action(state, time=time, training=False)
            next_state, _, terminated, truncated, info = env.step(action)

            reward_fn = GoalRewardFunction(goals=info["goals"])
            reward = reward_fn(state, action, next_state)

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                rewards.append(episode_reward)
                break

    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results_dir = Path("results_single")
    results_dir.mkdir(exist_ok=True)

    # 1) Build env
    env = make_env()

    # 2) Reward-free exploration
    print("\n" + "=" * 60)
    print("STEP 1: Reward-free exploration")
    rfe = UCRLRFE(
        env,
        num_steps=CONFIG["num_exploration_steps"],
        seed=CONFIG["seed"],
    )
    dataset = rfe.collect_data()
    print(f"Collected {len(dataset)} transitions.")

    # 3) Representation pretraining
    print("\n" + "=" * 60)
    print("STEP 2: Representation training")

    rep_cfg = RepresentationTrainerConfig(
        num_epochs=10, batch_size=256, lr=1e-3
    )

    fixed_encoder = FixedEncoder()
    fixed_trainer = RepresentationTrainer(
        fixed_encoder, rep_cfg, device=device
    )
    fixed_encoder = fixed_trainer.train(dataset)

    drift_encoder = DriftAwareEncoder()
    drift_trainer = RepresentationTrainer(
        drift_encoder, rep_cfg, device=device
    )
    drift_encoder = drift_trainer.train(dataset)

    # 4) Downstream RL training
    print("\n" + "=" * 60)
    print("STEP 3: Downstream RL training")

    env_train_fixed = make_env()
    fixed_agent, fixed_metrics = train_dqn(
        env_train_fixed, fixed_encoder, drift_aware=False, device=device
    )

    env_train_drift = make_env()
    drift_agent, drift_metrics = train_dqn(
        env_train_drift, drift_encoder, drift_aware=True, device=device
    )

    # 5) Evaluation
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")

    env_eval_fixed_before = make_env()
    fixed_before_mean, fixed_before_std = eval_policy(
        env_eval_fixed_before,
        fixed_agent,
        CONFIG["num_eval_episodes"],
        before_drift=True,
        device=device,
    )

    env_eval_fixed_after = make_env()
    fixed_after_mean, fixed_after_std = eval_policy(
        env_eval_fixed_after,
        fixed_agent,
        CONFIG["num_eval_episodes"],
        before_drift=False,
        device=device,
    )

    env_eval_drift_before = make_env()
    drift_before_mean, drift_before_std = eval_policy(
        env_eval_drift_before,
        drift_agent,
        CONFIG["num_eval_episodes"],
        before_drift=True,
        device=device,
    )

    env_eval_drift_after = make_env()
    drift_after_mean, drift_after_std = eval_policy(
        env_eval_drift_after,
        drift_agent,
        CONFIG["num_eval_episodes"],
        before_drift=False,
        device=device,
    )

    # 6) Plot results
    print("\n" + "=" * 60)
    print("STEP 5: Plotting results")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training curves
    axes[0].plot(
        fixed_metrics.get("train_reward"),
        label="Fixed encoder",
    )
    axes[0].plot(
        drift_metrics.get("train_reward"),
        label="Drift-aware encoder",
    )
    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    # Before/after drift bar plot
    x = np.arange(2)
    width = 0.35
    axes[1].bar(
        x - width / 2,
        [fixed_before_mean, fixed_after_mean],
        width,
        yerr=[fixed_before_std, fixed_after_std],
        label="Fixed encoder",
        alpha=0.8,
    )
    axes[1].bar(
        x + width / 2,
        [drift_before_mean, drift_after_mean],
        width,
        yerr=[drift_before_std, drift_after_std],
        label="Drift-aware encoder",
        alpha=0.8,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Before drift", "After drift"])
    axes[1].set_ylabel("Average Return")
    axes[1].set_title("Performance Before/After Drift")
    axes[1].legend()

    # Performance drop plot
    def pct_drop(before: float, after: float) -> float:
        if abs(before) < 1e-6:
            return 0.0
        return 100.0 * (before - after) / abs(before)

    fixed_drop = pct_drop(fixed_before_mean, fixed_after_mean)
    drift_drop = pct_drop(drift_before_mean, drift_after_mean)

    axes[2].bar(
        [0, 1],
        [fixed_drop, drift_drop],
        tick_label=["Fixed", "Drift-aware"],
    )
    axes[2].set_ylabel("Performance Drop (%)")
    axes[2].set_title("Robustness to Drift\n(lower is better)")
    axes[2].grid(True, axis="y")

    plt.tight_layout()
    out_path = results_dir / "final_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nResults saved to {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
