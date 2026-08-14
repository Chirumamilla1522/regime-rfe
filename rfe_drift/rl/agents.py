"""
Downstream RL Agents

Simple RL agents that use the pretrained representations for downstream tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict
from collections import deque
import random


class QLearningAgent:
    """
    Tabular Q-learning agent that uses state representations.
    """
    
    def __init__(
        self,
        action_dim: int,
        encoder,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            action_dim: Number of actions
            encoder: State encoder (FixedEncoder or DriftAwareEncoder)
            learning_rate: Q-learning learning rate
            gamma: Discount factor
            epsilon: Initial epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
        """
        self.action_dim = action_dim
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: maps state embeddings to Q-values
        # Using a dictionary with discretized embeddings as keys
        self.Q = {}
        self.embedding_cache = {}
    
    def _discretize_embedding(self, embedding: np.ndarray, bins: int = 10) -> tuple:
        """Discretize continuous embedding for tabular Q-learning"""
        # Flatten embedding if needed
        embedding_flat = embedding.flatten()
        # Simple binning approach
        discretized = tuple(int(x) for x in np.digitize(embedding_flat, bins=np.linspace(-1, 1, bins)))
        return discretized
    
    def _get_state_key(self, state: np.ndarray, time: Optional[float] = None) -> tuple:
        """Get key for Q-table lookup"""
        # Encode state
        if time is not None and hasattr(self.encoder, 'encode'):
            # Try drift-aware encoder (check if it accepts time parameter)
            try:
                embedding = self.encoder.encode(state, time=time)
            except TypeError:
                # Fall back to regular encoding
                embedding = self.encoder.encode(state)
        else:
            embedding = self.encoder.encode(state)
        
        # Discretize for tabular representation
        return self._discretize_embedding(embedding)
    
    def get_q_value(self, state: np.ndarray, action: int, time: Optional[float] = None) -> float:
        """Get Q-value for state-action pair"""
        state_key = self._get_state_key(state, time)
        action_key = (state_key, action)
        return self.Q.get(action_key, 0.0)
    
    def select_action(self, state: np.ndarray, time: Optional[float] = None, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        # Greedy action
        q_values = [self.get_q_value(state, a, time) for a in range(self.action_dim)]
        return np.argmax(q_values)
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        time: Optional[float] = None,
    ):
        """Update Q-values using Q-learning"""
        state_key = self._get_state_key(state, time)
        next_state_key = self._get_state_key(next_state, time)
        
        action_key = (state_key, action)
        
        # Current Q-value
        current_q = self.Q.get(action_key, 0.0)
        
        # Next state max Q-value
        next_q_values = [
            self.Q.get((next_state_key, a), 0.0) 
            for a in range(self.action_dim)
        ]
        max_next_q = max(next_q_values) if not done else 0.0
        
        # Q-learning update
        target = reward + self.gamma * max_next_q
        new_q = current_q + self.learning_rate * (target - current_q)
        
        self.Q[action_key] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQNAgent:
    """
    Deep Q-Network agent that uses state representations.
    """
    
    def __init__(
        self,
        action_dim: int,
        encoder,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = "cpu",
        seed: int = 0,
    ):
        """
        Initialize DQN agent.
        
        Args:
            action_dim: Number of actions
            encoder: State encoder (FixedEncoder or DriftAwareEncoder)
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run on
        """
        self.action_dim = action_dim
        self.encoder = encoder
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.step_count = 0
        self.rng = np.random.RandomState(seed)
        self.python_rng = random.Random(seed)
        
        # Q-network: takes encoded state, outputs Q-values
        self.q_network = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)
        
        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Move encoder to device
        self.encoder.to(device)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
    
    def select_action(self, state: np.ndarray, time: Optional[float] = None, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.randint(0, self.action_dim))
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Check if encoder is drift-aware
            is_drift_aware = hasattr(self.encoder, 'use_time_embedding') and self.encoder.use_time_embedding
            
            if time is not None and is_drift_aware:
                # Drift-aware encoder
                time_tensor = torch.FloatTensor([[time]]).to(self.device)
                embedding = self.encoder(state_tensor, time_tensor)
            else:
                # Fixed encoder or no time information
                embedding = self.encoder(state_tensor)
            
            q_values = self.q_network(embedding)
            action = q_values.argmax().item()
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        time: Optional[float] = None,
        next_time: Optional[float] = None,
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.append(
            (state, action, reward, next_state, done, time, next_time)
        )
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.python_rng.sample(list(self.replay_buffer), self.batch_size)
        
        # Convert to numpy arrays first, then to tensors (faster)
        states = torch.FloatTensor(np.array([record[0] for record in batch])).to(self.device)
        actions = torch.LongTensor(np.array([record[1] for record in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([record[2] for record in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([record[3] for record in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([record[4] for record in batch])).to(self.device)
        times = [record[5] for record in batch]
        next_times = [
            record[6] if record[6] is not None else record[5] for record in batch
        ]
        
        # Encode states
        # Check if encoder is drift-aware (has use_time_embedding attribute)
        is_drift_aware = hasattr(self.encoder, 'use_time_embedding') and self.encoder.use_time_embedding
        has_time_info = len(times) > 0 and any(t is not None for t in times)
        
        if has_time_info and is_drift_aware:
            # Drift-aware encoder with time information
            time_tensors = torch.FloatTensor([[t] if t is not None else [0] for t in times]).to(self.device)
            state_embeddings = self.encoder(states, time_tensors)
            next_time_tensors = torch.FloatTensor(
                [[t] if t is not None else [0] for t in next_times]
            ).to(self.device)
            next_state_embeddings = self.encoder(next_states, next_time_tensors)
        else:
            # Fixed encoder or no time information
            state_embeddings = self.encoder(states)
            next_state_embeddings = self.encoder(next_states)
        
        # Current Q-values
        current_q_values = self.q_network(state_embeddings)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next state Q-values (target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_embeddings)
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (self.gamma * max_next_q * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        time: Optional[float] = None,
        next_time: Optional[float] = None,
    ):
        """Update agent (store transition and train)"""
        self.store_transition(
            state, action, reward, next_state, done, time, next_time
        )
        return self.train_step()

