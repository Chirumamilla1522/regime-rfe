"""
State Representation Encoders

Implements two types of encoders:
1. FixedEncoder: Pretrained on data from a single/early environment configuration
2. DriftAwareEncoder: Pretrained across time with drifting dynamics, conditions on time/drift
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class FixedEncoder(nn.Module):
    """
    Fixed-environment encoder.
    
    Pretrained on data from a single (or early) environment configuration.
    Ignores later drift during pretraining.
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # For gridworld: (x, y)
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        """
        Initialize fixed encoder.
        
        Args:
            input_dim: Input state dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super(FixedEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to embedding.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
        
        Returns:
            Embedding tensor of shape (batch_size, output_dim)
        """
        # Normalize state if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        return self.encoder(state)
    
    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state (numpy interface)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            embedding = self.forward(state_tensor)
            return embedding.cpu().numpy()


class DriftAwareEncoder(nn.Module):
    """
    Drift-aware encoder.
    
    Pretrained across time with drifting dynamics.
    Conditions on time, drift index, or inferred context.
    Goal: learn stable or adaptable features under nonstationarity.
    """
    
    def __init__(
        self,
        input_dim: int = 2,  # For gridworld: (x, y)
        hidden_dim: int = 64,
        output_dim: int = 32,
        context_dim: int = 8,  # Dimension for drift context
        use_time_embedding: bool = True,
    ):
        """
        Initialize drift-aware encoder.
        
        Args:
            input_dim: Input state dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            context_dim: Dimension for drift context embedding
            use_time_embedding: Whether to use time/drift index as input
        """
        super(DriftAwareEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.use_time_embedding = use_time_embedding
        
        # Time/drift embedding
        if use_time_embedding:
            self.time_embedding = nn.Sequential(
                nn.Linear(1, context_dim),
                nn.ReLU(),
            )
            encoder_input_dim = input_dim + context_dim
        else:
            encoder_input_dim = input_dim
        
        # Encoder with context
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Optional: context predictor (infers drift context from state sequences)
        self.context_predictor = nn.Sequential(
            nn.Linear(input_dim * 3, context_dim),  # Uses last 3 states
            nn.ReLU(),
            nn.Linear(context_dim, context_dim),
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        time: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode state to embedding with drift awareness.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            time: Time/drift index tensor of shape (batch_size, 1) or (batch_size,)
            context: Optional context tensor of shape (batch_size, context_dim)
        
        Returns:
            Embedding tensor of shape (batch_size, output_dim)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if self.use_time_embedding:
            if time is None:
                # Default to zero if time not provided
                time = torch.zeros(state.shape[0], 1)
            elif isinstance(time, (int, float)):
                time = torch.FloatTensor([[time]])
            elif time.dim() == 1:
                time = time.unsqueeze(1)
            
            time_emb = self.time_embedding(time)
            
            if context is not None:
                # Combine time and inferred context
                if context.dim() == 1:
                    context = context.unsqueeze(0)
                combined_context = time_emb + context
            else:
                combined_context = time_emb
            
            # Concatenate state and context
            encoder_input = torch.cat([state, combined_context], dim=1)
        else:
            encoder_input = state
        
        return self.encoder(encoder_input)
    
    def encode(
        self, 
        state: np.ndarray, 
        time: Optional[float] = None,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode state (numpy interface)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            time_tensor = torch.FloatTensor([[time]]) if time is not None else None
            context_tensor = torch.FloatTensor(context) if context is not None else None
            embedding = self.forward(state_tensor, time_tensor, context_tensor)
            return embedding.cpu().numpy()
    
    def predict_context(self, state_history: List[np.ndarray]) -> np.ndarray:
        """
        Predict drift context from state history.
        
        Args:
            state_history: List of recent states
        
        Returns:
            Predicted context vector
        """
        if len(state_history) < 3:
            # Pad with zeros if not enough history
            state_history = [np.zeros(self.input_dim)] * (3 - len(state_history)) + state_history
        
        # Use last 3 states
        recent_states = np.concatenate(state_history[-3:])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(recent_states).unsqueeze(0)
            context = self.context_predictor(state_tensor)
            return context.cpu().numpy()


class RepresentationTrainer:
    """
    Trainer for representation learning from reward-free exploration data.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
    ):
        """
        Initialize representation trainer.
        
        Args:
            encoder: Encoder model to train
            learning_rate: Learning rate
            batch_size: Batch size for training
        """
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_forward_dynamics(
        self,
        replay_buffer: List[Tuple],
        num_epochs: int = 10,
        device: str = "cpu",
        seed: int = 0,
    ):
        """
        Train encoder using forward dynamics prediction.
        
        The encoder learns to predict next state from current state and action.
        This encourages the representation to capture transition-relevant information.
        
        Args:
            replay_buffer: List of (state, action, next_state, reward, done) tuples
            num_epochs: Number of training epochs
            device: Device to train on
        """
        if len(replay_buffer) < self.batch_size:
            return []
        
        # Build forward dynamics predictor
        action_dim = 4  # For gridworld
        state_dim = 2
        
        dynamics_predictor = nn.Sequential(
            nn.Linear(self.encoder.output_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        ).to(device)
        
        dynamics_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(dynamics_predictor.parameters()),
            lr=self.learning_rate
        )
        
        self.encoder.to(device)
        
        rng = np.random.RandomState(seed)
        losses = []
        for epoch in range(num_epochs):
            # Shuffle data
            indices = rng.permutation(len(replay_buffer))
            
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(replay_buffer), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch = [replay_buffer[idx] for idx in batch_indices]
                
                # Canonical records are (s, a, s', done, time, next_time).
                # Five-field legacy buffers remain readable.
                states = torch.tensor(
                    np.asarray([record[0] for record in batch]), dtype=torch.float32,
                    device=device,
                )
                actions = torch.tensor(
                    [record[1] for record in batch], dtype=torch.long, device=device
                )
                next_states = torch.tensor(
                    np.asarray([record[2] for record in batch]), dtype=torch.float32,
                    device=device,
                )
                
                # One-hot encode actions
                action_onehot = torch.zeros(len(actions), action_dim).to(device)
                action_onehot.scatter_(1, actions.unsqueeze(1), 1)
                
                # Encode states
                if isinstance(self.encoder, DriftAwareEncoder):
                    if any(len(record) < 6 for record in batch):
                        raise ValueError(
                            "DriftAwareEncoder training requires temporal metadata"
                        )
                    times = torch.tensor(
                        [[record[4]] for record in batch],
                        dtype=torch.float32,
                        device=device,
                    )
                    state_embeddings = self.encoder(states, time=times)
                else:
                    state_embeddings = self.encoder(states)
                
                # Predict next state
                predictor_input = torch.cat([state_embeddings, action_onehot], dim=1)
                predicted_next = dynamics_predictor(predictor_input)
                
                # Compute loss
                loss = self.criterion(predicted_next, next_states)
                
                # Backward pass
                dynamics_optimizer.zero_grad()
                loss.backward()
                dynamics_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                losses.append(avg_loss)
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.encoder.to("cpu")
        return losses
    
    def train_contrastive(
        self,
        replay_buffer: List[Tuple],
        num_epochs: int = 10,
        device: str = "cpu",
        temperature: float = 0.1,
    ):
        """
        Train encoder using contrastive learning.
        
        Positive pairs: consecutive states in trajectory
        Negative pairs: random states from different trajectories
        
        Args:
            replay_buffer: List of (state, action, next_state, reward, done) tuples
            num_epochs: Number of training epochs
            device: Device to train on
            temperature: Temperature for contrastive loss
        """
        if len(replay_buffer) < self.batch_size:
            return
        
        self.encoder.to(device)
        
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(replay_buffer) - 1)
            
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                # Positive pairs: (state, next_state) from same trajectory
                states = torch.FloatTensor([
                    replay_buffer[idx][0] for idx in batch_indices
                ]).to(device)
                next_states = torch.FloatTensor([
                    replay_buffer[idx][2] for idx in batch_indices
                ]).to(device)
                
                # Encode
                state_emb = self.encoder(states)
                next_state_emb = self.encoder(next_states)
                
                # Normalize embeddings
                state_emb = torch.nn.functional.normalize(state_emb, dim=1)
                next_state_emb = torch.nn.functional.normalize(next_state_emb, dim=1)
                
                # Positive similarity
                positive_sim = (state_emb * next_state_emb).sum(dim=1) / temperature
                
                # Negative samples: random states
                neg_indices = np.random.choice(
                    len(replay_buffer), 
                    size=len(batch_indices),
                    replace=True
                )
                neg_states = torch.FloatTensor([
                    replay_buffer[idx][0] for idx in neg_indices
                ]).to(device)
                neg_emb = torch.nn.functional.normalize(self.encoder(neg_states), dim=1)
                
                # Negative similarity
                negative_sim = (state_emb * neg_emb).sum(dim=1) / temperature
                
                # Contrastive loss: maximize positive, minimize negative
                loss = -torch.log(torch.sigmoid(positive_sim - negative_sim)).mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Contrastive Loss: {avg_loss:.4f}")
        
        self.encoder.to("cpu")

