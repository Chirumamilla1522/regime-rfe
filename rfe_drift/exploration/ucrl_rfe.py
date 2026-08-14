"""Tabular count-bonus reward-free data collection.

This is deliberately not called UCRL-RFE: it does not implement the episodic
optimistic planning and confidence sets required by that published algorithm.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import math


class CountBasedExplorer:
    """
    Lightweight count-based exploration without rewards or oracle drift access.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        confidence: float = 0.95,
        exploration_bonus: float = 1.0,
        seed: int = 0,
    ):
        """
        Initialize the count-based collector.
        
        Args:
            state_dim: Dimensionality of state space (for gridworld, this is grid_size^2)
            action_dim: Number of actions
            confidence: Confidence level for confidence intervals
            exploration_bonus: Bonus for exploring less visited states
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.confidence = confidence
        self.exploration_bonus = exploration_bonus
        self.rng = np.random.RandomState(seed)
        
        # Counters: N(s, a) = number of times action a was taken in state s
        self.N = defaultdict(lambda: defaultdict(int))
        
        # Transition counts: N(s, a, s') = number of times (s, a) led to s'
        self.N_trans = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Visit counts: N(s) = number of times state s was visited
        self.N_state = defaultdict(int)
        
        # State-action visit counts
        self.N_sa = defaultdict(lambda: defaultdict(int))
        
        # Total step count
        self.t = 0
        
        # Replay buffer: list of (state, action, next_state, reward, done)
        self.replay_buffer = []
        
        # State coverage tracking
        self.visited_states = set()
        
    def state_to_index(self, state: np.ndarray) -> int:
        """Convert state array to index for tabular representation"""
        if isinstance(state, np.ndarray) and len(state) == 2:
            # For 2D gridworld states (x, y)
            # Assuming grid_size can be inferred from state space
            # For now, use a simple hash
            return hash(tuple(state)) % (self.state_dim ** 2)
        return int(state) if isinstance(state, (int, np.integer)) else hash(tuple(state))
    
    def index_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert index back to state (for gridworld)"""
        # This is a simplified version - in practice, you'd store the mapping
        # For now, we'll work with state tuples directly
        pass
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        time: Optional[float] = None,
        next_time: Optional[float] = None,
    ):
        """
        Update the model with a new transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward (not used in RFE, but stored for later)
            done: Whether episode terminated
        """
        self.t += 1
        
        # Convert states to indices for tabular representation
        s = tuple(state) if isinstance(state, np.ndarray) else state
        s_next = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
        
        # Update counters
        self.N[s][action] += 1
        self.N_trans[s][action][s_next] += 1
        self.N_state[s] += 1
        self.N_sa[s][action] += 1
        
        # Track visited states
        self.visited_states.add(s)
        
        # Store in replay buffer
        if time is None:
            # Legacy shape for callers that do not provide temporal metadata.
            self.replay_buffer.append((s, action, s_next, reward, done))
        else:
            self.replay_buffer.append(
                (s, action, s_next, done, float(time), float(next_time or time))
            )
    
    def get_confidence_radius(self, n: int) -> float:
        """
        Compute confidence radius for a state-action pair visited n times.
        
        This radius-inspired heuristic is not used as a statistical guarantee.
        """
        if n == 0:
            return float('inf')
        return math.sqrt((2 * math.log(2 * self.state_dim * self.action_dim * self.t / (1 - self.confidence))) / n)
    
    def get_transition_estimate(self, state, action, next_state) -> float:
        """
        Get estimated transition probability P(s'|s,a).
        """
        n_sa = self.N_sa[state][action]
        if n_sa == 0:
            return 0.0
        return self.N_trans[state][action][next_state] / n_sa
    
    def get_exploration_bonus(self, state, action) -> float:
        """
        Compute exploration bonus for state-action pair.
        
        States and actions that have been visited less get higher bonuses.
        """
        n_sa = self.N_sa[state][action]
        if n_sa == 0:
            return self.exploration_bonus
        
        # Bonus decreases with visit count
        bonus = self.exploration_bonus / (1 + n_sa)
        
        # Add confidence radius
        confidence_radius = self.get_confidence_radius(n_sa)
        
        return bonus + confidence_radius
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select an action using local count bonuses.
        
        Args:
            state: Current state
            epsilon: Epsilon for epsilon-greedy (optional, for baseline comparison)
        
        Returns:
            Selected action
        """
        s = tuple(state) if isinstance(state, np.ndarray) else state
        
        # Compute Q-values with exploration bonuses
        q_values = []
        for a in range(self.action_dim):
            # Base value: negative of visit count (encourage exploration)
            base_value = -self.N_sa[s][a]
            
            # Exploration bonus
            exploration_bonus = self.get_exploration_bonus(s, a)
            
            # Optimistic value: base + bonus
            q_value = base_value + exploration_bonus
            q_values.append(q_value)
        
        # Select action with highest Q-value
        if len(q_values) == 0:
            return self.rng.randint(0, self.action_dim)
        
        # Break ties randomly
        max_value = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_value]
        return int(self.rng.choice(best_actions))
    
    def get_state_coverage(self, reachable_states: Optional[Set] = None) -> float:
        """Coverage over reachable states when that set is available."""
        if reachable_states is None:
            denominator = self.state_dim
            numerator = len(self.visited_states)
        else:
            reachable = {tuple(state) for state in reachable_states}
            denominator = len(reachable)
            numerator = len(self.visited_states & reachable)
        return numerator / denominator if denominator else 0.0
    
    def get_replay_buffer(self) -> List[Tuple]:
        """Get the replay buffer of collected trajectories"""
        return self.replay_buffer
    
    def reset(self):
        """Reset the algorithm (keep learned model, reset episode-specific state)"""
        # Keep the learned model, just reset episode tracking if needed
        pass


class UCRLRFE(CountBasedExplorer):
    """Deprecated compatibility alias for :class:`CountBasedExplorer`."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "UCRLRFE was a misnomer; use CountBasedExplorer. "
            "This code is a count-bonus heuristic, not faithful UCRL-RFE.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

