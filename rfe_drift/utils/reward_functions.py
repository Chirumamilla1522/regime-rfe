"""
Reward functions for downstream tasks
"""

import numpy as np
from typing import List, Tuple


class GoalRewardFunction:
    """Reward function that gives reward for reaching goals"""
    
    def __init__(self, goals: List[Tuple[int, int]], goal_reward: float = 1.0):
        """
        Initialize goal reward function.
        
        Args:
            goals: List of goal positions
            goal_reward: Reward for reaching a goal
        """
        self.goals = goals
        self.goal_reward = goal_reward
    
    def __call__(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """Compute reward"""
        next_pos = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
        if next_pos in self.goals:
            return self.goal_reward
        return 0.0


class DistanceRewardFunction:
    """Reward function based on distance to goals"""
    
    def __init__(
        self, 
        goals: List[Tuple[int, int]], 
        goal_reward: float = 1.0,
        distance_penalty: float = 0.01,
    ):
        """
        Initialize distance-based reward function.
        
        Args:
            goals: List of goal positions
            goal_reward: Reward for reaching a goal
            distance_penalty: Penalty per unit distance from goal
        """
        self.goals = goals
        self.goal_reward = goal_reward
        self.distance_penalty = distance_penalty
    
    def _distance_to_nearest_goal(self, pos: Tuple[int, int]) -> float:
        """Compute distance to nearest goal"""
        if len(self.goals) == 0:
            return 0.0
        distances = [np.sqrt((pos[0] - g[0])**2 + (pos[1] - g[1])**2) for g in self.goals]
        return min(distances)
    
    def __call__(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """Compute reward"""
        next_pos = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state
        
        # Check if reached goal
        if next_pos in self.goals:
            return self.goal_reward
        
        # Distance penalty
        distance = self._distance_to_nearest_goal(next_pos)
        return -self.distance_penalty * distance

