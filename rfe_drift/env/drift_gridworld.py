"""
Drift-Enabled GridWorld Environment

Implements a GridWorld MDP with tunable distributional drift:
- Shifting goal locations
- Changing transition noise
- Altering blocked cells/wall layouts
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, List
from enum import Enum


class DriftType(Enum):
    """Types of drift that can occur in the environment"""
    GOAL_SHIFT = "goal_shift"
    TRANSITION_NOISE = "transition_noise"
    WALL_CHANGE = "wall_change"
    COMBINED = "combined"


class DriftSchedule(Enum):
    """When drift occurs"""
    SUDDEN = "sudden"  # Single jump at a specific time
    GRADUAL = "gradual"  # Smooth transition over time
    PERIODIC = "periodic"  # Periodic changes


class DriftGridWorld(gym.Env):
    """
    A GridWorld environment with tunable distributional drift.
    
    The environment supports:
    - Configurable grid size
    - Multiple drift types (goal shift, transition noise, wall changes)
    - Different drift schedules (sudden, gradual, periodic)
    - Tunable drift strength
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Action space: 0=up, 1=right, 2=down, 3=left
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    
    def __init__(
        self,
        grid_size: int = 10,
        drift_type: DriftType = DriftType.GOAL_SHIFT,
        drift_strength: float = 0.5,
        drift_schedule: DriftSchedule = DriftSchedule.SUDDEN,
        drift_time: int = 1000,  # When drift occurs (for sudden) or period (for periodic)
        transition_noise: float = 0.0,  # Probability of random action
        num_goals: int = 1,
        num_walls: int = 5,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the DriftGridWorld environment.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            drift_type: Type of drift to apply
            drift_strength: Strength of drift (0.0 to 1.0)
            drift_schedule: When drift occurs
            drift_time: Time step when drift occurs (sudden) or period (periodic)
            transition_noise: Base transition noise probability
            num_goals: Number of goal locations
            num_walls: Number of wall cells
            seed: Random seed
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.drift_type = drift_type
        self.drift_strength = drift_strength
        self.drift_schedule = drift_schedule
        self.drift_time = drift_time
        self.base_transition_noise = transition_noise
        self.num_goals = num_goals
        self.num_walls = num_walls
        self.render_mode = render_mode
        
        # State space: (x, y) position
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Environment state
        self.agent_pos = None
        self.goals = []  # List of goal positions
        self.walls = set()  # Set of wall positions
        # The drift clock is global across episodes. ``step_count`` is kept as
        # a compatibility alias; episode_step_count is reset by reset().
        self.step_count = 0
        self.episode_step_count = 0
        self.episode_count = 0
        
        # Drift state
        self.initial_goals = []
        self.initial_walls = set()
        self.drifted_goals = []
        self.drifted_walls = set()
        self.current_transition_noise = transition_noise
        
        # Initialize random state
        self.np_random = np.random.RandomState(seed)
        self._seed = seed
        
        # Initialize environment
        self._initialize_environment()
        
    def _initialize_environment(self):
        """Initialize the environment with initial configuration"""
        # Start position (top-left corner)
        self.start_pos = (0, 0)
        
        # Generate initial goals (avoid start position)
        self.initial_goals = []
        for _ in range(self.num_goals):
            while True:
                goal = (
                    self.np_random.randint(1, self.grid_size),
                    self.np_random.randint(1, self.grid_size)
                )
                if goal != self.start_pos and goal not in self.initial_goals:
                    self.initial_goals.append(goal)
                    break
        
        # Generate initial walls
        self.initial_walls = set()
        for _ in range(self.num_walls):
            while True:
                wall = (
                    self.np_random.randint(0, self.grid_size),
                    self.np_random.randint(0, self.grid_size)
                )
                if (wall != self.start_pos and 
                    wall not in self.initial_goals and 
                    wall not in self.initial_walls):
                    self.initial_walls.add(wall)
                    break
        
        # Generate drifted configuration
        self._generate_drifted_config()
        
        # Set initial state
        self.goals = self.initial_goals.copy()
        self.walls = self.initial_walls.copy()
        self.current_transition_noise = self.base_transition_noise
        
    def _generate_drifted_config(self):
        """Generate the drifted configuration based on drift type"""
        if self.drift_type == DriftType.GOAL_SHIFT or self.drift_type == DriftType.COMBINED:
            # Shift goals by drift_strength * grid_size
            shift_amount = int(self.drift_strength * self.grid_size)
            self.drifted_goals = []
            for goal in self.initial_goals:
                # Shift goal position
                new_goal = (
                    min(self.grid_size - 1, max(0, goal[0] + shift_amount)),
                    min(self.grid_size - 1, max(0, goal[1] + shift_amount))
                )
                # Ensure it's not overlapping with start or other goals
                if new_goal != self.start_pos and new_goal not in self.drifted_goals:
                    self.drifted_goals.append(new_goal)
                else:
                    # Fallback: random position
                    while True:
                        new_goal = (
                            self.np_random.randint(1, self.grid_size),
                            self.np_random.randint(1, self.grid_size)
                        )
                        if (new_goal != self.start_pos and 
                            new_goal not in self.drifted_goals):
                            self.drifted_goals.append(new_goal)
                            break
        else:
            self.drifted_goals = self.initial_goals.copy()
        
        if self.drift_type == DriftType.WALL_CHANGE or self.drift_type == DriftType.COMBINED:
            # Change some walls
            num_changes = int(self.drift_strength * len(self.initial_walls))
            self.drifted_walls = self.initial_walls.copy()
            
            # Remove some walls
            walls_to_remove = list(self.drifted_walls)[:num_changes]
            for wall in walls_to_remove:
                self.drifted_walls.remove(wall)
            
            # Add new walls
            for _ in range(num_changes):
                while True:
                    new_wall = (
                        self.np_random.randint(0, self.grid_size),
                        self.np_random.randint(0, self.grid_size)
                    )
                    if (new_wall != self.start_pos and 
                        new_wall not in self.initial_goals and 
                        new_wall not in self.drifted_goals and
                        new_wall not in self.drifted_walls):
                        self.drifted_walls.add(new_wall)
                        break
        else:
            self.drifted_walls = self.initial_walls.copy()
        
        if self.drift_type == DriftType.TRANSITION_NOISE or self.drift_type == DriftType.COMBINED:
            # Increase transition noise
            self.drifted_transition_noise = min(1.0, 
                self.base_transition_noise + self.drift_strength * 0.3)
        else:
            self.drifted_transition_noise = self.base_transition_noise
    
    def _apply_drift(self):
        """Apply drift based on current step and drift schedule"""
        if self.drift_schedule == DriftSchedule.SUDDEN:
            if self.step_count >= self.drift_time:
                self.goals = self.drifted_goals.copy()
                self.walls = self.drifted_walls.copy()
                self.current_transition_noise = self.drifted_transition_noise
            else:
                self.goals = self.initial_goals.copy()
                self.walls = self.initial_walls.copy()
                self.current_transition_noise = self.base_transition_noise
        
        elif self.drift_schedule == DriftSchedule.GRADUAL:
            # Deterministic interpolation makes the schedule reproducible and
            # prevents reset() from retaining a stale prior configuration.
            if self.step_count < self.drift_time:
                self.goals = self.initial_goals.copy()
                self.walls = self.initial_walls.copy()
                self.current_transition_noise = self.base_transition_noise
            elif self.step_count < self.drift_time * 2:
                alpha = (self.step_count - self.drift_time) / self.drift_time
                self.goals = [
                    (
                        int(round((1 - alpha) * initial[0] + alpha * drifted[0])),
                        int(round((1 - alpha) * initial[1] + alpha * drifted[1])),
                    )
                    for initial, drifted in zip(
                        self.initial_goals, self.drifted_goals
                    )
                ]
                removed = sorted(self.initial_walls - self.drifted_walls)
                added = sorted(self.drifted_walls - self.initial_walls)
                remove_count = int(round(alpha * len(removed)))
                add_count = int(round(alpha * len(added)))
                self.walls = self.initial_walls - set(removed[:remove_count])
                self.walls |= set(added[:add_count])
                self.walls -= set(self.goals)
                self.current_transition_noise = (
                    self.base_transition_noise * (1 - alpha) +
                    self.drifted_transition_noise * alpha
                )
            elif self.step_count >= self.drift_time * 2:
                self.goals = self.drifted_goals.copy()
                self.walls = self.drifted_walls.copy()
                self.current_transition_noise = self.drifted_transition_noise
        
        elif self.drift_schedule == DriftSchedule.PERIODIC:
            # Periodic drift every drift_time steps
            period = (self.step_count // self.drift_time) % 2
            if period == 0:
                self.goals = self.initial_goals.copy()
                self.walls = self.initial_walls.copy()
                self.current_transition_noise = self.base_transition_noise
            else:
                self.goals = self.drifted_goals.copy()
                self.walls = self.drifted_walls.copy()
                self.current_transition_noise = self.drifted_transition_noise
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            self._seed = seed
        
        self.agent_pos = self.start_pos
        self.episode_step_count = 0
        self.episode_count += 1
        self._apply_drift()
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {
            "goals": self.goals,
            "walls": list(self.walls),
            "drift_applied": self.is_drifted,
            "global_step": self.step_count,
            "episode_step": self.episode_step_count,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.step_count += 1
        self.episode_step_count += 1
        
        # Apply drift
        self._apply_drift()
        
        # Apply transition noise
        if self.np_random.random() < self.current_transition_noise:
            action = self.np_random.randint(0, 4)
        
        # Move agent
        new_pos = self._move(self.agent_pos, action)
        
        # Check if new position is valid (not a wall)
        if new_pos not in self.walls:
            self.agent_pos = new_pos
        
        # Check if agent reached a goal
        terminated = self.agent_pos in self.goals
        truncated = False
        
        # Reward (for downstream tasks, can be overridden)
        reward = 1.0 if terminated else 0.0
        
        observation = np.array(self.agent_pos, dtype=np.int32)
        info = {
            "goals": self.goals,
            "walls": list(self.walls),
            "drift_applied": self.is_drifted,
            "global_step": self.step_count,
            "episode_step": self.episode_step_count,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Compute new position after taking action"""
        x, y = pos
        
        if action == self.ACTION_UP:
            y = max(0, y - 1)
        elif action == self.ACTION_RIGHT:
            x = min(self.grid_size - 1, x + 1)
        elif action == self.ACTION_DOWN:
            y = min(self.grid_size - 1, y + 1)
        elif action == self.ACTION_LEFT:
            x = max(0, x - 1)
        
        return (x, y)
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid[:] = "."
            
            # Mark walls
            for wall in self.walls:
                grid[wall[1], wall[0]] = "#"
            
            # Mark goals
            for goal in self.goals:
                grid[goal[1], goal[0]] = "G"
            
            # Mark agent
            grid[self.agent_pos[1], self.agent_pos[0]] = "A"
            
            print("\n" + "\n".join(" ".join(row) for row in grid) + "\n")
        
        elif self.render_mode == "rgb_array":
            # Return RGB array representation
            grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            grid[:] = [255, 255, 255]  # White background
            
            # Walls: black
            for wall in self.walls:
                grid[wall[1], wall[0]] = [0, 0, 0]
            
            # Goals: green
            for goal in self.goals:
                grid[goal[1], goal[0]] = [0, 255, 0]
            
            # Agent: red
            grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]
            
            return grid
    
    @property
    def is_drifted(self) -> bool:
        """Whether the active configuration differs from the initial one."""
        return (
            self.goals != self.initial_goals
            or self.walls != self.initial_walls
            or not np.isclose(
                self.current_transition_noise, self.base_transition_noise
            )
        )

    def set_global_step(self, step: int) -> None:
        """Set the public drift clock, primarily for controlled evaluation."""
        if step < 0:
            raise ValueError("global step must be non-negative")
        self.step_count = int(step)
        self._apply_drift()

    def reachable_states(self) -> set:
        """Return free cells reachable from the start under active walls."""
        if self.start_pos in self.walls:
            return set()
        reached = {self.start_pos}
        frontier = [self.start_pos]
        while frontier:
            position = frontier.pop()
            for action in range(self.action_space.n):
                candidate = self._move(position, action)
                if candidate not in self.walls and candidate not in reached:
                    reached.add(candidate)
                    frontier.append(candidate)
        return reached

    def get_state_coverage(self, visited_states: set) -> float:
        """Fraction of currently reachable cells that were visited."""
        reachable = self.reachable_states()
        visited = {
            tuple(state) if isinstance(state, np.ndarray) else tuple(state)
            for state in visited_states
        }
        return len(visited & reachable) / len(reachable) if reachable else 0.0

