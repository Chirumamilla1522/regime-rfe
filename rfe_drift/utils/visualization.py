"""
Visualization utilities for the environment and results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


def visualize_gridworld(
    grid_size: int,
    agent_pos: Tuple[int, int],
    goals: List[Tuple[int, int]],
    walls: List[Tuple[int, int]],
    title: str = "GridWorld",
    save_path: Optional[str] = None,
):
    """
    Visualize the GridWorld environment.
    
    Args:
        grid_size: Size of the grid
        agent_pos: Agent position (x, y)
        goals: List of goal positions
        walls: List of wall positions
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Create grid
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Draw walls
    for wall in walls:
        rect = patches.Rectangle(
            (wall[0], wall[1]), 1, 1,
            linewidth=2, edgecolor='black', facecolor='black', alpha=0.7
        )
        ax.add_patch(rect)
    
    # Draw goals
    for goal in goals:
        circle = patches.Circle(
            (goal[0] + 0.5, goal[1] + 0.5), 0.3,
            color='green', alpha=0.7
        )
        ax.add_patch(circle)
    
    # Draw agent
    agent_circle = patches.Circle(
        (agent_pos[0] + 0.5, agent_pos[1] + 0.5), 0.25,
        color='red', alpha=0.8
    )
    ax.add_patch(agent_circle)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.invert_yaxis()  # Invert y-axis to match typical grid coordinates
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Agent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='Goal'),
        patches.Patch(facecolor='black', alpha=0.7, label='Wall'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    
    plt.close()


def plot_trajectory(
    trajectory: List[Tuple[int, int]],
    grid_size: int,
    goals: List[Tuple[int, int]],
    walls: List[Tuple[int, int]],
    title: str = "Agent Trajectory",
    save_path: Optional[str] = None,
):
    """
    Plot an agent's trajectory through the gridworld.
    
    Args:
        trajectory: List of (x, y) positions
        grid_size: Size of the grid
        goals: List of goal positions
        walls: List of wall positions
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Create grid
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Draw walls
    for wall in walls:
        rect = patches.Rectangle(
            (wall[0], wall[1]), 1, 1,
            linewidth=2, edgecolor='black', facecolor='black', alpha=0.7
        )
        ax.add_patch(rect)
    
    # Draw goals
    for goal in goals:
        circle = patches.Circle(
            (goal[0] + 0.5, goal[1] + 0.5), 0.3,
            color='green', alpha=0.7
        )
        ax.add_patch(circle)
    
    # Draw trajectory
    if len(trajectory) > 1:
        traj_array = np.array(trajectory)
        ax.plot(
            traj_array[:, 0] + 0.5,
            traj_array[:, 1] + 0.5,
            'b-', alpha=0.5, linewidth=2, label='Trajectory'
        )
        ax.scatter(
            traj_array[0, 0] + 0.5,
            traj_array[0, 1] + 0.5,
            color='blue', s=100, marker='s', label='Start', zorder=5
        )
        ax.scatter(
            traj_array[-1, 0] + 0.5,
            traj_array[-1, 1] + 0.5,
            color='red', s=100, marker='o', label='End', zorder=5
        )
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    
    plt.close()

