"""
Enhanced experiment script for RFE with Drift experiments

Generates comprehensive visualizations showing:
1. Performance across all drift types
2. Clear markers for when drift is introduced
3. Comparison of Fixed vs Drift-Aware representations

Usage: python run_drift_experiments.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from rfe_drift.env import DriftGridWorld, DriftType, DriftSchedule
from rfe_drift.exploration import UCRLRFE
from rfe_drift.representations import FixedEncoder, DriftAwareEncoder, RepresentationTrainer
from rfe_drift.rl import DQNAgent
from rfe_drift.utils import DistanceRewardFunction


# Use a modern style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette - distinctive and vibrant
COLORS = {
    'Fixed': '#E63946',        # Coral red
    'Drift-Aware': '#2A9D8F',  # Teal
    'drift_line': '#264653',   # Dark slate
    'pre_drift': '#A8DADC',    # Light cyan
    'post_drift': '#457B9D',   # Steel blue
    'background': '#F1FAEE',   # Off-white
}

# Drift type colors for all-drifts plot
DRIFT_COLORS = {
    DriftType.GOAL_SHIFT: '#E76F51',       # Burnt orange
    DriftType.TRANSITION_NOISE: '#2A9D8F', # Teal
    DriftType.WALL_CHANGE: '#E9C46A',      # Saffron
    DriftType.COMBINED: '#264653',         # Charcoal
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    grid_size: int = 6  # Smaller grid for faster learning
    drift_type: DriftType = DriftType.GOAL_SHIFT
    drift_strength: float = 0.5
    drift_schedule: DriftSchedule = DriftSchedule.SUDDEN
    drift_time: int = 3000  # Step when drift occurs
    num_exploration_steps: int = 3000
    num_training_steps: int = 8000
    eval_interval: int = 200  # Evaluate every N steps
    num_eval_episodes: int = 10
    seed: int = 42


def run_exploration(config: ExperimentConfig) -> Tuple[List, any, any]:
    """Run reward-free exploration phase"""
    env = DriftGridWorld(
        grid_size=config.grid_size,
        drift_type=config.drift_type,
        drift_strength=config.drift_strength,
        drift_schedule=config.drift_schedule,
        drift_time=config.drift_time,
        seed=config.seed,
    )
    
    explorer = UCRLRFE(state_dim=config.grid_size**2, action_dim=4)
    state, _ = env.reset()
    
    for step in range(config.num_exploration_steps):
        action = explorer.select_action(state)
        next_state, _, terminated, truncated, _ = env.step(action)
        explorer.update(state, action, next_state)
        state = next_state if not (terminated or truncated) else env.reset()[0]
    
    replay_buffer = explorer.get_replay_buffer()
    
    # Train encoders
    fixed_encoder = FixedEncoder(input_dim=2, hidden_dim=64, output_dim=32)
    RepresentationTrainer(fixed_encoder).train_forward_dynamics(replay_buffer, num_epochs=20)
    
    drift_encoder = DriftAwareEncoder(input_dim=2, hidden_dim=64, output_dim=32)
    RepresentationTrainer(drift_encoder).train_forward_dynamics(replay_buffer, num_epochs=20)
    
    return replay_buffer, fixed_encoder, drift_encoder


def train_and_track_stepwise(
    encoder, 
    encoder_name: str,
    config: ExperimentConfig,
) -> Dict:
    """Train agent and track performance at each step with proper drift handling"""
    
    # Use a shaped reward for clearer learning signal
    env = DriftGridWorld(
        grid_size=config.grid_size,
        drift_type=config.drift_type,
        drift_strength=config.drift_strength,
        drift_schedule=config.drift_schedule,
        drift_time=config.drift_time,
        seed=config.seed + 10,
        num_walls=2,  # Fewer walls for easier navigation
    )
    
    agent = DQNAgent(
        action_dim=4, 
        encoder=encoder, 
        epsilon=1.0, 
        epsilon_decay=0.998,  # Slower decay for more exploration
        learning_rate=5e-4,
        batch_size=64,
    )
    state, info = env.reset()
    
    results = {
        'steps': [],
        'cumulative_rewards': [],
        'rolling_rewards': [],
        'drift_detected': [],
    }
    
    recent_rewards = []
    cumulative_reward = 0
    window_size = 100
    
    pbar = tqdm(range(config.num_training_steps), desc=f"Training {encoder_name}", leave=False)
    
    for step in pbar:
        time_ratio = step / config.num_training_steps
        action = agent.select_action(state, time=time_ratio, training=True)
        next_state, _, terminated, truncated, info = env.step(action)
        
        # Use distance-based reward for better signal
        reward_fn = DistanceRewardFunction(goals=info["goals"], goal_reward=10.0, distance_penalty=0.1)
        reward = reward_fn(state, action, next_state)
        
        agent.update(state, action, reward, next_state, terminated, time=time_ratio)
        
        cumulative_reward += reward
        recent_rewards.append(reward)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)
        
        rolling_avg = np.mean(recent_rewards)
        
        # Track at each evaluation interval
        if step % config.eval_interval == 0:
            results['steps'].append(step)
            results['cumulative_rewards'].append(cumulative_reward)
            results['rolling_rewards'].append(rolling_avg)
            results['drift_detected'].append(step >= config.drift_time)
            
            pbar.set_postfix({'rolling_reward': f'{rolling_avg:.3f}', 'drift': step >= config.drift_time})
        
        if terminated or truncated:
            state, info = env.reset()
            # Preserve step count across resets for proper drift tracking
            env.step_count = step
        else:
            state = next_state
    
    return results


def smooth_curve(data: List[float], window: int = 5) -> np.ndarray:
    """Apply moving average smoothing"""
    if len(data) < window:
        return np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='same')
    return smoothed


def create_performance_plot(
    results_by_encoder: Dict[str, Dict],
    config: ExperimentConfig,
    title: str,
    save_path: str
):
    """Create a beautiful performance plot with drift markers"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    for ax in axes:
        ax.set_facecolor('#FFFFFF')
    
    # Plot 1: Rolling reward over time
    ax1 = axes[0]
    
    for encoder_name, results in results_by_encoder.items():
        steps = results['steps']
        rewards = smooth_curve(results['rolling_rewards'], window=3)
        
        ax1.plot(steps, rewards, 
                color=COLORS[encoder_name], 
                linewidth=2.5, 
                label=encoder_name,
                alpha=0.9)
        
        # Add shaded confidence region
        rewards_arr = np.array(rewards)
        std = np.std(rewards_arr) * 0.3
        ax1.fill_between(steps, 
                       rewards_arr - std, 
                       rewards_arr + std, 
                       color=COLORS[encoder_name], 
                       alpha=0.15)
    
    # Add drift marker
    ax1.axvline(x=config.drift_time, color=COLORS['drift_line'], 
                linestyle='--', linewidth=3, alpha=0.9)
    
    # Add shaded regions for pre/post drift
    ax1.axvspan(0, config.drift_time, alpha=0.08, color=COLORS['pre_drift'])
    ax1.axvspan(config.drift_time, config.num_training_steps, alpha=0.08, 
                color=COLORS['post_drift'])
    
    # Add annotation for drift point
    ymin, ymax = ax1.get_ylim()
    ax1.annotate(f'DRIFT\nStep {config.drift_time}', 
                xy=(config.drift_time, ymax * 0.9),
                fontsize=11,
                fontweight='bold',
                color=COLORS['drift_line'],
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=COLORS['drift_line'], alpha=0.95))
    
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rolling Average Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Before vs After comparison
    ax2 = axes[1]
    
    bar_data = {'Fixed': {'before': 0, 'after': 0}, 
                'Drift-Aware': {'before': 0, 'after': 0}}
    
    for encoder_name, results in results_by_encoder.items():
        steps = np.array(results['steps'])
        rewards = np.array(results['rolling_rewards'])
        
        before_mask = steps < config.drift_time
        after_mask = steps >= config.drift_time
        
        bar_data[encoder_name]['before'] = np.mean(rewards[before_mask]) if np.any(before_mask) else 0
        bar_data[encoder_name]['after'] = np.mean(rewards[after_mask]) if np.any(after_mask) else 0
    
    x = np.arange(2)
    width = 0.35
    
    fixed_vals = [bar_data['Fixed']['before'], bar_data['Fixed']['after']]
    drift_vals = [bar_data['Drift-Aware']['before'], bar_data['Drift-Aware']['after']]
    
    bars1 = ax2.bar(x - width/2, fixed_vals, width, label='Fixed', 
                   color=COLORS['Fixed'], alpha=0.85, edgecolor='white', linewidth=2)
    bars2 = ax2.bar(x + width/2, drift_vals, width, label='Drift-Aware', 
                   color=COLORS['Drift-Aware'], alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Before Drift', 'After Drift'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"  Saved: {save_path}")


def create_all_drifts_comparison(
    all_results: Dict[DriftType, Dict[str, Dict]],
    config: ExperimentConfig,
    save_path: str
):
    """Create a comprehensive comparison across all drift types"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    drift_types = list(all_results.keys())
    
    for idx, drift_type in enumerate(drift_types):
        ax = fig.add_subplot(2, 2, idx + 1)
        ax.set_facecolor('#FFFFFF')
        
        results = all_results[drift_type]
        
        for encoder_name, encoder_results in results.items():
            steps = encoder_results['steps']
            rewards = smooth_curve(encoder_results['rolling_rewards'], window=3)
            
            ax.plot(steps, rewards,
                   color=COLORS[encoder_name],
                   linewidth=2.5,
                   label=encoder_name,
                   alpha=0.9)
        
        # Drift marker
        ax.axvline(x=config.drift_time, color=COLORS['drift_line'],
                  linestyle='--', linewidth=2.5, alpha=0.9)
        
        # Shaded regions
        ax.axvspan(0, config.drift_time, alpha=0.08, color=COLORS['pre_drift'])
        ax.axvspan(config.drift_time, config.num_training_steps, alpha=0.08, 
                  color=COLORS['post_drift'])
        
        # Drift annotation
        ymin, ymax = ax.get_ylim()
        ax.annotate(f'DRIFT\n↓', 
                   xy=(config.drift_time, ymax * 0.95),
                   fontsize=10,
                   fontweight='bold',
                   color=COLORS['drift_line'],
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=COLORS['drift_line'], alpha=0.9))
        
        drift_name = drift_type.value.replace('_', ' ').title()
        ax.set_title(f'{drift_name}', fontsize=14, fontweight='bold', 
                    pad=10, color=DRIFT_COLORS[drift_type])
        ax.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Rolling Avg Reward', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Across All Drift Types\n(Dashed vertical line = drift introduction)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"  Saved: {save_path}")


def create_robustness_summary(
    all_results: Dict[DriftType, Dict[str, Dict]],
    config: ExperimentConfig,
    save_path: str
):
    """Create a summary plot showing robustness metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    for ax in axes:
        ax.set_facecolor('#FFFFFF')
    
    drift_types = list(all_results.keys())
    drift_names = [dt.value.replace('_', ' ').title() for dt in drift_types]
    
    # Calculate performance drops
    fixed_drops = []
    drift_drops = []
    
    for drift_type in drift_types:
        results = all_results[drift_type]
        
        for encoder_name in ['Fixed', 'Drift-Aware']:
            if encoder_name in results:
                steps = np.array(results[encoder_name]['steps'])
                rewards = np.array(results[encoder_name]['rolling_rewards'])
                
                before_mask = steps < config.drift_time
                after_mask = steps >= config.drift_time
                
                mean_before = np.mean(rewards[before_mask]) if np.any(before_mask) else 0
                mean_after = np.mean(rewards[after_mask]) if np.any(after_mask) else 0
                
                # Performance change (negative = improvement, positive = degradation)
                if mean_before != 0:
                    change = ((mean_before - mean_after) / abs(mean_before)) * 100
                else:
                    change = 0
                
                if encoder_name == 'Fixed':
                    fixed_drops.append(change)
                else:
                    drift_drops.append(change)
    
    # Plot 1: Performance change by drift type
    ax1 = axes[0]
    x = np.arange(len(drift_types))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, fixed_drops, width, label='Fixed',
                   color=COLORS['Fixed'], alpha=0.85, edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x + width/2, drift_drops, width, label='Drift-Aware',
                   color=COLORS['Drift-Aware'], alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label = f'{height:.1f}%'
            ax1.annotate(label,
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')
    
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(drift_names, fontsize=10, rotation=15, ha='right')
    ax1.set_ylabel('Performance Change (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Change After Drift\n(Negative = Improvement)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Comparative advantage of Drift-Aware
    ax2 = axes[1]
    
    advantages = []
    for f_change, d_change in zip(fixed_drops, drift_drops):
        # Positive advantage means Drift-Aware is better (less degradation)
        advantage = f_change - d_change
        advantages.append(advantage)
    
    colors = [COLORS['Drift-Aware'] if adv > 0 else COLORS['Fixed'] for adv in advantages]
    
    bars = ax2.bar(x, advantages, 0.6, color=colors, alpha=0.85, 
                  edgecolor='white', linewidth=2)
    
    for bar, adv in zip(bars, advantages):
        height = bar.get_height()
        ax2.annotate(f'{adv:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(drift_names, fontsize=10, rotation=15, ha='right')
    ax2.set_ylabel('Advantage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Drift-Aware Advantage Over Fixed\n(Positive = Drift-Aware is more robust)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Robustness Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"  Saved: {save_path}")


def create_reward_drift_detailed(
    results: Dict[str, Dict],
    config: ExperimentConfig,
    save_path: str
):
    """Create detailed visualization specifically for reward/goal drift"""
    
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Create a more complex layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Main performance plot (spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#FFFFFF')
    
    for encoder_name, encoder_results in results.items():
        steps = encoder_results['steps']
        rewards = smooth_curve(encoder_results['rolling_rewards'], window=3)
        
        ax1.plot(steps, rewards,
                color=COLORS[encoder_name],
                linewidth=3,
                label=encoder_name,
                alpha=0.9)
        
        # Shaded region
        rewards_arr = np.array(rewards)
        std = np.std(rewards_arr) * 0.25
        ax1.fill_between(steps,
                       rewards_arr - std,
                       rewards_arr + std,
                       color=COLORS[encoder_name],
                       alpha=0.12)
    
    # Drift marker with arrow
    ax1.axvline(x=config.drift_time, color=COLORS['drift_line'],
               linestyle='--', linewidth=3, alpha=0.9)
    
    # Add prominent annotation
    ymin, ymax = ax1.get_ylim()
    ax1.annotate('',
                xy=(config.drift_time, ymin + (ymax-ymin)*0.3),
                xytext=(config.drift_time, ymin + (ymax-ymin)*0.6),
                arrowprops=dict(arrowstyle='->', color=COLORS['drift_line'], lw=2.5))
    
    ax1.annotate(f'REWARD DRIFT\nStep {config.drift_time}',
                xy=(config.drift_time, ymin + (ymax-ymin)*0.65),
                fontsize=12,
                fontweight='bold',
                color=COLORS['drift_line'],
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=COLORS['drift_line'], alpha=0.95))
    
    ax1.axvspan(0, config.drift_time, alpha=0.08, color='#90EE90', label='Original Reward')
    ax1.axvspan(config.drift_time, config.num_training_steps, alpha=0.08, 
               color='#FFB6C1', label='Shifted Reward')
    
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rolling Average Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Under Reward Drift (Goal Shift)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative rewards plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#FFFFFF')
    
    for encoder_name, encoder_results in results.items():
        steps = encoder_results['steps']
        cumulative = encoder_results['cumulative_rewards']
        ax2.plot(steps, cumulative, 
                color=COLORS[encoder_name], linewidth=2, 
                label=encoder_name, alpha=0.85)
    
    ax2.axvline(x=config.drift_time, color=COLORS['drift_line'],
               linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Steps', fontsize=11)
    ax2.set_ylabel('Cumulative Reward', fontsize=11)
    ax2.set_title('Cumulative Rewards', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Before/After bars
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#FFFFFF')
    
    bar_data = {}
    for encoder_name, encoder_results in results.items():
        steps = np.array(encoder_results['steps'])
        rewards = np.array(encoder_results['rolling_rewards'])
        
        before_mask = steps < config.drift_time
        after_mask = steps >= config.drift_time
        
        bar_data[encoder_name] = {
            'before': np.mean(rewards[before_mask]) if np.any(before_mask) else 0,
            'after': np.mean(rewards[after_mask]) if np.any(after_mask) else 0
        }
    
    x = np.arange(2)
    width = 0.35
    
    if 'Fixed' in bar_data and 'Drift-Aware' in bar_data:
        bars1 = ax3.bar(x - width/2, 
                       [bar_data['Fixed']['before'], bar_data['Fixed']['after']], 
                       width, label='Fixed', color=COLORS['Fixed'], alpha=0.85)
        bars2 = ax3.bar(x + width/2, 
                       [bar_data['Drift-Aware']['before'], bar_data['Drift-Aware']['after']], 
                       width, label='Drift-Aware', color=COLORS['Drift-Aware'], alpha=0.85)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Before Drift', 'After Drift'], fontsize=10, fontweight='bold')
    ax3.set_ylabel('Avg Reward', fontsize=11)
    ax3.set_title('Before vs After', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Performance change comparison
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#FFFFFF')
    
    changes = {}
    for encoder_name in ['Fixed', 'Drift-Aware']:
        if encoder_name in bar_data:
            before = bar_data[encoder_name]['before']
            after = bar_data[encoder_name]['after']
            if before != 0:
                changes[encoder_name] = ((before - after) / abs(before)) * 100
            else:
                changes[encoder_name] = 0
    
    if changes:
        colors = [COLORS['Fixed'], COLORS['Drift-Aware']]
        bars = ax4.bar(list(changes.keys()), list(changes.values()), 
                      color=colors, alpha=0.85, edgecolor='white', linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=11, fontweight='bold')
    
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_ylabel('Performance Change (%)', fontsize=11)
    ax4.set_title('Change After Drift\n(Negative = Improvement)', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Timeline visualization
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#FFFFFF')
    
    # Create a simple timeline
    timeline_y = 0.5
    ax5.plot([0, config.num_training_steps], [timeline_y, timeline_y], 
            color='gray', linewidth=3, alpha=0.4)
    
    # Mark phases
    ax5.scatter([0], [timeline_y], s=200, c='#2ECC71', marker='o', zorder=5, edgecolor='white', linewidth=2)
    ax5.scatter([config.drift_time], [timeline_y], s=300, c=COLORS['drift_line'], 
               marker='D', zorder=5, edgecolor='white', linewidth=2)
    ax5.scatter([config.num_training_steps], [timeline_y], s=200, c='#E74C3C', 
               marker='s', zorder=5, edgecolor='white', linewidth=2)
    
    # Phase regions
    ax5.axvspan(0, config.drift_time, ymin=0.3, ymax=0.7, alpha=0.3, color='#2ECC71')
    ax5.axvspan(config.drift_time, config.num_training_steps, ymin=0.3, ymax=0.7, alpha=0.3, color='#E74C3C')
    
    ax5.annotate('Training\nStarts', xy=(0, timeline_y), xytext=(0, 0.8),
                ha='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=1.5))
    ax5.annotate(f'DRIFT\nStep {config.drift_time}', xy=(config.drift_time, timeline_y), 
                xytext=(config.drift_time, 0.2),
                ha='center', fontsize=10, fontweight='bold', color=COLORS['drift_line'],
                arrowprops=dict(arrowstyle='->', color=COLORS['drift_line'], lw=1.5))
    ax5.annotate(f'End\n{config.num_training_steps}', 
                xy=(config.num_training_steps, timeline_y),
                xytext=(config.num_training_steps, 0.8),
                ha='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    
    ax5.set_xlim(-500, config.num_training_steps + 500)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('Steps', fontsize=11, fontweight='bold')
    ax5.set_title('Experiment Timeline', fontsize=12, fontweight='bold')
    ax5.set_yticks([])
    
    # Add phase labels
    ax5.text(config.drift_time/2, 0.1, 'Pre-Drift', ha='center', fontsize=10, 
            fontweight='bold', color='#2ECC71')
    ax5.text((config.drift_time + config.num_training_steps)/2, 0.1, 'Post-Drift', 
            ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
    
    plt.suptitle('Reward Drift (Goal Shift) - Detailed Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    """Run all experiments and create visualizations"""
    
    Path("results").mkdir(exist_ok=True)
    
    print("=" * 70)
    print("RFE with Distributional Drift - Enhanced Visualization Experiments")
    print("=" * 70)
    
    # Base configuration (smaller grid, shaped reward for clearer learning)
    base_config = ExperimentConfig(
        grid_size=6,
        drift_strength=0.5,
        drift_schedule=DriftSchedule.SUDDEN,
        drift_time=3000,
        num_exploration_steps=2000,
        num_training_steps=8000,
        eval_interval=100,
        num_eval_episodes=5,
        seed=42,
    )
    
    # Store results for all drift types
    all_drift_results = {}
    
    # ========================================
    # Experiment 1: All Drift Types
    # ========================================
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: All Drift Types")
    print("=" * 50)
    
    drift_types = [
        DriftType.GOAL_SHIFT,
        DriftType.TRANSITION_NOISE,
        DriftType.WALL_CHANGE,
        DriftType.COMBINED,
    ]
    
    for drift_type in drift_types:
        print(f"\n--- {drift_type.value.replace('_', ' ').title()} ---")
        
        config = ExperimentConfig(
            grid_size=base_config.grid_size,
            drift_type=drift_type,
            drift_strength=base_config.drift_strength,
            drift_schedule=base_config.drift_schedule,
            drift_time=base_config.drift_time,
            num_exploration_steps=base_config.num_exploration_steps,
            num_training_steps=base_config.num_training_steps,
            eval_interval=base_config.eval_interval,
            num_eval_episodes=base_config.num_eval_episodes,
            seed=base_config.seed,
        )
        
        print("  Running exploration...")
        _, fixed_encoder, drift_encoder = run_exploration(config)
        
        results = {}
        for encoder, name in [(fixed_encoder, "Fixed"), (drift_encoder, "Drift-Aware")]:
            results[name] = train_and_track_stepwise(encoder, name, config)
        
        all_drift_results[drift_type] = results
        
        # Individual plot for this drift type
        create_performance_plot(
            results, 
            config,
            f"Performance: {drift_type.value.replace('_', ' ').title()}",
            f"results/drift_{drift_type.value}.png"
        )
    
    # Create combined visualization for all drift types
    print("\n  Creating all-drifts comparison plot...")
    create_all_drifts_comparison(all_drift_results, base_config, "results/all_drifts_comparison.png")
    
    # Create robustness summary
    print("  Creating robustness summary...")
    create_robustness_summary(all_drift_results, base_config, "results/robustness_summary.png")
    
    # ========================================
    # Experiment 2: Reward Drift (Goal Shift) - Detailed
    # ========================================
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Reward Drift (Goal Shift) - Detailed")
    print("=" * 50)
    
    config = ExperimentConfig(
        grid_size=base_config.grid_size,
        drift_type=DriftType.GOAL_SHIFT,  # Goal shift = reward drift
        drift_strength=0.6,  # Moderate drift
        drift_schedule=DriftSchedule.SUDDEN,
        drift_time=base_config.drift_time,
        num_exploration_steps=base_config.num_exploration_steps,
        num_training_steps=base_config.num_training_steps,
        eval_interval=base_config.eval_interval,
        num_eval_episodes=base_config.num_eval_episodes,
        seed=base_config.seed + 100,
    )
    
    print("  Running exploration...")
    _, fixed_encoder, drift_encoder = run_exploration(config)
    
    reward_drift_results = {}
    for encoder, name in [(fixed_encoder, "Fixed"), (drift_encoder, "Drift-Aware")]:
        reward_drift_results[name] = train_and_track_stepwise(encoder, name, config)
    
    # Detailed reward drift visualization
    print("  Creating detailed reward drift visualization...")
    create_reward_drift_detailed(reward_drift_results, config, "results/reward_drift_detailed.png")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\n📊 Generated Visualizations:")
    print("  1. results/all_drifts_comparison.png - All drift types comparison")
    print("  2. results/robustness_summary.png - Robustness metrics summary")
    print("  3. results/reward_drift_detailed.png - Detailed reward drift analysis")
    print("  4. Individual drift type plots in results/drift_*.png")
    
    print("\n📈 Key Metrics by Drift Type:")
    for drift_type, results in all_drift_results.items():
        print(f"\n  {drift_type.value.replace('_', ' ').title()}:")
        for encoder_name in ['Fixed', 'Drift-Aware']:
            if encoder_name in results:
                steps = np.array(results[encoder_name]['steps'])
                rewards = np.array(results[encoder_name]['rolling_rewards'])
                
                before_mask = steps < base_config.drift_time
                after_mask = steps >= base_config.drift_time
                
                mean_before = np.mean(rewards[before_mask]) if np.any(before_mask) else 0
                mean_after = np.mean(rewards[after_mask]) if np.any(after_mask) else 0
                
                if mean_before != 0:
                    change = ((mean_before - mean_after) / abs(mean_before)) * 100
                else:
                    change = 0
                    
                print(f"    {encoder_name}: Before={mean_before:.3f}, After={mean_after:.3f}, Change={change:+.1f}%")
    
    print("\n" + "=" * 70)
    print("Done! Check the 'results' folder for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
