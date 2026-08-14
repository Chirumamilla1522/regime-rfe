"""
Metrics tracking for evaluation
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


class MetricsTracker:
    """Track metrics during training and evaluation"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
    
    def log(self, key: str, value: float, episode: bool = False):
        """Log a metric value"""
        if episode:
            self.episode_metrics[key].append(value)
        else:
            self.metrics[key].append(value)
    
    def get_mean(self, key: str, window: int = None) -> float:
        """Get mean of a metric, optionally over a window"""
        values = self.metrics.get(key, [])
        if len(values) == 0:
            return 0.0
        if window is None:
            return np.mean(values)
        return np.mean(values[-window:])
    
    def get_episode_mean(self, key: str, window: int = None) -> float:
        """Get mean of an episode metric"""
        values = self.episode_metrics.get(key, [])
        if len(values) == 0:
            return 0.0
        if window is None:
            return np.mean(values)
        return np.mean(values[-window:])
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        summary = {}
        for key in self.metrics:
            summary[key] = self.get_mean(key)
        for key in self.episode_metrics:
            summary[f"episode_{key}"] = self.get_episode_mean(key)
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.episode_metrics.clear()

