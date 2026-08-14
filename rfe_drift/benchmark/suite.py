"""Named RFE-Recurrent-Bench task specs and profiles.

Budgets are chosen so one dwell is too short to cover the state-action table,
while several recurrences can be. That is the accounting regime of the
recurrence-versus-restart lemma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# Door cells on the 5x5 four-room layout: 16 room cells + 4 door cells.
FOUR_ROOMS_STATES = 20

TASK_SPECS: Dict[str, Dict] = {
    "swap_chain": {
        "task": "swap_chain",
        "state_count": 9,
        "action_count": 2,
        "mode_count": 2,
        "separation": 0.9,
        "dwell": 120,
        "cycles": 4,
        "quarantine_steps": 8,
        "confirmation_steps": 24,
        "deployment_prefix": 54,
        "horizon": 20,
    },
    "riverswim": {
        "task": "riverswim",
        "state_count": 6,
        "action_count": 2,
        "mode_count": 2,
        "dwell": 80,
        "cycles": 5,
        "quarantine_steps": 6,
        "confirmation_steps": 16,
        "deployment_prefix": 36,
        "horizon": 20,
        "slip": 0.0,
    },
    "deepsea": {
        "task": "deepsea",
        "state_count": 8,
        "action_count": 2,
        "mode_count": 2,
        "dwell": 80,
        "cycles": 4,
        "quarantine_steps": 6,
        "confirmation_steps": 16,
        "deployment_prefix": 48,
        "horizon": 16,
        "slip": 0.1,
    },
    "four_rooms": {
        "task": "four_rooms",
        "state_count": FOUR_ROOMS_STATES,
        "action_count": 4,
        "mode_count": 2,
        "dwell": 160,
        "cycles": 4,
        "quarantine_steps": 10,
        "confirmation_steps": 24,
        "deployment_prefix": 120,
        "horizon": 16,
        "slip": 0.1,
    },
}


@dataclass(frozen=True)
class BenchmarkStudyConfig:
    profile: str = "pilot"
    seeds: Tuple[int, ...] = tuple(range(10))
    tasks: Tuple[str, ...] = (
        "swap_chain",
        "riverswim",
        "deepsea",
        "four_rooms",
    )
    value_gap_target: float = 0.05
    gate_gap_margin: float = 0.01
    gate_bootstrap_samples: int = 20_000
    # Recurrence must also beat pooling, unlike the saturated old suite.
    require_pooling_loss: bool = True
    certified: bool = False
    detector_window: int | None = None


def benchmark_profile(profile: str) -> BenchmarkStudyConfig:
    if profile == "quick":
        return BenchmarkStudyConfig(
            profile="quick",
            seeds=(0, 1),
            tasks=("swap_chain", "riverswim"),
        )
    if profile == "pilot":
        return BenchmarkStudyConfig()
    if profile == "full":
        return BenchmarkStudyConfig(
            profile="full",
            seeds=tuple(range(30)),
        )
    if profile in ("certified", "certified-full"):
        window = 160
        seeds = tuple(range(30)) if profile == "certified-full" else tuple(range(10))
        return BenchmarkStudyConfig(
            profile=profile,
            seeds=seeds,
            tasks=("swap_chain", "deepsea"),
            certified=True,
            detector_window=window,
        )
    raise ValueError(f"unknown benchmark profile: {profile}")
