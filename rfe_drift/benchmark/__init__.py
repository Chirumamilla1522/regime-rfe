"""RFE-Recurrent-Bench: generative environments for regime-conditional RFE."""

from rfe_drift.benchmark.envs import make_benchmark_env
from rfe_drift.benchmark.suite import TASK_SPECS, benchmark_profile
from rfe_drift.benchmark.study import run_benchmark_study

__all__ = [
    "TASK_SPECS",
    "benchmark_profile",
    "make_benchmark_env",
    "run_benchmark_study",
]
