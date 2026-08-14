# Reward-Free Exploration under Gridworld Drift

This repository is a small, reproducible study of frozen representations under
nonstationary gridworld dynamics. The implemented explorer is a **count-bonus
heuristic**, not UCRL-RFE: it does not perform optimistic model-based planning.
The time-conditioned encoder observes a normalized public clock, but never the
drift boundary, active goal/map, or an oracle drift flag.

The repository now also contains a stronger, separate experimental direction:
an online latent-regime signal inferred only from forward-model residuals,
matched stationary/inferred/oracle-context empirical planners, and a reliable
tabular planning instrument. The previous clock/DQN outputs remain unchanged.
See `EXPERIMENTS.md`.

## Canonical implementation

`run.py` and the `rfe_drift/` package are canonical. The root `rfe_drift.py` is
a deprecated historical prototype retained for history; do not use it for
reported results. `run_drift_experiments.py` is also legacy visualization code.

Corrected semantics include:

- one global drift clock that survives episode resets;
- the transition reaching the sudden-drift index uses the drifted model;
- reachable-state, rather than all-cell, coverage;
- explicit temporal metadata in conditioned representation training;
- identical clocks for environment transitions and encoder evaluation;
- seeded Python, NumPy, PyTorch, environment, collector, replay, and policies;
- raw episode/seed CSV and JSON output with bounded 95% cluster-bootstrap
  intervals over independent seed means.

## Install and verify

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q

# New CPU-only inferred-regime pilot (tests included)
./scripts/run_regime_pilot.sh
```

## Reproduce experiments and paper

```bash
# Ten seeds, three drift types, three methods
python3 run.py --output results/canonical

# Fast integration check
python3 run.py --quick --seeds 0 --output results/smoke

# Regenerate manuscript tables and figures from raw canonical outputs
python3 scripts/generate_paper_assets.py

# Self-contained article fallback with NeurIPS-style sections
cd paper
tectonic main.tex
# Alternatives: latexmk -pdf main.tex, or pdflatex/bibtex passes.
```

Outputs are `episodes.csv`, `per_seed.csv`, `coverage.csv`, `summary.csv`,
`results.json`, and plots. Confidence intervals resample seed-level means as
independent clusters and are bounded to `[0,1]`.

## Methods

- `fixed_count`: time-unaware encoder trained on all count-collected records.
- `time_count`: clock-conditioned encoder trained on exactly the same ordered
  count-collected records.
- `fixed_random`: fixed encoder with uniformly random reward-free collection.

All encoders use forward-state prediction and are frozen for downstream DQN.
The default experiment is intentionally modest and should be treated as a
diagnostic ablation, not a state-of-the-art benchmark.

## New inferred-regime protocol

Run `python3 run.py --protocol regime --profile quick --output
results/inferred_regime_quick` for a short check, or use the pilot script above.
The detector API accepts only `(state, action, next_state)`. Raw transitions,
detection/recovery metrics, coverage, seed-level summaries, and resumable unit
checkpoints are written to a result directory distinct from
`results/canonical`. The oracle-regime comparison is explicitly labeled as an
unrealistic upper bound.

## Recurrent tabular Detect--Match--Explore

The recurrence-focused study is separate from, and does not overwrite, the
earlier clock or inferred-regime results:

```bash
# Smoke test, predeclared 10-seed pilot, and larger sweep
python3 run.py --protocol recurrent-tabular --profile quick \
  --output results/recurrent_tabular_quick
python3 run.py --protocol recurrent-tabular --profile pilot \
  --output results/recurrent_tabular_pilot
python3 run.py --protocol recurrent-tabular --profile full \
  --output results/recurrent_tabular_full
```

Completed units resume from configuration-hashed JSON checkpoints; pass
`--no-resume` to recompute. The protocol compares pooled,
restart-without-reuse, clustering without quarantine, recurrence-aware
Detect--Match--Explore, oracle-boundary, oracle-mode, and sliding-window
models. Rewards are introduced only after collection and deployment mode
diagnosis uses only a transition prefix. Raw transitions, diagnosis decisions,
per-reward value gaps, worst-reward gaps, and recurrence sample savings are
retained.

Planning and evaluation use a 20-stage episodic dynamic program and
stage-indexed nonstationary policies. Every per-stage reward is at most
`1 / 20`, so every pathwise return, value, and gap is in `[0, 1]`, matching the
formal reward convention.

The corrected 10-seed, 40-unit CPU pilot **fails** the strengthened recurrence
gate. Recurrence improves the paired normalized worst-reward gap by 0.00110
(95% paired bootstrap CI `[0.00057, 0.00178]`) over restart, below the
predeclared meaningful margin of 0.01. Deployment acceptance is 1.0 and mean
sample savings are 48.12, but a statistically positive, numerically tiny effect
is not treated as practically meaningful.

## Recurring MiniGrid scale study

The controlled scale study uses the maintained `minigrid==3.1.0` package and
native 7x7 symbolic partial observations. Install dependencies and reproduce
the checked quick and pilot runs with:

```bash
python3 -m pip install -r requirements.txt
python3 run.py --protocol minigrid-recurrent --profile quick \
  --output results/minigrid_recurrent_quick
python3 run.py --protocol minigrid-recurrent --profile pilot \
  --output results/minigrid_recurrent_pilot
```

Regime switches occur only between episodes. One regime uses native controls
and observations; the other swaps left/right dynamics and mirrors/remaps the
partial observation. Mission strings, environment rewards, regime IDs, and
switch metadata are unavailable to collection, learned PCA/k-means features,
transition-signature matching, and deployment diagnosis. Arbitrary task
rewards are attached to frozen transitions after collection.

The checked 10-seed pilot took 39.76 seconds wall time and the predeclared
MiniGrid gate **fails**. Recurrence-aware worst-task gap is 0.0492 versus
0.2648 for restart, deployment-ID error is 0.20, and mean recurrence savings
are 6.8 samples. However, pooled worst-task gap is 0.0, so recurrence-aware is
worse than the pooled baseline. This is a no-go for further visual scaling,
not evidence for recurrence-aware dominance.

Current limitations are important: observations are symbolic rather than RGB;
there are only two deterministic regimes in Empty-6x6; policies are
memoryless over learned latent states despite partial observability; and the
label-aware oracle is an empirical upper control, not an exact optimal value.

## Recurring MiniGrid scale study

The next scale gate uses the maintained `minigrid==3.1.0` package and symbolic
7x7 egocentric observations. Two reset-level regimes recur: one uses native
controls/observations and one swaps left/right dynamics while mirroring and
color-remapping observations. Regime labels and native rewards are removed
from every deployable interface.

```bash
# Five-seed integration profile
python3 run.py --protocol minigrid-recurrent --profile quick \
  --output results/minigrid_recurrent_quick

# Predeclared ten-seed CPU pilot (tests first)
./scripts/run_minigrid_pilot.sh
```

Collection and diagnosis are reward-free. Full partial observations are saved
in `raw_transitions.csv`, so new observation-based rewards can be attached
after collection. Pooled, restart, recurrence-aware, and explicitly privileged
oracle-upper-bound planners use the same learned PCA/k-means state abstraction.
Configuration-hashed per-seed checkpoints make interrupted runs resumable.
See `EXPERIMENTS.md` for outputs, gate criteria, and limitations.

## Current checked result

The included ten-seed run does **not** show an advantage for time conditioning.
Goal drift has zero success for all methods. Under transition-noise drift,
post-drift mean success is 0.335 (`fixed_count`, CI `[0.110, 0.575]`), 0.180
(`time_count`, `[0.000, 0.405]`), and 0.200 (`fixed_random`,
`[0.035, 0.410]`). Wall-drift success is zero for all methods. A predeclared
500/1500/4000-step diagnostic was non-monotonic, indicating sparse-reward DQN
instability rather than evidence that simply increasing this small budget
resolves the failures. See `paper/main.tex` for limitations.
