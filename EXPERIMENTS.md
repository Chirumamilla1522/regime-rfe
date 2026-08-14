# Inferred-regime experiment protocol

The new protocol tests whether transition-derived regime context helps a
matched empirical planner. It does not provide the detector with global time,
the drift boundary, active walls/goals, or `drift_applied`. The detector uses a
smoothed online forward model and a deterministic two-window shift test over
pre-update prediction residuals. This is a practical experimental mechanism,
not a theorem or a claim of optimal change-point detection.

## One-command CPU launch

No GPU or H100 is needed. The primary learner is tabular model estimation plus
value iteration; PyTorch is used only by the retained clock/DQN study.

```bash
./scripts/run_regime_pilot.sh
```

That command runs tests and then:

```bash
python3 run.py --protocol regime --profile pilot \
  --output results/inferred_regime_pilot
```

For a short integration run:

```bash
python3 run.py --protocol regime --profile quick \
  --output results/inferred_regime_quick
```

The `full` profile is intentionally larger:

```bash
python3 run.py --protocol regime --profile full \
  --output results/inferred_regime_full
```

Each grid/geometry/strength/drift/schedule/seed unit is checkpointed as JSON.
Rerunning the same command resumes completed units. Use `--no-resume` to
recompute them. A configuration hash prevents incompatible checkpoints from
being reused.

## Comparisons

- `stationary_no_context`: one pooled transition model.
- `inferred_regime`: separate transition models keyed by detector output.
- `oracle_regime_upper_bound`: separate models keyed by ground-truth schedule
  labels. This is explicitly an upper bound and is not deployable.

Every comparison receives the same ordered transition stream and uses the same
maximum-likelihood estimation and value-iteration settings. The old
clock-conditioned DQN outputs in `results/canonical` are not modified.

## Outputs and metrics

- `transitions.csv`: every raw transition, residual, inferred ID, truth label,
  and alarm.
- `per_seed.csv`: unit-level success and episode length.
- `detection.csv`: delay, detected-change fraction, false alarms, and
  permutation-invariant regime assignment accuracy.
- `coverage.csv`: state, state-action, and distinct-transition coverage.
- `recovery.csv`: post-drift success versus additional transition samples.
- `summary.csv`: seed/geometry/strength-unit means with seed-level bootstrap
  intervals.
- `detection_summary.csv`, `coverage_summary.csv`, and
  `recovery_summary.csv`: geometry/strength averages clustered at the seed
  level before bootstrap resampling.
- `stationary_check.csv` and `stationary_check_summary.csv`: the same tabular
  instrument on zero-drift tasks.
- `results.json`: exact configuration, hash, runtime, and summaries.
- `checkpoints/`: deterministic resumable unit results.

Goal drift changes reward rather than dynamics, so a forward-residual detector
should not be expected to identify it. The pilot defaults to transition and
wall drift; the full profile includes goal and combined drift as stress tests.

## Go/no-go criteria before scaling

Scale to MiniGrid only if all of the following are met on predeclared grid
profiles:

1. The tabular instrument solves stationary tasks with at least 0.9 mean
   success.
2. Inferred context improves post-drift success or recovery area under the
   curve over the matched stationary control on at least two transition-drift
   conditions, with seed-level intervals that do not indicate a large harmful
   effect elsewhere.
3. Median sudden-change delay is at most 25% of the drift interval and false
   alarms are below 1 per 1,000 transitions.
4. State-action coverage exceeds 0.8, or failures are explicitly stratified by
   coverage.
5. Results reproduce after resume and across at least 10 independent seeds.

Move from MiniGrid to Procgen only after the same qualitative effect survives
partial observability, learned visual forward models, and a reliable
non-tabular downstream learner. Otherwise the correct decision is no-go:
improve detection/collection in the controlled domain rather than spend GPU
budget.

## Recurrent tabular protocol

The additional recurrence study uses the synthetic suite in
`rfe_drift/synthetic.py` and the implementation in
`rfe_drift/recurrent_study.py`. Its non-oracle path exposes no mode, boundary,
clock, or reward input to detection, matching, collection, or deployment
diagnosis. Boundary quarantine samples are deliberately excluded from mode
models. Oracle-boundary and oracle-mode are labeled upper controls.

```bash
python3 run.py --protocol recurrent-tabular --profile quick \
  --output results/recurrent_tabular_quick
python3 run.py --protocol recurrent-tabular --profile pilot \
  --output results/recurrent_tabular_pilot
python3 run.py --protocol recurrent-tabular --profile full \
  --output results/recurrent_tabular_full

# Override seeds or force recomputation
python3 run.py --protocol recurrent-tabular --profile pilot --seeds 0 1 2 \
  --output results/recurrent_tabular_custom
python3 run.py --protocol recurrent-tabular --profile pilot --no-resume \
  --output results/recurrent_tabular_pilot
```

Outputs are `raw_transitions.csv`, `value_gaps.csv`,
`deployment_diagnosis.csv`, `recurrence_savings.csv`,
`detector_diagnostics.csv`, `paired_seed_gaps.csv`, `summary.csv`,
`results.json`, and
configuration-hashed `checkpoints/`. The primary quality metric is the mean
worst post-hoc reward value gap over true deployment modes. Sample savings are
the difference between restart and recurrence-aware post-recurrence samples
needed to reach the predeclared value-gap target.

The downstream problem is finite-horizon and episodic: `H=20`, policies are
stage-indexed, and planning uses backward induction without discounting.
Rewards are stage-indexed and bounded by `1/H=0.05`, which guarantees pathwise
total reward at most one. The exact true-model optimum and every learned value
and gap therefore lie in `[0,1]`. The sample-savings target is a normalized
worst-reward gap of 0.05.

The strengthened recurrence gate requires all of: at least 10 seeds; all
values in `[0,1]`; restart-gap minus recurrence-gap improvement of at least 0.01
on that normalized scale; a paired seed-bootstrap 95% interval excluding zero;
positive recurrence sample savings; and at least 0.5 deployment-prefix
acceptance.

The corrected 10-seed/40-unit CPU pilot completed in approximately 24.8
seconds wall time (22.1 seconds summed unit runtime) and **failed** the gate.
Mean normalized worst-reward gaps were: pooled 0.13472,
restart-without-reuse 0.00248, cluster-without-quarantine 0.00127,
recurrence-aware 0.00138, oracle-boundary 0.00163, empirical oracle-mode
0.00157, and sliding-window 0.15458. Recurrence's paired improvement over
restart was 0.00110 (95% paired bootstrap CI `[0.00057, 0.00178]`), which is
positive but below the 0.01 meaningful-effect margin. Mean recurrence sample
savings were 48.12 and deployment acceptance was 1.0.

`oracle_value` is computed from the exact true kernel and is the actual
finite-horizon upper reference. The `oracle_mode` method still estimates its
kernel from finite samples, with privileged true labels; it is not the exact
kernel. Consequently tiny ordering reversals among empirical oracle-mode,
oracle-boundary, and clustered models can occur because plug-in planning is not
monotone in sample count. All three corrected gaps are near zero, while the
pooled and sliding-window controls remain substantially worse.

## Controlled recurring MiniGrid protocol

The tabular recurrence gate authorized this scale study, implemented in
`rfe_drift/minigrid_study.py`. It requires the maintained, exactly pinned
`minigrid==3.1.0`; if unavailable, the command raises an installation error
and does not silently substitute the repository's custom gridworld.

```bash
# Five-seed integration profile
python3 run.py --protocol minigrid-recurrent --profile quick \
  --output results/minigrid_recurrent_quick

# Predeclared ten-seed decision profile
python3 run.py --protocol minigrid-recurrent --profile pilot \
  --output results/minigrid_recurrent_pilot

# Resume is default; force recomputation only when intended
python3 run.py --protocol minigrid-recurrent --profile pilot --no-resume \
  --output results/minigrid_recurrent_pilot
```

The family alternates recurring regimes only at episode reset. Regime 0 uses
native MiniGrid left/right transitions and partial observations. Regime 1
swaps left/right controls and mirrors plus bijectively remaps colors in the
7x7 egocentric observation. Thus both transition and observation processes
change; this is not an undetectable reward-only shift. The learner receives
only image/action/image triples. Mission strings, native rewards, regime
labels, boundaries, and a clock are stripped. True labels in raw CSV files
carry the suffix `evaluation_only` and are consumed only by metrics and
`oracle_upper_bound`.

Reward-free observations first fit a shared PCA and MiniBatchKMeans latent
state encoder. Detect--Match--Explore is instantiated by random reward-free
exploration, action-conditional learned-feature delta signatures at episode
boundaries, recurrence matching against archived prototypes, and reuse of the
matched latent transition model. Baselines are pooled, restart from the
deployment prefix, recurrence-aware, and label-aware oracle. Three
observation-transition rewards (`goal_visible`, `goal_centered`, and
`visual_change`) are declared only after records are frozen; the primary
metric is the worst gap to the empirical oracle over those tasks.

Outputs are configuration-hashed resumable JSON checkpoints plus
`raw_transitions.csv`, `value_gaps.csv`, `deployment_diagnosis.csv`,
`recurrence_savings.csv`, `detector_diagnostics.csv`, `summary.csv`, and
`results.json`. The predeclared pilot gate requires all of:

1. at least 10 independent seeds;
2. lower worst-task gap than restart;
3. no larger worst-task gap than pooled;
4. deployment-ID error at most 0.25; and
5. positive mean recurrence sample savings at the 0.15 gap target.

The checked quick run took 13.15 seconds wall time (10.78 summed unit seconds):
recurrence gap 0.1438, restart gap 0.4750, pooled gap 0.0, ID error 0.20, and
2.0 mean saved samples. It fails the pooled and seed-count checks.

The checked pilot took 39.76 seconds wall time (37.15 summed unit seconds):
recurrence gap 0.0492, restart gap 0.2648, pooled gap 0.0, ID error 0.20, and
6.8 mean saved samples. It therefore **fails** the MiniGrid gate solely on the
predeclared pooled comparison. The correct decision is no-go for a larger
visual benchmark.

Limitations: this is symbolic 7x7 input rather than RGB; Empty-6x6 has only
two deterministic regimes; the downstream policy is memoryless in learned
latent state; task rewards are a small finite diagnostic family rather than a
uniform guarantee over all bounded rewards; and the empirical oracle is not
an exact POMDP optimum. Pooled succeeds on these tasks, so this setup does not
establish that regime separation is necessary.

## Recurring MiniGrid scale protocol

`rfe_drift/minigrid_study.py` is a separate scale study on the maintained
Farama MiniGrid package. `MiniGrid-Empty-6x6-v0` supplies 7x7 symbolic
egocentric observations. Regime 0 is native MiniGrid; regime 1 swaps left and
right actions and applies a bijective mirror/color transform to each partial
observation. Regimes follow `0,0,1,1,0,0,1,1` across resets. The wrapper emits
only the partial image, zero reward, termination flags, and an empty info
mapping.

The reward-free stream trains one PCA plus MiniBatchKMeans abstraction.
Action-conditional feature-delta signatures drive collection-time recurrence
matching and deployment diagnosis. Neither path accepts a regime label,
boundary, clock, reward, mission string, or native MiniGrid reward.
Ground-truth labels are retained only in `*_evaluation_only` columns and by the
explicit `oracle_upper_bound`.

Compared methods are:

- `pooled`: all historical modes plus the deployment prefix;
- `restart`: deployment-prefix data only;
- `recurrence_aware`: the matched historical archive plus the prefix;
- `oracle_upper_bound`: the true-label historical archive plus the prefix.

Rewards are attached only after collection. The predeclared family currently
contains goal-visible, goal-centered, and visual-change rewards, while the raw
JSON observations permit additional observation-based rewards without
recollection.

```bash
python3 run.py --protocol minigrid-recurrent --profile quick \
  --output results/minigrid_recurrent_quick
python3 run.py --protocol minigrid-recurrent --profile pilot \
  --output results/minigrid_recurrent_pilot
python3 run.py --protocol minigrid-recurrent --profile full \
  --output results/minigrid_recurrent_full
```

Outputs are `raw_transitions.csv`, `value_gaps.csv`,
`deployment_diagnosis.csv`, `recurrence_savings.csv`,
`detector_diagnostics.csv`, `summary.csv`, `results.json`, and atomic
configuration-hashed checkpoints. Reruns resume by default; `--no-resume`
recomputes units.

The pilot gate requires all of: at least ten seeds, lower recurrence-aware
worst-task gap than restart, recurrence-aware gap no worse than pooled,
deployment identification error at most 0.25, and positive mean recurrence
sample savings. A quick or seed-overridden run cannot pass the ten-seed check
and must not be reported as passing the pilot gate.

This is a controlled scale test, not a general visual-RL result: observations
are symbolic rather than RGB, there are two deterministic regimes, planning
uses a memoryless learned state under partial observability, and the oracle is
an empirical privileged-label control rather than an exact optimal value.

## RFE-Recurrent-Bench

Generative four-task suite in `rfe_drift/benchmark/`. Not a frozen trajectory
file: the learner collects reward-free transitions. Tasks: `swap_chain`,
`riverswim` (reversed current), `deepsea` (swapped dive action), `four_rooms`
(moving doors). Short dwell so one block under-covers the state-action table.

```bash
python3 run.py --protocol recurrent-bench --profile quick \
  --output results/recurrent_bench_quick
python3 run.py --protocol recurrent-bench --profile pilot \
  --output results/recurrent_bench_pilot
```

The ten-seed pilot (hash `90f18f4bd17c4218`, 40 units) **passes** the
benchmark gate after the detector fix. DME mean gap 0.070 versus restart
0.143 and pooling 0.333; paired improvement 0.073, CI [0.040, 0.109].
Oracle-mode is 0.026. Four Rooms remains easy for pooling. See
`rfe_drift/benchmark/README.md`.

## Stress tabular protocol

Occupancy imbalance, \(M\in\{2,3,5\}\), short dwell. Does not overwrite the
retained hashes.

```bash
python3 run.py --protocol recurrent-tabular --profile stress \
  --output results/recurrent_tabular_stress
```

Collection uses the greedy covering collector. DME rolls back the pre-alarm
tail. `mbcd_like` is a reward-free likelihood matcher without quarantine.
These runs are a new protocol; do not mix them with hash `6e54042d78e70be4`
or `90f18f4bd17c4218`.

