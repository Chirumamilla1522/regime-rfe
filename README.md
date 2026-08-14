# Regime-Conditional Reward-Free Exploration

[![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9B47?logo=pytest&logoColor=white)](#try-it)
[![Env](https://img.shields.io/badge/env-Gymnasium%20%2B%20MiniGrid-black)](#pick-a-study)
[![Code](https://img.shields.io/badge/remote-code%20only-555)](https://github.com/Chirumamilla1522/regime-rfe)

Infer a **latent dynamics regime** from transitions `(state, action, next_state)` only. Rewards are attached later. No method sees the drift boundary, active goal/map, or an oracle regime flag.

```mermaid
flowchart LR
  A[Collect without rewards] --> B[Infer or match regimes]
  B --> C[Fit transition models]
  C --> D[Reveal a reward]
  D --> E[Plan and diagnose]
```

<p align="center">
  <a href="#try-it"><strong>Try it</strong></a>
  ·
  <a href="#pick-a-study"><strong>Pick a study</strong></a>
  ·
  <a href="#what-gets-written"><strong>Outputs</strong></a>
  ·
  <a href="#layout"><strong>Layout</strong></a>
  ·
  <a href="#checked-results"><strong>Checked results</strong></a>
</p>

## Try it

```bash
git clone https://github.com/Chirumamilla1522/regime-rfe.git
cd regime-rfe
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Then run one command from the chooser below. Start with **smoke test** if you just want to confirm the harness works.

<details open>
<summary><strong>Smoke test</strong> — one seed, seconds on CPU</summary>

```bash
python3 run.py --quick --seeds 0 --output results/smoke
```

</details>

<details>
<summary><strong>Inferred-regime pilot</strong> — residual detector + matched planner</summary>

```bash
./scripts/run_regime_pilot.sh
```

Equivalent:

```bash
python3 run.py --protocol regime --profile pilot \
  --output results/inferred_regime_pilot
```

</details>

<details>
<summary><strong>Recurrent tabular Detect–Match–Explore</strong></summary>

```bash
python3 run.py --protocol recurrent-tabular --profile quick \
  --output results/recurrent_tabular_quick
```

</details>

<details>
<summary><strong>MiniGrid recurrence</strong></summary>

```bash
python3 run.py --protocol minigrid-recurrent --profile quick \
  --output results/minigrid_recurrent_quick
```

</details>

<details>
<summary><strong>RFE-Recurrent-Bench</strong></summary>

```bash
python3 run.py --protocol recurrent-bench --profile quick \
  --output results/recurrent_bench_quick
```

</details>

Interrupted runs resume from configuration-hashed checkpoints. Pass `--no-resume` to recompute.

## Pick a study

Click a protocol, copy the command, change `--profile` if you need a larger budget.

| Protocol | What it does | Profiles |
| --- | --- | --- |
| `clock` | Frozen encoders + DQN under gridworld drift | `--quick` |
| `regime` | Online residual detector, no public clock | `quick`, `pilot`, `full` |
| `recurrent-tabular` | Detect–Match–Explore on recurring MDPs | `quick`, `pilot`, `full`, `certified`, `certified-full` |
| `minigrid-recurrent` | Recurring MiniGrid with swapped controls | `quick`, `pilot`, `fourrooms`, `conflict`, `conflict-full` |
| `recurrent-bench` | Generative four-task recurrence bench | `quick`, `pilot`, `full`, `stress` |

<details>
<summary><code>clock</code> — frozen representations under drift</summary>

The explorer is a **count-bonus heuristic**, not UCRL-RFE. The time-conditioned encoder sees a normalized public clock, never the drift boundary.

```bash
python3 run.py --protocol clock --output results/canonical
python3 run.py --quick --seeds 0 --output results/smoke
```

Methods trained on the same ordered transitions:

- `fixed_count` — time-unaware encoder, count-bonus collection
- `time_count` — clock-conditioned encoder, same collection
- `fixed_random` — fixed encoder, uniform random collection

Canonical CLI: `run.py`. Root `rfe_drift.py` and `run_drift_experiments.py` are legacy.

</details>

<details>
<summary><code>regime</code> — infer context from forward-model residuals</summary>

The detector API accepts only `(state, action, next_state)`.

```bash
python3 run.py --protocol regime --profile quick \
  --output results/inferred_regime_quick
python3 run.py --protocol regime --profile pilot \
  --output results/inferred_regime_pilot
python3 run.py --protocol regime --profile full \
  --output results/inferred_regime_full
```

Comparisons: `stationary_no_context`, `inferred_regime`, and `oracle_regime_upper_bound` (unrealistic upper bound).

</details>

<details>
<summary><code>recurrent-tabular</code> — Detect–Match–Explore</summary>

```bash
python3 run.py --protocol recurrent-tabular --profile quick \
  --output results/recurrent_tabular_quick
python3 run.py --protocol recurrent-tabular --profile pilot \
  --output results/recurrent_tabular_pilot
python3 run.py --protocol recurrent-tabular --profile certified \
  --output results/recurrent_tabular_certified
```

Compares pooled, restart, clustering, recurrence-aware DME, oracle-boundary, oracle-mode, and sliding-window models. Rewards appear only after collection. Values and gaps are in `[0, 1]`.

</details>

<details>
<summary><code>minigrid-recurrent</code> — symbolic MiniGrid scale gate</summary>

Uses `minigrid==3.1.0` and 7×7 symbolic partial observations. Mission strings, native rewards, and regime IDs are hidden.

```bash
./scripts/run_minigrid_pilot.sh
python3 run.py --protocol minigrid-recurrent --profile fourrooms \
  --output results/minigrid_fourrooms_pilot
```

</details>

<details>
<summary><code>recurrent-bench</code> — swap-chain, RiverSwim, DeepSea, four rooms</summary>

```bash
python3 run.py --protocol recurrent-bench --profile quick \
  --output results/recurrent_bench_quick
python3 run.py --protocol recurrent-bench --profile pilot \
  --output results/recurrent_bench_pilot
```

Task table and gate criteria: [`rfe_drift/benchmark/README.md`](rfe_drift/benchmark/README.md).

</details>

## What gets written

Every study writes a result directory you pass with `--output`. Typical files:

| File | Contents |
| --- | --- |
| `summary.csv` / `results.json` | Seed-level means and bounded 95% cluster-bootstrap CIs |
| `per_seed.csv` | Independent units used for those intervals |
| `transitions.csv` / `raw_transitions.csv` | Reward-free trajectories (protocol-dependent) |
| `checkpoints/` | Resumable per-unit JSON |

Clock-study extras: `episodes.csv`, `coverage.csv`, `post_drift_success.png`.

## Layout

```text
run.py                 canonical harness
rfe_drift/             environment, detectors, planners, MiniGrid, bench
scripts/               pilots and asset helpers
tests/                 pytest suite
results/canonical/     checked clock-study CSV/JSON
```

<details>
<summary>Package map</summary>

```text
rfe_drift/
├── env/             drift gridworld
├── exploration/     count-bonus explorer
├── representations/ frozen encoders
├── rl/              downstream DQN
├── protocol.py      inferred-regime study
├── tabular.py       recurring MDP + DME
├── recurrent_study.py
├── minigrid_study.py
└── benchmark/       RFE-Recurrent-Bench
```

</details>

## Checked results

These are diagnostic, not SOTA claims.

| Study | Gate | Takeaway |
| --- | --- | --- |
| Clock / time conditioning | fail | No advantage for time conditioning; goal and wall drift stay at 0 success |
| Recurrence-aware tabular pilot | fail | Tiny positive gap vs restart, below the 0.01 margin |
| MiniGrid Empty-6x6 pilot | fail | Recurrence-aware worse than pooled |
| RFE-Recurrent-Bench 10-seed pilot | pass | DME beats restart and pooling on the predeclared gate |

Clock-study post-drift success under **transition noise** (10 seeds, 95% CI):

| Method | Mean | CI |
| --- | --- | --- |
| `fixed_count` | 0.335 | [0.110, 0.575] |
| `time_count` | 0.180 | [0.000, 0.405] |
| `fixed_random` | 0.200 | [0.035, 0.410] |

More protocol detail: [`EXPERIMENTS.md`](EXPERIMENTS.md) and [`USAGE.md`](USAGE.md).

## Design constraints

- One global drift clock that survives episode resets
- The transition that hits the sudden-drift index uses the drifted model
- Coverage is over reachable states, not every grid cell
- Identical clocks for environment transitions and encoder evaluation
- Seeded Python, NumPy, PyTorch, env, collector, replay, and policies
- Bootstrap CIs are clustered on independent seed means and clipped to `[0, 1]`
