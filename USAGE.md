# Usage

Run commands from the repository root with `python3`:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 run.py --output results/canonical
```

Override seeds without editing source:

```bash
python3 run.py --seeds 10 11 12 13 14 --output results/alternate
```

Use the bounded smoke configuration:

```bash
python3 run.py --quick --seeds 0 --output results/smoke
```

Protocol chooser and copy-paste commands are in `README.md`.

The canonical method names are `fixed_count`, `time_count`, and
`fixed_random`. “Count” denotes a local count-bonus heuristic. It must not be
described as UCRL-RFE because no optimistic planning problem is solved.
The fixed/count and time/count encoders receive the identical ordered
transition list; only time/count consumes each record's clock field.

Evaluation phases are controlled at global step 0 (pre) and `drift_time`
(post). Before each evaluation episode, both the environment and encoder are
placed on that same clock. Episode resets do not reset training time. The clock
is observable context, not a hidden oracle: methods are not given the boundary,
drift type, active goal, map, or drift indicator.

Raw files:

- `episodes.csv`: every evaluation episode;
- `per_seed.csv`: independent units used for confidence intervals;
- `coverage.csv`: reachable-state coverage by collector;
- `summary.csv`: mean and bounded 95% cluster-bootstrap interval over seeds;
- `results.json`: exact configuration and summary.

The default CPU run is small (10 seeds × 3 drifts × 3 downstream methods), but
runtime depends on PyTorch and hardware. Do not infer expected superiority:
the checked run is a negative result for time conditioning.

The predeclared downstream-budget diagnostic is:

```bash
python3 scripts/check_downstream_budget.py
```

It uses only goal drift, seeds 0--2, and budgets 500/1500/4000. It diagnoses
training stability and is not used to choose whichever budget looks best.
