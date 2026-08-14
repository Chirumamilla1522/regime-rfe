# RFE-Recurrent-Bench

Generative benchmark for regime-conditional reward-free exploration.
This is not a frozen trajectory dataset: the learner must collect its own
reward-free transitions, then plan for post-hoc rewards.

## Tasks

| Task | Source | Recurring change | States | Dwell | Cycles |
|---|---|---|---|---|---|
| `swap_chain` | this repo | conflicting action offsets | 9 | 120 | 4 |
| `riverswim` | Strehl & Littman | current reverses | 6 | 80 | 5 |
| `deepsea` | Osband-style chain | dive action swapped | 8 | 80 | 4 |
| `four_rooms` | Sutton four rooms | door locations move | 20 | 160 | 4 |

Dwell is intentionally shorter than a covering sample of the state-action
table, so restarting after every switch is under-sampled while recurrence
can accumulate.

## Protocol

Reward-free collection, hidden piecewise-stationary modes, quarantine,
recurrence matching, then a diagnostic prefix and a revealed reward family.
Non-oracle methods see only `(s, a, s')`.

```bash
python3 run.py --protocol recurrent-bench --profile quick \
  --output results/recurrent_bench_quick
python3 run.py --protocol recurrent-bench --profile pilot \
  --output results/recurrent_bench_pilot
```

## Gate

Ten seeds; values in `[0,1]`; DME beats restart by at least 0.01 with a
positive paired bootstrap CI; positive sample savings; deployment acceptance
at least 0.5; DME strictly better than pooling.

The retained ten-seed pilot (hash `90f18f4bd17c4218`) **passes**.

