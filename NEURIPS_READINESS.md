# NeurIPS readiness review

## Current status

The repository is now an auditable negative-result study rather than an
unsupported algorithm claim. The canonical implementation, raw outputs,
generated paper assets, tests, and compiled manuscript are present.

The artifact is suitable for internal review or a workshop submission. It is
not yet competitive for the NeurIPS main track because the environment is
small, downstream learning is unstable, the principal result is null, and the
evaluation does not include a faithful reward-free exploration algorithm.

## Resolved issues

- Removed the inaccurate `UCRL-RFE` method label. The implemented explorer is
  described as a count-bonus collector.
- Preserved global drift time across episode resets and aligned environment and
  encoder clocks.
- Made state coverage reachable-state aware.
- Trained fixed and clock-conditioned encoders on the same ordered transitions.
- Reused the trained downstream task during controlled evaluation.
- Seeded Python, NumPy, PyTorch, collection, replay, and action selection.
- Added ten-seed raw CSV/JSON outputs and bounded seed-cluster bootstrap
  intervals.
- Added regression tests for drift timing, coverage, temporal plumbing,
  matched representation data, and the end-to-end path.
- Generated all manuscript tables and figures from committed raw outputs.
- Compiled `paper/main.pdf` with no unresolved references.

## Evidence and interpretation

The corrected experiment does not support an advantage for observable-clock
conditioning. Post-drift transition-noise success is 0.335 for fixed/count,
0.180 for time/count, and 0.200 for fixed/random; intervals overlap. Every
method has zero observed success under goal and wall drift. Coverage is
saturated, so coordinate visitation alone is not the limiting factor.

These outcomes should be described as diagnostic. They do not show that clock
conditioning is generally ineffective, that count-based collection is worse,
or that reward-free exploration cannot handle drift.

## Required work before a main-track submission

1. Replace the fragile sparse-reward DQN protocol with a learner that reliably
   solves each stationary task, and report a predeclared learning-budget study.
2. Add a faithful published reward-free exploration baseline and a
   nonstationary or change-detection baseline.
3. Evaluate inferred context against observable time; the current clock is a
   privileged covariate.
4. Scale beyond one grid geometry and one drift strength. At minimum include
   larger grids, partial observability, transition-balanced coverage, and
   multiple drift schedules.
5. Separate reward drift from transition drift and report recovery time,
   sample efficiency, transition coverage, and representation diagnostics.
6. Increase statistical power after stabilizing the learner. All-zero samples
   produce deceptively narrow bootstrap intervals.
7. Use the official submission-year NeurIPS style and checklist when released
   or supplied; the compiled PDF currently uses the documented article
   fallback.

## Likely reviewer objections

- The proposed conditioning mechanism is technically modest.
- The toy domain does not establish relevance to modern reward-free RL.
- Zero-success conditions make the principal comparison weakly identified.
- The current experiment cannot distinguish representation failure from
  downstream optimization failure.
- A global observable clock may not be available in realistic deployments.

## Reproduction

```bash
python3 -m pytest -q
python3 run.py --output results/canonical
python3 scripts/generate_paper_assets.py
cd paper && tectonic main.tex --keep-logs
```
