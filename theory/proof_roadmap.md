# Proof roadmap and research ledger

## Dependency graph

1. Prove a detector/matcher-specific `SEG` theorem using time-uniform
   concentration under adaptive probing.
2. Verify that accepted per-mode data satisfy the chosen stationary RFE
   routine's sampling interface; otherwise modify collection rather than
   reusing its theorem.
3. Prove confidence-set separation and a robust diagnostic error bound.
4. Insert those three module guarantees into Theorem 1.
5. Separately prove the product-class direct sum; only then compare the upper
   bound to an \(M\)-scaled minimax lower bound.

The elementary reductions and impossibilities in `theorem_targets.md` are
complete.  Steps 1, 3, and 5 are open targets.

## Detector proof obligations

A valid proof must define the alarm statistic, filtration, predictable probe
schedule, window purity event, false-alarm control over all times, detection
delay, reset behavior, and recurrence test.  It must handle the first/final
block and neighboring switches, and prove quota accounting without counting a
quarantined episode as RFE data.  Fixed-time concentration applied after a
data-dependent alarm is insufficient.

## Assumptions-to-experiment ledger

* **A1 probe reachability:** estimate probe hit rates with confidence
  intervals; move discrepancies into rarely reached cells.  Violation makes
  detection delay diverge.
* **A2 controlled separation:** estimate held-out likelihood ratios or
  affinities; shrink transition gaps toward zero.  Diagnosis then approaches
  chance.
* **A3 useful occupancy:** report accepted pure episodes by true mode and make
  one recurring mode rare.  An under-quota mode loses PAC eligibility.
* **A4 dwell time:** compare true block lengths with quarantine, probe, and
  RFE budgets while sweeping switch frequency.  Short blocks mix windows.
* **A5 reward normalization:** report maximum pathwise return and rescale
  rewards.  Regret must rescale identically.
* **A6 fixed seen deployment:** log the deployment label and test unseen or
  post-diagnosis switches.  The procedure must reject or lose its guarantee.
* **A7 transition-only modes:** compare transition distances with reward-only
  changes.  A changed goal with unchanged transitions gives no detector
  signal.
* **A8 adaptive validity:** measure confidence-set coverage over repeated
  optional-stopping simulations.  Invalid intervals cause false matching.

These diagnostics are evidence about assumptions, not proofs of them.

## Novelty-threat analysis

* **Mixture/latent MDP overlap.**  If rapid hidden switching is allowed, the
  problem approaches latent or block-MDP identification.  Novelty must rest on
  recurring piecewise-stationary regimes plus reward-uniform reuse, not merely
  hidden labels.
* **Change-point literature overlap.**  Detecting multinomial changes is
  classical.  A contribution must couple controlled reachability, recurrence
  matching, and downstream RFE guarantees.
* **Reward-free RL overlap.**  Per-mode exploration is inherited from
  stationary RFE.  The new burden is statistically valid demultiplexing and
  deployment diagnosis; stationary rates must be cited, not relabeled.
* **System-identification overlap.**  The max-min diagnostic design is an
  active hypothesis-testing problem.  Efficient computation or robust design
  may already follow from controlled-sensing results and requires a dedicated
  literature review.
* **Empirical-only threat.**  A heuristic detector that works on Gridworld is
  not evidence for `SEG`; the theorem needs explicit margins and time-uniform
  error control.
* **Minimax threat.**  Single-mode hardness does not automatically tensorize.
  Any claim of \(M\)-optimality is premature until the adaptive direct-sum
  target is proved.

## Falsification checkpoints

Abandon or weaken the end-to-end claim if robust information remains zero at
the model accuracy delivered by feasible dwell times, if detector data violate
the RFE input contract, or if recurrence matching cannot be made time-uniform.
In those cases, honest alternatives are abstention, an oracle-segmentation
result, or an empirical method without a PAC claim.
