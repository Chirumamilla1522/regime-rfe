# Regime-conditional RFE readiness

## Verdict

The project now has a distinct problem formulation, a modular algorithm,
proved conditional and impossibility results, reproducible tabular and
MiniGrid protocols, and a separate manuscript. It is not ready for NeurIPS
submission. Both predeclared empirical gates fail, and the theorem connecting
the implemented detector to uncontaminated recurring-mode samples is open.

## Established contributions

- Formal reward-free objective with hidden recurring transition regimes and
  reward-independent deployment diagnosis.
- Detect–Match–Explore implementation with boundary quarantine, recurring-mode
  reuse, arbitrary post-hoc rewards, unseen-mode rejection, and resumable runs.
- Proved modular theory: simulation, KL chain rule, Pinsker/Le Cam,
  pooling inconsistency, boundary contamination, matching uniqueness,
  recurrence accounting, conditional PAC, error decomposition, known-model
  diagnosis rates, i.i.d. probe-stream two-window test, zero-observation
  impossibility, diagnostic testing lower bound, occupancy obstruction, and
  behavioral unidentifiability.
- Ten-seed finite-horizon tabular and MiniGrid pilots with raw outputs.

The direct-sum RFE lower bound, detector segmentation guarantee (`SEG` for
the implemented DME), robust estimated-model diagnosis rate, and matching
minimax switch overhead are research targets—not established theorems.

## Measured evidence

In the tabular pilot (hash `6e54042d78e70be4`, 40 units), pooled learning has
worst-reward gap 0.135, restart-only 0.00248, clustering without quarantine
0.00127, and recurrence-aware DME 0.00138. Mean transition savings versus
restart are 48.12. The paired restart-minus-DME improvement is 0.00110 with
a 95% seed-bootstrap interval of [0.00057, 0.00178], below the predeclared
0.01 practical margin. The tabular gate therefore fails.

In MiniGrid, DME improves over restart (0.0492 versus 0.2648), but pooling has
zero measured gap. Deployment-ID error is 0.20. The MiniGrid gate also fails.

## Required scientific improvements

1. Prove a time-uniform segmentation and recurrence-matching guarantee under
   explicit reachability, separation, and dwell-time assumptions.
2. Instantiate a published stationary RFE algorithm or prove a complete
   reward-uniform guarantee for the implemented base learner.
3. Construct predeclared benchmark families where recurrence creates a
   meaningful sample-allocation problem without simply hard-coding method
   success.
4. Increase the recurrence advantage beyond statistical significance to a
   practically meaningful margin across mode counts, occupancy imbalance, and
   diagnostic separation.
5. Replace the easy MiniGrid configuration with partial-observation tasks where
   pooled transition models are genuinely incompatible, then validate on held
   out layouts.
6. Prove or remove the proposed direct-sum lower bound. A single-mode lower
   bound cannot be multiplied by the number of modes without an adaptive
   direct-sum argument.
7. Add DARLING- and MBCD-style executable baselines rather than relying only on
   conceptual positioning.

## Likely reviewer objections

- The central PAC theorem is conditional on a segmentation event not yet proved
  for the implementation.
- DME composes known detection, clustering, and stationary RFE components.
- The tabular gain over restart is too small to justify recurrence machinery.
- MiniGrid favors the pooled baseline.
- The empirical reward family is finite, whereas the formal objective is
  uniform over all normalized rewards.
- The current stationary collector is a transparent optimistic baseline, not a
  faithful implementation of a named minimax RFE method.

## Reproduction

```bash
./scripts/reproduce_regime_paper.sh test
./scripts/reproduce_regime_paper.sh experiments
./scripts/reproduce_regime_paper.sh assets
./scripts/reproduce_regime_paper.sh paper
```

The complete pipeline is:

```bash
./scripts/reproduce_regime_paper.sh all
```

Main manuscript:

```text
paper/regime_rfe_submission.pdf
```
# Recurring-regime RFE manuscript readiness

## Status

**Buildable theory/pilot package; not scientifically submission-ready.**

`paper/regime_rfe.tex` builds successfully with Tectonic into a seven-page
letter-size PDF. `neurips_2026.sty` is not present in the repository, so the
checked build used the explicit one-inch `geometry` fallback. The build has no
LaTeX errors or overfull boxes. BibTeX emits two underfull-box typography
warnings for one long bibliography entry.

No Cursor plan file, core implementation, test, `theory/` file, or
`paper/main.tex` was edited for this package. No commit was created.

## Package files

Authored:

- `paper/regime_rfe.tex`
- `paper/regime_rfe_references.bib`
- `scripts/generate_regime_rfe_assets.py`
- `REGIME_RFE_READINESS.md`

Generated from `results/recurrent_tabular_pilot`:

- `paper/generated/regime_pilot_metrics.json`
- `paper/generated/tables/pilot_config.tex`
- `paper/generated/tables/pilot_summary.tex`
- `paper/generated/figures/pilot_value_gaps.pdf`
- `paper/generated/figures/pilot_recurrence_savings.pdf`

Checked build outputs:

- `paper/regime_rfe.pdf`
- `paper/regime_rfe.log`
- `paper/regime_rfe.blg`

The bibliography `.bbl` and other intermediates are managed internally by
Tectonic and were not retained.

## Reproduction

From the repository root:

```bash
python3 scripts/generate_regime_rfe_assets.py
cd paper
tectonic regime_rfe.tex --keep-logs
```

The asset generator completed without error and the Python file has no IDE
linter diagnostics. The retained pilot is identified by configuration hash
`0b7f0a4e02e3717c`.

## Exact checked empirical facts

- 40 units: 10 seeds, one 9-state setting, mode counts 2 and 3, separations
  0.75 and 0.9.
- Recurrence-aware mean worst-reward gap:
  `0.001192635756209756`.
- Restart-without-reuse mean worst-reward gap:
  `0.002660033024747276`.
- Pooled mean worst-reward gap: `0.14272792430777145`.
- Sliding-window mean worst-reward gap: `0.1909746328004328`.
- Mean recurrence sample savings: `49.44` transitions.
- Positive savings: 200/200 recurrence rows.
- Recurrence-aware deployment label agreement: 100/100 synthetic decisions.
- Correct learned mode count: 40/40 recurrence-aware units.
- Mean recurrence-aware alarms/reuse events per unit: `6.55` / `5.05`.
- Stored deployment acceptance rate: `1.0` for every reported method.
- Paired gap improvement: `0.0014673972685375198`; stored seed-bootstrap 95%
  interval: `[0.0006718183858685261, 0.002362931423338377]`.
- The overall gate fails because that mean improvement is below the
  predeclared `0.01` practical-effect margin, even though the interval excludes
  zero and the other stored checks pass.

These are exact retained-output summaries, not confidence bounds.

## Scientific blockers

1. **No end-to-end detector theorem.** The concrete detector has no proved
   time-uniform false-alarm, delay, boundary-purity, recurrence-matching, or
   adaptive-confidence guarantee, so `SEG` is only an interface.
2. **No robust estimated-model diagnosis theorem.** The known-model
   Bhattacharyya/MAP result does not justify confidence-set diagnosis without
   a positive robust separation margin.
3. **No direct sum.** An adaptive \(M\)-mode stationary-RFE lower bound remains
   a target; multiplying a single-mode lower bound is invalid.
4. **Stationary module contract is unchecked.** The empirical
   model/value-iteration component is not a cited stationary RFE algorithm,
   and detector-selected samples have not been proved admissible for one.
5. **Interaction-unit mismatch.** The pilot now uses a 20-stage finite-horizon
   value with pathwise total reward at most one, but data collection and dwell
   are measured in transition steps rather than complete theory episodes.
6. **Finite reward suite.** Pilot “worst reward” means the maximum over an
   implemented finite suite, not the theorem's supremum over all normalized
   post-hoc rewards.
7. **Limited empirical scope.** There are only ten seeds, one state count, two
   mode counts/separations, one dwell, fixed heuristic thresholds, and no
   uncertainty intervals or held-out tuning.
8. **MiniGrid gate fails.** A ten-seed symbolic Empty-6x6 pilot exists and is
   reported; pooling has zero measured gap, so the scale gate fails and this
   is not a verified visual-domain result.
9. **Style dependency absent.** The official `neurips_2026.sty` must be added
   and the page budget rechecked before submission.

The package is ready for internal circulation as a precise problem statement,
proof-status ledger, and reproducible tabular pilot. It is not ready for a
claim of end-to-end PAC performance, minimax optimality, or visual-domain
validation.
