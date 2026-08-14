# Regime-conditional RFE submission status

## Verdict

The new problem, algorithm scaffold, theory ledger, tabular protocol, MiniGrid
stress test, and manuscript are implemented and reproducible. The work is not
scientifically ready for NeurIPS: both predeclared empirical gates fail, and
the detector-to-segmentation theorem required by the PAC result is open.

## Current checked evidence

The retained tabular pilot has 40 units over ten seeds. Mean worst-reward gaps
are generated from `results/recurrent_tabular_pilot/results.json`:

- pooled: 0.13472;
- restart without reuse: 0.00248;
- recurrence-aware DME: 0.00138;
- sliding window: 0.15458.

The paired restart-minus-DME improvement is 0.00110 with a 95% seed-bootstrap
interval of [0.00057, 0.00178]. It is positive but below the predeclared 0.01
practical margin. Mean recurrence saving is 48.1 transitions and deployment
acceptance is 1.0.

The ten-seed MiniGrid pilot reports pooled gap 0, restart gap 0.26484, DME gap
0.04922, deployment-ID error 0.20, and 6.8 mean saved samples. It fails because
DME does not match pooling.

## Proved versus targeted

Proved in `theory/theorem_targets.md`:

- conditional modular PAC reduction;
- deployment-error decomposition;
- known-model likelihood diagnosis bound;
- zero-observation impossibility;
- binary diagnostic lower bound.

Not proved:

- the implemented detector satisfies the uncontaminated segmentation contract;
- robust diagnosis from estimated confidence sets;
- an adaptive multi-mode direct-sum RFE lower bound;
- matching minimax detection/switch overhead.

## Required improvements

1. Prove a time-uniform detection, quarantine, and recurrence-matching theorem.
2. Instantiate a published stationary RFE algorithm or prove the implemented
   base learner's reward-uniform contract.
3. Develop held-out recurring-mode tasks where reuse has a meaningful advantage
   without constructing the benchmark around DME.
4. Add executable DARLING- and MBCD-style baselines.
5. Replace the easy MiniGrid setting and validate on unseen layouts.
6. Evaluate a substantially larger reward family or compute an exact
   worst-reward certificate.

## Reproduction

```bash
./scripts/reproduce_regime_paper.sh test
./scripts/reproduce_regime_paper.sh experiments
./scripts/reproduce_regime_paper.sh paper
```

Canonical manuscript:

```text
paper/regime_rfe_submission.tex
paper/regime_rfe_submission.pdf
```
