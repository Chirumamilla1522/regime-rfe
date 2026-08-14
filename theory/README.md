# Theory package: recurring-regime reward-free exploration

This directory is a self-contained specification for the theory direction in
the accepted project plan. It does not describe the current heuristic
implementation and makes no claim that Detect--Match--Explore (DME) has already
been proved minimax optimal.

## Files

- `formal_model.md`: protocol, notation, PAC objective, assumptions ledger,
  exact segmentation interface, and per-mode RFE contract.
- `algorithm.md`: DME pseudocode and required statistical interfaces.
- `theorem_targets.md`: theorem statements, complete elementary proofs, and
  explicitly labelled proof targets.
- `proof_roadmap.md`: proof dependencies, assumptions-to-experiment ledger,
  novelty threats, and falsification points.
- `references.bib`: BibTeX records with stable DOI, proceedings, arXiv, or
  publisher URLs.
- `bibliography_ledger.md`: what each citation supports and what it does not.
- `theory.tex`: standalone LaTeX synopsis that compiles independently of the
  manuscript.

## Claim-status convention

- **PROVED HERE** means every step is supplied in `theorem_targets.md`. These
  are elementary reductions or impossibility arguments, not novelty claims.
- **CONDITIONAL COROLLARY** means the deduction is complete once the named
  module contracts hold.
- **TARGET (UNPROVED)** means a proposed theorem whose missing lemmas are
  itemized. It must not be described as established.
- **EXTERNAL RESULT** means only the cited paper establishes the result; its
  assumptions must be checked before use.

## Current result inventory

Proved here:

1. A conditional modular PAC reduction from correct segmentation, matching,
   per-mode stationary RFE, and deployment identification.
2. A deployment-error decomposition under total episodic reward at most one.
3. A zero-diagnostic impossibility, even for two deterministic
   transition-only one-step modes.
4. Binary diagnostic lower bounds from total variation and
   Bretagnolle--Huber, stated for any fixed adaptive diagnostic strategy.
5. A finite-family Bhattacharyya upper bound for exact-model MAP diagnosis.
6. A precise distinction between the single-mode RFE lower bound that embeds
   immediately and the \(M\)-fold direct-sum statement that remains unproved.

Not proved here:

1. An end-to-end finite-sample segmentation and recurrence-matching theorem for
   DME.
2. A sharp controlled-information characterization or an efficient solver for
   the max-min diagnostic problem.
3. A minimax direct-sum stationary-RFE lower bound with sharp tabular
   dependence.
4. Matching dwell-time and switch-overhead lower bounds.
5. Robustness of diagnostic design to estimated confidence sets without a
   separation margin.

The unproved items are precise research targets, not claims about the repository
or the literature.

## Reading order

Start with `formal_model.md`, then `algorithm.md` and
`theorem_targets.md`.  The roadmap records the missing end-to-end lemmas and
the bibliography ledger states exactly what each external citation can and
cannot justify.
