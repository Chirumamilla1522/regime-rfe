# Theorems, proofs, and research targets

Throughout, rewards obey (R), so every value and policy gap lies in \([0,1]\).

## Proved in the manuscript

The following are proved in `paper/regime_rfe.tex` (Appendix B):

1. Finite-horizon simulation under (R) (Lemma sim).
2. Diagnostic KL chain rule (Lemma chain).
3. Pinsker and Le Cam comparisons (Lemma info).
4. Inconsistency of pooled kernels (Theorem pool).
5. Boundary contamination identity (Theorem contam).
6. Unique certified confidence-set match (Lemma match).
7. Recurrence versus restart accounting (Lemma account).
8. Conditional modular PAC reduction (Theorem modular).
9. Deployment-error decomposition (Theorem decomp).
10. Rates under SEG (Corollary rate).
11. Known-model ML diagnosis (Proposition aff).
12. Known-model diagnostic sample complexity (Theorem known-rate).
13. Probe-stream two-window test for i.i.d. observations (Lemma twowindow).
14. SEG for probe-window DME (Theorem seg-probe), under an explicit window
    length, rollback quarantine, and episodic i.i.d. probes. This does **not**
    cover the short-window residual heuristic.
15. Robust diagnosis under a TV margin (Proposition robust).
16. Zero-diagnostic impossibility (Theorem zero).
17. Finite-diagnostic testing lower bound (Theorem diag-lb).
18. Occupancy obstruction, self-contained (Theorem occupancy).
19. Behavioral unidentifiability (Theorem behavior).

## Open targets (not theorems)

* **Target 1:** adaptive direct-sum lower bound of order
  \(M N^\star_{\rm stat}\). Multiplying Jin et al. by \(M\) is not a proof.
* **Target 2:** matching minimax dwell / switch-overhead lower bounds.
* **Not claimed:** SEG for the short-window residual detector used in
  RFE-Recurrent-Bench.

## Direct-sum note

Let \(N_{\rm stat}^\star(S,A,H,\varepsilon,\delta;\mathfrak P)\) denote the
minimax episode complexity of the *single-mode* stationary RFE problem under
exactly the same horizon convention, reward normalization, confidence level,
and kernel class \(\mathfrak P\).

Restricting the recurring-regime class to one deployment-eligible mode shows
that any algorithm needs at least \(N_{\rm stat}^\star\) episodes. Occupancy
obstruction shows that an under-sampled mode blocks PAC. Neither is an
\(M N_{\rm stat}^\star\) total lower bound.
