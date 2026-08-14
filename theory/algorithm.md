# Detect--Match--Explore (DME): interface-level algorithm

This is pseudocode for a modular algorithm.  It is not an assertion that the
suggested detector satisfies the segmentation contract.

```text
Inputs: K, Mbar, probe policy family, detector DETECT,
        matcher MATCH, stationary routine RFE, diagnostic designer DESIGN
State: candidate block buffer W; quarantine Q; libraries L = empty

for training episode k = 1,...,K:
    choose either a declared probe policy or the policy requested by RFE
    execute it without rewards and append the trajectory to W
    update DETECT using only predictable policies and observed transitions

    if DETECT alarms:
        quarantine the declared pre/post-alarm windows
        close the accepted portion of W
        form a time-valid confidence set Cnew from accepted data
        i = MATCH(Cnew, L)                 # recurrence or new mode
        if MATCH is ambiguous: quarantine/collect probes; do not pool
        else: pool the accepted block into library entry i
        reset W according to DETECT's stated reset convention

close the final block in the same way
if a computable module certificate fails: return ABSTAIN
# SEG(B,n,delta_seg) is an analysis event involving hidden truth, not a
# runtime-checkable predicate

for each library entry i:
    run/finalize RFE on its pure assigned episodes, producing planner Pi_i

choose sigma_hat = DESIGN(C_1,...,C_Mhat,q) using the robust objective (RDI)
if its certified pairwise information is zero: return ABSTAIN
execute sigma_hat for q fresh deployment episodes
identify m_hat by a declared robust likelihood/test rule
if no label is certified: return REJECT/UNSEEN

after the full reward r is revealed:
    return Pi_m_hat(r)
```

## Required interfaces

`DETECT` must state its filtration, false-alarm probability, delay/localization
event, reset rule, and behavior near the first and last block.  `MATCH` must
be permutation-invariant and may pool data only on a joint confidence event.
`RFE` must satisfy the contract in `formal_model.md`; arbitrary data collected
by a detector is not automatically valid input.  `DESIGN` must specify whether
it solves the exact-model objective (DI), the robust objective (RDI), or an
approximation with a certified factor.

All ambiguous outcomes abstain.  Silently assigning an uncertain block or
deployment transcript would invalidate purity and the modular theorem.

## Budget convention

For a true block \([b,e]\), at most the first and last \(B\) episodes are
quarantined.  Probe, matching, and useful RFE episodes in the remaining
interior are disjoint sets.  A sufficient (not necessary) dwell condition is
\[
D_{\min}\ge 2B+L_{\rm probe}+L_{\rm match}+L_{\rm useful}.
\]
Deriving these quantities for a concrete detector is a **TARGET
(UNPROVED)**.
