# Bibliography ledger

All URLs below are public and directly verifiable.

## Reward-free exploration

**Jin, Krishnamurthy, Simchowitz, and Yu (2020), “Reward-Free
Exploration for Reinforcement Learning.”**

* Proceedings: https://proceedings.mlr.press/v119/jin20d.html
* arXiv: https://arxiv.org/abs/2002.02794
* Supports: the reward-free protocol and Theorem 4.1's single-mode expected
  lower bound, including its explicit parameter restrictions.
* Does not support: hidden segmentation, recurring-mode matching, deployment
  diagnosis, or an \(M\)-fold direct sum.

**Kaufmann, Ménard, Domingues, Jonsson, Leurent, and Valko (2021),
“Adaptive Reward-Free Exploration.”**

* Proceedings: https://proceedings.mlr.press/v132/kaufmann21a.html
* Supports: adaptive model-estimation-based RFE upper bounds and discussion of
  time-homogeneous versus time-inhomogeneous horizon dependence.
* Does not support: using arbitrarily selected detector data as valid input or
  recurring hidden regimes.

**Zhang, Du, and Ji (2021), “Nearly Minimax Optimal Reward-free
Reinforcement Learning.”**

* arXiv: https://arxiv.org/abs/2010.05901
* Supports: near-minimax stationary/time-homogeneous RFE under the paper's
  total-reward normalization.
* Does not support: time-inhomogeneous rates or a multimode direct sum.

## Information inequalities and sequential validity

**Bretagnolle and Huber (1978), “Estimation des densités: risque
minimax.”**

* Archive: https://www.numdam.org/item/SPS_1978__12__342_0/
* DOI: https://doi.org/10.1007/BFb0064610
* Supports: the KL--TV inequality used in Theorem 5.

**Canonne (2023), “A short note on an inequality between KL and TV.”**

* arXiv: https://arxiv.org/abs/2202.07198
* Supports: a modern self-contained proof and precise form of the
  Bretagnolle--Huber inequality.

**Howard, Ramdas, McAuliffe, and Sekhon (2021), “Time-uniform,
nonparametric, nonasymptotic confidence sequences.”**

* Journal/DOI: https://doi.org/10.1214/20-AOS1991
* arXiv: https://arxiv.org/abs/1810.08240
* Supports: general confidence-sequence methodology under continuous
  monitoring.
* Does not by itself prove multinomial detector or matching guarantees here.

## Citation discipline

External rates may be imported only after matching kernel stationarity,
episode/horizon conventions, reward scale, expected versus high-probability
sample complexity, and the required confidence level.  The package makes no
priority or novelty claim based solely on these references.
