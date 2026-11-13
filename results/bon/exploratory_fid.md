# Exploratory FID on the Best-of-N subsets

**Not a protocol endpoint.** FID at these sample sizes is biased upward, so these values are not comparable with the paper (9.25) or our full-split baseline (9.81). Compare only within a set. The pre-registered distribution metric is KID ([bon_results.md](../../docs/bon_results.md)).

## R2000 (2000 tiles)

| Rule | N=1 | N=2 | N=4 | N=8 | N=16 | N=16 − N=1 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 29.38 | 28.99 | 29.33 | 28.86 | 28.79 | -0.59 [-1.41, +0.03] |
| v2 | 29.38 | 29.05 | 29.05 | 29.41 | 29.90 | +0.51 [-0.58, +1.26] |
| v1+v2 | 29.38 | 29.05 | 28.90 | 28.66 | 29.45 | +0.06 [-1.01, +0.77] |

Sampling floor, FID between two random halves of the real tiles: 42.67.
fp32 baseline images on the same tiles (same-size reference for N = 1): 29.04.

## S500 (500 tiles)

| Rule | N=1 | N=2 | N=4 | N=8 | N=16 | N=16 − N=1 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 68.83 | 68.28 | 66.29 | 66.66 | 66.73 | -2.10 [-5.57, +1.74] |
| v2 | 68.83 | 68.15 | 66.55 | 67.48 | 68.80 | -0.03 [-3.60, +3.77] |
| v1+v2 | 68.83 | 68.15 | 66.67 | 66.52 | 66.81 | -2.02 [-6.14, +1.09] |

Sampling floor, FID between two random halves of the real tiles: 74.31.
