"""KID from cached features, and the kernel quantities the bootstrap needs.

Kernel and estimator follow torchmetrics.image.kid exactly: cubic polynomial kernel
k(x, y) = (x·y / d + 1)^3 and the unbiased MMD² estimator, averaged over random subsets.
"""
from __future__ import annotations

import numpy as np


def poly_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a.shape[1]
    return (a.astype(np.float64) @ b.astype(np.float64).T / d + 1.0) ** 3


def mmd2_unbiased(kxx, kxy, kyy) -> float:
    m = kxx.shape[0]
    xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(xx + yy - 2 * kxy.sum() / (m * m))


def kid(fake: np.ndarray, real: np.ndarray, subset_size: int = 1000, subsets: int = 100, seed: int = 0):
    """KID mean and std over subsets, as torchmetrics computes it (independent subsets of each set)."""
    rng = np.random.default_rng(seed)
    k = min(subset_size, len(fake), len(real))
    vals = []
    for _ in range(subsets):
        f = fake[rng.permutation(len(fake))[:k]]
        r = real[rng.permutation(len(real))[:k]]
        vals.append(mmd2_unbiased(poly_kernel(r, r), poly_kernel(r, f), poly_kernel(f, f)))
    vals = np.array(vals)
    return {"kid": float(vals.mean()), "kid_std": float(vals.std()), "kid_subset_size": int(k)}


def resampled_mmd2_parts(fake: np.ndarray, real: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fake-dependent part of the unbiased MMD² under tile resampling, one value per resample.

    ``weights`` is (B, n): how many times each tile is drawn in each bootstrap resample. Tiles
    are paired (fake i and real i belong to tile i). Within-set pairs drawn from the same tile
    are excluded, as the unbiased estimator excludes i == j.

    Returns  mean_{i≠j} k(f_i, f_j) − 2 · mean_{i,j} k(f_i, r_j)  for each resample. The real-only
    term is the same for every selection rule on the same resample, so differences between
    rules are exact differences of MMD².
    """
    kff = poly_kernel(fake, fake)
    kfr = poly_kernel(fake, real)
    w = weights.astype(np.float64)                      # (B, n)
    n = w.sum(1)                                        # n per resample (constant)
    ff = ((w @ kff) * w).sum(1) - (w * w * np.diag(kff)[None]).sum(1)
    ff_count = n * n - (w * w).sum(1)
    fr = ((w @ kfr) * w).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ff_count > 0, ff / np.where(ff_count > 0, ff_count, 1), np.nan) - 2 * fr / (n * n)


def full_sample_mmd2_part(fake: np.ndarray, real: np.ndarray) -> float:
    """Same quantity on the original sample (all weights 1)."""
    return float(resampled_mmd2_parts(fake, real, np.ones((1, len(fake))))[0])
