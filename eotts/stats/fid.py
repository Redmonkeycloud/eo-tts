"""FID from cached Inception features, with the same formula as torchmetrics.

FID = ||mu_g - mu_r||² + tr(S_g + S_r - 2 (S_g S_r)^½), covariances with the n−1 normaliser,
tr((S_g S_r)^½) computed as the sum of square roots of the eigenvalues of S_g S_r
(torchmetrics.image.fid._compute_fid).
"""
from __future__ import annotations

import numpy as np


def gaussian_stats(feats: np.ndarray):
    x = np.asarray(feats, np.float64)
    return x.mean(0), np.cov(x, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2) -> float:
    a = float(((mu1 - mu2) ** 2).sum())
    b = float(np.trace(s1) + np.trace(s2))
    c = float(np.sqrt(np.linalg.eigvals(s1 @ s2).astype(np.complex128)).real.sum())
    return a + b - 2 * c


def fid(fake: np.ndarray, real: np.ndarray) -> float:
    if len(fake) < 2 or len(real) < 2:      # covariance undefined (only in tiny smoke tests)
        return float("nan")
    return frechet_distance(*gaussian_stats(fake), *gaussian_stats(real))
