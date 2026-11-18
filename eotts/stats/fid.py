"""FID from cached Inception features, with the same formula as torchmetrics, and its tile bootstrap.

FID = ||mu_g - mu_r||² + tr(S_g + S_r - 2 (S_g S_r)^½), covariances with the n−1 normaliser.
``fid`` computes tr((S_g S_r)^½) as torchmetrics does (sum of square roots of the eigenvalues
of S_g S_r). The bootstrap uses an equivalent symmetric form that is faster and numerically
steadier: with A = S_r^½, S_g S_r = S_g A² has the same eigenvalues as A S_g A, which is
symmetric positive semi-definite, so tr((S_g S_r)^½) = Σ √eigvalsh(A S_g A). The two forms
agree to floating-point error (tests/test_bon_stats.py).

Small sets (fewer distinct tiles in a resample than feature dimensions, e.g. 500 or 2,000 tiles
with 2,048 Inception features) use an exact n × n form instead: with M_ij = √w_i (f_i − μ_f)·(r_j − μ_r) √w_j,
the nonzero eigenvalues of S_g S_r are those of M Mᵀ / (Σw − 1)², so tr((S_g S_r)^½) is the sum of the
singular values of M divided by Σw − 1. The cost then scales with the number of tiles, not with
2,048³ (on S500 over 100× faster). Both forms are tested against plain FID. With singular
covariances the torchmetrics form also sums square roots of rounding noise in the null space;
the n × n form differs from it by about 1e-6 relative (2e-5 on an FID of 60).

Tile resampling: weight w_i = how often tile i is drawn. The weighted mean and the weighted
covariance with normaliser (Σw − 1) equal the plain mean and covariance of the data with
each row repeated w_i times, i.e. exactly FID on the resampled set.
"""
from __future__ import annotations

import numpy as np


def gaussian_stats(feats):
    x = np.asarray(feats, np.float64)
    return x.mean(0), np.cov(x, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2) -> float:
    a = float(((mu1 - mu2) ** 2).sum())
    b = float(np.trace(s1) + np.trace(s2))
    c = float(np.sqrt(np.linalg.eigvals(s1 @ s2).astype(np.complex128)).real.sum())
    return a + b - 2 * c


def fid(fake, real) -> float:
    if len(fake) < 2 or len(real) < 2:      # covariance undefined (only in tiny smoke tests)
        return float("nan")
    return frechet_distance(*gaussian_stats(fake), *gaussian_stats(real))


def _weighted_stats(xc: np.ndarray, w: np.ndarray):
    """Mean and covariance of rows of xc repeated w times. xc should be roughly centred."""
    n = w.sum()
    mu = w @ xc / n
    cov = (xc.T * w) @ xc
    cov -= n * np.outer(mu, mu)
    cov /= n - 1
    return mu, cov


def _sqrt_psd(s: np.ndarray) -> np.ndarray:
    ev, vec = np.linalg.eigh(s)
    return (vec * np.sqrt(np.clip(ev, 0, None))) @ vec.T


def _fid_sym(mu_f, s_f, mu_r, s_r, a_r) -> float:
    m = a_r @ s_f @ a_r
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu_f - mu_r) ** 2).sum() + np.trace(s_f) + np.trace(s_r) - 2 * tr_sqrt)


def fid_resampled(fakes: dict, real: np.ndarray, weights: np.ndarray, progress=None) -> dict:
    """FID of each fake set against the real set on every tile resample.

    fakes: name -> (n, d) features, row i belonging to tile i, as real. weights: (B, n) counts.
    Returns name -> (B,) array. All fakes share the real statistics of each resample, so
    differences between fakes are paired.
    """
    real = np.asarray(real, np.float64)
    centre = real.mean(0)
    rc = real - centre
    fc = {k: np.asarray(v, np.float64) - centre for k, v in fakes.items()}
    out = {k: np.empty(len(weights)) for k in fakes}
    if len(real) < real.shape[1]:
        return _fid_resampled_gram(fc, rc, np.asarray(weights, np.float64), out, progress)
    for b, w in enumerate(np.asarray(weights, np.float64)):
        if w.sum() < 2:
            for k in fakes:
                out[k][b] = np.nan
            continue
        mu_r, s_r = _weighted_stats(rc, w)
        a_r = _sqrt_psd(s_r)
        for k, x in fc.items():
            mu_f, s_f = _weighted_stats(x, w)
            out[k][b] = _fid_sym(mu_f, s_f, mu_r, s_r, a_r)
        if progress and ((b + 1) % 50 == 0 or b + 1 == len(weights)):
            progress(f"fid resample {b + 1}/{len(weights)}")
    return out


def _fid_resampled_gram(fc, rc, weights, out, progress):
    """Exact n × n form for sets with fewer tiles than feature dimensions (see module doc)."""
    gram = {k: x @ rc.T for k, x in fc.items()}            # (n, n) once per fake set
    rsq = (rc * rc).sum(1)
    fsq = {k: (x * x).sum(1) for k, x in fc.items()}
    for b, w in enumerate(weights):
        n = w.sum()
        if n < 2:
            for k in out:
                out[k][b] = np.nan
            continue
        idx = np.flatnonzero(w)
        wi, sw = w[idx], np.sqrt(w[idx])
        r = rc[idx]
        mu_r = wi @ r / n
        tr_r = (wi @ rsq[idx] - n * mu_r @ mu_r) / (n - 1)
        for k, x in fc.items():
            f = x[idx]
            mu_f = wi @ f / n
            tr_f = (wi @ fsq[k][idx] - n * mu_f @ mu_f) / (n - 1)
            m = gram[k][np.ix_(idx, idx)] - (f @ mu_r)[:, None] - (r @ mu_f)[None, :] + mu_f @ mu_r
            m *= sw[:, None]
            m *= sw[None, :]
            ev = np.linalg.eigvalsh(m @ m.T)
            tr_sqrt = np.sqrt(np.clip(ev, 0, None)).sum() / (n - 1)
            out[k][b] = float(((mu_f - mu_r) ** 2).sum() + tr_f + tr_r - 2 * tr_sqrt)
        if progress and ((b + 1) % 50 == 0 or b + 1 == len(weights)):
            progress(f"fid resample {b + 1}/{len(weights)}")
    return out


def fid_full_sample(fakes: dict, real: np.ndarray) -> dict:
    """Same quantity on the original sample (all weights 1)."""
    r = fid_resampled(fakes, real, np.ones((1, len(real))))
    return {k: float(v[0]) for k, v in r.items()}
