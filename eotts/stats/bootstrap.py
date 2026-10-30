"""Paired bootstrap over tiles, two-sided bootstrap p-values, and Holm correction."""
from __future__ import annotations

import numpy as np


def tile_weights(n_tiles: int, n_resamples: int, seed: int) -> np.ndarray:
    """(B, n) multinomial counts: resample n tiles with replacement, B times."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_tiles, np.full(n_tiles, 1.0 / n_tiles), size=n_resamples).astype(np.int32)


def paired_mean_diff(a: np.ndarray, b: np.ndarray, weights: np.ndarray, alpha: float = 0.05) -> dict:
    """Mean of a − b over tiles with a percentile bootstrap CI. a, b: per-tile values."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    boot = weights @ d / weights.sum(1)
    return summarize(float(d.mean()), boot, alpha)


def summarize(estimate: float, boot: np.ndarray, alpha: float = 0.05) -> dict:
    """Percentile CI and two-sided bootstrap p-value, 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)), floored at 1/B.

    Resamples whose statistic is undefined (only possible with a handful of tiles, e.g. a smoke
    test where every draw is the same tile) are dropped and counted.
    """
    boot = np.asarray(boot, np.float64)
    ok = np.isfinite(boot)
    b = boot[ok]
    if len(b) == 0:
        return {"estimate": estimate, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_boot": float("nan"), "n_resamples": 0, "n_dropped": int((~ok).sum())}
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return {"estimate": estimate, "ci_low": float(lo), "ci_high": float(hi),
            "p_boot": float(min(1.0, max(p, 1.0 / len(b)))), "n_resamples": int(len(b)),
            "n_dropped": int((~ok).sum())}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down: adjusted p-values and reject decisions, keyed like the input."""
    keys = sorted(pvalues, key=pvalues.get)
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = {"p": pvalues[k], "p_holm": running, "reject": running < alpha}
    return out
