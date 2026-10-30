import numpy as np
import pytest
import torch

from eotts.bon.select import best_of, oracle, rank_average, select, take, V1, V2
from eotts.bon.sets import sample_seed
from eotts.stats.bootstrap import holm, paired_mean_diff, tile_weights
from eotts.stats.kid import kid, mmd2_unbiased, poly_kernel, resampled_mmd2_parts


def test_seeds_deterministic_and_distinct():
    assert sample_seed("random", 123, 0) == sample_seed("random", 123, 0)
    seeds = {sample_seed(s, i, k) for s in ("random", "spatial") for i in range(50) for k in range(16)}
    assert len(seeds) == 2 * 50 * 16
    assert all(0 <= x < 2**63 for x in seeds)


def test_mmd2_matches_torchmetrics():
    from torchmetrics.image.kid import poly_mmd
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 64)), rng.normal(0.2, 1, size=(300, 64))
    ours = mmd2_unbiased(poly_kernel(a, a), poly_kernel(a, b), poly_kernel(b, b))
    theirs = float(poly_mmd(torch.tensor(a), torch.tensor(b)))
    assert ours == pytest.approx(theirs, rel=1e-9)


def test_resampled_parts_reduce_to_full_sample():
    rng = np.random.default_rng(1)
    f, r = rng.normal(size=(200, 32)), rng.normal(0.3, 1, size=(200, 32))
    part = resampled_mmd2_parts(f, r, np.ones((1, 200)))[0]
    krr = poly_kernel(r, r)
    real_term = (krr.sum() - np.trace(krr)) / (200 * 199)
    assert part + real_term == pytest.approx(mmd2_unbiased(poly_kernel(r, r), poly_kernel(r, f), poly_kernel(f, f)))


def test_resampled_parts_exclude_same_tile_pairs():
    rng = np.random.default_rng(2)
    f, r = rng.normal(size=(30, 8)), rng.normal(size=(30, 8))
    w = np.zeros((1, 30)); w[0, 0] = 2; w[0, 1:29] = 1      # tile 0 twice, tile 29 dropped
    idx = [0, 0] + list(range(1, 29))
    fi, ri = f[idx], r[idx]
    kff, kfr = poly_kernel(fi, fi), poly_kernel(fi, ri)
    same = np.equal.outer(idx, idx)
    expected = kff[~same].mean() - 2 * kfr.mean()
    assert resampled_mmd2_parts(f, r, w)[0] == pytest.approx(expected)


def test_kid_detects_shift():
    rng = np.random.default_rng(3)
    real = rng.normal(size=(500, 16))
    assert kid(rng.normal(size=(500, 16)), real, subset_size=200, subsets=20)["kid"] < \
           kid(rng.normal(1.0, 1, size=(500, 16)), real, subset_size=200, subsets=20)["kid"]


def test_bootstrap_ci_covers_and_holm():
    rng = np.random.default_rng(4)
    a = rng.normal(0.1, 1, 2000); b = rng.normal(0, 1, 2000)
    w = tile_weights(2000, 2000, 0)
    assert (w.sum(1) == 2000).all()
    d = paired_mean_diff(a, b, w)
    assert d["ci_low"] < d["estimate"] < d["ci_high"]
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"]["p_holm"] == pytest.approx(0.03) and h["c"]["p_holm"] == pytest.approx(0.06)
    assert h["b"]["p_holm"] == pytest.approx(0.06) and h["a"]["reject"] and not h["b"]["reject"]


def test_selection_rules_nested_and_correct():
    s1 = np.array([[0.1, 0.9, 0.5, 0.2], [0.4, 0.3, 0.8, 0.1]])
    s2 = np.array([[0.2, 0.1, 0.9, 0.3], [0.9, 0.2, 0.1, 0.8]])
    vs = {V1: s1, V2: s2}
    assert select("v1", vs, 1).tolist() == [0, 0]
    assert best_of(s1, 2).tolist() == [1, 0]
    assert select("v2", vs, 4).tolist() == [2, 0]
    # tile 0, n=4: ranks v1 [1,4,3,2], v2 [2,1,4,3] -> mean [1.5,2.5,3.5,2.5] -> sample 2
    # tile 1, n=4: ranks v1 [3,2,4,1], v2 [4,2,1,3] -> mean [3.5,2,2.5,2]   -> sample 0
    assert rank_average(s1, s2, 4).tolist() == [2, 0]
    # tie in mean rank broken by v2: n=2 tile 0: v1 ranks [1,2], v2 ranks [2,1] -> tie, v2 prefers 0
    assert rank_average(s1, s2, 2).tolist()[0] == 0
    assert oracle(s1, 4, False).tolist() == [0, 3]
    assert take(s1, np.array([1, 2])).tolist() == [0.9, 0.8]
