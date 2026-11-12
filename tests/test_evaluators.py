"""PairwiseEval (stateless, batched) must give the values of the upstream per-image code.

The upstream TerraDiT code (terradit/eval/metrics.py, compute_pairwise) calls the stateful
torchmetrics LPIPS module once per image and the SSIM module once per batch. PairwiseEval
calls the same LPIPS network batched and the functional SSIM. This test checks the values
agree, and that the cost per call does not grow with the number of calls.
"""
import time

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("torchmetrics")


def _images(n, seed, size=64):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):    # smooth, image-like content rather than white noise
        small = rng.integers(0, 256, (size // 8, size // 8, 3), dtype=np.uint8)
        out.append(Image.fromarray(small).resize((size, size), Image.BICUBIC))
    return out


def test_pairwise_matches_upstream_per_image():
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from eotts.scoring.evaluators import PairwiseEval, to_u8

    gen, gt = _images(12, 0), _images(12, 1)
    lp, ss = PairwiseEval("cpu")(gen, gt)

    lp_mod = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
    ss_mod = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
    g = torch.stack([to_u8(i) for i in gen]).float() / 255
    t = torch.stack([to_u8(i) for i in gt]).float() / 255
    with torch.no_grad():
        lp_ref = np.array([float(lp_mod(2 * a[None] - 1, 2 * b[None] - 1)) for a, b in zip(g, t)])
        ss_ref = ss_mod(g, t).numpy()
    assert np.abs(lp - lp_ref).max() < 1e-5      # float32 conv order differs between batch sizes
    assert np.abs(ss - ss_ref).max() < 1e-7


def test_pairwise_cost_does_not_grow():
    from eotts.scoring.evaluators import PairwiseEval
    ev = PairwiseEval("cpu")
    gen, gt = _images(4, 2), _images(4, 3)
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        ev(gen, gt)
        times.append(time.perf_counter() - t0)
    first, last = np.median(times[20:70]), np.median(times[-50:])
    assert last < 1.5 * first, (first, last)
