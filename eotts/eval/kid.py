"""Kernel Inception Distance between generated and ground-truth tiles.

Not in the TerraDiT paper; added because KID is unbiased at small sample sizes, where FID
is biased upward, so it is the distribution metric we use on the Best-of-N subsets.
Same Inception-v3 features (torchmetrics) and image handling as the repo's FID.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


def _load_u8(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)


def _find(d, s):
    for ext in (".png", ".jpg"):
        p = os.path.join(d, s + ext)
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def compute_kid(gen_dir, gt_dir, stems, device, batch_size=64, subset_size=1000, subsets=100):
    from torchmetrics.image.kid import KernelInceptionDistance
    pairs = [(g, t) for s in stems if (g := _find(gen_dir, s)) and (t := _find(gt_dir, s))]
    if len(pairs) < 50:
        return {}
    k = min(subset_size, len(pairs))
    kid = KernelInceptionDistance(feature=2048, subset_size=k, subsets=subsets).to(device)
    size = Image.open(pairs[0][1]).size
    for real, col in ((True, 1), (False, 0)):
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"kid[{'real' if real else 'gen'}]"):
            x = torch.stack([_load_u8(p[col], size) for p in pairs[i:i + batch_size]]).to(device)
            kid.update(x, real=real)
    mean, std = kid.compute()
    return {"kid": float(mean), "kid_std": float(std), "kid_subset_size": k}
