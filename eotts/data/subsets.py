"""Fixed evaluation subsets of the TerraDiT test splits, and checks that they are representative.

A subset of size n is the first n entries of one seeded permutation of the split's row
order. Larger subsets therefore contain smaller ones, and the selection depends only on the
seed and the split metadata, never on any result.
"""
from __future__ import annotations

import json
import os

import numpy as np

PERMUTATION_SEED = 0


def permutation(num_rows: int, seed: int = PERMUTATION_SEED) -> np.ndarray:
    return np.random.default_rng(seed).permutation(num_rows)


def subset_rows(rows: list[dict], n: int, seed: int = PERMUTATION_SEED) -> list[dict]:
    if n > len(rows):
        raise ValueError(f"subset of {n} from a split of {len(rows)} rows")
    return [rows[i] for i in permutation(len(rows), seed)[:n]]


def tile_stats(rows_idx: np.ndarray, rows: list[dict], inst: dict) -> dict:
    """Descriptive statistics of a set of tiles (indices into the split's row order)."""
    start, end = inst["tile_inst_start"][rows_idx], inst["tile_inst_end"][rows_idx]
    n_inst = (end - start).astype(int)
    gtype = inst["geom_type"]
    types = np.concatenate([gtype[s:e] for s, e in zip(start, end)]) if n_inst.sum() else np.array([], int)
    lat = np.array([rows[i]["latitude"] for i in rows_idx])
    lon = np.array([rows[i]["longitude"] for i in rows_idx])
    return {
        "tiles": int(len(rows_idx)),
        "instances_per_tile_mean": float(n_inst.mean()),
        "instances_per_tile_median": float(np.median(n_inst)),
        "instances_per_tile_p90": float(np.percentile(n_inst, 90)),
        "empty_tile_fraction": float((n_inst == 0).mean()),
        "dense_tile_fraction_ge15": float((n_inst >= 15).mean()),
        "geometry_type_share": {int(t): float((types == t).mean()) for t in np.unique(gtype)},
        "lat_mean": float(lat.mean()), "lat_std": float(lat.std()),
        "lon_mean": float(lon.mean()), "lon_std": float(lon.std()),
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
    }


def tag_distribution(rows_idx: np.ndarray, inst: dict, top: int = 20) -> dict[int, float]:
    start, end = inst["tile_inst_start"][rows_idx], inst["tile_inst_end"][rows_idx]
    if (end - start).sum() == 0:
        return {}
    ptr, ids = inst["tag_ids_ptr"], inst["tag_ids"]
    first_tag = np.concatenate([ids[ptr[s:e]] for s, e in zip(start, end) if e > s])
    vals, counts = np.unique(first_tag, return_counts=True)
    order = np.argsort(-counts)[:top]
    return {int(vals[i]): float(counts[i] / counts.sum()) for i in order}


def representativeness(split_dir: str, n: int, seed: int = PERMUTATION_SEED) -> dict:
    """Subset vs full split on the statistics above, plus the overlap of the top-20 tags."""
    rows = json.load(open(os.path.join(split_dir, "metadata.json"), encoding="utf-8"))
    inst = dict(np.load(os.path.join(split_dir, "inst_metadata.npz")))
    full_idx = np.arange(len(rows))
    sub_idx = permutation(len(rows), seed)[:n]
    full_tags = tag_distribution(full_idx, inst)
    sub_tags = tag_distribution(sub_idx, inst)
    return {
        "seed": seed, "n": n,
        "full": tile_stats(full_idx, rows, inst),
        "subset": tile_stats(sub_idx, rows, inst),
        "top20_tag_share_full": full_tags,
        "top20_tag_share_subset": sub_tags,
        "top20_tag_overlap": len(set(full_tags) & set(sub_tags)),
        "total_variation_top20": 0.5 * sum(abs(full_tags.get(k, 0) - sub_tags.get(k, 0))
                                           for k in set(full_tags) | set(sub_tags)),
    }


def write_subset_list(split_dir: str, n: int, out_path: str, seed: int = PERMUTATION_SEED):
    rows = json.load(open(os.path.join(split_dir, "metadata.json"), encoding="utf-8"))
    sub = subset_rows(rows, n, seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"seed": seed, "n": n, "split_rows": len(rows),
               "tiles": [{"img_name": r["img_name"], "hf_idx": int(r["hf_idx"])} for r in sub]},
              open(out_path, "w", encoding="utf-8"), indent=1)
    return out_path
