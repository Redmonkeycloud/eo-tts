"""The evaluation sets of the Best-of-N protocol (docs/experiment_protocol.md)."""
from __future__ import annotations

import hashlib
import json
import os

from eotts.provenance import REPO_ROOT

SETS = {
    "R2000": dict(split="random", n=2000, list=os.path.join(REPO_ROOT, "configs", "subsets", "random_2000.json")),
    "S500": dict(split="spatial", n=500, list=os.path.join(REPO_ROOT, "configs", "subsets", "spatial_500.json")),
}
NUM_SAMPLES = 16
N_VALUES = (1, 2, 4, 8, 16)


def sample_seed(split: str, hf_idx: int, k: int) -> int:
    """Seed of sample k of a tile: first 8 bytes of sha256("{split}:{hf_idx}:{k}"), as int63."""
    h = hashlib.sha256(f"{split}:{int(hf_idx)}:{int(k)}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def load_set(name: str, data_root: str) -> dict:
    """Tiles of a set, in protocol order, with their row index in the split metadata."""
    spec = SETS[name]
    listed = json.load(open(spec["list"]))
    if listed["n"] != spec["n"] or len(listed["tiles"]) != spec["n"]:
        raise ValueError(f"{spec['list']} does not hold {spec['n']} tiles")
    rows = json.load(open(os.path.join(data_root, "splits", spec["split"], "metadata.json")))
    pos = {r["img_name"]: i for i, r in enumerate(rows)}
    tiles = []
    for t in listed["tiles"]:
        i = pos.get(t["img_name"])
        if i is None or int(rows[i]["hf_idx"]) != int(t["hf_idx"]):
            raise ValueError(f"tile {t['img_name']} not in {spec['split']} metadata as listed")
        tiles.append({"img_name": t["img_name"], "hf_idx": int(t["hf_idx"]), "row": i,
                      "stem": os.path.splitext(t["img_name"])[0]})
    return {"name": name, "split": spec["split"], "tiles": tiles}


def sample_path(run_dir: str, stem: str, k: int) -> str:
    return os.path.join(run_dir, "samples", stem, f"k{k:02d}.png")
