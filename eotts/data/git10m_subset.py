"""Exact, small local copy of the Git-10M rows a test split needs.

TerraDiT reads imagery and captions from ``lcybuaa/Git-10M`` at a pinned revision
(~394 GB, 787 Arrow shards) by row index (``hf_idx``). The test splits only touch a
handful of those shards (random: 151 shards, ~29 GB; spatial: 2 shards, ~0.3 GB), so we
download one shard at a time, keep only the rows the split needs, verify every row's
``img_name`` against the split metadata, write them to a small Parquet part, and delete
the shard. Peak extra disk is one shard (~0.3 GB).

``Git10MSubset`` then behaves like the ``datasets`` object TerraDiT expects:
``subset[hf_idx] -> {"image": PIL.Image, "caption": str, "img_name": str}``. Because the
pixels and captions are the same bytes as in the pinned snapshot (checked by img_name),
everything downstream (generation, ground-truth tiles, metrics) is unchanged.

Shard layout: ``datasets.save_to_disk`` splits the 10,503,567 rows contiguously into
787 shards, the first ``N mod 787`` shards one row longer. Verified on shard 308
(2,716 / 2,716 spatial-split rows matched by img_name).
"""
from __future__ import annotations

import bisect
import gc
import io
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass

GIT10M_REPO = "lcybuaa/Git-10M"
GIT10M_REVISION = "29f192b8d2aa28b5d4d8c8d7f0f608cdc61fb52f"
GIT10M_NUM_ROWS = 10_503_567
GIT10M_NUM_SHARDS = 787

_DIV, _MOD = divmod(GIT10M_NUM_ROWS, GIT10M_NUM_SHARDS)
SHARD_STARTS = [_DIV * i + min(i, _MOD) for i in range(GIT10M_NUM_SHARDS + 1)]


def shard_of(hf_idx: int) -> int:
    """Shard index holding global row ``hf_idx``."""
    if not 0 <= hf_idx < GIT10M_NUM_ROWS:
        raise IndexError(f"hf_idx {hf_idx} outside Git-10M (0..{GIT10M_NUM_ROWS - 1})")
    return bisect.bisect_right(SHARD_STARTS, hf_idx) - 1


def shard_num_rows(shard: int) -> int:
    return SHARD_STARTS[shard + 1] - SHARD_STARTS[shard]


def shard_filename(shard: int) -> str:
    return f"train/data-{shard:05d}-of-{GIT10M_NUM_SHARDS:05d}.arrow"


def group_by_shard(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        out[shard_of(int(r["hf_idx"]))].append(r)
    return dict(sorted(out.items()))


def plan(rows: list[dict], repo_id=GIT10M_REPO, revision=GIT10M_REVISION) -> dict:
    """Which shards a set of rows needs, and how many bytes they are (one Hub listing call)."""
    groups = group_by_shard(rows)
    sizes = {}
    try:
        from huggingface_hub import HfApi
        tree = HfApi().list_repo_tree(repo_id, repo_type="dataset", revision=revision,
                                      path_in_repo="train")
        sizes = {f.path: f.size for f in tree if hasattr(f, "size")}
    except Exception as e:  # offline: plan without sizes
        print(f"[subset] could not list shard sizes ({type(e).__name__}); continuing")
    total = sum(sizes.get(shard_filename(s), 0) for s in groups)
    return {"num_rows": len(rows), "num_shards": len(groups), "shards": list(groups),
            "download_bytes": total}


def _read_shard(path: str):
    """Read an Arrow IPC stream shard fully into memory (no mmap, so the file can be deleted on Windows)."""
    import pyarrow as pa
    with pa.OSFile(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all()


def _extract(table, shard: int, rows: list[dict]):
    """Take the split's rows out of one shard table, verifying img_name. Returns a pyarrow Table."""
    import pyarrow as pa
    expected = shard_num_rows(shard)
    if table.num_rows != expected:
        raise ValueError(f"shard {shard}: {table.num_rows} rows, expected {expected}; "
                         f"the shard layout assumption does not hold for this revision")
    local = [int(r["hf_idx"]) - SHARD_STARTS[shard] for r in rows]
    took = table.take(pa.array(local, type=pa.int64()))
    names = took.column("img_name").to_pylist()
    for r, n in zip(rows, names):
        if n != r["img_name"]:
            raise ValueError(f"hf_idx {r['hf_idx']}: shard has {n!r}, split metadata has "
                             f"{r['img_name']!r}")
    images = took.column("image").to_pylist()
    missing = [r["hf_idx"] for r, im in zip(rows, images) if not im or not im.get("bytes")]
    if missing:
        raise ValueError(f"shard {shard}: {len(missing)} rows carry no embedded image bytes "
                         f"(first hf_idx {missing[0]})")
    return pa.table({
        "hf_idx": pa.array([int(r["hf_idx"]) for r in rows], type=pa.int64()),
        "img_name": pa.array(names, type=pa.string()),
        "caption": took.column("caption"),
        "image_bytes": pa.array([im["bytes"] for im in images], type=pa.binary()),
    })


def build_subset(rows: list[dict], out_dir: str, *, repo_id=GIT10M_REPO, revision=GIT10M_REVISION,
                 tmp_dir: str | None = None, keep_shards: bool = False, token: str | None = None):
    """Download the shards ``rows`` need, one at a time, and write ``out_dir/parts/*.parquet``.

    Resumable: a shard whose part already exists with the right row count is skipped.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tqdm.auto import tqdm

    parts = os.path.join(out_dir, "parts")
    tmp_dir = tmp_dir or os.path.join(out_dir, "_shard_tmp")
    os.makedirs(parts, exist_ok=True)
    groups = group_by_shard(rows)
    done = 0
    for shard, srows in tqdm(groups.items(), desc="shards"):
        part = os.path.join(parts, f"shard-{shard:05d}.parquet")
        if os.path.exists(part) and pq.ParquetFile(part).metadata.num_rows == len(srows):
            done += 1
            continue
        path = hf_hub_download(repo_id, shard_filename(shard), repo_type="dataset",
                               revision=revision, local_dir=tmp_dir, token=token)
        table = _read_shard(path)
        sub = _extract(table, shard, srows)
        del table
        gc.collect()
        pq.write_table(sub, part + ".tmp", compression="zstd")
        os.replace(part + ".tmp", part)
        if not keep_shards:
            os.remove(path)
        done += 1
    if not keep_shards and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = {"repo_id": repo_id, "revision": revision, "num_rows": len(rows),
                "num_shards": len(groups), "shards": list(groups),
                "hf_idx_sorted": sorted(int(r["hf_idx"]) for r in rows)}
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"))
    print(f"[subset] {len(rows)} rows from {len(groups)} shards -> {out_dir}")
    return out_dir


@dataclass
class _Row:
    img_name: str
    caption: str
    image_bytes: bytes


class Git10MSubset:
    """Drop-in for the Git-10M ``train`` split, restricted to the rows built above."""

    def __init__(self, subset_dir: str):
        import pyarrow.parquet as pq
        parts = os.path.join(subset_dir, "parts")
        if not os.path.isdir(parts):
            raise FileNotFoundError(f"{parts} missing; build it with scripts/03_build_git10m_subset.py")
        self.rows: dict[int, _Row] = {}
        for f in sorted(os.listdir(parts)):
            if not f.endswith(".parquet"):
                continue
            t = pq.read_table(os.path.join(parts, f))
            for i, name, cap, b in zip(t.column("hf_idx").to_pylist(), t.column("img_name").to_pylist(),
                                       t.column("caption").to_pylist(), t.column("image_bytes").to_pylist()):
                self.rows[int(i)] = _Row(name, cap or "", b)
        self.subset_dir = subset_dir

    def __len__(self):
        return len(self.rows)

    def __contains__(self, hf_idx):
        return int(hf_idx) in self.rows

    def __getitem__(self, hf_idx):
        from PIL import Image
        try:
            r = self.rows[int(hf_idx)]
        except KeyError:
            raise KeyError(f"hf_idx {hf_idx} not in subset {self.subset_dir}; rebuild it for this split") from None
        return {"image": Image.open(io.BytesIO(r.image_bytes)), "caption": r.caption,
                "img_name": r.img_name}

    def check_covers(self, rows: list[dict]):
        missing = [r["hf_idx"] for r in rows if int(r["hf_idx"]) not in self.rows]
        if missing:
            raise KeyError(f"{len(missing)} split rows missing from {self.subset_dir} "
                           f"(first hf_idx {missing[0]}); rebuild the subset")
