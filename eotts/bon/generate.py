"""Generate K samples per tile with TerraDiT-Ω, one seeded generator per sample.

The model, conditioning, sampler and decoder are the upstream TerraDiT functions at the
pinned commit, exactly as in the baseline. Two things differ from the baseline, both set by
the protocol: each sample's initial noise comes from its own CPU generator seeded by
(split, hf_idx, k), and sampling runs under bf16 autocast unless fp32 is requested.

Resumable: a (tile, k) whose PNG exists is skipped. Samples are written one PNG each.
"""
from __future__ import annotations

import json
import os
import time

from eotts.bon.sets import load_set, sample_path, sample_seed
from eotts.data.git10m_subset import Git10MSubset
from eotts.provenance import ensure_terradit_importable, provenance

FAMILY = {"omega_xl": "omega", "omega_base": "omega"}


def generate(*, set_name, run_dir, model="omega_xl", num_samples=16, sample_ids=None,
             precision="bf16", data_root="data/terradit", subset_root="data/git10m_subset",
             weights_root="data/weights", batch_size=32, num_steps=100, limit_tiles=None):
    ensure_terradit_importable()
    import torch
    from torch.utils.data._utils.collate import default_collate
    from terradit.generation import build_inference_model, load_text_vae, decode_latents, save_images
    from terradit.data.dataset import build_dataset
    from terradit.sampling.samplers import euler_sampler
    from terradit.training.train_terradit import build_model_kwargs
    from terradit.generate import split_paths
    from terradit.hf import resolve_checkpoint

    if precision not in ("bf16", "fp32"):
        raise ValueError(precision)
    ks = list(sample_ids) if sample_ids is not None else list(range(num_samples))
    s = load_set(set_name, data_root)
    tiles = s["tiles"][:limit_tiles] if limit_tiles else s["tiles"]
    todo = [(t, k) for t in tiles for k in ks if not os.path.exists(sample_path(run_dir, t["stem"], k))]

    os.makedirs(run_dir, exist_ok=True)
    meta_path = os.path.join(run_dir, "generation.json")
    meta = {"set": set_name, "split": s["split"], "model": model, "precision": precision,
            "sample_ids": ks, "num_tiles": len(tiles), "num_steps": num_steps, "cfg_scale": 0.0,
            "condition_type": "omega", "batch_size": batch_size,
            "seed_rule": 'first 8 bytes of sha256("{split}:{hf_idx}:{k}"), CPU torch.Generator',
            "provenance_at_start": provenance(), "runs": []}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path, encoding="utf-8"))
        for key in ("set", "model", "precision", "num_steps"):
            if old.get(key) != meta[key]:
                raise ValueError(f"{run_dir} was generated with {key}={old.get(key)!r}, now {meta[key]!r}")
        meta["runs"] = old.get("runs", [])
        meta["provenance_at_start"] = old["provenance_at_start"]
    run_entry = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pending_at_start": len(todo),
                 "provenance": provenance()}
    meta["runs"].append(run_entry)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"[bon] {set_name}: {len(tiles)} tiles x {len(ks)} samples, {len(todo)} to generate -> {run_dir}")
    if not todo:
        return meta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    family = FAMILY[model]
    ckpt = os.path.dirname(resolve_checkpoint(model, checkpoints_root=weights_root))
    net = build_inference_model(family, None, ckpt, device)
    _, clip, vae = load_text_vae(device)
    sp = split_paths(data_root, s["split"])
    ds = build_dataset(family, data_root, hf_dataset=Git10MSubset(os.path.join(subset_root, s["split"])),
                       split="test", vae_on_the_fly=True, metadata_path=sp["metadata_path"],
                       inst_meta_path=sp["inst_meta_path"], loc_embed_path=sp["loc_embed_path"])

    cache = {}

    def item(row):
        if row not in cache:
            cache.clear()
            it = ds[row]
            if os.path.splitext(it[-1])[0] != os.path.splitext(ds.metadata[row]["img_name"])[0]:
                raise ValueError("dataset row mismatch")
            cache[row] = [x for x in it if torch.is_tensor(x)]
        return cache[row]

    if precision == "bf16" and device == "cuda":
        ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        ctx = lambda: torch.autocast(device if device == "cuda" else "cpu", enabled=False)

    t0, done = time.time(), 0
    for b in range(0, len(todo), batch_size):
        chunk = todo[b:b + batch_size]
        batch = default_collate([item(t["row"]) for t, _ in chunk])
        _, _, kw = build_model_kwargs(family, batch, clip, device, 0, 1)
        kw["dropout_probs"] = None                       # condition_type omega: every primitive
        noise = torch.stack([
            torch.randn(net.in_channels, 32, 32,
                        generator=torch.Generator().manual_seed(sample_seed(s["split"], t["hf_idx"], k)))
            for t, k in chunk]).to(device)
        with torch.no_grad(), ctx():
            lat = euler_sampler(net, noise, num_steps=num_steps, cfg_scale=0.0, path_type="linear", **kw)
        imgs = decode_latents(vae, lat.to(torch.float32), device)
        paths = [sample_path(run_dir, t["stem"], k) for t, k in chunk]
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        save_images(imgs, paths)
        done += len(chunk)
        if (b // batch_size) % 10 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  {el / done:.2f} s/image  eta {(len(todo) - done) * el / done / 3600:.1f} h")

    run_entry["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_entry["s_per_image"] = round((time.time() - t0) / max(done, 1), 3)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    return meta
