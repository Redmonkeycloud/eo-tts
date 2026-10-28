"""Reproduce the TerraDiT paper evaluation with the pinned upstream code.

Generation and metrics are TerraDiT's own functions (``terradit/generate.py`` and
``terradit/eval/metrics.py`` at the submodule commit), called in-process. The one change:
the Git-10M object they index by ``hf_idx`` is our ``Git10MSubset`` (same bytes, verified
by img_name) instead of the 394 GB ``datasets`` download.

Known differences from the paper's stated protocol, recorded in every result file:
  * batch size: paper 32, repo default 16 (noise is drawn per batch, so it changes the
    exact samples, not the distribution); configurable.
  * seed: not reported in the paper; we use the repo default 42.
  * CLIP: the repo reports the CLIP logit (cosine x 100); the paper reports cosine.
    We store both (``clip_score`` and ``clip_cos``).
  * sFID: reported in the paper, not computed by the repo; not reported here.
"""
from __future__ import annotations

import json
import os
import sys
import time

from eotts.data.git10m_subset import Git10MSubset
from eotts.provenance import ensure_terradit_importable, provenance, REPO_ROOT

PAPER_TARGETS = os.path.join(REPO_ROOT, "configs", "paper_targets.json")
FAMILY = {"alpha_xl": "alpha", "sigma_xl": "sigma", "omega_xl": "omega", "omega_base": "omega"}


def _paper_target(model, cond, split):
    try:
        return json.load(open(PAPER_TARGETS))[model][cond][split]
    except KeyError:
        return None


def run_baseline(*, model="omega_xl", split="random", condition_type="omega",
                 data_root="data/terradit", subset_root="data/git10m_subset",
                 weights_root="data/weights", out_root="runs/baseline", num_steps=100,
                 batch_size=16, seed=42, limit=None, metrics=("fid", "clip", "lpips", "ssim"),
                 kid=True, regenerate=False, metrics_only=False):
    ensure_terradit_importable()
    import torch
    import terradit.generate as tg
    from terradit.hf import resolve_checkpoint
    from terradit.eval.metrics import (load_split_rows, attach_captions, ensure_gt_tiles,
                                       evaluate_folder, stem)

    family = FAMILY[model]
    cond = condition_type if family == "omega" else "text"
    run = f"{model}_{split}_{cond}" + (f"_n{limit}" if limit else "")
    gen_dir = os.path.join(out_root, run, "images")
    res_path = os.path.join(out_root, run, "results.json")
    os.makedirs(gen_dir, exist_ok=True)

    rows = load_split_rows(data_root, split)
    if limit:
        rows = rows[:limit]
    subset = Git10MSubset(os.path.join(subset_root, split))
    subset.check_covers(rows)
    ckpt = os.path.dirname(resolve_checkpoint(model, checkpoints_root=weights_root))

    timings = {}
    have = sum(os.path.exists(os.path.join(gen_dir, stem(r["img_name"]) + ".png")) for r in rows)
    if not metrics_only and (regenerate or have < len(rows)):
        tg.load_git10m = lambda *a, **k: subset          # the only substitution
        argv = ["generate.py", "--family", family, "--split", split, "--ckpt", ckpt,
                "--data-root", data_root, "--out-dir", gen_dir, "--num-steps", str(num_steps),
                "--batch-size", str(batch_size), "--seed", str(seed)]
        if family == "omega":
            argv += ["--condition-type", condition_type]
        if limit:
            argv += ["--limit", str(limit)]
        print("[baseline] generating:", " ".join(argv[1:]))
        old_argv, sys.argv = sys.argv, argv
        t0 = time.time()
        try:
            tg.main()
        finally:
            sys.argv = old_argv
        timings["generation_s"] = round(time.time() - t0, 1)
        timings["s_per_image"] = round(timings["generation_s"] / len(rows), 3)
        torch.cuda.empty_cache()
    else:
        print(f"[baseline] {have}/{len(rows)} images already in {gen_dir}; skipping generation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attach_captions(rows, subset)
    gt_dir = ensure_gt_tiles(rows, os.path.join(data_root, "gt", split), hf=subset)
    t0 = time.time()
    res = evaluate_folder(gen_dir, rows, gt_dir, device, tuple(metrics))
    if "clip_score" in res:
        res["clip_cos"] = res["clip_score"] / 100.0
    if kid:
        from eotts.eval.kid import compute_kid
        res.update(compute_kid(gen_dir, gt_dir, [stem(r["img_name"]) for r in rows], device))
    timings["metrics_s"] = round(time.time() - t0, 1)

    target = _paper_target(model, cond, split)
    out = {
        "run": run, "model": model, "family": family, "split": split, "condition_type": cond,
        "metrics": res,
        "paper": target,
        "delta_vs_paper": {k: round(res[k] - v, 4) for k, v in (target or {}).items() if k in res},
        "protocol": {"num_steps": num_steps, "cfg_scale": 0.0, "sampler": "euler (terradit)",
                     "batch_size": batch_size, "seed": seed, "dtype": "fp32 (terradit default)",
                     "num_tiles": len(rows), "limit": limit,
                     "image_source": "Git10MSubset (img_name-verified rows of the pinned snapshot)"},
        "timings": timings,
        "provenance": provenance(),
    }
    if limit:
        out["note"] = "limited run: FID on a few hundred tiles is biased upward and not comparable to the paper"
    json.dump(out, open(res_path, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("run", "metrics", "paper", "delta_vs_paper")}, indent=2))
    print(f"[baseline] -> {res_path}")
    return out
