"""Qualitative figure: what each verifier picks out of 16 samples (Study 2, Ω-B).

    python scripts/17_examples_v2.py --set S500 R2000            # -> docs/assets/bon_v2_examples_<set>.png
    python scripts/17_examples_v2.py --set R2000 --tiles 6 --seed 1

For --tiles tiles drawn at random from the set (fixed seed, so nobody picks flattering
examples), one row per tile: the real image, sample 0 (N = 1), and the sample chosen at N = 16 by
V1, V2 and, when its scores exist, V3. The caption is printed above each row. Reads the cached
scores, the sample PNGs and the Git-10M subset; no model is loaded.
"""
import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.select import V1, V2, V3, select
from eotts.bon.sets import load_set, sample_path
from eotts.data.git10m_subset import Git10MSubset
from eotts.provenance import REPO_ROOT
from eotts.scoring.score_run import load_scores

ap = argparse.ArgumentParser()
ap.add_argument("--set", nargs="+", default=["S500", "R2000"])
ap.add_argument("--tiles", type=int, default=4)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--v3", default="v3_internvl3_2b")
ap.add_argument("--runs-root", default="runs/bon")
ap.add_argument("--suffix", default="")
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "assets"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for set_name in args.set:
    run = os.path.join(args.runs_root, f"{set_name}_omega_base{args.suffix}")
    sc = load_scores(run)
    vs = {V1: sc[f"verifiers.{V1}"], V2: sc[f"verifiers.{V2}"]}
    cols = [("V1 RemoteCLIP", "v1"), ("V2 GeoRSCLIP", "v2")]
    if f"vlm_{args.v3}.samples" in sc:
        vs[V3] = sc[f"vlm_{args.v3}.samples"]
        cols.append(("V3 VLM", "v3"))
    s = load_set(set_name, args.data_root)
    tiles = s["tiles"][:len(vs[V1])]
    sub = Git10MSubset(os.path.join(args.subset_root, s["split"]))
    rows = sorted(np.random.default_rng(args.seed).choice(len(tiles), min(args.tiles, len(tiles)), replace=False))
    picks = {rule: select(rule, vs, args.n) for _, rule in cols}

    ncol = 2 + len(cols)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 3.05 * len(rows)), squeeze=False)
    for i, t in enumerate(rows):
        tile = tiles[t]
        item = sub[tile["hf_idx"]]
        panels = [("Real image", item["image"].convert("RGB")),
                  ("Sample 0 (N = 1)", Image.open(sample_path(run, tile["stem"], 0)))]
        for label, rule in cols:
            k = int(picks[rule][t])
            panels.append((f"{label}: best of {args.n} (k={k})", Image.open(sample_path(run, tile["stem"], k))))
        for j, (label, im) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=8)
        cap = textwrap.shorten(item["caption"], width=230, placeholder=" …")
        axes[i][0].text(0, -0.2, "\n".join(textwrap.wrap(cap, 115)), transform=axes[i][0].transAxes,
                        fontsize=7, va="bottom", ha="left")
    fig.suptitle(f"{set_name}: the sample each verifier selects (TerraDiT-Ω-B; tiles drawn at random, seed "
                 f"{args.seed})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=3.2)
    out = os.path.join(args.out, f"bon_v2_examples_{set_name}{args.suffix}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[examples] {set_name}: tiles {[tiles[t]['stem'] for t in rows]} -> {out}")
