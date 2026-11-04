"""Scaling curves of the Best-of-N results (reads results/bon/*.json only).

    python scripts/08_plot_bon.py          # -> docs/assets/bon_scaling.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RULES = {"v1": ("V1 RemoteCLIP-B/32", "#4C78A8"), "v2": ("V2 GeoRSCLIP-L/14", "#E45756"),
         "v1+v2": ("V1+V2 rank average", "#54A24B")}
PANELS = [("clip_cos", "Δ CLIP score (cosine)", 1.0), ("kid_dino", "Δ KID-DINOv2 ×10³", 1e3),
          ("lpips", "Δ LPIPS (lower is better)", 1.0), ("ssim", "Δ SSIM", 1.0)]

sets = {s: json.load(open(os.path.join(ROOT, "results", "bon", f"{s}.json"), encoding="utf-8"))
        for s in ("R2000", "S500") if os.path.exists(os.path.join(ROOT, "results", "bon", f"{s}.json"))}
if not sets:
    sys.exit("no results/bon/*.json")

fig, axes = plt.subplots(len(PANELS), len(sets), figsize=(5.2 * len(sets), 3.0 * len(PANELS)),
                         sharex=True, squeeze=False)
for j, (sname, r) in enumerate(sets.items()):
    ns = [int(n) for n in r["n_values"]]
    for i, (metric, label, scale) in enumerate(PANELS):
        ax = axes[i][j]
        ax.axhline(0, color="#999", lw=0.8)
        for rule, (name, color) in RULES.items():
            d = [r["rules"][rule][str(n)][metric]["diff_vs_n1"] for n in ns]
            est = [x["estimate"] * scale for x in d]
            lo = [x["ci_low"] * scale for x in d]
            hi = [x["ci_high"] * scale for x in d]
            ax.plot(ns, est, marker="o", ms=3.5, lw=1.6, color=color, label=name)
            ax.fill_between(ns, lo, hi, color=color, alpha=0.15, lw=0)
        if metric in r.get("oracle", {}):
            base = r["oracle"][metric]["1"]
            ax.plot(ns, [(r["oracle"][metric][str(n)] - base) * scale for n in ns], ls="--", lw=1,
                    color="#555", label="oracle (upper bound)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        if i == 0:
            ax.set_title(f"{sname} ({r['tiles']} tiles)", fontsize=11)
        if j == 0:
            ax.set_ylabel(label, fontsize=9)
        if i == len(PANELS) - 1:
            ax.set_xlabel("N (candidates per tile)")
        ax.grid(alpha=0.25, lw=0.5)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Best-of-N on TerraDiT-Ω-XL: change vs N = 1 (95 % paired bootstrap CI)", fontsize=12)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = os.path.join(ROOT, "docs", "assets", "bon_scaling.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print(f"[plot] -> {out}")
