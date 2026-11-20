"""Figures for Study 2, from results/bon_v2/<set>.json (no model, no images).

    python scripts/16_plot_v2.py                     # -> docs/assets/bon_v2_scaling.png, bon_v2_tradeoff.png
    python scripts/16_plot_v2.py --sets S500 R2000   # before RFULL exists

Figure 1, scaling: FID, CLIP score and KID-DINOv2 of the selected images at N = 1 … 16 for every
verifier, one column per set. Bands: 95 % CI of the difference to sample 0, drawn around the
value. Dotted line: the average single draw (amendment 3).

Figure 2, trade-off at N = 16: change in CLIP score (x) against change in FID and in KID-DINOv2
(y), both against the average single draw, with 95 % CIs, one point per verifier and set.
Right means better caption agreement, down means more realistic images.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", default=["S500", "R2000", "RFULL"])
ap.add_argument("--dir", default=os.path.join(ROOT, "results", "bon_v2"))
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "assets"))
args = ap.parse_args()

RULES = {"v1": ("V1 RemoteCLIP", "#4C78A8", "o"), "v2": ("V2 GeoRSCLIP", "#E45756", "s"),
         "v1+v2": ("V1+V2", "#B279A2", "D"), "v3": ("V3 VLM (secondary)", "#54A24B", "^"),
         "v1+v3": ("V1+V3 (secondary)", "#F58518", "v")}
TITLE = {"S500": "S500 (500 Git-Spatial tiles)", "R2000": "R2000 (2,000 Git-Rand tiles)",
         "RFULL": "RFULL (14,932 Git-Rand tiles)"}
SET_MARK = {"S500": "o", "R2000": "s", "RFULL": "D"}

sets = {}
for s in args.sets:
    p = os.path.join(args.dir, f"{s}.json")
    if os.path.exists(p):
        sets[s] = json.load(open(p, encoding="utf-8"))
if not sets:
    sys.exit(f"no <set>.json in {args.dir}")
os.makedirs(args.out, exist_ok=True)


def value(cell, m):
    return {"fid": lambda: cell["fid"]["full_sample"], "clip_cos": lambda: cell["clip_cos"]["mean"],
            "kid_dino": lambda: cell["kid_dino"]["mmd2_full_sample"] * 1e3}[m]()


def mean_draw(r, m):
    if m == "fid":
        return r.get("mean_draw_fid_full_sample")
    if m == "kid_dino":
        return r["mean_draw_kid_full_sample"]["kid_dino"] * 1e3 if "mean_draw_kid_full_sample" in r else None
    c = r["rules"]["v1"]["16"]["clip_cos"]
    return c["mean"] - c["diff_vs_mean_draw"]["estimate"] if "diff_vs_mean_draw" in c else None


# ---------- figure 1: scaling ----------
PANELS = [("fid", "FID ↓"), ("clip_cos", "CLIP score ↑"), ("kid_dino", "KID-DINOv2 ×10³ ↓")]
fig, axes = plt.subplots(len(PANELS), len(sets), figsize=(5.0 * len(sets), 3.1 * len(PANELS)), squeeze=False)
for j, (s, r) in enumerate(sets.items()):
    ns = [int(n) for n in r["n_values"]]
    for i, (m, label) in enumerate(PANELS):
        ax = axes[i][j]
        scale = 1e3 if m == "kid_dino" else 1.0
        for rule in r["rules"]:
            name, color, mk = RULES.get(rule, (rule, "#777", "o"))
            cells = [r["rules"][rule][str(n)] for n in ns]
            vals = [value(c, m) for c in cells]
            ref = vals[0]
            lo = [ref + c[m]["diff_vs_n1"]["ci_low"] * scale for c in cells]
            hi = [ref + c[m]["diff_vs_n1"]["ci_high"] * scale for c in cells]
            ls = "--" if rule in ("v3", "v1+v3") else "-"
            ax.plot(ns, vals, marker=mk, ms=4, lw=1.6, ls=ls, color=color, label=name)
            ax.fill_between(ns, lo, hi, color=color, alpha=0.10, lw=0)
        md = mean_draw(r, m)
        if md is not None:
            ax.axhline(md, color="#333", lw=0.9, ls=":", label="average single draw (N = 1)")
        if m in ("clip_cos",) and "oracle" in r:
            ax.plot(ns, [r["oracle"]["clip_cos"][str(n)] for n in ns], color="#999", lw=1, ls="-.",
                    label="oracle (upper bound)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [f"{n}\n({100 * n:,})" for n in ns])
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(TITLE.get(s, s), fontsize=11)
        if j == 0:
            ax.set_ylabel(label, fontsize=10)
        if i == len(PANELS) - 1:
            ax.set_xlabel("N candidates per tile (NFE per tile = 100 · N)")
handles, labels = axes[1][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Best-of-N with TerraDiT-Ω-B: metric values vs N (bands: 95 % CI of the change from sample 0)",
             fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
f1 = os.path.join(args.out, "bon_v2_scaling.png")
fig.savefig(f1, dpi=150)
plt.close(fig)

# ---------- figure 2: trade-off ----------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, (m, label, scale) in zip(axes, [("fid", "Δ FID vs average draw ↓", 1.0),
                                         ("kid_dino", "Δ KID-DINOv2 ×10³ vs average draw ↓", 1e3)]):
    for s, r in sets.items():
        for rule in r["rules"]:
            c = r["rules"][rule]["16"]
            if "diff_vs_mean_draw" not in c["clip_cos"]:
                continue
            name, color, _ = RULES.get(rule, (rule, "#777", "o"))
            x, y = c["clip_cos"]["diff_vs_mean_draw"], c[m]["diff_vs_mean_draw"]
            ax.errorbar(x["estimate"], y["estimate"] * scale,
                        xerr=[[x["estimate"] - x["ci_low"]], [x["ci_high"] - x["estimate"]]],
                        yerr=[[(y["estimate"] - y["ci_low"]) * scale], [(y["ci_high"] - y["estimate"]) * scale]],
                        fmt=SET_MARK.get(s, "o"), color=color, ms=7, lw=1, capsize=2,
                        mfc=color if s != "S500" else "white")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Δ CLIP score vs average draw ↑")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25, lw=0.5)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, marker="o", ls="", label=n) for n, c, _ in RULES.values()]
h += [Line2D([], [], color="#333", marker=SET_MARK[s], ls="", mfc="white" if s == "S500" else "#333",
             label=TITLE.get(s, s)) for s in sets]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.suptitle("Best-of-N at N = 16: caption agreement vs realism (95 % CIs; right and down is better)", fontsize=12)
fig.tight_layout(rect=(0, 0.13, 1, 0.95))
f2 = os.path.join(args.out, "bon_v2_tradeoff.png")
fig.savefig(f2, dpi=150)
plt.close(fig)
print(f"[plot] -> {f1}, {f2}")
