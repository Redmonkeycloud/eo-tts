"""EXPLORATORY: FID per selection rule and N on R2000 and S500 (not a protocol endpoint).

    python scripts/09_exploratory_fid.py                    # point estimates, a few minutes
    python scripts/09_exploratory_fid.py --bootstrap 100    # + CIs for N=16 vs N=1 (slow on CPU)

Reads the cached Inception features and verifier scores of the scored runs; no model is
loaded and nothing is generated. Writes results/bon/exploratory_fid.json and .md.

Why exploratory: the protocol reports KID, not FID, on these subsets, because FID is biased
upward at small n and the bias also depends on the distribution, so even differences
between rules at the same n can mislead (Chong & Forsyth, CVPR 2020). The numbers here
are not comparable with the paper's or our full-split FID (9.81 on 14,932 tiles). As a
same-size reference, the fp32 baseline images on the same R2000 tiles are scored too.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.select import RULES, V1, V2, select, take
from eotts.bon.sets import N_VALUES
from eotts.provenance import REPO_ROOT, provenance
from eotts.scoring.score_run import load_scores
from eotts.stats.bootstrap import summarize, tile_weights
from eotts.stats.fid import fid

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", default="runs/bon")
ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "bon"))
ap.add_argument("--bootstrap", type=int, default=0, help="resamples for the N=16 vs N=1 CIs (0 = off)")
ap.add_argument("--seed", type=int, default=2468)
ap.add_argument("--suffix", default="", help="e.g. _smoke (writes under runs/ instead of results/)")
args = ap.parse_args()


def boot_fid_diff(a, b, real, weights):
    """FID(a) − FID(b) on tile resamples (rows of a, b, real are paired by tile)."""
    out = []
    for w in weights:
        idx = np.repeat(np.arange(len(w)), w)
        out.append(fid(a[idx], real[idx]) - fid(b[idx], real[idx]))
    return np.array(out)


results = {"note": "EXPLORATORY, not a protocol endpoint. FID at n = 2,000 / 500 is biased upward and "
                   "not comparable with full-split FID (paper 9.25, our baseline 9.81). Compare values "
                   "only within one set.",
           "sets": {}, "provenance": provenance()}

for set_name in ("R2000", "S500"):
    d = os.path.join(args.runs_root, set_name + args.suffix)
    if not os.path.isdir(os.path.join(d, "scores")):
        print(f"[fid] {set_name}: no scores, skipped")
        continue
    t0 = time.time()
    sc = load_scores(d)
    feats, real = sc["inception.samples"], sc["inception.gt"]
    vs = {V1: sc[f"verifiers.{V1}"], V2: sc[f"verifiers.{V2}"]}
    T = len(real)
    r = {"tiles": T, "rules": {}}

    # sampling floor: FID between two random halves of the real tiles, at n/2 each
    perm = np.random.default_rng(args.seed).permutation(T)
    r["fid_real_half_vs_half"] = fid(real[perm[: T // 2]], real[perm[T // 2:]])

    for rule in RULES:
        r["rules"][rule] = {}
        for n in N_VALUES:
            idx = select(rule, vs, n)
            r["rules"][rule][str(n)] = {"fid": fid(take(feats, idx), real)}
        print(f"[fid] {set_name} {rule}: " + ", ".join(
            f"N={n} {r['rules'][rule][str(n)]['fid']:.2f}" for n in N_VALUES))

    if args.bootstrap:
        W = tile_weights(T, args.bootstrap, args.seed)
        k0 = feats[:, 0]
        for rule in RULES:
            chosen = take(feats, select(rule, vs, 16))
            est = r["rules"][rule]["16"]["fid"] - r["rules"][rule]["1"]["fid"]
            r["rules"][rule]["16"]["diff_vs_n1"] = summarize(est, boot_fid_diff(chosen, k0, real, W))
            print(f"[fid] {set_name} {rule}: N=16 − N=1 CI done")

    base = os.path.join(args.runs_root, "R2000_baseline_fp32" + args.suffix)
    if set_name == "R2000" and os.path.isdir(os.path.join(base, "scores")):
        bs = load_scores(base)
        r["fp32_baseline_same_tiles"] = fid(bs["inception.samples"][:, 0], bs["inception.gt"])
    r["seconds"] = round(time.time() - t0, 1)
    results["sets"][set_name] = r

if args.suffix:
    args.out = os.path.join(args.runs_root, "analysis" + args.suffix)
os.makedirs(args.out, exist_ok=True)
json.dump(results, open(os.path.join(args.out, "exploratory_fid.json"), "w", encoding="utf-8"), indent=1)

L = ["# Exploratory FID on the Best-of-N subsets", "",
     "**Not a protocol endpoint.** FID at these sample sizes is biased upward, so these values are "
     "not comparable with the paper (9.25) or our full-split baseline (9.81). Compare only within "
     "a set. The pre-registered distribution metric is KID ([bon_results.md](../../docs/bon_results.md)).", ""]
for set_name, r in results["sets"].items():
    L += [f"## {set_name} ({r['tiles']} tiles)", "",
          "| Rule | " + " | ".join(f"N={n}" for n in N_VALUES) + " | N=16 − N=1 |",
          "| --- |" + " --- |" * (len(N_VALUES) + 1)]
    for rule in RULES:
        c = r["rules"][rule]
        diff = c["16"].get("diff_vs_n1")
        dtxt = (f"{diff['estimate']:+.2f} [{diff['ci_low']:+.2f}, {diff['ci_high']:+.2f}]" if diff
                else f"{c['16']['fid'] - c['1']['fid']:+.2f}")
        L.append(f"| {rule} | " + " | ".join(f"{c[str(n)]['fid']:.2f}" for n in N_VALUES) + f" | {dtxt} |")
    L += ["", f"Sampling floor, FID between two random halves of the real tiles: {r['fid_real_half_vs_half']:.2f}."]
    if "fp32_baseline_same_tiles" in r:
        L.append(f"fp32 baseline images on the same tiles (same-size reference for N = 1): "
                 f"{r['fp32_baseline_same_tiles']:.2f}.")
    L.append("")
open(os.path.join(args.out, "exploratory_fid.md"), "w", encoding="utf-8").write("\n".join(L))
print(f"[fid] -> {args.out}/exploratory_fid.json, exploratory_fid.md")
