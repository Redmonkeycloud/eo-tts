"""POST-HOC diagnostic: how much do FID and KID move between independent draws of the same model?

    python scripts/13_null_draws.py --runs R2000_omega_base S500_omega_base
    python scripts/13_null_draws.py --runs R2000_omega_base R2000      # also the Ω-XL pilot

Written after the stop rule of protocol v2 triggered on R2000 (anchor 2: FID of BoN sample 0,
bf16, vs the fp32 baseline images, two different noise draws; |ΔFID| = 1.13 > 1.06, the half-width
of the narrowest N = 16 FID CI). It is not part of the pre-registered analysis and is labelled
post-hoc wherever it is reported.

Each tile has 16 samples drawn with independent seeds. Taking sample k of every tile gives 16
independent "draws" of the whole set from the same model, with no selection at all. The spread
of FID (and KID) across these 16 draws is the size of difference that two draws of the same
model show by chance at this n. That is the right yardstick for anchor 2, and it is also the
null distribution for a selection rule that picks an arbitrary sample instead of using a
verifier.

Reads cached scores only. Writes results/bon_v2/null_draws.json and .md.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.provenance import REPO_ROOT, provenance
from eotts.scoring.score_run import load_scores
from eotts.stats.fid import fid
from eotts.stats.kid import mmd2_unbiased, poly_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["R2000_omega_base", "S500_omega_base"])
ap.add_argument("--runs-root", default="runs/bon")
ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "bon_v2"))
args = ap.parse_args()


def kid_full(fake, real):
    return mmd2_unbiased(poly_kernel(fake, fake), poly_kernel(fake, real), poly_kernel(real, real))


def spread(values):
    v = np.asarray(values)
    pair = np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), 1)]
    return {"per_draw": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "diff_vs_draw0": (v[1:] - v[0]).tolist(),
            "abs_pairwise_diff": {"median": float(np.median(pair)), "p90": float(np.percentile(pair, 90)),
                                  "p95": float(np.percentile(pair, 95)), "max": float(pair.max()),
                                  "pairs": int(len(pair))}}


res = {"note": "POST-HOC diagnostic after the protocol v2 stop rule triggered on R2000; not pre-registered. "
               "Draw k = sample k of every tile (independent seeds, same model, no selection).",
       "runs": {}}
for run in args.runs:
    d = os.path.join(args.runs_root, run)
    if not os.path.isdir(os.path.join(d, "scores")):
        print(f"[null] {run}: no scores, skipped")
        continue
    sc = load_scores(d)
    inc, inc_gt = sc["inception.samples"], sc["inception.gt"]
    dino, dino_gt = sc["dino.samples"], sc["dino.gt"]
    K = inc.shape[1]
    r = {"tiles": int(inc.shape[0]), "draws": int(K),
         "fid": spread([fid(inc[:, k], inc_gt) for k in range(K)]),
         "kid_inception": spread([kid_full(inc[:, k].astype(np.float64), inc_gt.astype(np.float64)) for k in range(K)]),
         "kid_dino": spread([kid_full(dino[:, k].astype(np.float64), dino_gt.astype(np.float64)) for k in range(K)])}
    res["runs"][run] = r
    f = r["fid"]
    print(f"[null] {run}: FID over {K} draws {f['mean']:.2f} ± {f['sd']:.2f} (min {f['min']:.2f}, max {f['max']:.2f}); "
          f"|pairwise diff| median {f['abs_pairwise_diff']['median']:.2f}, p95 {f['abs_pairwise_diff']['p95']:.2f}")

res["provenance"] = provenance()
os.makedirs(args.out, exist_ok=True)
json.dump(res, open(os.path.join(args.out, "null_draws.json"), "w", encoding="utf-8"), indent=1)

L = ["# POST-HOC: FID and KID across independent draws of the same model", "",
     "Not pre-registered. Written after the stop rule triggered on R2000 (protocol v2, Reports). "
     "Draw k = sample k of every tile: 16 independent draws of the set from the same model, no selection.", "",
     "| Run | Tiles | FID mean ± SD over draws | FID min – max | abs. difference between two draws: median / p95 / max | "
     "KID-Inception ×10³ mean ± SD | KID-DINOv2 ×10³ mean ± SD |",
     "| --- | --- | --- | --- | --- | --- | --- |"]
for run, r in res["runs"].items():
    f, a = r["fid"], r["fid"]["abs_pairwise_diff"]
    L.append(f"| {run} | {r['tiles']} | {f['mean']:.2f} ± {f['sd']:.2f} | {f['min']:.2f} – {f['max']:.2f} | "
             f"{a['median']:.2f} / {a['p95']:.2f} / {a['max']:.2f} | "
             f"{r['kid_inception']['mean'] * 1e3:.3f} ± {r['kid_inception']['sd'] * 1e3:.3f} | "
             f"{r['kid_dino']['mean'] * 1e3:.1f} ± {r['kid_dino']['sd'] * 1e3:.1f} |")
open(os.path.join(args.out, "null_draws.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"[null] -> {args.out}/null_draws.md")
