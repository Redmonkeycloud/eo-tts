"""POST-HOC diagnostic: is sample 0 different from samples 1-15?

    python scripts/14_sample0_check.py --runs R2000_omega_base S500_omega_base R2000 S500

The null-draw diagnostic (scripts/13_null_draws.py) found that with TerraDiT-Ω-B, sample 0 has
the worst FID of all 16 draws on both R2000 (40.47; others 39.17-40.15) and S500 (73.17; others
69.75-72.74), and the worst KID-Inception on both. With Ω-XL (same seeds) sample 0 is ordinary.
Sample 0 is the N = 1 reference of every Best-of-N comparison, so this matters.

For each run and each k this reports, from the cached scores, the mean of every evaluator and
verifier, and, from the PNG files of the first --tiles tiles, simple image statistics (file size,
mean and spread of brightness, saturation, sharpness as variance of the Laplacian). For each
quantity it gives the z-score of sample 0 against samples 1-15. Reads files only.

Writes results/bon_v2/sample0_check.json and .md.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.sets import load_set, sample_path
from eotts.provenance import REPO_ROOT, provenance
from eotts.scoring.score_run import load_scores

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", default=["R2000_omega_base", "S500_omega_base", "R2000", "S500"])
ap.add_argument("--runs-root", default="runs/bon")
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--tiles", type=int, default=300, help="tiles whose PNGs are read for image statistics")
ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "bon_v2"))
args = ap.parse_args()


def img_stats(path):
    a = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0
    lum = a @ np.array([0.299, 0.587, 0.114])
    sat = a.max(2) - a.min(2)
    lap = (-4 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:])
    return {"file_kb": os.path.getsize(path) / 1024, "brightness": lum.mean(), "contrast": lum.std(),
            "saturation": sat.mean(), "sharpness_lapvar": lap.var() * 1e3}


def z0(per_k):
    v = np.asarray(per_k, np.float64)
    rest = v[1:]
    sd = rest.std(ddof=1)
    return {"k0": float(v[0]), "mean_k1_15": float(rest.mean()), "sd_k1_15": float(sd),
            "z_k0": float((v[0] - rest.mean()) / sd) if sd > 0 else float("nan"),
            "rank_k0_of_16": int((v > v[0]).sum()) + 1}


res = {"note": "POST-HOC diagnostic; not pre-registered. z_k0 = (sample 0 - mean of samples 1-15) / SD of samples 1-15, "
               "over per-k set means. rank_k0_of_16: 1 = highest value.", "runs": {}}
for run in args.runs:
    d = os.path.join(args.runs_root, run)
    if not os.path.isdir(os.path.join(d, "scores")):
        print(f"[k0] {run}: no scores, skipped")
        continue
    sc = load_scores(d)
    K = sc["evaluators.clip_cos"].shape[1]
    r = {"scores": {}, "images": {}}
    for key, v in sc.items():
        if v.ndim == 2 and v.shape[1] == K and (key.startswith("evaluators.") or key.startswith("verifiers.")
                                                   or key.startswith("vlm_")):
            r["scores"][key] = z0(v.mean(0))
    set_name = run.split("_")[0]
    tiles = load_set(set_name, args.data_root)["tiles"][:args.tiles]
    stats = [[img_stats(sample_path(d, t["stem"], k)) for k in range(K)] for t in tiles]
    for q in stats[0][0]:
        r["images"][q] = z0([np.mean([s[k][q] for s in stats]) for k in range(K)])
    gen = os.path.join(d, "generation.json")
    if os.path.exists(gen):
        g = json.load(open(gen, encoding="utf-8"))
        r["generation"] = {k: g.get(k) for k in ("model", "precision", "batch_size", "sample_ids", "num_steps")}
        r["generation"]["runs"] = [{k: x.get(k) for k in ("started", "finished", "pending_at_start", "s_per_image")}
                                   for x in g.get("runs", [])]
    res["runs"][run] = r
    flagged = {k: v["z_k0"] for k, v in {**r["scores"], **r["images"]}.items() if abs(v["z_k0"]) >= 2.5}
    print(f"[k0] {run}: |z| >= 2.5 for {flagged or 'nothing'}")

res["provenance"] = provenance()
os.makedirs(args.out, exist_ok=True)
json.dump(res, open(os.path.join(args.out, "sample0_check.json"), "w", encoding="utf-8"), indent=1)
L = ["# POST-HOC: is sample 0 different from samples 1-15?", "",
     "z = (sample 0 − mean of samples 1–15) / SD of samples 1–15, over per-sample set means; rank 1 = highest of 16.", ""]
for run, r in res["runs"].items():
    L += [f"## {run}", "", "| Quantity | sample 0 | mean 1–15 | SD 1–15 | z | rank of sample 0 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in {**r["scores"], **{"image." + q: x for q, x in r["images"].items()}}.items():
        L.append(f"| {k} | {v['k0']:.5f} | {v['mean_k1_15']:.5f} | {v['sd_k1_15']:.5f} | {v['z_k0']:+.2f} | {v['rank_k0_of_16']} |")
    if "generation" in r:
        L += ["", f"generation.json: `{json.dumps(r['generation'])}`"]
    L.append("")
open(os.path.join(args.out, "sample0_check.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"[k0] -> {args.out}/sample0_check.md")
