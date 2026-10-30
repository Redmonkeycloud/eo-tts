"""Analyse the scored Best-of-N runs: effects, primary endpoints, anchors, report.

    python scripts/07_analyze_bon.py                 # whatever runs are scored
    python scripts/07_analyze_bon.py --suffix _smoke # the smoke-test runs

Writes results/bon/<set>.json, results/bon/endpoints.json, results/bon/anchors.json and a
readable results/bon/report.md. Only scores are read; no model is loaded.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.analyze import (FEATURES, PER_IMAGE, analyze_set, anchor_per_image, anchor_vs_reference,
                               endpoints, psnr_same_noise, save)
from eotts.bon.report import write_report
from eotts.bon.select import RULES
from eotts.bon.sets import SETS, load_set, sample_path
from eotts.provenance import REPO_ROOT, provenance
from eotts.scoring.score_run import load_scores

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", default="runs/bon")
ap.add_argument("--out", default="results/bon")
ap.add_argument("--suffix", default="", help="e.g. _smoke")
ap.add_argument("--data-root", default="data/terradit")
args = ap.parse_args()
# smoke-test analyses stay under runs/ (not committed); real results go to results/bon/
out_dir = os.path.join(args.runs_root, "analysis" + args.suffix) if args.suffix else args.out


def scores(run):
    d = os.path.join(args.runs_root, run + args.suffix)
    return load_scores(d) if os.path.isdir(os.path.join(d, "scores")) else None


per_set = {}
for s in SETS:
    sc = scores(s)
    if sc and all(f"{c}" in " ".join(sc) for c in ("verifiers.", "evaluators.", "inception.", "dino.")):
        print(f"[analyze] {s}")
        per_set[s] = analyze_set(sc, s)
        save(per_set[s], os.path.join(out_dir, f"{s}.json"))

ends = endpoints(per_set) if per_set else {}
if ends:
    save(ends, os.path.join(out_dir, "endpoints.json"))

targets = json.load(open(os.path.join(REPO_ROOT, "configs", "paper_targets.json")))
baseline = json.load(open(os.path.join(REPO_ROOT, "results", "baseline", "omega_xl_random_omega.json")))
anchors = {}
base_sc, r_sc = scores("R2000_baseline_fp32"), scores("R2000")
if base_sc:
    full = baseline["metrics"]
    anchors["1_R2000_vs_full_split"] = anchor_vs_reference(
        base_sc, {"clip_cos": full["clip_cos"], "lpips": full["lpips"], "ssim": full["ssim"]},
        "fp32 baseline images on the R2000 tiles vs the full Git-Rand baseline means")
    if r_sc:
        anchors["2_bf16_vs_fp32_R2000"] = anchor_per_image(
            r_sc, base_sc, "BoN sample 0 (bf16, per-sample seeds)", "fp32 baseline (different noise)")
s_sc, s32_sc = scores("S500"), scores("S500_fp32_k0")
if s_sc and s32_sc:
    a3 = anchor_per_image(s_sc, s32_sc, "BoN sample 0 (bf16)", "sample 0 (fp32, same noise)")
    st = load_set("S500", args.data_root)["tiles"]
    gen = json.load(open(os.path.join(args.runs_root, "S500_fp32_k0" + args.suffix, "generation.json")))
    st = st[:gen["num_tiles"]]
    a3["pixels"] = psnr_same_noise(
        [sample_path(os.path.join(args.runs_root, "S500" + args.suffix), t["stem"], 0) for t in st],
        [sample_path(os.path.join(args.runs_root, "S500_fp32_k0" + args.suffix), t["stem"], 0) for t in st])
    anchors["3_bf16_vs_fp32_S500_same_noise"] = a3
if s32_sc:
    p = targets["omega_xl"]["omega"]["spatial"]
    anchors["4_S500_vs_paper_spatial"] = anchor_vs_reference(
        s32_sc, {"clip_cos": p["clip_cos"], "lpips": p["lpips"], "ssim": p["ssim"]},
        "fp32 sample 0 on S500 vs the paper's full Git-Spatial values (plausibility only)")

# stop rule: an anchor difference larger than the half-width of the narrowest N=16 CI on that metric
flags = []
for key, set_name in (("2_bf16_vs_fp32_R2000", "R2000"), ("3_bf16_vs_fp32_S500_same_noise", "S500")):
    if key in anchors and set_name in per_set:
        for m in PER_IMAGE:
            hw = min((per_set[set_name]["rules"][r]["16"][m]["diff_vs_n1"]["ci_high"]
                      - per_set[set_name]["rules"][r]["16"][m]["diff_vs_n1"]["ci_low"]) / 2 for r in RULES)
            d = abs(anchors[key][m]["diff"]["estimate"])
            if d > hw:
                flags.append({"anchor": key, "metric": m, "abs_diff": d, "bon_ci_half_width": hw})
anchors["stop_rule"] = {"rule": "stop if |anchor difference| exceeds the half-width of the narrowest "
                                "N=16 vs N=1 CI for that metric on the same set",
                        "flags": flags, "stop": bool(flags)}
anchors["provenance"] = provenance()
save(anchors, os.path.join(out_dir, "anchors.json"))

write_report(per_set, ends, anchors, os.path.join(out_dir, "report.md"))
print(f"[analyze] -> {out_dir}/report.md")
if flags:
    print("[analyze] STOP RULE TRIGGERED:", json.dumps(flags, indent=1))
