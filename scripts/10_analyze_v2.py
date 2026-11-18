"""Study 2 analysis (docs/experiment_protocol_v2.md with amendment 1): TerraDiT-Ω-B.

    python scripts/10_analyze_v2.py --check-gpu-kid      # step 0: GPU KID vs the committed pilot R2000.json
    python scripts/10_analyze_v2.py --set S500
    python scripts/10_analyze_v2.py --set R2000
    python scripts/10_analyze_v2.py --set RFULL
    python scripts/10_analyze_v2.py --set S500 --suffix _smoke   # smoke-test runs (output under runs/)

Reads cached scores only; no image model is loaded. Writes results/bon_v2/<set>.json,
endpoints_<set>.json, anchors_<set>.json and report_<set>.md.
Needs a GPU for KID on RFULL (protocol v2, Statistics); FID runs on the CPU in float64.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.analyze import (PER_IMAGE, analyze_set, anchor_per_image, endpoints_v2, psnr_same_noise, save)
from eotts.bon.report import write_report_v2
from eotts.bon.select import RULES
from eotts.bon.sets import load_set, sample_path
from eotts.provenance import REPO_ROOT, provenance
from eotts.scoring.score_run import load_scores

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="RFULL", choices=["S500", "R2000", "RFULL"])
ap.add_argument("--runs-root", default="runs/bon")
ap.add_argument("--v3", default=None, help="VLM verifier fixed by amendment 2 (e.g. v3_qwen3vl_4b); adds the "
                                          "secondary rules v3 and v1+v3 if its scores exist")
ap.add_argument("--out", default="results/bon_v2")
ap.add_argument("--suffix", default="", help="e.g. _smoke")
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--device", default=None, help="KID device; default cuda if available, else the float64 CPU path")
ap.add_argument("--check-gpu-kid", action="store_true",
                help="re-run the pilot R2000 KID with the GPU path and compare with results/bon/R2000.json")
args = ap.parse_args()
B_KID_RUN = 20 if args.suffix else 1000      # smoke tests only: fewer KID/FID resamples

import torch
device = args.device or ("cuda" if torch.cuda.is_available() else None)
if device == "none":
    device = None
out_dir = os.path.join(args.runs_root, "analysis_v2" + args.suffix) if args.suffix else args.out
os.makedirs(out_dir, exist_ok=True)
t_start = time.time()
log = lambda msg: print(f"[v2 {time.time() - t_start:7.0f}s] {msg}", flush=True)


def scores(run):
    d = os.path.join(args.runs_root, run + args.suffix)
    return load_scores(d) if os.path.isdir(os.path.join(d, "scores")) else None


def head(sc, n):
    """Scores of the first n tiles (every array is tile-major)."""
    return {k: v[:n] for k, v in sc.items()}


if args.check_gpu_kid:
    if device is None:
        sys.exit("no GPU: nothing to check (the float64 path is the reference)")
    committed = json.load(open(os.path.join(REPO_ROOT, "results", "bon", "R2000.json"), encoding="utf-8"))
    sc = scores("R2000")
    if sc is None:
        sys.exit("runs/bon/R2000/scores not found")
    log(f"pilot R2000 with KID on {device}")
    new = analyze_set(sc, "R2000", device=device, progress=log)
    worst, rows = 0.0, []
    for rule in RULES:
        for n in committed["rules"][rule]:
            for name in ("kid_inception", "kid_dino"):
                a, b = committed["rules"][rule][n][name]["diff_vs_n1"], new["rules"][rule][n][name]["diff_vs_n1"]
                width = a["ci_high"] - a["ci_low"]
                if width <= 0:
                    continue
                dev = max(abs(a[k] - b[k]) for k in ("estimate", "ci_low", "ci_high")) / width
                worst = max(worst, dev)
                rows.append({"rule": rule, "n": n, "metric": name, "max_abs_dev_over_ci_width": dev})
    res = {"check": "GPU float32 KID vs committed float64 pilot results (protocol v2, Statistics)",
           "device": device, "threshold": 0.01, "worst": worst, "pass": worst <= 0.01, "cells": rows,
           "provenance": provenance()}
    save(res, os.path.join(out_dir, "gpu_kid_check.json"))
    log(f"worst deviation = {worst:.2e} of the CI width -> {'PASS' if res['pass'] else 'FAIL: use --device none'}")
    sys.exit(0)

set_name = args.set
run = f"{set_name}_omega_base"
# anchor runs per set (protocol v2, amendment 1): the paper-protocol baseline exists for Git-Rand
# only; the same-noise fp32 anchor of R2000 (first 500 tiles) serves RFULL too, same tiles and seeds
BASELINE_RUN = {"R2000": "R2000_omega_base_baseline_fp32", "RFULL": "RFULL_omega_base_baseline_fp32"}.get(set_name)
SAME_NOISE_RUN = {"S500": "S500_omega_base_fp32_k0", "R2000": "R2000_omega_base_fp32_k0",
                  "RFULL": "R2000_omega_base_fp32_k0"}[set_name]
sc = scores(run)
if sc is None:
    sys.exit(f"{args.runs_root}/{run}{args.suffix}/scores not found")
T = sc[f"evaluators.clip_cos"].shape[0]
log(f"{run}: {T} tiles; KID on {device or 'cpu float64'}; FID float64 CPU, {B_KID_RUN:,} resamples")
v3 = None
if args.v3:
    key = f"vlm_{args.v3}.samples"
    if key not in sc:
        sys.exit(f"--v3 {args.v3}: no scores in {run}; run 06_score_bon.py --components none --vlm {args.v3}")
    v3 = sc[key]
    log(f"V3 = {args.v3} (secondary rules v3, v1+v3)")
r = analyze_set(sc, set_name, with_fid=True, device=device, progress=log, b_kid=B_KID_RUN, v3=v3,
                mean_draw=True)       # amendment 3: also against the average single draw
if args.v3:
    r["v3_model"] = args.v3
save(r, os.path.join(out_dir, f"{set_name}.json"))
ends = endpoints_v2(r, set_name)
save(ends, os.path.join(out_dir, f"endpoints_{set_name}.json"))
log("main analysis saved")

prefix = None     # R2000 is analysed as its own set (its samples are the first 2,000 tiles of RFULL)

anchors = {}
base_dir = os.path.join(REPO_ROOT, "results", "baseline", "omega_base_random_omega.json")
if not os.path.exists(base_dir):
    base_dir = os.path.join("runs", "baseline", "omega_base_random_omega", "results.json")
if os.path.exists(base_dir) and set_name != "S500":
    b = json.load(open(base_dir, encoding="utf-8"))
    m = b["metrics"]
    anchors["1_paper_protocol_baseline"] = {
        "label": "Ω-B paper-protocol baseline on all of Git-Rand (fp32, upstream seed) vs the paper's Ω-B "
                 "ablation (Table 3; split not named, plausibility only)",
        "ours": {k: m[k] for k in ("fid", "sfid", "clip_cos", "lpips", "ssim", "kid") if k in m},
        "paper_ablation": {"fid": 21.97, "sfid": 6.25, "ssim": 0.2632, "clip_cos": 0.2759},
        "source": base_dir}
base_sc = scores(BASELINE_RUN) if BASELINE_RUN else None
if base_sc is not None:
    log("anchor 2")
    anchors["2_bf16_vs_fp32_baseline"] = anchor_per_image(
        sc, base_sc, "BoN sample 0 (bf16, per-sample seeds)", "fp32 paper-protocol baseline (different noise)",
        with_fid=True, b_kid=B_KID_RUN)
a3_sc = scores(SAME_NOISE_RUN)
if a3_sc is not None:
    log("anchor 3")
    n3 = a3_sc["evaluators.clip_cos"].shape[0]
    a3 = anchor_per_image(head(sc, n3), a3_sc, "BoN sample 0 (bf16)", "sample 0 (fp32, same noise)")
    tiles = load_set(set_name, args.data_root)["tiles"][:n3]
    a3["pixels"] = psnr_same_noise(
        [sample_path(os.path.join(args.runs_root, run + args.suffix), t["stem"], 0) for t in tiles],
        [sample_path(os.path.join(args.runs_root, SAME_NOISE_RUN + args.suffix), t["stem"], 0)
         for t in tiles])
    anchors["3_bf16_vs_fp32_same_noise"] = a3

flags = []


def narrowest_half_width(metric):
    return min((r["rules"][q]["16"][metric]["diff_vs_n1"]["ci_high"]
                - r["rules"][q]["16"][metric]["diff_vs_n1"]["ci_low"]) / 2 for q in RULES)


for key in ("2_bf16_vs_fp32_baseline", "3_bf16_vs_fp32_same_noise"):
    if key in anchors:
        for m in PER_IMAGE:
            d, hw = abs(anchors[key][m]["diff"]["estimate"]), narrowest_half_width(m)
            if d > hw:
                flags.append({"anchor": key, "metric": m, "abs_diff": d, "bon_ci_half_width": hw})
if "2_bf16_vs_fp32_baseline" in anchors and "fid" in anchors["2_bf16_vs_fp32_baseline"]:
    d, hw = abs(anchors["2_bf16_vs_fp32_baseline"]["fid"]["diff"]["estimate"]), narrowest_half_width("fid")
    if d > hw:
        flags.append({"anchor": "2_bf16_vs_fp32_baseline", "metric": "fid", "abs_diff": d, "bon_ci_half_width": hw})
anchors["stop_rule"] = {"rule": "stop if |anchor difference| exceeds the half-width of the narrowest N=16 vs N=1 "
                                f"CI for that metric on {set_name} (E1-E3 on anchors 2-3, FID on anchor 2)",
                        "flags": flags, "stop": bool(flags)}
anchors["provenance"] = provenance()
save(anchors, os.path.join(out_dir, f"anchors_{set_name}.json"))

write_report_v2(r, ends, anchors, prefix, os.path.join(out_dir, f"report_{set_name}.md"), set_label=set_name)
log(f"-> {out_dir}/report_{set_name}.md")
if flags:
    print("[v2] STOP RULE TRIGGERED:", json.dumps(flags, indent=1))
