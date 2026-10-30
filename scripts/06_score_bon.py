"""Score generated images with the verifiers (V1, V2) and evaluators (E1–E5).

    python scripts/06_score_bon.py --run R2000                # the 32,000 Best-of-N samples
    python scripts/06_score_bon.py --run S500
    python scripts/06_score_bon.py --run S500_fp32_k0         # anchor 3
    python scripts/06_score_bon.py --run R2000_baseline_fp32  # anchors 1-2: fp32 baseline images on R2000

Each component (verifiers, evaluators, inception, dino) is saved separately under
runs/bon/<run>/scores/ and skipped if present, so scoring resumes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.sets import load_set, sample_path
from eotts.data.git10m_subset import Git10MSubset
from eotts.scoring.score_run import score_images

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="R2000 | S500 | S500_fp32_k0 | R2000_baseline_fp32 | *_smoke")
ap.add_argument("--out-root", default="runs/bon")
ap.add_argument("--baseline-images", default="runs/baseline/omega_xl_random_omega/images")
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--limit-tiles", type=int, default=None, help="baseline anchor only: first N tiles (smoke test)")
args = ap.parse_args()

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
set_name = args.run.split("_")[0]
s = load_set(set_name, args.data_root)
run_dir = os.path.join(args.out_root, args.run)

if args.run.startswith("R2000_baseline_fp32"):
    tiles, ks = s["tiles"][:args.limit_tiles] if args.limit_tiles else s["tiles"], [0]
    image_path = lambda t, k: os.path.join(args.baseline_images, t["stem"] + ".png")
else:
    gen = json.load(open(os.path.join(run_dir, "generation.json")))
    tiles, ks = s["tiles"][:gen["num_tiles"]], gen["sample_ids"]
    image_path = lambda t, k: sample_path(run_dir, t["stem"], k)

score_images(tiles=tiles, image_path=image_path, ks=ks, out_dir=run_dir,
             subset=Git10MSubset(os.path.join(args.subset_root, s["split"])), device=device,
             batch_size=args.batch_size)
print(f"[score] done -> {run_dir}/scores")
