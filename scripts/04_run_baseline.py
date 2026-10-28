"""Paper-protocol baseline for TerraDiT (generation + FID / CLIP / LPIPS / SSIM, plus KID).

    python scripts/04_run_baseline.py --split spatial --limit 64      # smoke test (minutes)
    python scripts/04_run_baseline.py --split random                  # the paper number
    python scripts/04_run_baseline.py --split spatial

Generation resumes: tiles already on disk are kept, so an interrupted run can be restarted
with the same command. Results: runs/baseline/<run>/results.json, with the paper's numbers
and the difference next to ours.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.baselines.terradit_baseline import run_baseline

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="omega_xl", choices=["omega_xl", "omega_base", "alpha_xl", "sigma_xl"])
ap.add_argument("--split", default="random", choices=["random", "spatial", "dense"])
ap.add_argument("--condition-type", default="omega", choices=["omega", "box", "point"])
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--weights-root", default="data/weights")
ap.add_argument("--out-root", default="runs/baseline")
ap.add_argument("--num-steps", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=16, help="repo default 16; the paper states 32")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=None, help="first N tiles only (smoke test)")
ap.add_argument("--no-kid", action="store_true")
ap.add_argument("--regenerate", action="store_true")
ap.add_argument("--metrics-only", action="store_true")
args = ap.parse_args()

run_baseline(model=args.model, split=args.split, condition_type=args.condition_type,
             data_root=args.data_root, subset_root=args.subset_root, weights_root=args.weights_root,
             out_root=args.out_root, num_steps=args.num_steps, batch_size=args.batch_size,
             seed=args.seed, limit=args.limit, kid=not args.no_kid, regenerate=args.regenerate,
             metrics_only=args.metrics_only)
