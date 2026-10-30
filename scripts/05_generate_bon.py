"""Generate the Best-of-N samples of the protocol (docs/experiment_protocol.md).

    python scripts/05_generate_bon.py --set R2000                 # 2,000 tiles x 16, bf16  (~56 h)
    python scripts/05_generate_bon.py --set S500                  # 500 tiles x 16, bf16    (~14 h)
    python scripts/05_generate_bon.py --set S500 --anchor-fp32    # sample 0 in fp32, same noise (~1.6 h)
    python scripts/05_generate_bon.py --set S500 --limit-tiles 2  # smoke test

Resumable: rerun the same command after an interruption.
Output: runs/bon/<set>/ (or runs/bon/<set>_fp32_k0/ for the anchor).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.generate import generate
from eotts.bon.sets import NUM_SAMPLES, SETS

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=list(SETS))
ap.add_argument("--anchor-fp32", action="store_true", help="sample 0 only, fp32 (protocol anchor 3)")
ap.add_argument("--model", default="omega_xl", choices=["omega_xl", "omega_base"])
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--num-steps", type=int, default=100)
ap.add_argument("--limit-tiles", type=int, default=None, help="first N tiles of the set (smoke test)")
ap.add_argument("--out-root", default="runs/bon")
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--weights-root", default="data/weights")
args = ap.parse_args()

smoke = args.limit_tiles is not None or args.num_steps != 100 or args.model != "omega_xl"
name = args.set + ("_fp32_k0" if args.anchor_fp32 else "") + ("_smoke" if smoke else "")
generate(set_name=args.set, run_dir=os.path.join(args.out_root, name), model=args.model,
         sample_ids=[0] if args.anchor_fp32 else range(NUM_SAMPLES),
         precision="fp32" if args.anchor_fp32 else "bf16", data_root=args.data_root,
         subset_root=args.subset_root, weights_root=args.weights_root, batch_size=args.batch_size,
         num_steps=args.num_steps, limit_tiles=args.limit_tiles)
