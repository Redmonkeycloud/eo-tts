"""Build the local Git-10M subset for a test split (only the shards it touches).

    python scripts/02_build_git10m_subset.py --split spatial            # 2 shards, ~0.3 GB download
    python scripts/02_build_git10m_subset.py --split random             # 151 shards, ~29 GB download
    python scripts/02_build_git10m_subset.py --split random --plan-only # sizes, no download

Shards are downloaded one at a time and deleted after their rows are extracted, so peak
extra disk is about one shard. Interrupted runs resume where they stopped.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.data.git10m_subset import build_subset, plan

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["random", "spatial", "dense"])
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--plan-only", action="store_true")
ap.add_argument("--keep-shards", action="store_true", help="keep downloaded shards (debug)")
args = ap.parse_args()

meta = os.path.join(args.data_root, "splits", args.split, "metadata.json")
if not os.path.exists(meta):
    sys.exit(f"{meta} missing: run scripts/01_download_assets.py first")
rows = json.load(open(meta))
p = plan(rows)
print(f"[subset] {args.split}: {p['num_rows']} tiles in {p['num_shards']} shards, "
      f"download {p['download_bytes'] / 1e9:.1f} GB (one shard on disk at a time)")
if args.plan_only:
    sys.exit(0)
build_subset(rows, os.path.join(args.subset_root, args.split), token=os.environ.get("HF_TOKEN"),
             keep_shards=args.keep_shards)
