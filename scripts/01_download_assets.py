"""Download what evaluation needs from the TerraDiT Hub repos (~2.5 GB for omega_xl).

    python scripts/01_download_assets.py                    # omega_xl weights + split files
    python scripts/01_download_assets.py --models omega_xl alpha_xl

* MVRL/TerraDiT-data: only ``splits/**`` (test metadata, instance geometry, RANGE+
  embeddings, ~120 MB) and ``osm/*`` (tag vocabulary). The 8 GB of training artifacts
  are not needed for evaluation.
* MVRL/TerraDiT: ``<model>/model.safetensors`` + ``config.json``.

The text encoder (LongCLIP) and the SDXL VAE are fetched into the normal Hugging Face
cache on first use.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.provenance import ensure_terradit_importable

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--weights-root", default="data/weights")
ap.add_argument("--models", nargs="+", default=["omega_xl"])
args = ap.parse_args()

ensure_terradit_importable()
from huggingface_hub import snapshot_download
from terradit.hf import DATA_REPO, download_weights

os.makedirs(args.data_root, exist_ok=True)
print(f"[assets] {DATA_REPO}: splits + osm -> {args.data_root}")
snapshot_download(DATA_REPO, repo_type="dataset", local_dir=args.data_root,
                  allow_patterns=["splits/**", "osm/*"])
for m in args.models:
    print(f"[assets] weights {m} -> {args.weights_root}/{m}")
    download_weights(m, checkpoints_root=args.weights_root)

for root, _, files in os.walk(args.data_root):
    for f in sorted(files):
        p = os.path.join(root, f)
        if ".cache" not in p:
            print(f"  {os.path.relpath(p, args.data_root):45s} {os.path.getsize(p) / 1e6:8.1f} MB")
print("[assets] done")
