"""Check that every sample PNG of a run opens and decodes; optionally delete the broken ones.

    python scripts/11_check_samples.py runs/bon/S500_omega_base runs/bon/R2000_omega_base
    python scripts/11_check_samples.py runs/bon/R2000_omega_base --delete

Use after an interruption such as a power cut. Before the generation code wrote PNGs
atomically, an interrupted write could leave a truncated kNN.png, which a resumed run would
skip as done. A deleted sample is regenerated identically on resume (its seed depends only on
split, tile and k). Leftover *.partial.png files are always removed with --delete.
"""
import argparse
import glob
import os
import sys

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--delete", action="store_true", help="delete broken samples so the next run regenerates them")
args = ap.parse_args()

bad_total = 0
for run in args.runs:
    files = sorted(glob.glob(os.path.join(run, "samples", "*", "k*.png")))
    partial = [f for f in files if f.endswith(".partial.png")]
    files = [f for f in files if not f.endswith(".partial.png")]
    bad = []
    for f in files:
        try:
            if os.path.getsize(f) == 0:
                raise ValueError("empty file")
            with Image.open(f) as im:
                im.load()
        except Exception as e:                     # noqa: BLE001 - any failure to decode counts
            bad.append((f, repr(e)))
    print(f"[check] {run}: {len(files)} samples, {len(bad)} broken, {len(partial)} partial")
    for f, e in bad:
        print(f"   broken: {f}  ({e})")
    if args.delete:
        for f in [b[0] for b in bad] + partial:
            os.remove(f)
        if bad or partial:
            print(f"[check] deleted {len(bad) + len(partial)} files; rerun the generation command to regenerate")
    bad_total += len(bad)
sys.exit(1 if bad_total and not args.delete else 0)
