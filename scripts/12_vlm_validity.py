"""Choose V3 among the VLM candidates on REAL tiles only (protocol v2, amendment 2).

    python scripts/12_vlm_validity.py                 # all candidates, ~1 h on an RTX 3060
    python scripts/12_vlm_validity.py --tiles 20      # quick check that every candidate runs

For each of the first 500 R2000 tiles, the real image is scored against its own caption and
against 20 captions of other R2000 tiles (drawn once, seed 2027). Measure per verifier:
the share of those 20 that score above the own caption (0 = own caption first, 0.5 = chance),
and how often the own caption is first. V1 and V2 are measured on the same 21-way sets, so the
candidates are compared with the verifiers we already use. Pairs are scored one at a time, so
no prompt is padded (see eotts/scoring/vlm.py).

Pre-registered choice (amendment 2): V3 is the candidate with the lowest mean share. No
generated sample and no evaluator is used, so the choice cannot fit the endpoints.

Writes results/bon_v2/vlm_validity.json and .md. Resumes per candidate.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.bon.sets import load_set
from eotts.data.git10m_subset import Git10MSubset
from eotts.provenance import REPO_ROOT, provenance
from eotts.scoring.verifiers import VERIFIERS, ClipVerifier
from eotts.scoring.vlm import QUESTION, VLMS, VlmVerifier
from eotts.stats.bootstrap import summarize, tile_weights

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="R2000")
ap.add_argument("--tiles", type=int, default=500)
ap.add_argument("--distractors", type=int, default=20)
ap.add_argument("--seed", type=int, default=2027)
ap.add_argument("--pool-tiles", type=int, default=None, help="draw distractors from the first N tiles (smoke tests only)")
ap.add_argument("--models", default=",".join(VLMS))
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "bon_v2"))
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
s = load_set(args.set, args.data_root)
sub = Git10MSubset(os.path.join(args.subset_root, s["split"]))
pool = s["tiles"][:args.pool_tiles] if args.pool_tiles else s["tiles"]
tiles = pool[:args.tiles]
rng = np.random.default_rng(args.seed)
# distractor captions: other tiles of the set, without replacement, never the tile itself
dis = [rng.choice(np.delete(np.arange(len(pool)), i), args.distractors, replace=False) for i in range(len(tiles))]
cap = {j: sub[pool[j]["hf_idx"]]["caption"] for j in set(range(len(tiles))) | {int(x) for d in dis for x in d}}
images = [sub[t["hf_idx"]]["image"].convert("RGB") for t in tiles]
captions = [[cap[i]] + [cap[int(j)] for j in dis[i]] for i in range(len(tiles))]   # own caption first

smoke = args.tiles != 500 or args.pool_tiles is not None
os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, "vlm_validity" + ("_smoke" if smoke else "") + ".json")
res = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
if res and (res.get("tiles") != args.tiles or res.get("seed") != args.seed or res.get("set") != args.set):
    sys.exit(f"{path} was written with other settings; move it away first")
res.update({"set": args.set, "tiles": args.tiles, "distractors": args.distractors, "seed": args.seed,
            "question": QUESTION, "measure": "share of distractor captions scoring above the own caption "
            "(0 = own first, 0.5 = chance); top1 = own caption scores highest",
            "choice_rule": "V3 = VLM candidate with the lowest mean share (protocol v2, amendment 2)",
            "verifiers": res.get("verifiers", {})})
W = tile_weights(len(tiles), 1000, args.seed)


def summary(scores):
    """scores: (tiles, 1 + distractors), own caption in column 0."""
    share = (scores[:, 1:] > scores[:, :1]).mean(1)
    top1 = (scores[:, 1:] < scores[:, :1]).all(1).astype(float)
    boot = lambda x: summarize(float(x.mean()), (W @ x) / W.sum(1))
    return {"share_above_own": boot(share), "top1": boot(top1)}


def save():
    res["provenance"] = provenance()
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)


for name in VERIFIERS:                      # V1, V2 on the same 21-way sets
    if name in res["verifiers"]:
        continue
    v = ClipVerifier.load(name, device)
    with torch.no_grad():
        im = torch.cat([v.image_features(images[i:i + 32]) for i in range(0, len(images), 32)])
        sc = np.stack([(v.text_features(captions[i]) @ im[i]).cpu().numpy() for i in range(len(tiles))])
    res["verifiers"][name] = {"kind": "clip", **summary(sc)}
    save()
    del v

for name in args.models.split(","):
    if name in res["verifiers"]:
        print(f"[validity] {name}: done, skipped")
        continue
    t0 = time.time()
    v = VlmVerifier(name, device)
    yes = np.zeros((len(tiles), 1 + args.distractors))
    mass = np.zeros_like(yes)
    for i in range(len(tiles)):
        for j, c in enumerate(captions[i]):
            y, n = v.score([images[i]], [c], return_no=True)          # one pair at a time: no padding
            yes[i, j], mass[i, j] = float(y[0]), float(y[0] + n[0])
        if (i + 1) % 50 == 0:
            print(f"[validity] {name}: {i + 1}/{len(tiles)} tiles, {(time.time() - t0) / (i + 1):.1f} s/tile")
    res["verifiers"][name] = {"kind": "vlm", **VLMS[name], **summary(yes),
                              "p_yes_plus_no_mean": float(mass.mean()),
                              "p_yes_own_mean": float(yes[:, 0].mean()),
                              "p_yes_distractor_mean": float(yes[:, 1:].mean()),
                              "seconds_per_pair": (time.time() - t0) / yes.size}
    save()
    del v
    torch.cuda.empty_cache() if device == "cuda" else None

vlm_names = [k for k, d in res["verifiers"].items() if d["kind"] == "vlm"]
if vlm_names and not smoke:
    res["chosen_v3"] = min(vlm_names, key=lambda k: res["verifiers"][k]["share_above_own"]["estimate"])
save()

L = [f"# V3 candidates on real tiles ({res['set']}, first {res['tiles']} tiles, {res['distractors']} distractor captions)",
     "", "Share of distractor captions scoring above the tile's own caption (0 = own caption first, "
     "0.5 = chance) and how often the own caption is first; 95 % bootstrap CIs over tiles.", "",
     "| Verifier | Kind | Share above own [95 % CI] | Own caption first [95 % CI] | P(yes)+P(no) | s/pair |",
     "| --- | --- | --- | --- | --- | --- |"]
for k, d in res["verifiers"].items():
    a, t = d["share_above_own"], d["top1"]
    L.append(f"| {k} | {d['kind']} | {a['estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] | "
             f"{t['estimate']:.3f} [{t['ci_low']:.3f}, {t['ci_high']:.3f}] | "
             f"{d.get('p_yes_plus_no_mean', float('nan')):.3f} | {d.get('seconds_per_pair', float('nan')):.3f} |")
if "chosen_v3" in res:
    L += ["", f"**V3 = {res['chosen_v3']}** (lowest mean share; rule fixed in amendment 2)."]
open(path[:-5] + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
