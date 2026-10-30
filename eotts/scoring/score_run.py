"""Score every generated image of a run with the verifiers and evaluators, once.

Writes ``<out_dir>/scores/<component>.npz``; each component is computed only if its file is
missing, so scoring resumes and a new verifier can be added later without touching the rest.

Arrays are (tiles, K) for scores and (tiles, K, D) for features, in protocol tile order.
Ground-truth features and the verifiers' scores of the ground-truth tiles are stored too.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from eotts.provenance import provenance


def _batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(i + bs, n))


def score_images(*, tiles, image_path, ks, out_dir, subset, device, batch_size=32,
                 components=("verifiers", "evaluators", "inception", "dino")):
    """tiles: protocol tiles; image_path(tile, k) -> path; subset: Git10MSubset for GT + captions."""
    import torch
    from eotts.scoring.verifiers import ClipVerifier, VERIFIERS
    from eotts.scoring.evaluators import ClipEval, PairwiseEval, InceptionFeatures, DinoFeatures

    sdir = os.path.join(out_dir, "scores")
    os.makedirs(sdir, exist_ok=True)
    T, K = len(tiles), len(ks)
    pairs = [(t, k) for t in tiles for k in ks]                  # tile-major
    missing = [image_path(t, k) for t, k in pairs if not os.path.exists(image_path(t, k))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, first {missing[0]}; finish generation first")
    captions = [subset[t["hf_idx"]]["caption"] for t in tiles]
    gts = lambda sl: [subset[t["hf_idx"]]["image"].convert("RGB") for t in tiles[sl]]
    imgs = lambda sl: [Image.open(image_path(*p)).convert("RGB") for p in pairs[sl]]
    caps_of = lambda sl: [captions[i // K] for i in range(sl.start, sl.stop)]
    meta = {"tiles": [t["img_name"] for t in tiles], "ks": list(ks)}

    def save(name, **arrays):
        np.savez(os.path.join(sdir, name + ".npz"), **arrays)
        log = os.path.join(sdir, "scoring_log.json")
        entries = json.load(open(log, encoding="utf-8")) if os.path.exists(log) else []
        entries.append({"component": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "provenance": provenance(), **meta})
        json.dump(entries, open(log, "w", encoding="utf-8"), indent=1)

    def done(name):
        return os.path.exists(os.path.join(sdir, name + ".npz"))

    if "verifiers" in components and not done("verifiers"):
        out = {}
        for name in VERIFIERS:
            v = ClipVerifier.load(name, device)
            s = np.zeros(len(pairs), np.float32)
            for sl in tqdm(list(_batches(len(pairs), batch_size)), desc=name):
                s[sl] = v.score(imgs(sl), caps_of(sl)).numpy()
            out[name] = s.reshape(T, K)
            # ground truth: own-caption score and its rank among all captions of the set
            tf = torch.cat([v.text_features(captions[sl]) for sl in _batches(T, 256)])
            gf = torch.cat([v.image_features(gts(sl)) for sl in _batches(T, batch_size)])
            sim = (gf @ tf.T).cpu().numpy()
            out[name + "__gt"] = np.diag(sim).astype(np.float32)
            out[name + "__gt_rank_pct"] = ((sim > np.diag(sim)[:, None]).sum(1) / (T - 1)).astype(np.float32)
            del v
            torch.cuda.empty_cache() if device == "cuda" else None
        save("verifiers", **out)

    if "evaluators" in components and not done("evaluators"):
        ce, pw = ClipEval(device), PairwiseEval(device)
        clip = np.zeros(len(pairs), np.float32)
        lp = np.zeros(len(pairs), np.float32)
        ss = np.zeros(len(pairs), np.float32)
        gt_cache = {}
        for sl in tqdm(list(_batches(len(pairs), batch_size)), desc="E1-E3"):
            im = imgs(sl)
            clip[sl] = ce(im, caps_of(sl))
            tid = [i // K for i in range(sl.start, sl.stop)]
            for i in set(tid) - set(gt_cache):
                gt_cache[i] = subset[tiles[i]["hf_idx"]]["image"].convert("RGB")
            l, s_ = pw(im, [gt_cache[i] for i in tid])
            lp[sl], ss[sl] = l, s_
            for i in [i for i in gt_cache if i < min(tid)]:
                del gt_cache[i]
        save("evaluators", clip_cos=clip.reshape(T, K), lpips=lp.reshape(T, K), ssim=ss.reshape(T, K))
        del ce, pw

    for name, cls in (("inception", InceptionFeatures), ("dino", DinoFeatures)):
        if name in components and not done(name):
            fx = cls(device)
            feats = [fx(imgs(sl)) for sl in tqdm(list(_batches(len(pairs), batch_size)), desc=name)]
            gtf = [fx(gts(sl)) for sl in tqdm(list(_batches(T, batch_size)), desc=name + "[gt]")]
            f = np.concatenate(feats).astype(np.float32)
            save(name, samples=f.reshape(T, K, -1), gt=np.concatenate(gtf).astype(np.float32))
            del fx
    return sdir


def load_scores(out_dir) -> dict:
    sdir = os.path.join(out_dir, "scores")
    out = {}
    for f in sorted(os.listdir(sdir)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(sdir, f))
            for k in z.files:
                out[f"{f[:-4]}.{k}"] = z[k]
    return out
