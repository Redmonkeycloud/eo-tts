"""Score every generated image of a run with the verifiers and evaluators, once.

Writes ``<out_dir>/scores/<component>.npz``; each component is computed only if its file is
missing, so scoring resumes and a new verifier can be added later without touching the rest.

Arrays are (tiles, K) for scores and (tiles, K, D) for features, in protocol tile order.
Ground-truth features and the verifiers' scores of the ground-truth tiles are stored too.

Within a component, progress is checkpointed to ``<component>.partial.npz`` every
``ckpt_every`` batches (written atomically), so an interrupted pass over the 238,912 images of
the full split resumes where it stopped instead of from the start.
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


def _resumable(sdir, tag, n, batch_size, compute, ckpt_every=200):
    """Run compute(slice) -> {field: array} over n items in batches, checkpointing as it goes."""
    part = os.path.join(sdir, tag + ".partial.npz")
    arrays, start = {}, 0
    if os.path.exists(part):
        z = np.load(part)
        if int(z["__n"]) != n or int(z["__batch"]) != batch_size:
            raise ValueError(f"{part} was written for n={int(z['__n'])}, batch={int(z['__batch'])}")
        start = int(z["__done"])
        arrays = {k: z[k] for k in z.files if not k.startswith("__")}
        print(f"[score] {tag}: resuming at item {start}/{n}")

    def checkpoint(done):
        tmp = part + ".tmp.npz"
        np.savez(tmp, __done=done, __n=n, __batch=batch_size, **arrays)
        os.replace(tmp, part)

    batches = [sl for sl in _batches(n, batch_size) if sl.start >= start]
    for b, sl in enumerate(tqdm(batches, desc=tag, initial=start // batch_size,
                                total=start // batch_size + len(batches))):
        res = compute(sl)
        for k, v in res.items():
            v = np.asarray(v)
            if k not in arrays:
                arrays[k] = np.zeros((n,) + v.shape[1:], np.float32)
            arrays[k][sl] = v
        if (b + 1) % ckpt_every == 0:
            checkpoint(sl.stop)
    return arrays, part


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
        out, parts = {}, []
        for name in VERIFIERS:
            v = ClipVerifier.load(name, device)
            arr, part = _resumable(sdir, f"verifiers_{name}", len(pairs), batch_size,
                                   lambda sl: {"s": v.score(imgs(sl), caps_of(sl)).numpy()})
            out[name] = arr["s"].reshape(T, K)
            parts.append(part)
            # ground truth: own-caption score and its rank among all captions of the set
            tf = torch.cat([v.text_features(captions[sl]) for sl in _batches(T, 256)])
            gf = torch.cat([v.image_features(gts(sl)) for sl in _batches(T, batch_size)])
            sim = (gf @ tf.T).cpu().numpy()
            out[name + "__gt"] = np.diag(sim).astype(np.float32)
            out[name + "__gt_rank_pct"] = ((sim > np.diag(sim)[:, None]).sum(1) / (T - 1)).astype(np.float32)
            del v
            torch.cuda.empty_cache() if device == "cuda" else None
        save("verifiers", **out)
        for p in parts:
            os.remove(p) if os.path.exists(p) else None

    if "evaluators" in components and not done("evaluators"):
        ce, pw = ClipEval(device), PairwiseEval(device)
        gt_cache = {}

        def e123(sl):
            im = imgs(sl)
            tid = [i // K for i in range(sl.start, sl.stop)]
            for i in set(tid) - set(gt_cache):
                gt_cache[i] = subset[tiles[i]["hf_idx"]]["image"].convert("RGB")
            l, s_ = pw(im, [gt_cache[i] for i in tid])
            for i in [i for i in gt_cache if i < min(tid)]:
                del gt_cache[i]
            return {"clip": ce(im, caps_of(sl)), "lp": l, "ss": s_}

        arr, part = _resumable(sdir, "evaluators", len(pairs), batch_size, e123)
        save("evaluators", clip_cos=arr["clip"].reshape(T, K), lpips=arr["lp"].reshape(T, K),
             ssim=arr["ss"].reshape(T, K))
        os.remove(part) if os.path.exists(part) else None
        del ce, pw

    for name, cls in (("inception", InceptionFeatures), ("dino", DinoFeatures)):
        if name in components and not done(name):
            fx = cls(device)
            arr, part = _resumable(sdir, name, len(pairs), batch_size, lambda sl: {"f": fx(imgs(sl))})
            gtf = [fx(gts(sl)) for sl in tqdm(list(_batches(T, batch_size)), desc=name + "[gt]")]
            save(name, samples=arr["f"].reshape(T, K, -1), gt=np.concatenate(gtf).astype(np.float32))
            os.remove(part) if os.path.exists(part) else None
            del fx
    return sdir


def load_scores(out_dir) -> dict:
    sdir = os.path.join(out_dir, "scores")
    out = {}
    for f in sorted(os.listdir(sdir)):
        if f.endswith(".npz") and ".partial" not in f:
            z = np.load(os.path.join(sdir, f))
            for k in z.files:
                out[f"{f[:-4]}.{k}"] = z[k]
    return out
