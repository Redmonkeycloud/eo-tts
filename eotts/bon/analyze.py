"""Analysis of the Best-of-N protocol: selection, effects with CIs, endpoints, anchors.

Everything here is computed from the cached scores and features; no model is loaded.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

from eotts.bon.select import RULES, V1, V2, V3, V3_RULES, oracle, select, take
from eotts.bon.sets import N_VALUES
from eotts.stats.bootstrap import holm, paired_mean_diff, summarize, tile_weights
from eotts.stats.fid import fid_resampled
from eotts.stats.kid import kid, poly_kernel, resampled_mmd2_parts

PER_IMAGE = {"clip_cos": True, "lpips": False, "ssim": True}      # metric -> higher is better
FEATURES = {"inception": "kid_inception", "dino": "kid_dino"}
B_MEAN, B_KID = 10_000, 1_000
SEED_MEAN, SEED_KID = 12345, 54321


def _real_term(real: np.ndarray, device=None) -> float:
    """mean_{i≠j} k(r_i, r_j), the real-only part of the unbiased MMD²."""
    m = len(real)
    if device is None:
        k = poly_kernel(real, real)
        return float((k.sum() - np.trace(k)) / (m * (m - 1)))
    import torch
    with torch.no_grad():
        r = torch.as_tensor(np.ascontiguousarray(real), dtype=torch.float32, device=device)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            k = r @ r.T
            k.div_(real.shape[1]).add_(1.0).pow_(3)
            total = float(sum(float(k[i:i + 2048].double().sum()) for i in range(0, m, 2048)))
            diag = float(torch.diagonal(k).double().sum())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        del k
    return (total - diag) / (m * (m - 1))


def analyze_set(sc: dict, set_name: str, with_fid: bool = False, device=None, progress=None,
                b_kid: int = B_KID, v3=None) -> dict:
    """sc: load_scores() of a K=16 run. Returns all per-set results.

    with_fid (protocol v2): also FID on Inception features for every rule × N, with tile
    bootstrap CIs on the same resamples as KID (B_KID, SEED_KID), so FID and KID CIs are paired.
    device: None = KID kernels in float64 NumPy (as for the pilot study); "cuda" = float32 GPU
    with float64 reductions (needed at the full-split size; see eotts.stats.kid).
    b_kid: KID/FID resamples; the protocol value B_KID, smaller only in smoke tests.
    v3: (T, K) scores of the VLM verifier fixed by amendment 2; adds the secondary rules
    V3_RULES. They are reported like the others but are not in the primary Holm family.
    """
    vs = {V1: sc[f"verifiers.{V1}"], V2: sc[f"verifiers.{V2}"]}
    rules = RULES
    if v3 is not None:
        vs[V3] = np.asarray(v3)
        rules = RULES + V3_RULES
    T, K = vs[V1].shape
    W = tile_weights(T, B_MEAN, SEED_MEAN).astype(np.float64)
    Wk = tile_weights(T, b_kid, SEED_KID)
    Wk_aug = np.vstack([np.ones((1, T), Wk.dtype), Wk])      # row 0 = the original sample
    ev = {m: sc[f"evaluators.{m}"] for m in PER_IMAGE}
    feats = {f: (sc[f"{f}.samples"], sc[f"{f}.gt"]) for f in FEATURES}
    real_term = {f: _real_term(feats[f][1], device) for f in FEATURES}
    sel = {"k0": np.zeros(T, int)}
    for rule in rules:
        for n in N_VALUES[1:]:
            sel[f"{rule}/{n}"] = select(rule, vs, n)
    key = lambda rule, n: "k0" if n == 1 else f"{rule}/{n}"
    parts = {f: {} for f in FEATURES}                          # (B_KID + 1,) per selection
    for f, (smp, gt) in feats.items():
        for kname, idx in sel.items():
            parts[f][kname] = resampled_mmd2_parts(take(smp, idx), gt, Wk_aug, device=device)
        if progress:
            progress(f"kid {f} done")

    out = {"set": set_name, "tiles": T, "samples_per_tile": K, "n_values": list(N_VALUES),
           "bootstrap": {"per_image_resamples": B_MEAN, "kid_resamples": b_kid,
                         "seeds": [SEED_MEAN, SEED_KID]},
           "kid_backend": "numpy-float64" if device is None else f"torch-float32-{device}",
           "rules": {}, "secondary_rules": [r for r in rules if r not in RULES]}
    for rule in rules:
        rr = {}
        for n in N_VALUES:
            idx = sel[key(rule, n)]
            cell = {"selected_index_hist": np.bincount(idx, minlength=K).tolist()}
            for m in PER_IMAGE:
                chosen = take(ev[m], idx)
                cell[m] = {"mean": float(chosen.mean()),
                           "diff_vs_n1": paired_mean_diff(chosen, ev[m][:, 0], W)}
            for f, name in FEATURES.items():
                p, p0 = parts[f][key(rule, n)], parts[f]["k0"]
                cell[name] = {"mmd2_full_sample": float(p[0]) + real_term[f],
                              "diff_vs_n1": summarize(float(p[0] - p0[0]), p[1:] - p0[1:])}
            rr[str(n)] = cell
        out["rules"][rule] = rr
    full_k0 = {f: float(parts[f]["k0"][0]) for f in FEATURES}

    if with_fid:
        smp, gt = feats["inception"]
        chosen = {k: take(smp, idx) for k, idx in sel.items()}
        fids = fid_resampled(chosen, gt, Wk_aug, progress=progress)
        out["n1_fid_full_sample"] = float(fids["k0"][0])
        for rule in rules:
            for n in N_VALUES:
                f_, f0 = fids[key(rule, n)], fids["k0"]
                out["rules"][rule][str(n)]["fid"] = {
                    "full_sample": float(f_[0]),
                    "diff_vs_n1": summarize(float(f_[0] - f0[0]), f_[1:] - f0[1:])}
        out["bootstrap"]["fid_resamples"] = b_kid

    out["oracle"] = {m: {str(n): float(take(ev[m], oracle(ev[m], n, hb)).mean()) for n in N_VALUES}
                     for m, hb in PER_IMAGE.items()}
    out["n1_kid_full_sample"] = {FEATURES[f]: full_k0[f] + real_term[f] for f in FEATURES}
    # torchmetrics-style subset KID at N = 1, for comparison with the baseline's reported KID.
    # The full-sample unbiased MMD² used above has the same expectation and is used for all
    # comparisons because it is computed on every tile.
    out["n1_kid_subsets"] = {FEATURES[f]: kid(feats[f][0][:, 0], feats[f][1]) for f in FEATURES}

    corr = {}
    for vname, v in vs.items():
        for m in PER_IMAGE:
            pooled = spearmanr(v.ravel(), ev[m].ravel()).statistic
            within = np.nanmean([spearmanr(v[t], ev[m][t]).statistic for t in range(T)])
            corr[f"{vname}~{m}"] = {"pooled": float(pooled), "within_tile_mean": float(within)}
    out["verifier_evaluator_spearman"] = corr
    out["verifier_validity_gt"] = {
        vname: {"own_caption_score_mean": float(sc[f"verifiers.{vname}__gt"].mean()),
                "own_caption_rank_pct_mean": float(sc[f"verifiers.{vname}__gt_rank_pct"].mean()),
                "note": "fraction of the set's other captions scoring above the tile's own caption on the "
                        "real image; 0 = always ranked first, 0.5 = chance"}
        for vname in (V1, V2)}
    return out


def endpoints(per_set: dict[str, dict]) -> dict:
    """Primary endpoints of the protocol, Holm-corrected over (verifier rule × set)."""
    pv, eff = {}, {}
    for s, r in per_set.items():
        for rule in RULES:
            d = r["rules"][rule]["16"]["clip_cos"]["diff_vs_n1"]
            pv[f"{s}/{rule}"] = d["p_boot"]
            eff[f"{s}/{rule}"] = d
    h = holm(pv)
    guard = {}
    for s, r in per_set.items():
        base = r["n1_kid_full_sample"]["kid_dino"]
        for rule in RULES:
            d = r["rules"][rule]["16"]["kid_dino"]["diff_vs_n1"]
            if base > 0:
                margin = 0.10 * base
                verdict = bool(d["ci_high"] < margin)
            else:   # amendment 2: a relative margin is undefined when the estimate at N = 1 is not positive
                margin, verdict = None, None
            guard[f"{s}/{rule}"] = {"diff": d, "margin": margin, "kid_dino_n1": base,
                                    "non_inferior": verdict}
    return {"primary_clip_n16_vs_n1": {k: {**eff[k], **h[k]} for k in eff},
            "guard_kid_dino_n16_vs_n1": guard}


def endpoints_v2(r: dict, set_name: str) -> dict:
    """Primary endpoints of protocol v2 (one set): Δ CLIP and Δ FID at N = 16 vs N = 1 for each
    verifier rule, Holm-corrected together over the 2 metrics × 3 rules = 6 tests. Guard as in v1:
    KID-DINOv2 at N = 16 non-inferior to N = 1 with a margin of 10 % of KID-DINOv2 at N = 1.
    """
    pv, eff = {}, {}
    for metric in ("clip_cos", "fid"):
        for rule in RULES:
            d = r["rules"][rule]["16"][metric]["diff_vs_n1"]
            pv[f"{metric}/{rule}"] = d["p_boot"]
            eff[f"{metric}/{rule}"] = d
    h = holm(pv)
    v1_style = endpoints({set_name: r})["guard_kid_dino_n16_vs_n1"]
    return {"set": set_name,
            "primary_n16_vs_n1": {k: {**eff[k], **h[k],
                                      "direction": "higher is better" if k.startswith("clip") else
                                                   "lower is better"} for k in eff},
            "guard_kid_dino_n16_vs_n1": {k.split("/", 1)[1]: v for k, v in v1_style.items()}}


def anchor_per_image(sc_a: dict, sc_b: dict, label_a: str, label_b: str, ka: int = 0, kb: int = 0,
                     with_fid: bool = False, b_kid: int = B_KID) -> dict:
    """Paired per-image differences a − b on the same tiles (E1–E3); KID of each; with_fid: FID
    of each and the paired FID difference with a tile-bootstrap CI."""
    T = sc_a["evaluators.clip_cos"].shape[0]
    W = tile_weights(T, B_MEAN, SEED_MEAN)
    out = {"a": label_a, "b": label_b}
    for m in PER_IMAGE:
        a, b = sc_a[f"evaluators.{m}"][:, ka], sc_b[f"evaluators.{m}"][:, kb]
        out[m] = {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": paired_mean_diff(a, b, W)}
    for f, name in FEATURES.items():
        if f"{f}.samples" in sc_a and f"{f}.samples" in sc_b:
            gt = sc_a[f"{f}.gt"]
            out[name] = {"a": kid(sc_a[f"{f}.samples"][:, ka], gt), "b": kid(sc_b[f"{f}.samples"][:, kb], gt)}
    if with_fid and "inception.samples" in sc_a and "inception.samples" in sc_b:
        gt = sc_a["inception.gt"]
        x = {"a": sc_a["inception.samples"][:, ka], "b": sc_b["inception.samples"][:, kb]}
        full = fid_resampled(x, gt, np.ones((1, T)))
        boot = fid_resampled(x, gt, tile_weights(T, b_kid, SEED_KID))
        out["fid"] = {"a": float(full["a"][0]), "b": float(full["b"][0]),
                      "diff": summarize(float(full["a"][0] - full["b"][0]), boot["a"] - boot["b"])}
    return out


def anchor_vs_reference(sc: dict, reference: dict, label: str, k: int = 0) -> dict:
    """Means with bootstrap CIs of E1–E3 next to reference values (full split or paper)."""
    T = sc["evaluators.clip_cos"].shape[0]
    W = tile_weights(T, B_MEAN, SEED_MEAN)
    out = {"label": label}
    for m in PER_IMAGE:
        v = sc[f"evaluators.{m}"][:, k].astype(np.float64)
        boot = W @ v / W.sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ref = reference.get(m)
        out[m] = {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "reference": ref,
                  "reference_inside_ci": None if ref is None else bool(lo <= ref <= hi)}
    return out


def psnr_same_noise(paths_a, paths_b) -> dict:
    from PIL import Image
    vals = []
    for a, b in zip(paths_a, paths_b):
        x = np.asarray(Image.open(a).convert("RGB"), np.float64) / 255
        y = np.asarray(Image.open(b).convert("RGB"), np.float64) / 255
        mse = max(((x - y) ** 2).mean(), 1e-12)
        vals.append(10 * np.log10(1 / mse))
    v = np.array(vals)
    return {"tiles": len(v), "psnr_db_mean": float(v.mean()), "psnr_db_min": float(v.min()),
            "psnr_db_p05": float(np.percentile(v, 5))}


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1)
