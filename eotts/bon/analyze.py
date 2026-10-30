"""Analysis of the Best-of-N protocol: selection, effects with CIs, endpoints, anchors.

Everything here is computed from the cached scores and features; no model is loaded.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import spearmanr

from eotts.bon.select import RULES, V1, V2, oracle, select, take
from eotts.bon.sets import N_VALUES
from eotts.stats.bootstrap import holm, paired_mean_diff, summarize, tile_weights
from eotts.stats.kid import kid, poly_kernel, resampled_mmd2_parts

PER_IMAGE = {"clip_cos": True, "lpips": False, "ssim": True}      # metric -> higher is better
FEATURES = {"inception": "kid_inception", "dino": "kid_dino"}
B_MEAN, B_KID = 10_000, 1_000
SEED_MEAN, SEED_KID = 12345, 54321


def _real_term(real: np.ndarray) -> float:
    k = poly_kernel(real, real)
    m = len(real)
    return float((k.sum() - np.trace(k)) / (m * (m - 1)))


def analyze_set(sc: dict, set_name: str) -> dict:
    """sc: load_scores() of a K=16 run. Returns all per-set results."""
    vs = {V1: sc[f"verifiers.{V1}"], V2: sc[f"verifiers.{V2}"]}
    T, K = vs[V1].shape
    W = tile_weights(T, B_MEAN, SEED_MEAN)
    Wk = tile_weights(T, B_KID, SEED_KID)
    ev = {m: sc[f"evaluators.{m}"] for m in PER_IMAGE}
    feats = {f: (sc[f"{f}.samples"], sc[f"{f}.gt"]) for f in FEATURES}
    real_term = {f: _real_term(feats[f][1]) for f in FEATURES}
    boot_parts_k0, full_k0 = {}, {}
    for f, (smp, gt) in feats.items():
        boot_parts_k0[f] = resampled_mmd2_parts(smp[:, 0], gt, Wk)
        full_k0[f] = float(resampled_mmd2_parts(smp[:, 0], gt, np.ones((1, T)))[0])

    out = {"set": set_name, "tiles": T, "samples_per_tile": K, "n_values": list(N_VALUES),
           "bootstrap": {"per_image_resamples": B_MEAN, "kid_resamples": B_KID,
                         "seeds": [SEED_MEAN, SEED_KID]}, "rules": {}}
    for rule in RULES:
        rr = {}
        for n in N_VALUES:
            idx = select(rule, vs, n)
            cell = {"selected_index_hist": np.bincount(idx, minlength=K).tolist()}
            for m in PER_IMAGE:
                sel = take(ev[m], idx)
                cell[m] = {"mean": float(sel.mean()),
                           "diff_vs_n1": paired_mean_diff(sel, ev[m][:, 0], W)}
            for f, name in FEATURES.items():
                smp, gt = feats[f]
                chosen = take(smp, idx)
                full = float(resampled_mmd2_parts(chosen, gt, np.ones((1, T)))[0]) + real_term[f]
                parts = resampled_mmd2_parts(chosen, gt, Wk)
                diff = summarize(full - real_term[f] - full_k0[f], parts - boot_parts_k0[f])
                cell[name] = {"mmd2_full_sample": full, "diff_vs_n1": diff}
            rr[str(n)] = cell
        out["rules"][rule] = rr

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
        for vname in vs}
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
            margin = 0.10 * base
            guard[f"{s}/{rule}"] = {"diff": d, "margin": margin, "kid_dino_n1": base,
                                    "non_inferior": d["ci_high"] < margin}
    return {"primary_clip_n16_vs_n1": {k: {**eff[k], **h[k]} for k in eff},
            "guard_kid_dino_n16_vs_n1": guard}


def anchor_per_image(sc_a: dict, sc_b: dict, label_a: str, label_b: str, ka: int = 0, kb: int = 0) -> dict:
    """Paired per-image differences a − b on the same tiles (E1–E3)."""
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
