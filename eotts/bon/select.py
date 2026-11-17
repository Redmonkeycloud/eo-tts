"""Best-of-N selection rules. Candidates of BoN-N are samples 0 … N−1 (nested sets)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

V1, V2 = "v1_remoteclip_b32", "v2_georsclip_l14"
V3 = "v3"                      # the VLM verifier fixed by protocol v2 amendment 2, under this key
RULES = ("v1", "v2", "v1+v2")  # primary rules: the Holm family of protocol v2
V3_RULES = ("v3", "v1+v3")     # secondary rules (amendment 2), used only when V3 scores exist


def best_of(scores: np.ndarray, n: int) -> np.ndarray:
    """Index of the highest-scoring sample among the first n, per tile. scores: (T, K)."""
    return np.argmax(scores[:, :n], axis=1)


def rank_average(s1: np.ndarray, s2: np.ndarray, n: int) -> np.ndarray:
    """Highest mean rank under two verifiers among the first n; ties broken by s2."""
    r1 = rankdata(s1[:, :n], axis=1)            # 1 = lowest score
    r2 = rankdata(s2[:, :n], axis=1)
    mean = (r1 + r2) / 2
    # lexicographic: mean rank first, then s2
    order = np.lexsort((s2[:, :n], mean), axis=1)  # ascending; last = best
    return order[:, -1]


def select(rule: str, verifier_scores: dict, n: int) -> np.ndarray:
    if n == 1:
        return np.zeros(len(verifier_scores[V1]), dtype=int)
    if rule == "v1":
        return best_of(verifier_scores[V1], n)
    if rule == "v2":
        return best_of(verifier_scores[V2], n)
    if rule == "v1+v2":
        return rank_average(verifier_scores[V1], verifier_scores[V2], n)
    if rule == "v3":
        return best_of(verifier_scores[V3], n)
    if rule == "v1+v3":                       # ties broken by V3
        return rank_average(verifier_scores[V1], verifier_scores[V3], n)
    raise ValueError(rule)


def oracle(values: np.ndarray, n: int, higher_is_better: bool) -> np.ndarray:
    return np.argmax(values[:, :n], 1) if higher_is_better else np.argmin(values[:, :n], 1)


def take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """values: (T, K) or (T, K, D); idx: (T,) -> (T,) or (T, D)."""
    return values[np.arange(len(idx)), idx]
