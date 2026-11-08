"""Candidate signal-vs-noise measures for non-negative vectors.

Every function maps an array (..., n) to (...,) and is vectorised over rows.
All measures except `nbr_corr` are permutation-invariant (functions of the
value multiset only); `nbr_corr` uses the image grid and exists as a contrast.
Orientation convention: larger value = more concentrated / "more signal",
except the entropies (nse, nse_minsub, nse_medsub, renyi2), where larger =
flatter, as in the model.
"""
import numpy as np

from ngm2_math import nse

TINY = 1e-12


def _sorted_desc(v):
    return -np.sort(-v, axis=-1)


def gini(v):
    s = np.sort(v, axis=-1)
    n = v.shape[-1]
    i = np.arange(1, n + 1)
    tot = s.sum(-1)
    return np.where(tot > TINY, (2 * (i * s).sum(-1) / (n * np.maximum(tot, TINY))) - (n + 1) / n, 0.0)


def hoyer(v):
    n = v.shape[-1]
    l1 = np.abs(v).sum(-1)
    l2 = np.sqrt((v * v).sum(-1))
    return np.where(l2 > TINY, (np.sqrt(n) - l1 / np.maximum(l2, TINY)) / (np.sqrt(n) - 1), 0.0)


def renyi2(v):
    """Normalised collision entropy log(1/sum p^2)/log n (1 = flat)."""
    n = v.shape[-1]
    s = v.sum(-1, keepdims=True)
    p = v / np.maximum(s, TINY)
    return np.log(1.0 / np.maximum((p * p).sum(-1), TINY)) / np.log(n)


def kurtosis(v):
    m = v.mean(-1, keepdims=True)
    d = v - m
    var = (d * d).mean(-1)
    return np.where(var > TINY, (d ** 4).mean(-1) / np.maximum(var, TINY) ** 2 - 3.0, 0.0)


def peak_to_mean(v):
    return v.max(-1) / np.maximum(v.mean(-1), TINY)


def cv(v):
    return v.std(-1) / np.maximum(v.mean(-1), TINY)


def topk_mass(v, frac=0.1):
    k = max(1, int(round(v.shape[-1] * frac)))
    s = _sorted_desc(v)
    return s[..., :k].sum(-1) / np.maximum(v.sum(-1), TINY)


def margin(v):
    s = _sorted_desc(v)
    return (s[..., 0] - s[..., 1]) / np.maximum(s[..., 0], TINY)


def robust_z(v):
    med = np.median(v, -1)
    mad = np.median(np.abs(v - med[..., None]), -1)
    return (v.max(-1) - med) / np.maximum(1.4826 * mad, TINY)


def nse_minsub(v):
    return nse(v - v.min(-1, keepdims=True))


def nse_medsub(v):
    return nse(np.maximum(v - np.median(v, -1, keepdims=True), 0.0))


def nbr_corr(v, shape):
    """Mean lag-1 Pearson correlation of horizontally and vertically adjacent
    pixels; shape = (h, w, c) with c-last memory layout of the vector."""
    h, w, c = shape
    x = v.reshape(v.shape[:-1] + (h, w, c)).astype(np.float64)

    def corr(a, b):
        a = a.reshape(a.shape[: -3] + (-1,))
        b = b.reshape(b.shape[: -3] + (-1,))
        a = a - a.mean(-1, keepdims=True)
        b = b - b.mean(-1, keepdims=True)
        den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
        return np.where(den > TINY, (a * b).sum(-1) / np.maximum(den, TINY), 0.0)

    return 0.5 * (corr(x[..., :, 1:, :], x[..., :, :-1, :]) +
                  corr(x[..., 1:, :, :], x[..., :-1, :, :]))


MEASURES = {
    "nse": nse,
    "nse_minsub": nse_minsub,
    "nse_medsub": nse_medsub,
    "renyi2": renyi2,
    "gini": gini,
    "hoyer": hoyer,
    "kurtosis": kurtosis,
    "peak_to_mean": peak_to_mean,
    "cv": cv,
    "top10_mass": topk_mass,
    "margin": margin,
    "robust_z": robust_z,
}


def compute_all(v):
    v = np.asarray(v, dtype=np.float64)
    return {k: np.asarray(f(v), dtype=np.float64) for k, f in MEASURES.items()}


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie) (Mann-Whitney); 0.5 = no separation."""
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
