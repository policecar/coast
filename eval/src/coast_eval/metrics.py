"""Metric kernels of the frozen evaluation (issue #82), ported from the Julia harness.

Codes are two (P, N) arrays over P presentations and N units: `amps` (float, the raw
reduced amplitudes) and `masks` (bool, the winner sets). Every function is pure and
raises ValueError on input it cannot measure; none returns a default for bad data.

Conventions that the reported numbers depend on (keep them):
- Cosine is taken over the winner union of the two codes. A degenerate pair (empty union,
  or zero norm on either side) scores 0.0 in the primary metrics and 1.0 ("not
  separated") in `separation`.
- Kendall tau-a keeps ties in its denominator, so ties count against agreement.
- Code distances are rounded to float32 before ranking, as the Julia `model_rdm` did.
"""

from __future__ import annotations

import numpy as np
import scipy.stats
from numpy.typing import NDArray

type Floats = NDArray[np.floating]
type Bools = NDArray[np.bool_]


def _tie_pairs(ranks: NDArray[np.int64]) -> int:
    """Number of pairs with equal rank: the sum of t(t-1)/2 over the tie groups."""
    counts = np.unique(ranks, return_counts=True)[1]
    return int((counts * (counts - 1) // 2).sum())


def _dense_ranks(v: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.unique(v, return_inverse=True)[1].astype(np.int64)


def _inversions(a: NDArray[np.int64]) -> int:
    """Pairs i < j with a[i] > a[j], counted by a bottom-up merge sort (Knight 1966).

    `a` holds dense ranks 0..k-1. Each level merges blocks of `width`: a right-half
    element passes every larger element of its left half, which is one inversion each.
    """
    n = a.size
    k = int(a.max()) + 1 if n else 1
    pos = np.arange(n)
    cur = a.copy()
    total = 0
    width = 1
    while width < n:
        pair = pos // (2 * width)
        right = (pos // width) % 2 == 1
        key = pair * k + cur  # sorted within each half, half blocks in position order
        left_keys = key[~right]
        passed = np.searchsorted(left_keys, key[right], side="right")
        left_end = np.searchsorted(left_keys, (pair[right] + 1) * k, side="left")
        total += int((left_end - passed).sum())
        cur = np.sort(key) - pair * k
        width *= 2
    return total


def kendall_tau_a(x: Floats, y: Floats) -> float:
    """Kendall tau-a = (concordant - discordant) / (m(m-1)/2), ties in the denominator.

    O(m log m): sort by (x, y); the discordant pairs are the inversions of y in that
    order. m < 2 gives 0.0 (no pairs).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"kendall_tau_a: need two equal-length vectors, got {x.shape}, {y.shape}")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("kendall_tau_a: input contains NaN")
    m = x.size
    if m < 2:
        return 0.0
    rx, ry = _dense_ranks(x), _dense_ranks(y)
    discordant = _inversions(ry[np.lexsort((ry, rx))])
    pairs = m * (m - 1) // 2
    both = _tie_pairs(rx * (int(ry.max()) + 1) + ry)
    concordant = pairs - discordant - _tie_pairs(rx) - _tie_pairs(ry) + both
    return (concordant - discordant) / pairs


def mann_whitney_auc(positive: Floats, negative: Floats) -> float:
    """P(positive > negative) + 0.5 * P(tie), from mid-ranks of the pooled values.

    0.5 is chance; an all-tied input reads exactly 0.5.
    """
    pos = np.asarray(positive, dtype=np.float64).ravel()
    neg = np.asarray(negative, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("mann_whitney_auc: both groups must be non-empty")
    if not (np.isfinite(pos).all() and np.isfinite(neg).all()):
        raise ValueError("mann_whitney_auc: non-finite value in the input")
    ranks = scipy.stats.rankdata(np.concatenate([pos, neg]))
    n1, n2 = pos.size, neg.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def _check_codes(amps: Floats, masks: Bools, what: str) -> None:
    if amps.ndim != 2 or amps.shape != masks.shape or masks.dtype != np.bool_:
        raise ValueError(
            f"{what}: amps {amps.shape} and masks {masks.shape} ({masks.dtype}) must be "
            "(presentations, units), masks bool"
        )
    if not np.isfinite(amps).all():
        raise ValueError(f"{what}: non-finite amplitude")


def cosine_matrix(
    a: Floats, wa: Bools, b: Floats, wb: Bools, *, degenerate: float
) -> NDArray[np.float64]:
    """S[i, j] = cosine of a[i] and b[j] over the winner union wa[i] | wb[j].

    Units in the union carry each code's raw amplitude, even where only the other code
    won; units outside it are dropped. A degenerate pair (empty union, or either masked
    norm zero) scores `degenerate`. Full-vector cosine is the case of all-true masks.
    """
    _check_codes(a, wa, "cosine_matrix")
    _check_codes(b, wb, "cosine_matrix")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"cosine_matrix: code widths differ ({a.shape[1]} vs {b.shape[1]})")
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    out = np.empty((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        union = wa[i] | wb
        ai = np.where(union, a64[i], 0.0)
        bj = np.where(union, b64, 0.0)
        na = (ai * ai).sum(axis=1)
        nb = (bj * bj).sum(axis=1)
        ok = (na > 0.0) & (nb > 0.0)
        dot = (ai * bj).sum(axis=1)
        out[i] = np.where(ok, dot / np.sqrt(np.where(ok, na * nb, 1.0)), degenerate)
    return out


def similarity_preservation(pixels: Floats, amps: Floats, masks: Bools) -> float:
    """Kendall tau-a between pixel-cosine and code-cosine distances over all image pairs.

    Pixel distances are 1 - full cosine; code distances are 1 - winner-union cosine
    (degenerate 0.0), rounded to float32. Rows of `pixels` and of the codes are the same
    presentations in the same order.
    """
    if pixels.ndim != 2 or pixels.shape[0] != amps.shape[0]:
        raise ValueError(f"similarity_preservation: pixels {pixels.shape} vs codes {amps.shape}")
    full = np.ones(pixels.shape, dtype=np.bool_)
    pixel_dist = 1.0 - cosine_matrix(pixels, full, pixels, full, degenerate=0.0)
    code_dist = (1.0 - cosine_matrix(amps, masks, amps, masks, degenerate=0.0)).astype(np.float32)
    upper = np.triu_indices(amps.shape[0], k=1)
    return kendall_tau_a(pixel_dist[upper], code_dist[upper].astype(np.float64))


def _check_ids(ids: NDArray[np.integer], n: int, what: str) -> None:
    if ids.shape != (n,):
        raise ValueError(f"{what}: {ids.shape} stim_ids for {n} presentations")


def matched_mismatched_auc(
    clean_amps: Floats,
    clean_masks: Bools,
    corrupt_amps: Floats,
    corrupt_masks: Bools,
    *,
    clean_ids: NDArray[np.integer],
    corrupt_ids: NDArray[np.integer],
) -> float:
    """AUC of clean-vs-own-corrupted cosine against clean-vs-other-corrupted cosine.

    Row i of both code sets must be the same dataset image: the two runs' `stim_ids`
    must be equal and unique. Matched pairs are (i, i), the mismatched pairs every
    (i, j), i != j; degenerate pairs score 0.0. 0.5 = chance.
    """
    n = clean_amps.shape[0]
    if n < 2 or corrupt_amps.shape[0] != n:
        raise ValueError(
            f"matched_mismatched_auc: need >= 2 aligned presentations, got {n} clean and "
            f"{corrupt_amps.shape[0]} corrupted"
        )
    _check_ids(clean_ids, n, "matched_mismatched_auc")
    _check_ids(corrupt_ids, n, "matched_mismatched_auc")
    if not np.array_equal(clean_ids, corrupt_ids) or np.unique(clean_ids).size != n:
        raise ValueError(
            "matched_mismatched_auc: clean and corrupted stim_ids must be equal and unique"
        )
    s = cosine_matrix(clean_amps, clean_masks, corrupt_amps, corrupt_masks, degenerate=0.0)
    off = ~np.eye(n, dtype=np.bool_)
    return mann_whitney_auc(np.diag(s), s[off])


def separation(amps: Floats, masks: Bools, *, stim_ids: NDArray[np.integer]) -> float:
    """Mean winner-union cosine over the pairs i < j whose `stim_ids` differ.

    A degenerate pair counts as 1.0: two codes without amplitude are indistinguishable,
    so a silent population does not read as well separated.
    """
    _check_ids(stim_ids, amps.shape[0], "separation")
    i, j = np.triu_indices(amps.shape[0], k=1)
    different = stim_ids[i] != stim_ids[j]
    if not different.any():
        raise ValueError("separation: need >= 2 presentations of different images")
    s = cosine_matrix(amps, masks, amps, masks, degenerate=1.0)
    return float(s[i[different], j[different]].mean())


def _check_masks(masks: Bools, what: str) -> None:
    if masks.ndim != 2 or masks.dtype != np.bool_ or 0 in masks.shape:
        raise ValueError(f"{what}: masks must be a non-empty bool (presentations, units) array")


def never_wins_fraction(masks: Bools) -> float:
    """Fraction of units that win on no presentation."""
    _check_masks(masks, "never_wins_fraction")
    return float((~masks.any(axis=0)).mean())


def domination(masks: Bools) -> float:
    """Largest share of presentations won by a single unit."""
    _check_masks(masks, "domination")
    return float(masks.sum(axis=0).max() / masks.shape[0])


def shuffled_units(amps: Floats, masks: Bools, seed: int) -> tuple[Floats, Bools]:
    """The codes with each presentation's units permuted independently. Code sizes and
    amplitudes stay and which units carry them goes, so `separation` and `domination` of the
    result are their chance levels: both rise with code size alone."""
    _check_masks(masks, "shuffled_units")
    if amps.shape != masks.shape:
        raise ValueError(f"shuffled_units: amps {amps.shape} vs masks {masks.shape}")
    order = np.argsort(np.random.default_rng(seed).random(masks.shape), axis=1)
    return np.take_along_axis(amps, order, axis=1), np.take_along_axis(masks, order, axis=1)
