"""Known answers for the metric kernels, ported from roast/test at 2bdf3aa0, plus property
tests against independent quadratic references (ties included)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from coast_eval import metrics as m
from coast_eval.observe import relative_winners

type F = NDArray[np.float64]
type B = NDArray[np.bool_]


def tau_a_naive(x: F, y: F) -> float:
    """O(m^2): sum over i < j of sign(dx) * sign(dy), over m(m-1)/2."""
    n = len(x)
    if n < 2:
        return 0.0
    s = sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i in range(n) for j in range(i + 1, n))
    return float(s) / (n * (n - 1) / 2)


def auc_naive(pos: F, neg: F) -> float:
    """O(n1 n2): mean over pairs of 1[p > q] + 0.5 * 1[p == q]."""
    return float(np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg]))


def cosine_naive(a: F, b: F, wa: B, wb: B, degenerate: float) -> float:
    """The Julia loop: accumulate over the union, degenerate on empty union or zero norm."""
    dot = na = nb = 0.0
    for d in range(len(a)):
        if wa[d] or wb[d]:
            dot += a[d] * b[d]
            na += a[d] * a[d]
            nb += b[d] * b[d]
    return degenerate if na == 0.0 or nb == 0.0 else dot / (math.sqrt(na) * math.sqrt(nb))


def arr(*v: float) -> F:
    return np.array(v, dtype=np.float64)


def bools(*v: int) -> B:
    return np.array(v, dtype=np.bool_)


def cos1(a: F, b: F, wa: B, wb: B, degenerate: float = 0.0) -> float:
    return float(m.cosine_matrix(a[None], wa[None], b[None], wb[None], degenerate=degenerate)[0, 0])


# --- Kendall tau-a ---------------------------------------------------------------------

TAU_CASES = {
    "cross-group ties": (arr(0, 0, 1, 1, 2), arr(0, 1, 1, 2, 2)),
    "x-ties only": (arr(1, 1, 1, 2, 3), arr(5, 3, 1, 4, 2)),
    "y-ties only": (arr(5, 3, 1, 4, 2), arr(1, 1, 1, 2, 3)),
    "singleton pair": (arr(0.7, 0.2), arr(0.9, 0.1)),
}


@pytest.mark.parametrize("name", TAU_CASES)
def test_tau_a_matches_naive_on_tie_patterns(name: str) -> None:
    x, y = TAU_CASES[name]
    assert m.kendall_tau_a(x, y) == pytest.approx(tau_a_naive(x, y), abs=1e-12)


def test_tau_a_known_values() -> None:
    v = arr(3, 1, 4, 7, 5, 9, 2, 6)
    assert m.kendall_tau_a(v, v) == pytest.approx(1.0, abs=1e-12)
    w = arr(1, 2, 3, 4, 5, 6)
    assert m.kendall_tau_a(w, w[::-1].copy()) == pytest.approx(-1.0, abs=1e-12)
    # One adjacent swap among 5: 9 concordant, 1 discordant; two swaps: 8 and 2.
    x = arr(1, 2, 3, 4, 5)
    assert m.kendall_tau_a(x, arr(1, 2, 4, 3, 5)) == pytest.approx(0.8, abs=1e-12)
    assert m.kendall_tau_a(x, arr(2, 1, 4, 3, 5)) == pytest.approx(0.6, abs=1e-12)
    # All-equal on either side: no untied pair.
    r = np.random.default_rng(1).standard_normal(6)
    assert m.kendall_tau_a(np.zeros(6), r) == 0.0
    assert m.kendall_tau_a(r, np.zeros(6)) == 0.0


def test_tau_a_ties_count_against_agreement() -> None:
    # tau-b would give 1 for identical inputs; tau-a keeps the tied pair in the denominator.
    vt = arr(3, 1, 4, 1, 5, 9, 2, 6)
    assert m.kendall_tau_a(vt, vt) == pytest.approx(27 / 28, abs=1e-12)
    assert m.kendall_tau_a(arr(1, 1, 2), arr(1, 1, 2)) == pytest.approx(2 / 3, abs=1e-12)


def test_tau_a_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        m.kendall_tau_a(arr(1, 2), arr(1))
    with pytest.raises(ValueError, match="NaN"):
        m.kendall_tau_a(np.full(5, np.nan), arr(1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="NaN"):
        m.kendall_tau_a(arr(1, 2, 3, 4, 5), np.full(5, np.nan))


@settings(deadline=None)
@given(st.data())
def test_tau_a_property_vs_naive(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 40))
    x = np.array(data.draw(st.lists(st.integers(-3, 3), min_size=n, max_size=n)), dtype=float)
    y = np.array(data.draw(st.lists(st.integers(-3, 3), min_size=n, max_size=n)), dtype=float)
    assert m.kendall_tau_a(x, y) == pytest.approx(tau_a_naive(x, y), abs=1e-12)


def test_tau_a_large_continuous_vs_naive() -> None:
    # Several merge levels with an odd tail block.
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal(301), rng.standard_normal(301)
    y[::7] = x[::7]
    assert m.kendall_tau_a(x, y) == pytest.approx(tau_a_naive(x, y), abs=1e-12)


# --- Mann-Whitney AUC ------------------------------------------------------------------


def test_auc_known_values() -> None:
    assert m.mann_whitney_auc(arr(2, 3), arr(0, 1)) == 1.0
    assert m.mann_whitney_auc(arr(0, 1), arr(2, 3)) == 0.0
    assert m.mann_whitney_auc(arr(1, 1, 1), arr(1, 1)) == 0.5
    # One tied cross pair gets half credit: (0.5 + 1 + 1 + 1) / 4.
    assert m.mann_whitney_auc(arr(1, 2), arr(1, 0)) == 0.875


def test_auc_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        m.mann_whitney_auc(arr(), arr(1))
    with pytest.raises(ValueError, match="non-finite"):
        m.mann_whitney_auc(arr(np.nan), arr(1))
    with pytest.raises(ValueError, match="non-finite"):
        m.mann_whitney_auc(arr(1), arr(np.inf))


@settings(deadline=None)
@given(
    st.lists(st.integers(0, 4), min_size=1, max_size=30),
    st.lists(st.integers(0, 4), min_size=1, max_size=30),
)
def test_auc_property_vs_naive(pos: list[int], neg: list[int]) -> None:
    p, q = np.array(pos, dtype=float), np.array(neg, dtype=float)
    assert m.mann_whitney_auc(p, q) == pytest.approx(auc_naive(p, q), abs=1e-12)


# --- Winner-union cosine ---------------------------------------------------------------


def test_cosine_full_vector_cases() -> None:
    full = bools(1, 1, 1)
    assert cos1(arr(1, 2, 3), arr(1, 2, 3), full, full) == pytest.approx(1.0)
    assert cos1(arr(1, 0, 0), arr(0, 1, 0), full, full) == 0.0
    assert cos1(arr(1, 0, 0), arr(-1, 0, 0), full, full) == pytest.approx(-1.0)
    assert cos1(arr(1, 1, 0), arr(1, 0, 0), full, full) == pytest.approx(1 / math.sqrt(2))


def test_cosine_masks_to_the_union() -> None:
    # Unit 2 is outside the union, so its 9 is dropped; unit 1 is in the union only
    # because b won there, and a's raw 2 still counts: [1, 2] . [1, 1] / (sqrt5 sqrt2).
    a, wa = arr(1, 2, 9), bools(1, 0, 0)
    b, wb = arr(1, 1, 0), bools(1, 1, 0)
    assert cos1(a, b, wa, wb) == pytest.approx(3 / math.sqrt(10), abs=1e-15)
    # Hand-computed cells of the Julia model_rdm test: e1, e1 + e2, 2 e3.
    codes = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 2, 0]], dtype=np.float32)
    s = m.cosine_matrix(codes, codes > 0, codes, codes > 0, degenerate=0.0)
    assert s[0, 1] == pytest.approx(1 / math.sqrt(2)) and s[0, 2] == 0.0 and s[1, 2] == 0.0
    assert np.array_equal(s, s.T)


@pytest.mark.parametrize("degenerate", [0.0, 1.0])
def test_cosine_degenerate_pairs(degenerate: float) -> None:
    # Empty union.
    none = bools(0, 0, 0)
    assert cos1(arr(1, 2, 3), arr(3, 2, 1), none, none, degenerate) == degenerate
    # Non-empty union, zero norm on one side only.
    assert cos1(arr(0, 0, 5), arr(1, 1, 0), bools(1, 0, 0), bools(0, 1, 0), degenerate) == (
        degenerate
    )
    # Not degenerate: the convention must not leak into a regular pair.
    assert cos1(arr(1, 0, 0), arr(0, 1, 0), bools(1, 0, 0), bools(0, 1, 0), degenerate) == 0.0


def test_cosine_refuses_bad_input() -> None:
    a = np.ones((2, 3))
    with pytest.raises(ValueError, match="widths differ"):
        m.cosine_matrix(a, a > 0, np.ones((2, 4)), np.ones((2, 4)) > 0, degenerate=0.0)
    with pytest.raises(ValueError, match="masks bool"):
        m.cosine_matrix(a, a, a, a > 0, degenerate=0.0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="non-finite"):
        m.cosine_matrix(np.array([[np.nan, 1, 1]]), a[:1] > 0, a, a > 0, degenerate=0.0)


@settings(deadline=None)
@given(st.data())
def test_cosine_property_vs_loop(data: st.DataObject) -> None:
    n = data.draw(st.integers(1, 6))
    p, q = data.draw(st.integers(1, 4)), data.draw(st.integers(1, 4))
    ints = st.lists(st.integers(-2, 2), min_size=n, max_size=n)
    flags = st.lists(st.booleans(), min_size=n, max_size=n)
    a = np.array([data.draw(ints) for _ in range(p)], dtype=float)
    b = np.array([data.draw(ints) for _ in range(q)], dtype=float)
    wa = np.array([data.draw(flags) for _ in range(p)], dtype=bool)
    wb = np.array([data.draw(flags) for _ in range(q)], dtype=bool)
    degenerate = data.draw(st.sampled_from([0.0, 1.0]))
    s = m.cosine_matrix(a, wa, b, wb, degenerate=degenerate)
    for i in range(p):
        for j in range(q):
            assert s[i, j] == pytest.approx(cosine_naive(a[i], b[j], wa[i], wb[j], degenerate))


# --- Identity AUC, separation, preservation --------------------------------------------


def relative(v: F) -> B:
    return relative_winners(v, 0.3)


CLEAN = arr(0.5, 1, 0, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1).reshape(3, 4)


IDS = np.arange(3)


def auc_of(clean: F, corrupt: F) -> float:
    ids = np.arange(len(clean))
    return m.matched_mismatched_auc(
        clean, relative(clean), corrupt, relative(corrupt), clean_ids=ids, corrupt_ids=ids
    )


def test_matched_mismatched_auc_known_values() -> None:
    collapsed = np.tile(arr(1, 0, 0, 0), (3, 1))
    assert auc_of(CLEAN, collapsed) == 0.5
    assert auc_of(CLEAN, CLEAN) == 1.0
    assert auc_of(CLEAN, CLEAN[[1, 2, 0]]) == 0.25
    assert auc_of(np.zeros((3, 4)), np.zeros((3, 4))) == 0.5


def test_matched_mismatched_auc_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match=">= 2 aligned"):
        auc_of(CLEAN[:1], CLEAN[:1])
    with pytest.raises(ValueError, match=">= 2 aligned"):
        auc_of(CLEAN, CLEAN[:2])
    with pytest.raises(ValueError, match="widths differ"):
        auc_of(CLEAN, np.ones((3, 3)))
    bad = CLEAN.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        m.matched_mismatched_auc(
            bad, relative(CLEAN), CLEAN, relative(CLEAN), clean_ids=IDS, corrupt_ids=IDS
        )


@pytest.mark.parametrize(
    ("clean_ids", "corrupt_ids", "message"),
    [
        ([0, 1, 2], [1, 2, 0], "equal and unique"),  # same count, different order
        ([0, 0, 2], [0, 0, 2], "equal and unique"),  # a repeated image
        ([0, 1], [0, 1], "stim_ids for 3"),
    ],
)
def test_matched_mismatched_auc_checks_alignment(
    clean_ids: list[int], corrupt_ids: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        m.matched_mismatched_auc(
            CLEAN,
            relative(CLEAN),
            CLEAN,
            relative(CLEAN),
            clean_ids=np.array(clean_ids),
            corrupt_ids=np.array(corrupt_ids),
        )


def test_separation_counts_degenerate_pairs_as_unseparated() -> None:
    # Two live codes on disjoint units and one silent (empty) code.
    amps = arr(1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2).reshape(3, 4)
    masks = relative(amps)
    assert not masks[1].any()
    # Pairs: (live, dead) = 1, (live, live) = 0, (dead, live) = 1.
    assert m.separation(amps, masks, stim_ids=IDS) == pytest.approx(2 / 3)
    # A silent population is not "perfectly separated".
    silent = np.zeros((6, 5)), np.zeros((6, 5), dtype=bool)
    assert m.separation(*silent, stim_ids=np.arange(6)) == 1.0
    # The primary keeps 0.0 for the same degenerate pairs.
    s = m.cosine_matrix(amps, masks, amps, masks, degenerate=0.0)
    assert s[0, 1] == 0.0 and s[1, 2] == 0.0
    with pytest.raises(ValueError, match="different images"):
        m.separation(amps[:1], masks[:1], stim_ids=IDS[:1])
    with pytest.raises(ValueError, match="stim_ids for 3"):
        m.separation(amps, masks, stim_ids=IDS[:2])


def test_separation_skips_pairs_of_the_same_image() -> None:
    # Images 0 and 1 shown twice each: only the four cross-image pairs count, and the
    # two same-image pairs (cosine 1) must not raise the mean.
    amps = arr(1, 0, 0, 1, 1, 0, 0, 1).reshape(4, 2)
    masks = amps > 0
    assert m.separation(amps, masks, stim_ids=np.array([0, 1, 0, 1])) == 0.0
    assert m.separation(amps, masks, stim_ids=np.arange(4)) == pytest.approx(2 / 6)
    with pytest.raises(ValueError, match="different images"):
        m.separation(amps, masks, stim_ids=np.zeros(4, dtype=np.int64))


def test_shuffled_units_keep_each_code_and_lose_its_units() -> None:
    rng = np.random.default_rng(0)
    amps = rng.random((40, 12)) + 0.1
    masks = rng.random((40, 12)) < np.linspace(0.1, 0.9, 40)[:, None]  # sizes vary by row
    masks[:, 0] = True  # unit 0 wins everywhere
    a, w = m.shuffled_units(amps, masks, 1)
    assert np.array_equal(w.sum(axis=1), masks.sum(axis=1))
    assert np.array_equal(np.sort(a, axis=1), np.sort(amps, axis=1))
    for row in range(40):  # each amplitude keeps its winner flag: one permutation for both
        assert sorted(a[row][w[row]]) == sorted(amps[row][masks[row]])
    assert not w[:, 0].all() and m.domination(w) < m.domination(masks) == 1.0
    again, _ = m.shuffled_units(amps, masks, 1)
    other, _ = m.shuffled_units(amps, masks, 2)
    assert np.array_equal(a, again) and not np.array_equal(a, other)
    with pytest.raises(ValueError, match="amps"):
        m.shuffled_units(amps[:, :3], masks, 1)


def test_chance_levels_rise_with_code_size_and_real_structure_stands_above_them() -> None:
    rng = np.random.default_rng(1)
    ids = np.arange(60)

    def amps(masks: B) -> F:  # strong winners over weak non-winners, as in a settled code
        return np.where(masks, 1 + rng.random(masks.shape), 0.01 * rng.random(masks.shape))

    def chance(size: int) -> tuple[float, float]:
        masks = np.argsort(rng.random((60, 50)), axis=1) < size
        a, w = m.shuffled_units(amps(masks), masks, 3)
        return m.separation(a, w, stim_ids=ids), m.domination(w)

    assert chance(5)[0] < chance(18)[0] and chance(5)[1] < chance(18)[1]
    # All presentations on the same 5 units: size 5, yet far above chance at size 5.
    same = np.zeros((60, 50), dtype=bool)
    same[:, :5] = True
    assert m.separation(amps(same), same, stim_ids=ids) > chance(5)[0] + 0.3
    assert m.domination(same) == 1.0 > chance(5)[1] + 0.3


def test_similarity_preservation_known_values() -> None:
    px = arr(1, 0, 0, 1, 1, 0, 0, 0, 1).reshape(3, 3)  # e1, e1 + e2, e3
    full = np.ones((3, 3), dtype=bool)
    # Distances (d01, d02, d12) = (1 - 1/sqrt2, 1, 1) on both sides: the tied pair is lost.
    assert m.similarity_preservation(px, px, full) == pytest.approx(2 / 3)
    # Codes e3, e1 + e2, e1: distances (1, 1, 1 - 1/sqrt2); pairs score 0, -1, 0.
    codes = arr(0, 0, 1, 1, 1, 0, 1, 0, 0).reshape(3, 3)
    assert m.similarity_preservation(px, codes, full) == pytest.approx(-1 / 3)
    with pytest.raises(ValueError, match="pixels"):
        m.similarity_preservation(px[:2], px, full)


def test_similarity_preservation_rounds_code_distances_to_float32() -> None:
    # Code distances (1, 1 - 1e-9, 1) are distinct in float64 but all round to 1.0f, so
    # the code side is fully tied; the pixel distances are distinct.
    px = arr(1, 0, 0, 1, 0.5, 0, 1, 2, 2).reshape(3, 3)
    codes = arr(1, 0, 0, 0, 1, 0, 1e-9, 0, 1).reshape(3, 3)
    full = np.ones((3, 3), dtype=bool)
    d = 1.0 - m.cosine_matrix(codes, full, codes, full, degenerate=0.0)
    assert d[0, 2] != d[0, 1] and np.float32(d[0, 2]) == np.float32(d[0, 1])
    pxd = 1.0 - m.cosine_matrix(px, full, px, full, degenerate=0.0)
    iu = np.triu_indices(3, k=1)
    expected = tau_a_naive(pxd[iu], d[iu].astype(np.float32).astype(np.float64))
    assert expected != tau_a_naive(pxd[iu], d[iu])
    assert m.similarity_preservation(px, codes, full) == pytest.approx(expected)


# --- Win statistics and the structural bound -------------------------------------------


def test_never_wins_and_domination() -> None:
    masks = bools(1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0).reshape(3, 4)  # unit 3 never wins
    assert m.never_wins_fraction(masks) == 0.25
    assert m.domination(masks) == pytest.approx(2 / 3)  # unit 0 wins 2 of 3
    for f in (m.never_wins_fraction, m.domination):
        with pytest.raises(ValueError, match="non-empty bool"):
            f(np.zeros((0, 4), dtype=bool))
        with pytest.raises(ValueError, match="non-empty bool"):
            f(masks.astype(float))  # pyright: ignore[reportArgumentType]
