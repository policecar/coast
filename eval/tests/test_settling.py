"""The settling-curve readout on synthetic per-step amplitudes with known answers.

Four presentations of K = 30 steps over 4 units, the reference code the mean of steps 3..15
(the readout the curve was measured against: lead 2, window 13), frac 0.3:
- `settles`: the previous image's pattern on steps 1-2, then one pattern from step 3 on,
  with one unit just above and one just below the 0.3 threshold.
- `never`: alternates between two single-unit patterns on every step.
- `empty`: a negative constant, so the maximum is not positive and every code is empty.
- `drifts`: one pattern up to step 15, another from step 16, so ref and ref30 differ.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from coast_eval import observe as ob
from coast_eval import settling as st

K = 30
REF = {"ref_first": 3, "ref_last": 15}


def presentations() -> NDArray[np.float32]:
    a = np.zeros((4, K, 4), dtype=np.float32)
    a[0, :2] = [1, 0, 0, 0]
    a[0, 2:] = [0, 1, 0.35, 0.25]
    a[1, 0::2] = [1, 0, 0, 0]  # odd k
    a[1, 1::2] = [0, 0, 0, 1]  # even k
    a[2] = [-1, -2, -3, -4]
    a[3, :15] = [0, 0, 1, 0]
    a[3, 15:] = [3, 0, 0, 0]
    return a


def by_step(*runs: tuple[int, float]) -> NDArray[np.float64]:
    """A per-step series from (number of steps, value) runs."""
    return np.concatenate([np.full(n, v) for n, v in runs])


def test_curve_measures() -> None:
    c = st.curve(presentations(), frac=0.3, **REF)
    # Per presentation, by k = 1..30: settles, never, empty, drifts.
    jac = [by_step((2, 0), (28, 1)), by_step((30, 0.5)), by_step((30, 1)),
           by_step((15, 1), (15, 0))]  # fmt: skip
    exact = [by_step((2, 0), (28, 1)), by_step((30, 0)), by_step((30, 1)),
             by_step((15, 1), (15, 0))]  # fmt: skip
    # cosine against step 15: never shows unit 0 there (k = 15 is odd)
    cos = [by_step((2, 0), (28, 1)), np.tile([1.0, 0.0], 15), by_step((30, 1)),
           by_step((15, 1), (15, 0))]  # fmt: skip
    size = [by_step((2, 1), (28, 2)), by_step((30, 1)), by_step((30, 0)), by_step((30, 1))]
    # avg_k for k = 3..30. never: step 3 alone is unit 0 only. drifts: the mean over 3..k
    # is (3 (k - 15), 0, 13, 0) / (k - 2); unit 0 joins at k = 17 (6 >= 0.3 * 13) and
    # unit 2 leaves at k = 30 (13 < 0.3 * 45).
    avg = [by_step((28, 1)), by_step((1, 0.5), (27, 1)), by_step((28, 1)),
           by_step((14, 1), (13, 0.5), (1, 0))]  # fmt: skip
    for got, want in [
        (c.jaccard_step_ref, jac),
        (c.exact_step_ref, exact),
        (c.cosine_step_ref_last, cos),
        (c.code_size_step, size),
        (c.jaccard_avg_ref, avg),
    ]:
        assert np.allclose(got, np.mean(want, axis=0))
    assert len(c.jaccard_step_ref) == K and len(c.jaccard_avg_ref) == K - 2
    # ref vs ref30: only drifts changes ({2} -> {0}); sizes 2, 2, 0, 1 for both
    assert c.jaccard_ref_full == pytest.approx(0.75)
    assert c.code_size_ref == pytest.approx(1.25) and c.code_size_full == pytest.approx(1.25)


def test_curve_ref_window_is_explicit() -> None:
    # The same amplitudes against the window 3..30: drifts' reference becomes {0}.
    c = st.curve(presentations(), frac=0.3, ref_first=3, ref_last=30)
    assert c.jaccard_step_ref[0] == pytest.approx((0 + 0.5 + 1 + 0) / 4)
    assert c.jaccard_step_ref[29] == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    assert c.jaccard_ref_full == 1.0
    with pytest.raises(ValueError, match="outside"):
        st.curve(presentations(), frac=0.3, ref_first=3, ref_last=31)


def test_jaccard_and_cosine_conventions() -> None:
    t, f = True, False
    a = np.array([[t, t, f], [f, f, f], [t, f, f], [f, f, f]])
    b = np.array([[t, f, t], [f, f, f], [f, t, f], [t, f, f]])
    assert st.jaccard(a, b).tolist() == [1 / 3, 1.0, 0.0, 0.0]  # two empty codes: 1
    with pytest.raises(ValueError, match="one shape"):
        st.jaccard(a, b[:, :2])
    x = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.float32)
    y = np.array([[2, 4, 0], [1, 1, 1], [0, 3, 0], [2, 0, 0]], dtype=np.float32)
    assert np.allclose(st.cosine(x, y), [1.0, 0.0, 0.0, -1.0])  # zero norm: 0


def test_window_mean_is_the_protocol_mean() -> None:
    # A 15-step segment read by presentation_means(lead 2, window 13) gives steps 3..15.
    rng = np.random.default_rng(3)
    amps = rng.random((5, 15, 6)).astype(np.float32)
    steps = np.arange(75, dtype=np.uint64)
    out = ob.Site("out", amps.reshape(75, 6), steps)
    index = ob.Site("index", np.repeat(np.arange(5), 15).astype(np.uint64), steps)
    _, want = ob.presentation_means(out, index, window=13, lead=2, min_segment=2)
    assert np.array_equal(st.window_mean(amps, 3, 15), want)
    assert np.array_equal(st.window_mean(amps, 4, 4), amps[:, 3])
    with pytest.raises(ValueError, match="outside"):
        st.window_mean(amps, 0, 3)


def test_variants() -> None:
    a = presentations()
    v = st.variants(a, frac=0.3, steps=(1, 3), windows={"avg5": (3, 5), "ref30": (3, 30)})
    assert list(v) == ["step1", "step3", "avg5", "ref30"]
    assert np.array_equal(v["step1"][0], a[:, 0])
    assert v["step1"][1][0].tolist() == [True, False, False, False]
    assert v["step3"][1][0].tolist() == [False, True, True, False]
    assert np.allclose(v["avg5"][0][1], [2 / 3, 0, 0, 1 / 3])
    assert v["ref30"][1][3].tolist() == [True, False, False, False]
    assert not v["ref30"][1][2].any()
    with pytest.raises(ValueError, match="twice"):
        st.variants(a, frac=0.3, steps=(1,), windows={"step1": (3, 5)})
    with pytest.raises(ValueError, match="outside"):
        st.variants(a, frac=0.3, steps=(31,), windows={})


def sites(index: list[int], n_units: int = 2) -> tuple[ob.Site, ob.Site]:
    steps = np.arange(100, 100 + len(index), dtype=np.uint64)
    values = np.arange(len(index) * n_units, dtype=np.float32).reshape(len(index), n_units)
    return ob.Site("out", values, steps), ob.Site("index", np.array(index, np.uint64), steps)


def test_presentation_steps() -> None:
    # A one-step transient (the loaded state's last index) is dropped.
    out, index = sites([9, 4, 4, 4, 7, 7, 7])
    ids, amps = st.presentation_steps(out, index, steps_per_image=3, min_segment=2)
    assert ids.tolist() == [4, 7]
    assert amps.shape == (2, 3, 2) and amps.dtype == np.float32
    assert np.array_equal(amps[1, 0], out.values[4])  # step 1 of the second presentation
    with pytest.raises(ValueError, match="must last 3 steps"):
        st.presentation_steps(*sites([4, 4, 4, 7, 7]), steps_per_image=3, min_segment=2)
    with pytest.raises(ValueError, match="no presentation"):
        st.presentation_steps(*sites([4, 7, 9]), steps_per_image=3, min_segment=2)
    other = ob.Site("index", index.values, index.steps + 1)
    with pytest.raises(ValueError, match="same steps"):
        st.presentation_steps(out, other, steps_per_image=3, min_segment=2)
