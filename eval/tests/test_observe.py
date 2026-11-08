"""Observe reader tests on small HDF5 files laid out like a real ngm25 run (flat root
datasets, `__step_index` siblings, `metadata@type`)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

from coast_eval import observe as ob

OUT = "ngm25_neuron_group[1]__out"
INDEX = "mnist_input[0]__index"
PROTOCOL = {"window": 15, "lead": 2, "min_segment": 2}


def write_log(path: Path, sites: dict[str, NDArray[np.generic]], steps: NDArray[np.uint64]) -> Path:
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["type"] = np.bytes_(b"observe_log")
        for name, values in sites.items():
            f[name] = values
            f[name + "__step_index"] = steps
    return path


def write_run(
    d: Path, out: NDArray[np.float32], index: list[int], steps: NDArray[np.uint64] | None = None
) -> Path:
    s = np.arange(100, 100 + len(index), dtype=np.uint64) if steps is None else steps
    write_log(d / "population.observe.h5", {OUT: out}, s)
    write_log(d / "input.observe.h5", {INDEX: np.array(index, dtype=np.uint64)}, s)
    return d


def means(d: Path, **kw: int) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    out = ob.read_site(d / "population.observe.h5", "out")
    index = ob.read_site(d / "input.observe.h5", "index")
    return ob.presentation_means(out, index, **(PROTOCOL | kw))


def test_segments_on_index_and_drops_the_transient(tmp_path: Path) -> None:
    # [t | a a a | b b b]: the length-1 transient goes; a length-3 block keeps only its
    # final step after the 2-step lead.
    out = np.zeros((7, 4), dtype=np.float32)
    out[1:4] = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [1, 2, 3, 4]]
    out[4:7] = [[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2], [5, 6, 7, 8]]
    ids, amps = means(write_run(tmp_path, out, [0, 1, 1, 1, 2, 2, 2]))
    assert ids.tolist() == [1, 2]
    assert amps.dtype == np.float32
    assert np.array_equal(amps, [[1, 2, 3, 4], [5, 6, 7, 8]])


def test_index_not_content_decides_presentations(tmp_path: Path) -> None:
    # Identical rows under three different indices stay three presentations.
    out = np.ones((10, 4), dtype=np.float32)
    ids, amps = means(write_run(tmp_path, out, [99, 5, 5, 5, 7, 7, 7, 9, 9, 9]))
    assert ids.tolist() == [5, 7, 9]
    assert amps.shape == (3, 4)


def test_short_segments(tmp_path: Path) -> None:
    # A segment of exactly min_segment steps is kept, and a segment no longer than the
    # lead still yields its final step.
    out = np.arange(6, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
    ids, amps = means(write_run(tmp_path, out, [0, 1, 1, 2, 2, 2]))
    assert ids.tolist() == [1, 2]
    assert np.array_equal(amps, [[2, 2], [5, 5]])


def test_mean_skips_the_lead(tmp_path: Path) -> None:
    out = np.array(
        [[0, 0, 0, 0], [0.1] * 4, [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [10, 20, 30, 40]],
        dtype=np.float32,
    )
    d = write_run(tmp_path, out, [9, 1, 1, 1, 1, 1])
    assert np.array_equal(means(d)[1], [[5, 9, 13, 17]])  # rows 3..5 of the block 1..5
    assert np.allclose(means(d, lead=0)[1], [[16.1 / 5, 29.1 / 5, 42.1 / 5, 55.1 / 5]])


def test_window_caps_the_settled_tail(tmp_path: Path) -> None:
    out = np.arange(20, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
    d = write_run(tmp_path, out, [4] * 20)
    assert np.allclose(means(d, window=5)[1], np.mean(np.arange(15, 20)))
    assert np.allclose(means(d, window=15)[1], np.mean(np.arange(5, 20)))
    assert np.allclose(means(d, window=30)[1], np.mean(np.arange(2, 20)))


def test_relative_winners() -> None:
    v = np.array(
        [[10, 3, 5, 2, 1], [-1, -2, -0.5, -3, -0.1], [0, -1, -2, 0, -3], [10, 10, 3, 2, 1]],
        dtype=np.float32,
    )
    w = ob.relative_winners(v, 0.5)
    assert w[0].tolist() == [True, False, True, False, False]  # 5 == 0.5 * 10 is inclusive
    assert not w[1].any() and not w[2].any()  # max < 0 and max == 0: empty codes
    assert ob.relative_winners(v, 1.0)[3].tolist() == [True, True, False, False, False]
    # v >= 0.3 * peak holds in float64 (Julia: a winner) but not in float32.
    near = np.array([[1.6501218, 0.49503654]], dtype=np.float32)
    assert ob.relative_winners(near, 0.3).tolist() == [[True, True]]
    for frac in (0.0, 1.5):
        with pytest.raises(ValueError, match="outside"):
            ob.relative_winners(v, frac)
    with pytest.raises(ValueError, match="finite"):
        ob.relative_winners(np.array([[np.nan, 1.0]]), 0.3)


def test_read_codes(tmp_path: Path) -> None:
    out = np.zeros((9, 3), dtype=np.float32)
    out[0:3] = [1.0, 0.2, 0.5]
    out[6:9] = [0.0, 0.0, 2.0]  # rows 3..5 stay silent
    codes = ob.read_codes(
        write_run(tmp_path, out, [3, 3, 3, 1, 1, 1, 8, 8, 8]), frac=0.3, **PROTOCOL
    )
    assert codes.stim_ids.tolist() == [3, 1, 8]
    assert codes.masks.tolist() == [
        [True, False, True],
        [False, False, False],
        [False, False, True],
    ]
    assert np.array_equal(codes.amps[0], np.array([1.0, 0.2, 0.5], dtype=np.float32))


MALFORMED = {
    "misaligned": "not observed on the same steps",
    "no-index": "expected one site 'index'",
    "two-outs": "expected one site 'out'",
    "no-sibling": "__step_index sibling",
    "not-observe": "not an observe log",
    "rows": "for 4 steps",
    "steps": "not strictly increasing",
    "duplicate-steps": "not strictly increasing",
    "dtype": "must be integer",
}


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_files_raise(tmp_path: Path, case: str) -> None:
    out = np.ones((4, 2), dtype=np.float32)
    s = np.arange(4, dtype=np.uint64)
    write_run(tmp_path, out, [0, 0, 1, 1], s)
    pop, inp = tmp_path / "population.observe.h5", tmp_path / "input.observe.h5"
    index = np.array([0, 0, 1, 1], dtype=np.uint64)
    if case == "misaligned":
        write_log(inp, {INDEX: index}, s + 5)
    elif case == "no-index":
        write_log(inp, {"mnist_input[0]__other": index}, s)
    elif case == "two-outs":
        write_log(pop, {OUT: out, "ngm25_neuron_group[2]__out": out}, s)
    elif case == "no-sibling":
        with h5py.File(pop, "a") as f:
            del f[OUT + "__step_index"]
    elif case == "not-observe":
        with h5py.File(pop, "a") as f:
            f["metadata"].attrs["type"] = np.bytes_(b"buffer_log")
    elif case == "rows":
        write_log(pop, {OUT: out[:3]}, s)
    elif case == "steps":
        write_run(tmp_path, out, [0, 0, 1, 1], np.array([0, 2, 1, 3], dtype=np.uint64))
    elif case == "duplicate-steps":
        write_run(tmp_path, out, [0, 0, 1, 1], np.array([0, 1, 1, 2], dtype=np.uint64))
    elif case == "dtype":
        write_log(inp, {INDEX: index.astype(np.float32)}, s)
    with pytest.raises(ValueError, match=MALFORMED[case]):
        ob.read_codes(tmp_path, frac=0.3, **PROTOCOL)
