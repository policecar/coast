"""The settling curve of a frozen trial: how a presentation's code develops step by step.

A frozen readout averages the settled steps of a presentation and applies the relative
winner rule. This module keeps every step of each presentation instead, so that the code a
single step or a shorter window would give can be compared with a reference window (the
measured curve used the readout of the time, steps 3..15; `protocol.settle_lead` has
since been set from it). `scripts/eval_settling.py` runs the measurement; its docstring
holds the pre-registered definitions, which these functions implement.

Steps are numbered k = 1..K within a presentation, k = 1 being the first step on which the
dataset's `input/index` shows the image. No shift for the buffered hop is applied: on the
pixel template the group's output at step k answers the input of step k - 1, so step 1
shows the response to the previous image.

Conventions:
- Jaccard of two empty codes is 1.0 (the same set); of an empty and a non-empty code, 0.0.
- Cosine is over the full amplitude vector; a zero norm on either side scores 0.0, the
  degenerate value of the primary metrics (`metrics.cosine_matrix`).
- Window means are taken in float64 and rounded to float32, as `observe.presentation_means`
  does, so a window code here is the code `read_codes` gives for the same steps.
"""

from __future__ import annotations

import msgspec
import numpy as np
from numpy.typing import NDArray

from coast_eval.observe import Site, relative_winners

type Amps = NDArray[np.float32]
type Bools = NDArray[np.bool_]


def presentation_steps(
    out: Site, index: Site, *, steps_per_image: int, min_segment: int
) -> tuple[NDArray[np.int64], Amps]:
    """Segment `out` on changes of the dataset index; return (stim_ids, amps) with amps of
    shape (presentations, steps_per_image, units), step k of presentation p at [p, k - 1].

    Segments shorter than `min_segment` steps are dropped (a loaded run's first step can
    still report the checkpoint's last index). Every kept segment must have exactly
    `steps_per_image` steps: a shorter one would leave steps of the curve undefined.
    """
    if steps_per_image < 1 or min_segment < 1:
        raise ValueError("presentation_steps: need steps_per_image >= 1, min_segment >= 1")
    if not np.array_equal(out.steps, index.steps):
        raise ValueError(f"'{out.name}' and '{index.name}' are not observed on the same steps")
    if out.values.ndim != 2 or index.values.ndim != 1:
        raise ValueError(
            f"expected 2-D '{out.name}' and 1-D '{index.name}', got shapes "
            f"{out.values.shape} and {index.values.shape}"
        )
    if not np.issubdtype(index.values.dtype, np.integer):
        raise ValueError(f"'{index.name}' must be integer, got {index.values.dtype}")
    ids = index.values.astype(np.int64)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], ids.size]
    kept = [(int(s), int(e)) for s, e in zip(starts, ends, strict=True) if e - s >= min_segment]
    if not kept:
        raise ValueError(f"'{index.name}' contains no presentation of >= {min_segment} steps")
    wrong = [(int(ids[s]), e - s) for s, e in kept if e - s != steps_per_image]
    if wrong:
        raise ValueError(
            f"presentations must last {steps_per_image} steps; (index, steps) {wrong[:5]}"
        )
    values = out.values.astype(np.float32)
    amps = np.stack([values[s:e] for s, e in kept])
    return np.array([ids[s] for s, _ in kept], dtype=np.int64), amps


def _check_steps(amps: Amps, first: int, last: int) -> None:
    if amps.ndim != 3 or not np.isfinite(amps).all():
        raise ValueError("amps must be a finite (presentations, steps, units) array")
    if not 1 <= first <= last <= amps.shape[1]:
        raise ValueError(f"steps {first}..{last} outside 1..{amps.shape[1]}")


def window_mean(amps: Amps, first: int, last: int) -> Amps:
    """(P, N) mean amplitude over steps first..last (1-based, inclusive), averaged in
    float64 and rounded to float32."""
    _check_steps(amps, first, last)
    return amps[:, first - 1 : last].astype(np.float64).mean(axis=1).astype(np.float32)


def jaccard(a: Bools, b: Bools) -> NDArray[np.float64]:
    """|a & b| / |a | b| along the last axis; two empty codes score 1.0."""
    if a.shape != b.shape or a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError(f"jaccard: need two bool arrays of one shape, got {a.shape}, {b.shape}")
    inter = (a & b).sum(axis=-1)
    union = (a | b).sum(axis=-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.float64]:
    """Cosine along the last axis over the full vector; a zero norm on either side scores 0.0."""
    if a.shape != b.shape:
        raise ValueError(f"cosine: shapes differ, {a.shape} vs {b.shape}")
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = (a64 * a64).sum(axis=-1), (b64 * b64).sum(axis=-1)
    ok = (na > 0.0) & (nb > 0.0)
    return np.where(ok, (a64 * b64).sum(axis=-1) / np.sqrt(np.where(ok, na * nb, 1.0)), 0.0)


class Curve(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Means over presentations. Lists indexed by step: `*_step` hold k = 1..K, `*_avg`
    hold the windows first..k for k = first..K (`ref_first` is the window's first step)."""

    jaccard_step_ref: list[float]  # Jaccard(code_k, ref)
    exact_step_ref: list[float]  # fraction of presentations with code_k == ref
    cosine_step_ref_last: list[float]  # cosine(amplitude_k, amplitude at step ref_last)
    code_size_step: list[float]  # |code_k|
    jaccard_avg_ref: list[float]  # Jaccard(avg_k, ref)
    jaccard_ref_full: float  # Jaccard(ref, mean of ref_first..K)
    code_size_ref: float
    code_size_full: float


def curve(amps: Amps, *, frac: float, ref_first: int, ref_last: int) -> Curve:
    """The settling curve of (P, K, N) per-step amplitudes against the reference code `ref`,
    the relative-rule code of the mean over steps ref_first..ref_last. `code_k` is the code
    of step k alone, `avg_k` that of the mean over ref_first..k."""
    _check_steps(amps, ref_first, ref_last)
    k_steps = amps.shape[1]
    ref = relative_winners(window_mean(amps, ref_first, ref_last), frac)
    full = relative_winners(window_mean(amps, ref_first, k_steps), frac)
    step = np.stack([relative_winners(amps[:, k], frac) for k in range(k_steps)], axis=1)
    avg = np.stack(
        [
            relative_winners(window_mean(amps, ref_first, k), frac)
            for k in range(ref_first, k_steps + 1)
        ],
        axis=1,
    )
    ref_b = ref[:, None, :]
    exact = (step == ref_b).all(axis=-1)
    last = amps[:, ref_last - 1 : ref_last, :]

    def mean_by_step(v: NDArray[np.generic]) -> list[float]:
        return [float(x) for x in v.astype(np.float64).mean(axis=0)]

    return Curve(
        jaccard_step_ref=mean_by_step(jaccard(step, np.broadcast_to(ref_b, step.shape))),
        exact_step_ref=mean_by_step(exact),
        cosine_step_ref_last=mean_by_step(cosine(amps, np.broadcast_to(last, amps.shape))),
        code_size_step=mean_by_step(step.sum(axis=-1)),
        jaccard_avg_ref=mean_by_step(jaccard(avg, np.broadcast_to(ref_b, avg.shape))),
        jaccard_ref_full=float(jaccard(ref, full).mean()),
        code_size_ref=float(ref.sum(axis=-1).mean()),
        code_size_full=float(full.sum(axis=-1).mean()),
    )


def variants(
    amps: Amps,
    *,
    frac: float,
    steps: tuple[int, ...],
    windows: dict[str, tuple[int, int]],
) -> dict[str, tuple[Amps, Bools]]:
    """Named (amps, masks) codes: `step<k>` for each single step k in `steps`, and each
    named window (first, last) of `windows` as its mean. The relative rule applies to all."""
    out: dict[str, tuple[Amps, Bools]] = {}
    for k in steps:
        _check_steps(amps, k, k)
        a = np.ascontiguousarray(amps[:, k - 1])
        out[f"step{k}"] = (a, relative_winners(a, frac))
    for name, (first, last) in windows.items():
        if name in out:
            raise ValueError(f"variants: name {name!r} used twice")
        a = window_mean(amps, first, last)
        out[name] = (a, relative_winners(a, frac))
    return out
