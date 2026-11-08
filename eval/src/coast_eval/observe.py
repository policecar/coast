"""Read presentation codes from a COAST observe directory.

A run with `observe.groups` writes one HDF5 file per group, `<output_dir>/<group>.observe.h5`.
Datasets sit flat at the file root, named `<entity path>__<local id>` (for example
`ngm25_neuron_group[1]__out`), each with a `<name>__step_index` uint64 sibling; the
`metadata` group carries `type = "observe_log"`. The producer-owned site list is
`tests/integration_observe/observe_contract.json`.

Presentations are segmented on the dataset loader's own `input/index` stream, not on the
input content: two presentations are the same segment only if they are the same dataset
row, even when two different rows render identically.

Every reader raises ValueError on a malformed file instead of returning a default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray

_STEP_SUFFIX = "__step_index"


@dataclass(frozen=True)
class Site:
    """One observe site: `values` has one row per observed step, `steps` the step numbers."""

    name: str
    values: NDArray[np.generic]
    steps: NDArray[np.uint64]


@dataclass(frozen=True)
class Codes:
    """Per-presentation codes: row p is the presentation of dataset index `stim_ids[p]`."""

    stim_ids: NDArray[np.int64]
    amps: NDArray[np.float32]
    masks: NDArray[np.bool_]


def read_site(path: Path, local_id: str) -> Site:
    """Read the single dataset in `path` whose last `__` component is `local_id`."""
    with h5py.File(path, "r") as f:
        meta = f.get("metadata")
        kind = meta.attrs.get("type") if isinstance(meta, h5py.Group) else None
        if kind not in (b"observe_log", "observe_log"):
            raise ValueError(f"{path}: not an observe log (metadata type {kind!r})")
        names = [
            name
            for name in f.keys()
            if name != "metadata"
            and not name.endswith(_STEP_SUFFIX)
            and name.rsplit("__", 1)[-1] == local_id
        ]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one site '{local_id}', found {names}")
        name = names[0]
        sibling = f.get(name + _STEP_SUFFIX)
        site = f[name]
        if not (isinstance(site, h5py.Dataset) and isinstance(sibling, h5py.Dataset)):
            raise ValueError(f"{path}: site '{name}' needs a dataset and a {_STEP_SUFFIX} sibling")
        values = cast(NDArray[np.generic], np.asarray(site[()]))
        steps = cast(NDArray[np.generic], np.asarray(sibling[()]))
    if steps.ndim != 1 or steps.dtype != np.uint64:
        raise ValueError(f"{path}: '{name}{_STEP_SUFFIX}' must be 1-D uint64, got {steps.dtype}")
    if values.ndim == 0 or values.shape[0] != steps.size:
        raise ValueError(f"{path}: '{name}' has shape {values.shape} for {steps.size} steps")
    if steps.size > 1 and not (np.diff(steps.astype(np.int64)) > 0).all():
        raise ValueError(f"{path}: '{name}' steps are not strictly increasing")
    return Site(name, values, steps.astype(np.uint64, copy=False))


def presentation_means(
    out: Site, index: Site, *, window: int, lead: int, min_segment: int
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Segment `out` on changes of the dataset index; return (stim_ids, mean amplitudes).

    Segments shorter than `min_segment` steps are dropped (a resumed run's first step can
    still report the checkpoint's last index). Each kept segment is averaged over its
    steps after the first `lead` (at least its final step survives), at most the last
    `window` of them.
    """
    if window < 1 or lead < 0 or min_segment < 1:
        raise ValueError("presentation_means: need window >= 1, lead >= 0, min_segment >= 1")
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
    kept = [(s, e) for s, e in zip(starts, ends, strict=True) if e - s >= min_segment]
    if not kept:
        raise ValueError(f"'{index.name}' contains no presentation of >= {min_segment} steps")
    values = out.values.astype(np.float64)
    means = np.stack(
        [values[max(min(s + lead, e - 1), e - window) : e].mean(axis=0) for s, e in kept]
    )
    stim_ids = np.array([ids[s] for s, _ in kept], dtype=np.int64)
    return stim_ids, means.astype(np.float32)


def relative_winners(amps: NDArray[np.floating], frac: float) -> NDArray[np.bool_]:
    """Winner masks: unit u wins presentation p iff amps[p, u] >= frac * max(amps[p]).

    A presentation whose maximum is not positive has no activity to be relative to and
    gets an empty code. The threshold is computed in float64, as Julia did; a float32
    product flips units within one ulp of it. The settled means themselves are averaged
    in float64 and rounded to float32, where Julia averaged in Float32, so they can
    differ from the Julia codes by about one ulp.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"relative_winners: frac={frac} outside (0, 1]")
    if amps.ndim != 2 or not np.isfinite(amps).all():
        raise ValueError("relative_winners: amps must be a finite (presentations, units) array")
    peak = amps.max(axis=1, keepdims=True)
    return (amps >= frac * peak.astype(np.float64)) & (peak > 0)


def read_codes(
    observe_dir: Path, *, window: int, lead: int, min_segment: int, frac: float
) -> Codes:
    """Codes of the one population in `observe_dir`, from its `population/out` and the
    dataset's `input/index` sites: settled mean per presentation, then the relative rule."""
    out = read_site(observe_dir / "population.observe.h5", "out")
    index = read_site(observe_dir / "input.observe.h5", "index")
    stim_ids, amps = presentation_means(
        out, index, window=window, lead=lead, min_segment=min_segment
    )
    return Codes(stim_ids, amps, relative_winners(amps, frac))
