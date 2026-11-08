"""The milestone-1 protocol of issue #82 as data: the coast configs it runs.

Training: one job per seed on the MNIST train split, from an explicit index manifest
reshuffled every pass, `change_interval` steps per image (1 in the shipped configs), learning on.
Frozen trials: per (seed, condition), the trained state loaded (by the runner, not the
config), learning off, the held-out test indices in order at 15 steps per image, observe
groups `population` and `input` over the whole trial. Conditions: `clean`, the 15 MNIST-C
corruptions, and `zeros` (an all-zero image set, the chance control).

The template (`eval/configs/mnist_ngm25_frozen_eval.json`) is the #243 configuration: the
roast template plus the max_softmax inhibition (strength 10) that run pinned.

A frozen trial is a frozen run (the config's top-level `frozen`, SPEC.md M.7.6): all
learning off. coast continues a checkpoint's image sequence only when the checkpoint was
written by the same run (`run_id`, S.5.14); the configs leave it "auto", which coastd turns
into a fresh id per job. So a trial's first launch, from the trained state (another job),
starts its test sequence fresh, and a coastd resume of the trial continues it, as a training
resume continues training's. Each run records its `frozen` (`runs.check_provenance` checks
it).

Training reshuffles its manifest every pass (`shuffle`), as the shipped configs do: at 1M steps
each image repeats about 16.7 times, and one fixed order would be identical for every seed.
A frozen trial keeps the manifest's order, so its presentations align with `test_indices`.

The seed goes into the config as every node's `random_seed_offset`, which composes like
coast's `--seed-offset`. The dataset's input noise is zeroed. Two noise floors stay at the
template's values: the neurons' baseline noise, because in the NGM model it encodes the
resting state, and a retina's output noise, its designed null signal, which coast refuses to
set to zero. A pixel-vs-retina comparison therefore also compares input without and with a
noise floor.

Tuning trials (`tuning_jobs`, for hyperparameter sweeps) are frozen trials on images held out
of the train set (`tuning_split`), never on test images: a setting chosen by its score on the
evaluation sample would be scored on the images it was chosen on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
from numpy.typing import NDArray

from coast_eval.datasets import (
    MNIST_C,
    TEST_IMAGES,
    TEST_LABELS,
    TRAIN_IMAGES,
    TRAIN_LABELS,
    corruption_files,
    corruption_tuning_images,
    read_idx_images,
    read_idx_labels,
    write_idx,
)

CONDITIONS: tuple[str, ...] = ("clean", *MNIST_C, "zeros")
FROZEN_STEPS_PER_IMAGE = 15
SAMPLE_SEED = 20261106
SAMPLE_POOL = 9000  # test indices from here on calibrate the retina (#258)
TRAIN_COUNT = 60_000  # MNIST's train split
TUNING_SEED = 20261108
TUNING_POOL = 1000  # rows of every tuning image file, MNIST-C's included (fetch_mnist_c.py)
# SHA-256 of `tuning_split(TUNING_POOL)` as little-endian int64 bytes: the rows MNIST-C's
# pinned tuning files hold. numpy does not promise a Generator stream across versions, so
# `write_tuning_inputs` refuses a permutation that is not this one
TUNING_POOL_SHA256 = "5c65c97414ea84cd2d76326f3f307aae7001c2287decb3f3c80fbedb1b8fb546"
_DATASET = "datasets.mnist.default"
_GROUP = "ngm25.neuron_groups.L4-mnist"


class Inputs(msgspec.Struct, frozen=True):
    """Index manifests and the all-zero image file, written by `write_inputs`. The paths
    are written into configs as given, so they must resolve from the run's working
    directory (for coastd: the run dir)."""

    test_indices: tuple[int, ...]
    train_manifest: str
    test_manifest: str
    zeros_images: str


class Job(msgspec.Struct, frozen=True, kw_only=True):
    """One coast run. `condition` is "train" or one of CONDITIONS. A frozen job loads the
    final state of the "train" job with the same seed."""

    seed: int
    condition: str
    steps: int
    config_text: str


def _indices(indices: Iterable[int], count: int, what: str) -> tuple[int, ...]:
    t = tuple(int(i) for i in indices)
    if not t or len(set(t)) != len(t) or min(t) < 0 or max(t) >= count:
        raise ValueError(f"{what} indices must be non-empty, unique and in [0, {count})")
    return t


def evaluation_sample(n: int) -> list[int]:
    """The `n` test images of an evaluation: a fixed random sample, sorted, of the indices
    below SAMPLE_POOL. Random, because MNIST's first 5000 test images are its cleaner half;
    fixed, so that every comparison reads the same images."""
    if not 1 <= n <= SAMPLE_POOL:
        raise ValueError(f"need 1..{SAMPLE_POOL} test images, got {n}")
    picked = np.random.default_rng(SAMPLE_SEED).choice(SAMPLE_POOL, n, replace=False)
    return sorted(int(i) for i in picked)


def tuning_split(n: int) -> list[int]:
    """The `n` train images of a sweep's tuning trials: the first `n` of a fixed random
    permutation of the 60k train indices, in that order. Held out of the train set, because
    tuning on the evaluation sample would score a setting on the images it was chosen on;
    random, for the reason `evaluation_sample` is. A prefix of one permutation, so that the
    tuning files hold TUNING_POOL images once and every `n` up to it reads their first rows."""
    if not 1 <= n <= TUNING_POOL:
        raise ValueError(f"need 1..{TUNING_POOL} tuning images, got {n}")
    order = np.random.default_rng(TUNING_SEED).permutation(TRAIN_COUNT)
    return [int(i) for i in order[:n]]


class TuningInputs(msgspec.Struct, frozen=True):
    """The files of `write_tuning_inputs`, as `Inputs` are for an evaluation. Row r of every
    tuning image file (`images`, `zeros_images`, MNIST-C's `tuning-images-idx3-ubyte`) and of
    `labels` is train image `tuning_split(TUNING_POOL)[r]`; `tuning_manifest` presents rows
    0..n-1, so a presentation's dataset index is its row and `tuning_indices[row]` the train
    image. `train_manifest` is the train set without `tuning_indices`."""

    tuning_indices: tuple[int, ...]
    train_manifest: str
    tuning_manifest: str
    images: str
    labels: str
    zeros_images: str


def write_tuning_inputs(out_dir: Path, data_root: Path, n: int) -> TuningInputs:
    """Write the training manifest without the `n` tuning images, the tuning manifest, and
    the clean tuning images, their labels and an all-zero set, TUNING_POOL rows each, into
    `out_dir`. The clean images come from the train split only. Refuses a tuning split that
    is not the one the pinned MNIST-C tuning files hold (TUNING_POOL_SHA256)."""
    images = read_idx_images(data_root / TRAIN_IMAGES)
    labels = read_idx_labels(data_root / TRAIN_LABELS)
    if images.shape[0] != TRAIN_COUNT or labels.size != TRAIN_COUNT:
        raise ValueError(f"{data_root}: need the {TRAIN_COUNT} MNIST train images and labels")
    tuning = tuning_split(n)
    rows = np.asarray(tuning_split(TUNING_POOL))
    if hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() != TUNING_POOL_SHA256:
        raise ValueError("tuning_split is not the permutation the pinned MNIST-C tuning files "
                         "were written for (numpy's Generator stream changed?)")  # fmt: skip
    held = set(tuning)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("train_indices.txt", "tuning_rows.txt", "tuning-images-idx3-ubyte",
             "tuning-labels-idx1-ubyte", "tuning-zeros-idx3-ubyte")  # fmt: skip
    paths = [out_dir / name for name in names]
    paths[0].write_text("".join(f"{i}\n" for i in range(TRAIN_COUNT) if i not in held))
    paths[1].write_text("".join(f"{r}\n" for r in range(n)))
    write_idx(paths[2], images[rows])
    write_idx(paths[3], labels[rows])
    write_idx(paths[4], np.zeros_like(images[rows]))
    return TuningInputs(tuple(tuning), *(str(p) for p in paths))


def write_inputs(
    out_dir: Path, data_root: Path, train_indices: Iterable[int], test_indices: Iterable[int]
) -> Inputs:
    """Write the two index manifests and a zero image set shaped like the clean test split
    into `out_dir` (the caller's directory; never the data root)."""
    clean = read_idx_images(data_root / TEST_IMAGES)
    train = _indices(train_indices, read_idx_labels(data_root / TRAIN_LABELS).size, "train")
    test = _indices(test_indices, clean.shape[0], "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / n for n in ("train_indices.txt", "test_indices.txt", "zeros-images-idx3")]
    for path, ids in zip(paths, (train, test), strict=False):
        path.write_text("".join(f"{i}\n" for i in ids))
    write_idx(paths[2], np.zeros_like(clean))
    return Inputs(test, *(str(p) for p in paths))


def _set(cfg: dict[str, Any], dotted: str, value: object) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    if leaf not in node:
        raise KeyError(f"template has no key '{dotted}'")
    node[leaf] = value


def _job(
    template: dict[str, Any],
    *,
    seed: int,
    condition: str,
    steps: int,
    files: tuple[str, str, str],
    change_interval: int,
    observe: dict[str, Any],
) -> Job:
    """`files` = (images, labels, index manifest) as the run's working directory sees them."""
    cfg: dict[str, Any] = json.loads(json.dumps(template))
    for key, value in {
        "image_file": files[0],
        "label_file": files[1],
        "index_manifest": files[2],
        "change_interval": change_interval,
        "noise_range": [0.0, 0.0],
        "shuffle": condition == "train",
    }.items():
        _set(cfg, f"{_DATASET}.{key}", value)
    _set(cfg, "frozen", condition != "train")
    _set(cfg, "observe.groups", observe)
    for node in cfg["sim"]["nodes"]:
        if "random_seed_offset" not in node:
            raise KeyError(f"template node {node['id']} has no 'random_seed_offset'")
        node["random_seed_offset"] = seed
    text = json.dumps(cfg, indent=2, sort_keys=True)
    return Job(seed=seed, condition=condition, steps=steps, config_text=text)


def _unique(values: Sequence[Any], what: str) -> None:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{what} must be non-empty and unique, got {list(values)}")


def train_jobs(
    template: dict[str, Any],
    inputs: Inputs | TuningInputs,
    *,
    data_ref: str,
    seeds: Sequence[int],
    steps: int,
    change_interval: int,
) -> list[Job]:
    """One learning run per seed; no observe groups. `data_ref` is the data root as the
    run's working directory sees it (coastd links it into each run dir as `./data`).

    A run presents steps // change_interval images, drawing the train manifest in a fresh
    random order each pass: 1M steps at 1 step per image (`eval_ab.py`'s default) pass over
    the 60k training images about 16.7 times.
    """
    _unique(seeds, "seeds")
    if steps < 1 or change_interval < 1:
        raise ValueError(f"need steps, change_interval >= 1, got {steps}, {change_interval}")
    files = (f"{data_ref}/{TRAIN_IMAGES}", f"{data_ref}/{TRAIN_LABELS}", inputs.train_manifest)
    return [
        _job(
            template,
            seed=s,
            condition="train",
            steps=steps,
            files=files,
            change_interval=change_interval,
            observe={},
        )
        for s in seeds
    ]


def _condition_files(condition: str, inputs: Inputs, data_ref: str) -> tuple[str, str, str]:
    if condition == "clean":
        images, labels = f"{data_ref}/{TEST_IMAGES}", f"{data_ref}/{TEST_LABELS}"
    elif condition == "zeros":
        images, labels = inputs.zeros_images, f"{data_ref}/{TEST_LABELS}"
    else:
        images, labels = (f"{data_ref}/{f}" for f in corruption_files(condition))
    return images, labels, inputs.test_manifest


def _frozen(
    template: dict[str, Any],
    files: Callable[[str], tuple[str, str, str]],
    images: int,
    *,
    seed: int,
    conditions: Sequence[str],
    base_step: int,
) -> list[Job]:
    _unique(conditions, "conditions")
    steps = images * FROZEN_STEPS_PER_IMAGE
    window = {"ranges": [[base_step, base_step + steps - 1]], "combine": "or"}
    return [
        _job(
            template,
            seed=seed,
            condition=c,
            steps=steps,
            files=files(c),
            change_interval=FROZEN_STEPS_PER_IMAGE,
            observe={"population": window, "input": window},
        )
        for c in conditions
    ]


def frozen_jobs(
    template: dict[str, Any],
    inputs: Inputs,
    *,
    data_ref: str,
    seed: int,
    conditions: Sequence[str],
    base_step: int,
) -> list[Job]:
    """One frozen trial per condition for the state trained with `seed`, which ends at
    `base_step`. Observe ranges are absolute, because coast continues the loaded step count."""
    return _frozen(template, lambda c: _condition_files(c, inputs, data_ref),
                   len(inputs.test_indices), seed=seed, conditions=conditions,
                   base_step=base_step)  # fmt: skip


def _tuning_files(condition: str, inputs: TuningInputs, data_ref: str) -> tuple[str, str, str]:
    if condition == "clean":
        images = inputs.images
    elif condition == "zeros":
        images = inputs.zeros_images
    else:
        images = f"{data_ref}/{corruption_tuning_images(condition)}"
    for path in (images, inputs.labels):
        if Path(path).name.startswith("t10k-"):
            raise ValueError(f"{condition}: tuning trials must not read the test split ({path})")
    return images, inputs.labels, inputs.tuning_manifest


def tuning_jobs(
    template: dict[str, Any],
    inputs: TuningInputs,
    *,
    data_ref: str,
    seed: int,
    conditions: Sequence[str],
    base_step: int,
) -> list[Job]:
    """`frozen_jobs` on the tuning images: `clean` and `zeros` from `write_tuning_inputs`,
    a corruption from MNIST-C's train split (`datasets.corruption_tuning_images`). A file of
    the test split (`t10k-...`) is refused, whatever names it: the tuning trials of a sweep
    must never score a setting on test images."""
    return _frozen(template, lambda c: _tuning_files(c, inputs, data_ref),
                   len(inputs.tuning_indices), seed=seed, conditions=conditions,
                   base_step=base_step)  # fmt: skip


def steps_per_image(cfg: dict[str, Any]) -> int:
    """The config's dataset `change_interval`: how many steps each image is presented."""
    node: Any = cfg
    for key in _DATASET.split("."):
        node = node[key]
    return int(node["change_interval"])


def settle_lead(cfg: dict[str, Any]) -> int:
    """Leading steps of a presentation to drop before averaging: the longest path from a
    source node (one without inputs, the dataset) to the one neuron group, in hops. 1 for
    the pixel template, 2 with a retina in between. The readout averages the remaining
    steps of the presentation, `lead + 1 .. FROZEN_STEPS_PER_IMAGE`.

    It assumes that every hop is one buffered step (the sim core swaps the io buffers once
    per step), so the group sees an image `hops` steps after the dataset presents it, and
    that the group needs no further steps to settle. The settling curve (brief §9) shows
    none beyond the hop for this model on the pixel and retina templates;
    `scripts/eval_settling.py --template` measures it for any template.
    """
    sources = {n["id"]: [src for src, _ in n["inputs"]] for n in cfg["sim"]["nodes"]}
    groups = [n["id"] for n in cfg["sim"]["nodes"] if n["entity_type"].endswith("neuron_group")]
    if len(groups) != 1:
        raise ValueError(f"settle_lead: need exactly one neuron group node, got {groups}")
    depth: dict[int, int] = {}

    def hops(node: int, path: frozenset[int]) -> int:
        if node in path:
            raise ValueError(f"settle_lead: the node graph has a cycle through node {node}")
        if node not in sources:
            raise ValueError(f"settle_lead: input from unknown node {node}")
        if node not in depth:
            ins = sources[node]
            depth[node] = 1 + max(hops(s, path | {node}) for s in ins) if ins else 0
        return depth[node]

    lead = hops(groups[0], frozenset())
    if lead == 0:
        raise ValueError(f"settle_lead: neuron group node {groups[0]} has no inputs")
    return lead


def check_aligned(observed: NDArray[np.int64], expected: Sequence[int]) -> None:
    """The metrics' precondition: row p of every condition's codes is dataset image
    `expected[p]`, each image exactly once. Raise unless the observed ids are exactly that."""
    _unique(expected, "expected ids")
    want = np.asarray(expected, dtype=np.int64)
    if observed.shape != want.shape or not np.array_equal(observed, want):
        n = min(observed.size, want.size)
        diff = np.flatnonzero(observed[:n] != want[:n])
        at = int(diff[0]) if diff.size else n
        raise ValueError(
            f"observed {observed.size} presentations, expected {want.size}; first "
            f"difference at position {at}"
        )
