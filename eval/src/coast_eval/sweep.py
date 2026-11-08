"""Hyperparameter sweeps of the frozen evaluation through coastd (`design/eval_sweep_plan.md`).

A sweep is variants x seeds x lengths over one template (D1). A spec
(`coast-eval-sweep-spec/1`) names the template, `axes` (JSON pointer -> values, expanded to
their full grid), named `variants` (pointer -> value sets) and the seeds, lengths, tuning
conditions and tuning image count; the template unchanged is always the variant `base`.

A (variant, seed) chain trains to `lengths[0]`, continues training to each further length in a
job of its own (`load_state_from` the previous one, under the same `run_id`, so the train
sequence continues where it stopped, SPEC.md S.5.14), and runs the tuning trials off each done
training job (D2). A point is one (variant, length): its report is `report.build` of one state,
the variant, over all seeds. Tuning trials read images held out of the train set
(`protocol.tuning_split`), never test images (D3).

Everything runs through `runner.run_jobs`: placement by seed, the ledger, adoption on a rerun
and the run-dir checks are the runner's. A point whose `report.json` exists is not run again
(D6). Output (D4): `<out>/sweep.json`, `<out>/<variant>/<length>/report.{json,md}`,
`<out>/ledger.json`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import msgspec
import numpy as np

from coast_eval import datasets, protocol, report, runner, runs
from coast_eval.runner import TRAIN, Daemon, Done, Planned

SPEC_SCHEMA = "coast-eval-sweep-spec/1"
SCHEMA = "coast-eval-sweep/1"
BASE = "base"
STEPS_PER_IMAGE = 1  # training: the shipped configs' rate, `eval_ab.py`'s default
PROJECTION_SEEDS = (1, 2, 3, 4, 5)
# coastd runs a higher priority first. Continuations and tuning trials outrank first training
# jobs, so that points finish one chain at a time instead of after every first training job; a
# hint to coastd's queue, not a schedule (D5)
FOLLOW_UP_PRIORITY = 1
FRAC = 0.3
_NAME = re.compile(r"[A-Za-z0-9_.,=+-]+")
# What the protocol writes into every config: an override there would be overwritten, or
# would break the checks the runner makes ("*" is any list index)
_PROTOCOL_KEYS = (
    ("frozen",),
    ("run_id",),
    ("observe", "groups"),
    ("sim", "nodes", "*", "random_seed_offset"),
    *(("datasets", "mnist", "default", k) for k in ("image_file", "label_file",
      "index_manifest", "change_interval", "noise_range", "shuffle")),
)  # fmt: skip


class Ratio(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A derived axis: per grid point, the value of axis `ratio_of` times each factor. It
    keeps a relation the grid must hold at every point (secondary <= primary learning rate),
    which an absolute list cannot."""

    ratio_of: str
    factors: list[float]


class Spec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema: Literal["coast-eval-sweep-spec/1"]
    template: str  # relative to the repository root, or absolute
    seeds: list[int]
    lengths: list[int]
    conditions: list[str]
    tuning_images: int
    axes: dict[str, list[Any] | Ratio] = msgspec.field(default_factory=lambda: {})
    variants: dict[str, dict[str, Any]] = msgspec.field(default_factory=lambda: {})


def load_spec(raw: bytes) -> Spec:
    """Decode and check a spec on its own; the pointers are checked against the template by
    `expand`. Lengths must increase (each continues the previous), seeds must be unique and
    non-negative (they are `random_seed_offset`s and place the chains, seed mod N)."""
    try:
        spec = msgspec.json.decode(raw, type=Spec)
    except (msgspec.DecodeError, msgspec.ValidationError) as e:
        raise ValueError(f"not a {SPEC_SCHEMA} spec: {e}") from e
    if not spec.seeds or len(set(spec.seeds)) != len(spec.seeds) or min(spec.seeds) < 0:
        raise ValueError(f"seeds must be non-empty, unique and >= 0, got {spec.seeds}")
    if (
        not spec.lengths
        or spec.lengths[0] < 1
        or any(b <= a for a, b in itertools.pairwise(spec.lengths))
    ):
        raise ValueError(f"lengths must be non-empty, >= 1 and increasing, got {spec.lengths}")
    known = set(protocol.CONDITIONS)
    c = spec.conditions
    if not c or len(set(c)) != len(c) or not set(c) <= known:
        raise ValueError(f"conditions must be non-empty, unique and from {sorted(known)}: {c}")
    protocol.tuning_split(spec.tuning_images)  # raises outside 1..TUNING_POOL
    for name in spec.variants:
        _check_name(name, "variant name")
        if name == BASE:
            raise ValueError(f"'{BASE}' is the template unchanged; name the variant otherwise")
        if not spec.variants[name]:
            raise ValueError(f"variant {name!r} overrides nothing: that is '{BASE}'")
    return spec


def _check_name(name: str, what: str) -> None:
    """A variant name is a directory and a part of every job's ledger key."""
    if not _NAME.fullmatch(name) or name in (".", ".."):
        raise ValueError(f"{what} {name!r} must be made of [A-Za-z0-9_.,=+-]")


def _segments(pointer: str) -> list[str]:
    """RFC 6901: '/a/b~1c' is ['a', 'b/c']."""
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError(f"{pointer!r} is not a JSON pointer to a key ('/a/b')")
    return [s.replace("~1", "/").replace("~0", "~") for s in pointer[1:].split("/")]


def _locate(doc: Any, pointer: str) -> tuple[Any, str | int]:
    """(container, key) of `pointer` in `doc`; raises unless it names an existing value. coast
    only warns on an unknown key (`json_loader.cpp:37-48`), so a misspelt pointer would run the
    template's value under the variant's name."""
    segs = _segments(pointer)
    if "*" in segs:
        raise ValueError(f"{pointer}: '*' is not a key")
    for owned in _PROTOCOL_KEYS:  # the pointer is, contains or lies under one of them
        if all(o in ("*", k) for o, k in zip(owned, segs, strict=False)):
            raise ValueError(f"{pointer}: the protocol writes this key; a sweep cannot set it")
    node: Any = doc
    for depth, key in enumerate(segs):
        if isinstance(node, dict) and key in node:
            k: str | int = key
        elif isinstance(node, list) and key.isdecimal() and int(key) < len(cast(list[Any], node)):
            k = int(key)
        else:
            break
        if depth == len(segs) - 1:
            return cast(Any, node), k
        node = cast(Any, node)[k]
    raise ValueError(f"{pointer}: not in the template")


# @decision DEC-001: overrides go into the template, and protocol-owned keys are refused
# @rationale: the plan applies overrides after the protocol writes its keys; refusing every
# pointer that is, contains or lies under such a key makes both orders give the same config,
# so the protocol's job builders run unchanged on the overridden template.
def apply(template: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of `template` with each pointer's value replaced. The protocol's own keys cannot
    be overridden (`_locate`), so applying the overrides to the template, before the protocol
    writes its keys, is the same as applying them after."""
    out: dict[str, Any] = json.loads(json.dumps(template))
    for pointer, value in overrides.items():
        parent, key = _locate(out, pointer)
        parent[key] = value
    return out


def _number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def same_value(a: Any, b: Any) -> bool:
    """The one rule for "the same value" of a sweep: numbers (int or float, not bool) within
    `math.isclose(rel_tol=1e-9, abs_tol=0.0)`, anything else of the same type and equal. It
    decides that a grid point is `base` (`expand`), that two axis values are one, and which grid
    point a report's value belongs to (`sweep_report.coordinate`), so that the driver and the
    aggregator cannot disagree. A derived value is a float product (`0.1 * 0.1` is
    `0.010000000000000002`), and `1` and `1.0` are one value to coast."""
    if _number(a) and _number(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    return type(a) is type(b) and a == b


def token(value: Any, pointer: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"axis {pointer}: {value!r} is not a scalar; put it in a named variant")


def expand(
    spec: Spec, template: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """(overrides of every variant by name, aliases). The variants are `base` (none), the grid
    of `axes` in their order, then the named variants. A grid name is `k=v` per axis joined by
    `,`, `k` the pointer's last key; a derived value appears as its factor
    (`secondary_learning_rate=x0.01`). Every pointer must exist in the template.

    A grid point whose every value is the template's (`same_value`) is `base`: it is not run a
    second time but listed in the aliases (grid name -> `base`), so that a grid view can still
    place `base` at its point."""
    plain = {p: v for p, v in spec.axes.items() if not isinstance(v, Ratio)}
    keys = [_segments(p)[-1] for p in spec.axes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"axes must end in distinct keys (they name the variants), got {keys}")
    choices: list[list[tuple[str, Any]]] = []  # per axis: (name token, value or factor)
    for pointer, axis in spec.axes.items():
        _locate(template, pointer)
        if isinstance(axis, Ratio):
            ref = plain.get(axis.ratio_of)
            if ref is None:
                raise ValueError(f"axis {pointer}: ratio_of {axis.ratio_of!r} is not a plain "
                                 "axis of this spec")  # fmt: skip
            if not all(_number(v) for v in ref):
                raise ValueError(f"axis {pointer}: ratio_of {axis.ratio_of} has non-numbers")
            values: Sequence[Any] = axis.factors
            choices.append([("x" + token(f, pointer), f) for f in values])
        else:
            values = axis
            choices.append([(token(v, pointer), v) for v in values])
        if not values:
            raise ValueError(f"axis {pointer}: values must be non-empty and unique")
        for a, b in itertools.combinations(values, 2):
            if same_value(a, b):
                raise ValueError(f"axis {pointer}: values must be non-empty and unique; {a!r} "
                                 f"and {b!r} are the same value (`same_value`)")  # fmt: skip
    out: dict[str, dict[str, Any]] = {BASE: {}}
    aliases: dict[str, str] = {}
    for point in itertools.product(*choices) if choices else ():  # no axes: no grid
        chosen = dict(zip(spec.axes, point, strict=True))
        overrides = {
            p: chosen[a.ratio_of][1] * v if isinstance(a, Ratio) else v
            for (p, a), (_, v) in zip(spec.axes.items(), point, strict=True)
        }
        name = ",".join(f"{k}={t}" for k, (t, _) in zip(keys, point, strict=True))
        _check_name(name, "grid variant")
        if all(same_value(_value(template, p), v) for p, v in overrides.items()):
            aliases[name] = BASE
        else:
            out[name] = overrides
    for name, overrides in spec.variants.items():
        if name in out or name in aliases:
            raise ValueError(f"variant {name!r} is also a grid point's name")
        for pointer in overrides:
            _locate(template, pointer)
        out[name] = dict(overrides)
    return out, aliases


def _value(doc: Any, pointer: str) -> Any:
    parent, key = _locate(doc, pointer)
    return parent[key]


def input_pins(conditions: Sequence[str]) -> dict[str, str]:
    """The pins of every dataset file a sweep's runs read, relative to the data root: the
    train split and each tuned corruption's tuning images (`datasets.TUNING_PINS`)."""
    pins = {f: datasets.PINS[f] for f in (datasets.TRAIN_IMAGES, datasets.TRAIN_LABELS)}
    for c in conditions:
        if c in ("clean", "zeros"):
            continue
        rel = datasets.corruption_tuning_images(c)
        pins[rel] = datasets.TUNING_PINS[rel]
    return pins


def tuning_pixels(data_root: Path, inputs: protocol.TuningInputs, condition: str) -> report.Floats:
    """(n, 784) pixels in [0, 1] of the presented tuning images: rows 0..n-1 of the
    condition's tuning file (`inputs` as written locally; MNIST-C's under `data_root`)."""
    n = len(inputs.tuning_indices)
    if condition == "zeros":
        return np.zeros((n, 784))
    path = (Path(inputs.images) if condition == "clean"
            else data_root / datasets.corruption_tuning_images(condition))  # fmt: skip
    images = datasets.read_idx_images(path)
    if images.shape[0] != protocol.TUNING_POOL:
        raise ValueError(f"{path}: {images.shape[0]} images, a tuning file holds "
                         f"{protocol.TUNING_POOL}")  # fmt: skip
    return images[:n].reshape(n, -1) / 255.0


@dataclass(frozen=True)
class Sweep:
    """A sweep ready to run. `inputs` are as a run dir sees them (shipped); `spec_doc` is the
    spec file's JSON, as submitted."""

    spec: Spec
    spec_doc: dict[str, Any]
    ref: str
    template_name: str
    template_sha256: str
    variants: dict[str, dict[str, Any]]  # overrides by name, `expand`
    aliases: dict[str, str]  # grid names that are another variant (`expand`)
    templates: dict[str, dict[str, Any]]  # the template with them applied, by name
    inputs: protocol.TuningInputs

    def report_path(self, variant: str, length: int) -> str:
        return f"{variant}/{length}/report.json"

    def keys(self) -> list[str]:
        return [_key(v, length, s, c) for v in self.variants for length in self.spec.lengths
                for s in self.spec.seeds for c in (TRAIN, *self.spec.conditions)]  # fmt: skip


def make_sweep(
    spec: Spec,
    spec_doc: dict[str, Any],
    *,
    ref: str,
    template_path: Path,
    inputs: protocol.TuningInputs,
) -> Sweep:
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError(f"ref must be a full commit SHA, got {ref!r}")
    template: dict[str, Any] = json.loads(template_path.read_text())
    if template.get("run_id", "auto") != "auto":
        raise ValueError(f'{template_path}: run_id must be "auto": each chain is its own run')
    variants, aliases = expand(spec, template)
    return Sweep(spec, spec_doc, ref, template_path.name, datasets.sha256_file(template_path),
                 variants, aliases, {v: apply(template, o) for v, o in variants.items()},
                 inputs)  # fmt: skip


def _label(variant: str, length: int) -> str:
    return f"{variant}/{length}"


def _key(variant: str, length: int, seed: int, condition: str) -> str:
    return f"{_label(variant, length)}/{seed}/{condition}"


def _point(p: Planned) -> tuple[str, int]:
    variant, length = p.label.rsplit("/", 1)
    return variant, int(length)


def _train(sw: Sweep, variant: str, seed: int) -> Planned:
    (job,) = protocol.train_jobs(sw.templates[variant], sw.inputs, data_ref=runner.DATA_REF,
                                 seeds=[seed], steps=sw.spec.lengths[0],
                                 change_interval=STEPS_PER_IMAGE)  # fmt: skip
    return Planned(_label(variant, sw.spec.lengths[0]), sw.ref, job, 0)


# @decision DEC-002: a further length is a continuation job under its source's run_id
# @rationale: coastd's load_state_from takes only a done job's final state (no snapshots), and
# coast continues a checkpoint's train sequence only for the run that wrote it (S.5.14).
def _continued(job: protocol.Job, run_id: str, steps: int) -> protocol.Job:
    """The training job continued under `run_id`, the run its source ran as: coast then
    continues the source checkpoint's train sequence (S.5.14) instead of starting it afresh."""
    cfg = json.loads(job.config_text)
    cfg["run_id"] = run_id
    text = json.dumps(cfg, indent=2, sort_keys=True)
    return protocol.Job(seed=job.seed, condition=TRAIN, steps=steps, config_text=text)


def _trials(sw: Sweep, variant: str, length: int, seed: int, base: int, source: str | None
            ) -> list[Planned]:  # fmt: skip
    jobs = protocol.tuning_jobs(sw.templates[variant], sw.inputs, data_ref=runner.DATA_REF,
                                seed=seed, conditions=sw.spec.conditions,
                                base_step=base)  # fmt: skip
    return [
        Planned(_label(variant, length), sw.ref, j, base, source, FOLLOW_UP_PRIORITY) for j in jobs
    ]


def _needed(sw: Sweep, variant: str, index: int, skip: Collection[tuple[str, int]]) -> bool:
    """A point of `variant` from `lengths[index]` on still needs its report."""
    return any((variant, length) not in skip for length in sw.spec.lengths[index:])


def first_jobs(sw: Sweep, skip: Collection[tuple[str, int]]) -> list[Planned]:
    """The first training job of every chain one of whose points still needs its report."""
    return [_train(sw, v, s) for v in sw.variants if _needed(sw, v, 0, skip)
            for s in sw.spec.seeds]  # fmt: skip


def after(sw: Sweep, done: Done, skip: Collection[tuple[str, int]]) -> list[Planned]:
    """What follows a done job: after a training job at `lengths[i]`, the point's tuning trials
    (unless its report exists) and the continuation to `lengths[i + 1]` (unless every later
    point's report exists); after a trial, nothing. The step a job continues from is its
    source's `n_steps_completed`."""
    p = done.planned
    if p.job.condition != TRAIN:
        return []
    variant, length = _point(p)
    base = runs.read_manifest(done.run).n_steps_completed
    out: list[Planned] = []
    if (variant, length) not in skip:
        out += _trials(sw, variant, length, p.job.seed, base, done.job_id)
    i = sw.spec.lengths.index(length)
    if i + 1 < len(sw.spec.lengths) and _needed(sw, variant, i + 1, skip):
        nxt = sw.spec.lengths[i + 1]
        run_id = json.loads(done.run.config.read_text())["run_id"]  # checked: `check_run_id`
        job = _continued(p.job, run_id, nxt - base)
        out.append(
            Planned(_label(variant, nxt), sw.ref, job, base, done.job_id, FOLLOW_UP_PRIORITY)
        )
    return out


def fingerprint(daemons: Sequence[Daemon], sw: Sweep, expected_inputs: Mapping[str, str]) -> str:
    """SHA256 of what decides a sweep's jobs and where they run (`runner.fingerprint`'s rule:
    the config texts are hashed, so an edited protocol refuses an old ledger). A continuation's
    text names its source's run_id, known only once the source is done; it is hashed with a
    placeholder in its place."""
    texts: list[str] = []
    for v in sw.variants:
        for s in sw.spec.seeds:
            train = _train(sw, v, s).job
            texts.append(train.config_text)
            for i, length in enumerate(sw.spec.lengths):
                if i:
                    texts.append(_continued(train, "<its source's run_id>", 1).config_text)
                texts += [t.job.config_text for t in _trials(sw, v, length, s, length, None)]
    doc = {
        "daemons": identities(daemons),
        "ref": sw.ref,
        "spec": msgspec.to_builtins(sw.spec),
        "variants": sw.variants,
        "inputs": msgspec.to_builtins(sw.inputs),
        "expected_inputs": dict(sorted(expected_inputs.items())),
        "config_texts": texts,
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def identities(daemons: Sequence[Daemon]) -> list[str]:
    return [f"{d.info.hostname}:{d.info.tcp_port}" for d in runner.canonical(daemons)]


class Point(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    variant: str
    length: int
    report: str  # relative to sweep.json's directory
    seeds: list[int]
    status: Literal["done", "pending", "failed"]


class Doc(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """`sweep.json`: the index the aggregator reads. A point is `done` once its report exists,
    `failed` when a job of it, or its report, stopped the last run. `fingerprint` is the
    sweep's (`fingerprint`): an `--out` holding another sweep's is refused (`check_out`)."""

    schema: Literal["coast-eval-sweep/1"]
    spec: dict[str, Any]
    template_sha256: str
    ref: str
    daemons: list[str]
    variants: dict[str, dict[str, Any]]
    points: list[Point]
    aliases: dict[str, str]  # a grid name -> the variant it is (`base`), not run again
    fingerprint: str


def write_doc(out: Path, sw: Sweep, daemons: Sequence[Daemon], fp: str,
              failed: Collection[tuple[str, int]]) -> Doc:  # fmt: skip
    """Write `<out>/sweep.json` (atomically: the aggregator may read it at any time)."""
    points = [
        Point(
            v,
            length,
            sw.report_path(v, length),
            list(sw.spec.seeds),
            "done"
            if (out / sw.report_path(v, length)).is_file()
            else "failed"
            if (v, length) in failed
            else "pending",
        )
        for v in sw.variants
        for length in sw.spec.lengths
    ]
    doc = Doc(SCHEMA, sw.spec_doc, sw.template_sha256, sw.ref, identities(daemons),
              sw.variants, points, sw.aliases, fp)  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    part = out / "sweep.json.part"
    part.write_text(json.dumps(msgspec.to_builtins(doc), indent=1) + "\n")
    os.replace(part, out / "sweep.json")
    return doc


def point_protocol(sw: Sweep, length: int) -> report.Protocol:
    """The report's protocol of a point. `test_indices` are the presented dataset indices,
    rows 0..n-1 of the tuning files (`protocol.TuningInputs`), not test images."""
    n = len(sw.inputs.tuning_indices)
    return report.Protocol(train_steps=length, train_steps_per_image=STEPS_PER_IMAGE,
                           train_images=protocol.TRAIN_COUNT - n, test_indices=list(range(n)),
                           conditions=list(sw.spec.conditions),
                           projection_seeds=list(PROJECTION_SEEDS), frac=FRAC)  # fmt: skip


def write_point(
    out: Path,
    sw: Sweep,
    variant: str,
    length: int,
    dirs: Mapping[tuple[int, str], runs.RunDir],
    pixels: Mapping[str, report.Floats],
) -> None:
    """`report.json` and `report.md` of one point: one state labelled by the variant. The
    report is written last, so a point is never `done` with a partial report."""
    p = point_protocol(sw, length)
    # measured under the point's label, so that a refusal names the sweep's --redo key
    state = report.measure_state(_label(variant, length), dirs, p, pixels,
                                 template=sw.template_name)  # fmt: skip
    state = msgspec.structs.replace(state, label=variant)
    ids = np.asarray(p.test_indices, dtype=np.int64)
    r = report.build(p, report.pixel_measures(pixels, ids), [state])
    path = out / sw.report_path(variant, length)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name("report.md").write_text(
        report.to_markdown(r, images="tuning images held out of the train set")
    )
    part = path.with_name("report.json.part")
    part.write_bytes(report.to_json(r))
    os.replace(part, path)


def redo_keys(sw: Sweep, redo: Collection[str], skip: Collection[tuple[str, int]]) -> set[str]:
    """`redo` with what depends on it: a training key redoes the later lengths of its chain
    and all their trials. A key is refused if a point it redoes has a report: that point is not
    run at all, so its report would stay from the old chain (remove the reports to redo it)."""
    keys = set(sw.keys())
    if unknown := sorted(set(redo) - keys):
        raise ValueError(f"--redo names no job of this sweep: {unknown}")
    out = set(redo)
    for k in redo:
        variant, length, seed, condition = k.rsplit("/", 3)
        i = sw.spec.lengths.index(int(length))
        lengths = sw.spec.lengths[i:] if condition == TRAIN else [int(length)]
        if reports := [sw.report_path(variant, n) for n in lengths if (variant, n) in skip]:
            verb, them = ("exists", "it") if len(reports) == 1 else ("exist", "them")
            raise ValueError(f"--redo {k}: {', '.join(reports)} {verb}; remove {them} to redo "
                             "it")  # fmt: skip
        if condition == TRAIN:
            out |= {_key(variant, n, int(seed), c) for n in lengths
                    for c in (TRAIN, *sw.spec.conditions)}  # fmt: skip
    return out


class _Owner(msgspec.Struct, frozen=True):
    fingerprint: str | None = None  # absent: written before sweeps recorded it


class _ReportState(msgspec.Struct, frozen=True):
    label: str
    git_sha: str


class _ReportStates(msgspec.Struct, frozen=True):
    states: list[_ReportState]


def check_out(out: Path, sw: Sweep, fp: str) -> set[tuple[str, int]]:
    """The points whose report exists in `out`, after checking that `out` is this sweep's:
    its `sweep.json` (if any) carries `fp`, and each report is one state, the variant, at the
    sweep's ref. A report is what makes a point done, so another sweep's reports in `out`
    (run at another ref, its ledger removed) would otherwise count as this sweep's."""

    def refuse(what: str) -> ValueError:
        return ValueError(f"{out}: another sweep's output ({what}); use another --out")

    index = out / "sweep.json"
    if index.is_file():
        owner = msgspec.json.decode(index.read_bytes(), type=_Owner).fingerprint
        if owner != fp:
            raise refuse(f"fingerprint {owner}, this sweep's is {fp}")
    done: set[tuple[str, int]] = set()
    for v in sw.variants:
        for n in sw.spec.lengths:
            path = out / sw.report_path(v, n)
            if not path.is_file():
                continue
            states = msgspec.json.decode(path.read_bytes(), type=_ReportStates).states
            if [(x.label, x.git_sha) for x in states] != [(v, sw.ref)]:
                raise refuse(f"{path} reports {[(x.label, x.git_sha) for x in states]}, not "
                             f"{v} at {sw.ref}")  # fmt: skip
            done.add((v, n))
    return done


def run_sweep(
    daemons: Sequence[Daemon],
    sw: Sweep,
    *,
    out: Path,
    pixels: Mapping[str, report.Floats],
    expected_inputs: Mapping[str, str],
    poll_s: float,
    timeout_s: float,
    retry_s: float = 60.0,
    priority: int = 0,
    redo: Collection[str] = (),
) -> Doc:
    """Run every point without a report and write each report as its last job is done;
    returns the final `sweep.json`. The runner's rules hold (`runner.run_milestone1`): inputs
    verified on every daemon before and after, every run dir checked, fail fast, resumable from
    `<out>/ledger.json`, `redo` keys submitted afresh. The caller checks the spec's conditions
    (`report.check_conditions`) before it writes anything, as `scripts/eval_sweep.py` does."""
    daemons = runner.canonical(daemons)
    placement = runner.assign(sw.spec.seeds, len(daemons))
    fp = fingerprint(daemons, sw, expected_inputs)
    skip = check_out(out, sw, fp)
    redone = redo_keys(sw, redo, skip)
    want = {(s, c) for s in sw.spec.seeds for c in (TRAIN, *sw.spec.conditions)}
    collected: dict[tuple[str, int], dict[tuple[int, str], runs.RunDir]] = {}
    failed: set[tuple[str, int]] = set()

    def follow(done: Done) -> list[Planned]:
        point = _point(done.planned)
        dirs = collected.setdefault(point, {})
        dirs[done.planned.job.seed, done.planned.job.condition] = done.run
        if point not in skip and dirs.keys() == want:
            try:
                write_point(out, sw, *point, dirs, pixels)
            except ValueError:
                failed.add(point)
                write_doc(out, sw, daemons, fp, failed)
                raise
            write_doc(out, sw, daemons, fp, failed)
        return after(sw, done, skip)

    def failing(p: Planned) -> None:
        failed.add(_point(p))
        write_doc(out, sw, daemons, fp, failed)

    def started() -> None:
        write_doc(out, sw, daemons, fp, failed)

    runner.run_jobs(daemons, placement, first_jobs(sw, skip), follow, fp=fp,
                    expected_inputs=expected_inputs, poll_s=poll_s, timeout_s=timeout_s,
                    ledger=out / "ledger.json", retry_s=retry_s, priority=priority,
                    redone=redone, started=started,
                    failed=failing)  # fmt: skip
    return write_doc(out, sw, daemons, fp, ())
