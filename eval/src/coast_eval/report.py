"""The milestone-1 report (issue #82): per-seed measurements of one or two code states, the
after-minus-before deltas and the reading rule, as JSON and Markdown.

The input is a state's runs keyed (training seed, condition), as `runner.run_milestone1`
returns them per state label. The training seeds are the replicates. A frozen trial loads
its training run's final state, whose restored random state wins over the config's seed, so
the trials of one seed differ by condition only; the seed is taken from the key, never from
a frozen manifest.

Per seed (the keys of #243):
- `kendall_tau_a`: pixel-cosine vs code-cosine distances over the clean images.
- `model/<condition>`: matched/mismatched cosine AUC of the clean codes against the
  condition's codes. `model/clean` is the clean run against itself; `model/zeros` is the
  chance control.
- `random/<s>/<condition>`: the same AUC for a random WTA projection of the pixels, code
  sizes matched presentation by presentation to this seed's clean codes, projection seed s.
- `<prefix>/mnistc_mean`, `<prefix>/mnistc_worst`: over the 15 MNIST-C corruptions.
- `separation`, `never_wins`, `domination` of the clean codes; `code_size/<condition>`, the
  mean number of winners per presentation.
- `separation_chance`, `domination_chance`: the same measures with each presentation's units
  shuffled (`metrics.shuffled_units`), the mean over the projection seeds. Both measures rise
  with code size alone, so read them against these. A schema /1 report written before
  5f0e17d8 lacks these two keys, and so does its conversion (`scripts/eval_report_upgrade.py`):
  they cannot be computed afterwards. A /1 report written after 5f0e17d8 keeps them.
`pixel/<condition>` depends on neither state nor seed and is stored once.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import msgspec
import numpy as np
from numpy.typing import NDArray

from coast_eval import metrics, protocol, runs
from coast_eval.datasets import MNIST_C, TEST_IMAGES, corruption_files, read_idx_images
from coast_eval.observe import Codes, read_codes

type Measures = dict[str, float]
type Floats = NDArray[np.float64]
PRIMARY = ("kendall_tau_a", "model/mnistc_mean", "model/mnistc_worst")
SECONDARY = ("separation", "separation_chance", "never_wins", "domination", "domination_chance")
_PRIMARY_ALL = (*PRIMARY, *(f"model/{c}" for c in MNIST_C))


class Protocol(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    train_steps: int
    train_steps_per_image: int
    train_images: int  # entries in the train index manifest
    test_indices: list[int]
    conditions: list[str]
    projection_seeds: list[int]
    frac: float  # relative winner rule
    window: int = protocol.FROZEN_STEPS_PER_IMAGE  # settle window; the lead is per state


class Host(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Where a seed's runs ran and the toolchain that built them (`coast_runtime.json`).
    Results are bit-identical across hosts only for the same compiler and target CPU
    (`design/eval_multi_daemon_plan.md` §3 A1); None: a binary built before they were
    recorded."""

    hostname: str
    compiler: str | None
    target_cpu: str | None


class State(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    label: str
    git_sha: str
    binary_sha256: str | None
    lead: int
    measures: dict[int, Measures]  # training seed -> measures
    worst: dict[int, list[str]]  # corruptions at model/mnistc_worst, ties kept
    template: str  # file name of the template the state's configs were made from
    hosts: dict[int, Host] = msgspec.field(default_factory=lambda: dict[int, Host]())  # seed->host


SCHEMA = "coast-eval-report/2"


class Report(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # /2 is /1 without `Protocol.extra_lead` and with `separation_chance`/`domination_chance`,
    # which a report converted from an early /1 lacks (see the module docstring).
    schema: Literal["coast-eval-report/2"]
    protocol: Protocol
    pixel: Measures
    states: list[State]
    deltas: dict[int, Measures]  # second state minus first; empty for one state
    directions: dict[str, str]  # the reading rule per primary; empty for one state


def check_conditions(conditions: Sequence[str]) -> None:
    """Raise unless a report can summarise `conditions`: unique and known, `clean` (every AUC
    is taken against the clean codes) and either none or all 15 MNIST-C corruptions (their
    mean and worst are over all 15). `zeros` is optional. Call it before any run."""
    known, given = set(protocol.CONDITIONS), set(conditions)
    if len(given) != len(conditions) or not given <= known or "clean" not in given:
        raise ValueError(f"conditions must be unique, known and include clean: {conditions}")
    n = len(given & set(MNIST_C))
    if n not in (0, len(MNIST_C)):
        raise ValueError(f"{n} of the 15 MNIST-C corruptions: give none or all of them")


def condition_pixels(data_root: Path, condition: str, indices: Sequence[int]) -> Floats:
    """(P, 784) pixels in [0, 1] of the presented images; `zeros` is the all-zero set."""
    if condition == "zeros":
        return np.zeros((len(indices), 784))
    rel = TEST_IMAGES if condition == "clean" else corruption_files(condition)[0]
    images = read_idx_images(data_root / rel)[np.asarray(indices)]
    return images.reshape(len(indices), -1) / 255.0


def random_codes(
    pixels: Floats, sizes: NDArray[np.integer], n_units: int, seed: int
) -> tuple[Floats, NDArray[np.bool_]]:
    """A random WTA code: Gaussian projection / sqrt(width), the `sizes[p]` largest units of
    presentation p win (ties to the lower unit). Same seed, same projection."""
    w = np.random.default_rng(seed).standard_normal((n_units, pixels.shape[1]))
    amps = pixels @ (w / np.sqrt(pixels.shape[1])).T
    ranks = np.argsort(np.argsort(-amps, axis=1, kind="stable"), axis=1)
    return amps, ranks < np.asarray(sizes)[:, None]


def _summarise(m: Measures, prefix: str) -> list[str]:
    """Add the MNIST-C mean and worst for `prefix`; return the worst corruptions."""
    present = [c for c in MNIST_C if f"{prefix}/{c}" in m]
    if not present:
        return []
    if len(present) != len(MNIST_C):
        raise ValueError(f"{prefix}: {len(present)} of the 15 MNIST-C corruptions present")
    aucs = {c: m[f"{prefix}/{c}"] for c in MNIST_C}
    m[f"{prefix}/mnistc_mean"] = float(np.mean(list(aucs.values())))
    m[f"{prefix}/mnistc_worst"] = min(aucs.values())
    return [c for c, v in aucs.items() if v == m[f"{prefix}/mnistc_worst"]]


def _auc(
    a: NDArray[np.floating],
    wa: NDArray[np.bool_],
    b: NDArray[np.floating],
    wb: NDArray[np.bool_],
    ids: NDArray[np.int64],
) -> float:
    return metrics.matched_mismatched_auc(a, wa, b, wb, clean_ids=ids, corrupt_ids=ids)


def pixel_measures(pixels: Mapping[str, Floats], ids: NDArray[np.int64]) -> Measures:
    """The raw-pixel reference: the model AUC with full-vector pixel cosine."""
    full = np.ones_like(pixels["clean"], dtype=np.bool_)
    m = {f"pixel/{c}": _auc(pixels["clean"], full, px, full, ids) for c, px in pixels.items()}
    _summarise(m, "pixel")
    return m


def seed_measures(
    codes: Mapping[str, Codes], pixels: Mapping[str, Floats], projection_seeds: Sequence[int]
) -> tuple[Measures, list[str]]:
    """One training seed's measures and its worst corruptions. `codes` and `pixels` are keyed
    by condition and hold the same presentations in the same order; `clean` is required."""
    if codes.keys() != pixels.keys() or "clean" not in codes:
        raise ValueError(f"conditions {sorted(codes)} vs pixels {sorted(pixels)}; need clean")
    clean = codes["clean"]
    ids = clean.stim_ids
    m: Measures = {
        "kendall_tau_a": metrics.similarity_preservation(pixels["clean"], clean.amps, clean.masks),
        "separation": metrics.separation(clean.amps, clean.masks, stim_ids=ids),
        "never_wins": metrics.never_wins_fraction(clean.masks),
        "domination": metrics.domination(clean.masks),
    }
    shuffled = [metrics.shuffled_units(clean.amps, clean.masks, s) for s in projection_seeds]
    m["separation_chance"] = statistics.fmean(
        metrics.separation(a, w, stim_ids=ids) for a, w in shuffled
    )
    m["domination_chance"] = statistics.fmean(metrics.domination(w) for _, w in shuffled)
    sizes = clean.masks.sum(axis=1)
    n_units = clean.amps.shape[1]
    references = {s: random_codes(pixels["clean"], sizes, n_units, s) for s in projection_seeds}
    for cond, c in codes.items():
        m[f"code_size/{cond}"] = float(c.masks.sum(axis=1).mean())
        m[f"model/{cond}"] = _auc(clean.amps, clean.masks, c.amps, c.masks, ids)
        for s, (a, wa) in references.items():
            b, wb = random_codes(pixels[cond], sizes, n_units, s)
            m[f"random/{s}/{cond}"] = _auc(a, wa, b, wb, ids)
    for s in projection_seeds:
        _summarise(m, f"random/{s}")
    return m, _summarise(m, "model")


def read_state(
    run_dirs: Mapping[tuple[int, str], runs.RunDir], p: Protocol, label: str = ""
) -> tuple[str, int, dict[int, dict[str, Codes]], dict[int, Host]]:
    """(git SHA, lead, codes by seed and condition, host by seed) of one state's frozen runs.
    Each run's lead is the settle lead of its own config.json; a presentation of
    `p.window` steps is averaged over its steps lead + 1 .. `p.window` (1-based). Every run
    must be one clean build and one lead, and all runs of a seed must name one host. A run
    this refuses is named by its key (`label`/seed/condition), the one to redo."""
    builds: set[tuple[str, bool]] = set()
    leads: set[int] = set()
    out: dict[int, dict[str, Codes]] = {}
    hosts: dict[int, set[Host]] = {}
    for (seed, cond), run in sorted(run_dirs.items()):
        try:
            m = runs.read_manifest(run)
            builds.add((m.coast_git_sha, m.coast_git_dirty))
            hosts.setdefault(seed, set()).add(
                Host(m.hostname, m.coast_compiler, m.coast_target_cpu)
            )
            if cond == "train":
                continue
            cfg = json.loads(run.config.read_text())
            if protocol.steps_per_image(cfg) != p.window:  # else the window would be its tail
                raise ValueError(
                    f"{run.config}: {protocol.steps_per_image(cfg)} steps per image, the report "
                    f"averages presentations of {p.window}"
                )
            lead = protocol.settle_lead(cfg)
            if lead >= p.window:  # presentation_means would clamp to the last step
                raise ValueError(
                    f"{run.config}: settle lead {lead} (hop count) leaves no step of the "
                    f"{p.window}-step window to average"
                )
            leads.add(lead)
            codes = read_codes(
                runs.observe_dir(run, m, ("population", "input")),
                window=p.window,
                lead=lead,
                min_segment=2,
                frac=p.frac,
            )
            protocol.check_aligned(codes.stim_ids, p.test_indices)
            out.setdefault(seed, {})[cond] = codes
        except ValueError as e:
            key = f"{label}/{seed}/{cond}"
            e.add_note(
                f"refused: {key} ({run.root}); to redo it, run the same evaluation "
                f"with --redo {key}"
            )
            raise
    if len(builds) != 1 or next(iter(builds))[1]:
        raise ValueError(f"a state's runs must come from one clean build, got {sorted(builds)}")
    if len(leads) != 1:
        raise ValueError(f"a state's runs must share one settle lead, got {sorted(leads)}")
    split = {seed: h for seed, h in hosts.items() if len(h) != 1}
    if split:
        raise ValueError(f"a seed's runs must name one host and toolchain, got {split}")
    seeds = {seed for seed, _ in run_dirs}
    if out.keys() != seeds or any(sorted(c) != sorted(p.conditions) for c in out.values()):
        raise ValueError(
            f"every seed of {sorted(seeds)} needs exactly the conditions {p.conditions}"
        )
    return next(iter(builds))[0], leads.pop(), out, {seed: h.pop() for seed, h in hosts.items()}


def measure_state(
    label: str,
    run_dirs: Mapping[tuple[int, str], runs.RunDir],
    p: Protocol,
    pixels: Mapping[str, Floats],
    *,
    template: str,
    binary_sha256: str | None = None,
) -> State:
    """The measures of one state's runs. `template` is the file name of the template its
    configs were made from, recorded as given."""
    sha, lead, codes, hosts = read_state(run_dirs, p, label)
    measures: dict[int, Measures] = {}
    worst: dict[int, list[str]] = {}
    for seed, by_condition in codes.items():
        measures[seed], worst[seed] = seed_measures(by_condition, pixels, p.projection_seeds)
    return State(label, sha, binary_sha256, lead, measures, worst, template, hosts)


def direction(deltas: Sequence[float]) -> str:
    """The reading rule: a measure moved only if every seed moved the same strict way."""
    n = len(deltas)
    if all(d > 0 for d in deltas):
        return f"all {n} increased"
    if all(d < 0 for d in deltas):
        return f"all {n} decreased"
    return f"not all {n} moved in the same direction"


def build(p: Protocol, pixel: Measures, states: Sequence[State]) -> Report:
    """The report of one state, or of two with deltas (second minus first) and directions."""
    if len(states) not in (1, 2):
        raise ValueError(f"a report compares one or two states, got {len(states)}")
    deltas: dict[int, Measures] = {}
    directions: dict[str, str] = {}
    if len(states) == 2:
        a, b = states
        if a.measures.keys() != b.measures.keys():
            raise ValueError(f"seeds differ: {sorted(a.measures)} vs {sorted(b.measures)}")
        for seed, ma in a.measures.items():
            ha, hb = a.hosts.get(seed), b.hosts.get(seed)
            if ha and hb and ha.hostname != hb.hostname:  # plan D4: the host is not the state
                raise ValueError(f"seed {seed} ran on {ha.hostname} for {a.label} but on "
                                 f"{hb.hostname} for {b.label}")  # fmt: skip
            # nor is the toolchain: a cached build from an older compiler would add its
            # drift to the deltas (results match across toolchains only by accident)
            if ha and hb and _recorded(ha) and _recorded(hb) and _toolchain(ha) != _toolchain(hb):
                raise ValueError(
                    f"seed {seed}: {a.label} was built with {_toolchain_text(ha)}, {b.label} "
                    f"with {_toolchain_text(hb)}; the deltas would mix the toolchain change "
                    "with the code change"
                )
            if ma.keys() != b.measures[seed].keys():
                raise ValueError(f"seed {seed}: the two states measured different keys")
            deltas[seed] = {k: b.measures[seed][k] - v for k, v in ma.items()}
        measured = a.measures[next(iter(a.measures))]
        keys = [k for k in _PRIMARY_ALL if k in measured]
        directions = {k: direction([d[k] for d in deltas.values()]) for k in keys}
    return Report(SCHEMA, p, pixel, list(states), deltas, directions)


def _recorded(h: Host) -> bool:
    return bool(h.compiler and h.target_cpu)


def _toolchain(h: Host) -> tuple[str | None, str | None]:
    return h.compiler, h.target_cpu


def _toolchain_text(h: Host) -> str:
    return f"{h.compiler} for {h.target_cpu}" if _recorded(h) else "an unrecorded toolchain"


def _toolchains_of(x: State) -> str:
    """How a state's binaries were built, for its line in the Markdown."""
    by: dict[str, list[int]] = {}
    for seed, h in sorted(x.hosts.items()):
        by.setdefault(_toolchain_text(h), []).append(seed)
    if not by:
        return ""
    if len(by) == 1:
        return f", built with {next(iter(by))}"
    return ", built with " + "; ".join(f"{t} (seeds {seeds})" for t, seeds in by.items())


def to_json(r: Report) -> bytes:
    """Canonical JSON: sorted keys (seed keys become strings, as JSON requires)."""
    return (json.dumps(msgspec.to_builtins(r), sort_keys=True, indent=1) + "\n").encode()


def _median(r: State, key: str) -> float:
    return statistics.median(m[key] for m in r.measures.values())


def to_markdown(r: Report, *, images: str = "held-out test images") -> str:
    """The report as Markdown. `images` says what the frozen trials presented (a sweep's
    tuning trials present train images held out of training)."""
    p = r.protocol
    s = r.states
    shown = p.train_steps // p.train_steps_per_image
    passes = shown / p.train_images
    first, last = p.test_indices[0], p.test_indices[-1]
    contiguous = last - first + 1 == len(p.test_indices)
    lines = [
        "# Frozen evaluation: " + " vs ".join(x.label for x in s),
        "",
        *(
            f"- **{x.label}**: git `{x.git_sha}`"
            + (f", binary SHA256 `{x.binary_sha256}`" if x.binary_sha256 else "")
            + _toolchains_of(x)
            + f", template `{x.template}`, settle lead {x.lead}."
            for x in s
        ),
        "",
        f"Training: {p.train_steps} steps at {p.train_steps_per_image} steps per image present "
        f"{shown} images, {passes:.1f} passes over the {p.train_images} training images, each "
        "pass in a fresh random order. "
        f"Frozen trials: {len(p.test_indices)} {images} ("
        f"{'indices' if contiguous else 'a sample of indices'} {first}..{last}), "
        f"{p.window} steps each, the settled steps "
        f"averaged, winners at >= {p.frac} of the presentation's maximum. Replicates are the "
        f"{len(s[0].measures)} training seeds; projection seeds {p.projection_seeds} are nested "
        "controls, not replicates.",
    ]
    hosts = {h for x in s for h in x.hosts.values()}
    if len({h.hostname for h in hosts}) > 1:
        per_seed = sorted(s[0].hosts.items())
        lines += [
            "",
            "Hosts (both states of a seed ran on the same one): "
            + ", ".join(f"seed {seed} on {h.hostname}" for seed, h in per_seed)
            + ".",
        ]
        known = sorted({_toolchain_text(h) for h in hosts if _recorded(h)})
        if len(known) > 1:
            lines += [
                "",
                "The hosts' toolchains differ (" + "; ".join(known) + "): a seed's numbers "
                "depend on its host as well as on its state, so compare a seed only with itself.",
            ]
    if len(s) == 2 and any(not _recorded(h) for h in hosts):
        lines += [
            "",
            "Some runs come from builds that do not record their toolchain, so whether both "
            "states of a seed were built with the same one is not checked for them.",
        ]
    if len(s) == 2:
        if s[0].git_sha != s[1].git_sha:
            lines += [
                "",
                f"The two states differ by every commit between `{s[0].git_sha[:8]}` and "
                f"`{s[1].git_sha[:8]}`, not by one model change.",
            ]
        if s[0].template != s[1].template:
            lines += [
                "",
                f"The two states run different templates, `{s[0].template}` and `{s[1].template}`.",
            ]
        lines += [
            "",
            "## Reading rule",
            "",
            "A measure moved only if all seeds moved in the same strict direction. No "
            "thresholds, no aggregate verdict, no significance claim.",
            "",
            *(f"- `{k}`: {r.directions[k]}." for k in _PRIMARY_ALL if k in r.directions),
        ]
    keys = [
        *PRIMARY,
        *SECONDARY,
        "model/clean",
        "model/zeros",
        "code_size/clean",
        "code_size/zeros",
        *(f"model/{c}" for c in MNIST_C),
    ]
    keys = [k for k in keys if k in s[0].measures[next(iter(s[0].measures))]]
    head = ["Seed", "Measure", *(x.label for x in s), *(["Delta"] if r.deltas else [])]
    lines += [
        "",
        "## Per seed",
        "",
        "| " + " | ".join(head) + " |",
        "|---|---|" + "---:|" * (len(head) - 2),
    ]
    for seed in s[0].measures:
        for k in keys:
            vals = [x.measures[seed][k] for x in s] + ([r.deltas[seed][k]] if r.deltas else [])
            lines.append(f"| {seed} | {k} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    if "separation_chance" in keys:
        lines += [
            "",
            "`separation` and `domination` rise with code size alone. `separation_chance` and "
            "`domination_chance` are the same measures with each presentation's code size and "
            "amplitudes kept and its units shuffled, the mean over the projection seeds.",
        ]
    elif "separation" in keys:
        lines += [
            "",
            "`separation` and `domination` rise with code size alone. This report has no "
            "chance levels for them (it was converted from schema /1), so read them with "
            "`code_size/clean`.",
        ]
    lines += _controls(r, mnistc="model/mnistc_mean" in keys, zeros="model/zeros" in keys)
    return "\n".join([*lines, ""])


def _controls(r: Report, *, mnistc: bool, zeros: bool) -> list[str]:
    """The controls table, with the columns of the controls that ran: the MNIST-C references
    (random and pixel) if the corruptions ran, the zero-input control if `zeros` ran."""
    head: list[str] = []
    rows: list[list[str]] = [[x.label] for x in r.states]
    notes: list[str] = []
    if mnistc:
        head += ["model mnistc_mean", "random mnistc_mean", "pixel mnistc_mean"]
        for row, x in zip(rows, r.states, strict=True):
            rand = statistics.median(
                m[f"random/{q}/mnistc_mean"]
                for m in x.measures.values()
                for q in r.protocol.projection_seeds
            )
            row += [f"{_median(x, 'model/mnistc_mean'):.4f}", f"{rand:.4f}",
                    f"{r.pixel['pixel/mnistc_mean']:.4f}"]  # fmt: skip
        notes.append(
            "The random projections come from numpy's default generator, so they agree with "
            "another implementation's only statistically."
        )
    if zeros:
        head += ["model/zeros", "code_size/zeros"]
        for row, x in zip(rows, r.states, strict=True):
            row += [f"{_median(x, 'model/zeros'):.4f}", f"{_median(x, 'code_size/zeros'):.1f}"]
        notes.insert(
            0,
            "The zeros control is full codes, not empty ones: with all-zero input the peak "
            "amplitude is baseline neuron noise, so under the relative rule nearly every unit "
            "wins every presentation. Its AUC is chance because the noise carries no image.",
        )
    if not head:
        return []
    return [
        "",
        "## Controls (medians over seeds; random also over projection seeds)",
        "",
        "| State | " + " | ".join(head) + " |",
        "|---|" + "---:|" * len(head),
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
        " ".join(notes),
    ]
