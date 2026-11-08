"""The sweep aggregator (`design/eval_sweep_plan.md` D4): what a hyperparameter sweep found,
per training length, as JSON and Markdown.

The input is a sweep's output directory: `sweep.json` (`coast-eval-sweep/1`) and, per point,
`<variant>/<length>/report.json`, an unchanged frozen-evaluation report (`report.Report`) of
one state labelled with the variant's name and holding all of its training seeds.

Per training length, one row per variant:
- the median over seeds of each measure, as `report.to_markdown` takes it;
- against the `base` variant (the template without overrides), per seed the variant minus
  base, paired by training seed. `deltas` is the median of these differences and
  `directions` the reading rule over them (`report.direction`): a measure moved only if
  every seed moved the same strict way;
- the collapse guards: a code size more than `CODE_SIZE_FACTOR` times larger or smaller than
  base's, and never-wins up in every seed. A variant can raise τ-a or an AUC by changing how
  many units win, not what they code, so a guarded row is read with that in mind; it is
  flagged, not dropped.

A ranking per length orders the variants by each primary measure that is present: τ-a and the
MNIST-C worst AUC (`model/mnistc_worst`). With both, `front` names the variants no other variant
beats on both. The random projection's MNIST-C mean is a reference column (`RANDOM`): the
median over seeds and projection seeds, as the report's controls table takes it, without Δ.

`aliases` are grid points whose every value is the template's: the driver does not run them
again, it names them as `base` (grid name -> `base`), so a grid view can place base there.

A point that is not done, or done without its report, is listed under `missing` with its
status, never dropped silently. A report that exists but does not match its point (another
schema, label, length or seed set) is refused with its path, as is a sweep whose reports do
not share one protocol apart from the training length.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import msgspec

from coast_eval import report as rp
from coast_eval import sweep as sw
from coast_eval.datasets import MNIST_C

SWEEP_SCHEMA = "coast-eval-sweep/1"
SCHEMA = "coast-eval-sweep-report/1"
BASE = "base"
# @decision DEC-SWEEP-001: a code size outside [1/2, 2] times base's flags the row
# @rationale: separation and domination rise with code size alone, and the AUCs compare codes
#   of different sizes; a factor of two is a different code regime, while seed-to-seed spread
#   in the shipped runs stays well inside it. It flags, it never drops or re-ranks.
CODE_SIZE_FACTOR = 2.0
MEAN = "model/mnistc_mean"
WORST = "model/mnistc_worst"
RANDOM = "random/mnistc_mean"  # median over seeds and projection seeds; no per-seed value
CODE_SIZE = "code_size/clean"
NEVER_WINS = "never_wins"
RANKED = ("kendall_tau_a", WORST)  # higher is better for both
REQUIRED = ("kendall_tau_a", "separation", "domination", NEVER_WINS, CODE_SIZE)


class Point(msgspec.Struct, frozen=True):
    """One (variant, length) of `sweep.json`. `status` is kept as the driver wrote it; only
    `done` with an existing report counts as a result. Unknown fields are ignored, so the
    driver may add some without breaking this reader."""

    variant: str
    length: int
    report: str  # relative to the sweep's output directory
    seeds: list[int]
    status: str


class Sweep(msgspec.Struct, frozen=True):
    """The fields of `sweep.json` this module reads (plan D4); others are ignored."""

    schema: Literal["coast-eval-sweep/1"]
    spec: dict[str, Any]
    ref: str
    template_sha256: str
    variants: dict[str, dict[str, Any]]  # name -> JSON pointer -> value; base is {}
    points: list[Point]
    aliases: dict[str, str] = msgspec.field(default_factory=lambda: dict[str, str]())


class Axis(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One grid axis of the spec, in spec order. A derived axis (plan D1: `{"ratio_of": <the
    pointer of another axis>, "factors": [...]}`) lists its factors as `values`: a variant's
    override on it is the other axis' value times one of them."""

    pointer: str
    values: list[Any]
    ratio_of: str | None = None


def spec_axes(spec: Mapping[str, Any]) -> list[Axis]:
    """The grid axes of a sweep spec, decoded by the driver's own type (`sweep.Spec.axes`), so
    a derived axis' factors are the floats the driver named its variants by."""
    axes = msgspec.convert(spec.get("axes", {}), dict[str, list[Any] | sw.Ratio])
    return [
        Axis(p, a.factors, a.ratio_of) if isinstance(a, sw.Ratio) else Axis(p, a)
        for p, a in axes.items()
    ]


def coordinate(axis: Axis, overrides: Mapping[str, Any]) -> Any:
    """A variant's value on `axis` (for a derived axis, its factor), or None when the variant
    sits on none of the axis' values. Values compare by the driver's rule (`sweep.same_value`):
    a derived value against the product of the other axis' value and a factor."""
    if axis.pointer not in overrides:
        return None
    v = overrides[axis.pointer]
    if axis.ratio_of is None:
        return next((x for x in axis.values if sw.same_value(v, x)), None)
    ref = overrides.get(axis.ratio_of)
    if not isinstance(ref, int | float):
        return None
    return next((f for f in axis.values if sw.same_value(v, ref * f)), None)


def axis_text(axis: Axis, value: Any) -> str:
    """A value of `axis` as a grid variant's name writes it (`sweep.token`), a derived value
    as its factor, `x0.01`."""
    return ("x" if axis.ratio_of is not None else "") + sw.token(value, axis.pointer)


def name_coordinates(axes: Sequence[Axis], name: str) -> tuple[Any, ...] | None:
    """The coordinates, in axis order, of a grid variant's name (`k=t` per axis joined by
    `,`, `k` the pointer's last key); None unless every axis is named once with one of its
    values. An alias has no overrides of its own, so its name is all there is to place it."""
    tokens = dict(part.split("=", 1) for part in name.split(",") if "=" in part)
    if len(tokens) != len(axes) or len(name.split(",")) != len(axes):
        return None
    coord: list[Any] = []
    for a in axes:
        t = tokens.get(a.pointer.rsplit("/", 1)[-1])
        hits = [v for v in a.values if axis_text(a, v) == t]
        if len(hits) != 1:
            return None
        coord.append(hits[0])
    return tuple(coord)


class Row(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One variant at one length. `deltas`, `directions` and `guards` are empty for base, and
    for every variant at a length whose base point is missing."""

    variant: str
    seeds: list[int]
    medians: dict[str, float]
    deltas: dict[str, float]  # median over seeds of (variant - base), paired by seed
    directions: dict[str, str]  # report.direction over the paired seeds
    guards: dict[str, str]  # guard name -> why it fired


class Ranking(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    measure: str
    variants: list[str]  # highest median first; ties by name


class Length(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    length: int
    rows: list[Row]  # in the order of sweep.json's variants
    rankings: list[Ranking]
    front: list[str]  # beaten on every ranked measure by no other variant; [] with one measure


class Missing(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    variant: str
    length: int
    status: str
    reason: str


class SweepReport(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    schema: Literal["coast-eval-sweep-report/1"]
    ref: str
    template_sha256: str
    base: str
    axes: list[Axis]  # the spec's grid axes, in spec order
    variants: dict[str, dict[str, Any]]
    seeds: list[int]
    measures: list[str]  # the keys of every row's medians, in table order
    lengths: list[Length]
    missing: list[Missing]
    aliases: dict[str, str] = msgspec.field(default_factory=lambda: dict[str, str]())


def read_sweep(out: Path) -> Sweep:
    path = out / "sweep.json"
    try:
        return msgspec.json.decode(path.read_bytes(), type=Sweep)
    except msgspec.ValidationError as e:
        raise ValueError(f"{path}: not a {SWEEP_SCHEMA} file: {e}") from e


def read_point(out: Path, p: Point) -> rp.Report:
    """The report of a done point; raise unless it is one state labelled with the variant, at
    the point's length, with exactly the point's seeds."""
    path = out / p.report
    try:
        r = msgspec.json.decode(path.read_bytes(), type=rp.Report)
    except msgspec.ValidationError as e:
        raise ValueError(f"{path}: not a coast-eval report of schema {rp.SCHEMA}: {e}") from e
    if len(r.states) != 1 or r.states[0].label != p.variant:
        labels = [x.label for x in r.states]
        raise ValueError(f"{path}: states {labels}, a sweep point is one state {p.variant!r}")
    if r.protocol.train_steps != p.length:
        raise ValueError(
            f"{path}: trained {r.protocol.train_steps} steps, sweep.json says {p.length}"
        )
    if sorted(r.states[0].measures) != sorted(p.seeds):
        raise ValueError(
            f"{path}: seeds {sorted(r.states[0].measures)}, sweep.json says {sorted(p.seeds)}"
        )
    return r


def table_measures(measured: Mapping[str, float]) -> list[str]:
    """The per-seed keys of the table, in order: primary, per corruption, secondary."""
    keys = [
        "kendall_tau_a",
        MEAN,
        WORST,
        *(f"model/{c}" for c in MNIST_C),
        "separation",
        "separation_chance",
        "domination",
        "domination_chance",
        NEVER_WINS,
        CODE_SIZE,
    ]
    missing = [k for k in REQUIRED if k not in measured]
    if missing:
        raise ValueError(f"lacks {missing}")
    return [k for k in keys if k in measured]


def guards(row: Row, base: Row) -> dict[str, str]:
    """The collapse guards of `row` against `base` (see the module docstring)."""
    out: dict[str, str] = {}
    v, b = row.medians[CODE_SIZE], base.medians[CODE_SIZE]
    if b <= 0.0:
        if v != b:
            out["code_size"] = f"{v:.1f} units, base {b:.1f}"
    elif not 1.0 / CODE_SIZE_FACTOR <= v / b <= CODE_SIZE_FACTOR:
        out["code_size"] = f"{v / b:.2f} times base ({v:.1f} vs {b:.1f} units)"
    d = row.directions.get(NEVER_WINS, "")
    if d.endswith("increased"):
        out["never_wins"] = (
            f"{d} ({row.medians[NEVER_WINS]:.3f} vs base {base.medians[NEVER_WINS]:.3f})"
        )
    return out


def ranking(rows: Sequence[Row], measure: str) -> Ranking:
    order = sorted(rows, key=lambda r: (-r.medians[measure], r.variant))
    return Ranking(measure, [r.variant for r in order])


def front(rows: Sequence[Row], measures: Sequence[str]) -> list[str]:
    """The variants no other variant beats on every measure (>= on all, > on one), in row
    order; [] for fewer than two measures, where the ranking already says it."""
    if len(measures) < 2:
        return []

    def beats(a: Row, b: Row) -> bool:
        ge = all(a.medians[k] >= b.medians[k] for k in measures)
        return ge and any(a.medians[k] > b.medians[k] for k in measures)

    return [r.variant for r in rows if not any(beats(o, r) for o in rows if o is not r)]


def _row(
    variant: str,
    x: rp.State,
    keys: Sequence[str],
    projection_seeds: Sequence[int],
    base: tuple[rp.Report, rp.State, Row] | None,
) -> Row:
    medians = {k: statistics.median(m[k] for m in x.measures.values()) for k in keys}
    if MEAN in keys and projection_seeds:
        medians[RANDOM] = statistics.median(
            m[f"random/{q}/mnistc_mean"] for m in x.measures.values() for q in projection_seeds
        )
    row = Row(variant, sorted(x.measures), medians, {}, {}, {})
    if base is None or variant == BASE:
        return row
    base_report, base_state, base_row = base
    # report.build pairs the seeds and refuses a pair that ran on different hosts or
    # toolchains, or measured different keys: the checks a two-state report makes.
    pair = rp.build(base_report.protocol, base_report.pixel, [base_state, x])
    deltas = {k: statistics.median(d[k] for d in pair.deltas.values()) for k in keys}
    directions = {k: rp.direction([d[k] for d in pair.deltas.values()]) for k in keys}
    row = msgspec.structs.replace(row, deltas=deltas, directions=directions)
    return msgspec.structs.replace(row, guards=guards(row, base_row))


def aggregate(out: Path) -> SweepReport:
    """The sweep report of the sweep in `out`; see the module docstring."""
    sweep = read_sweep(out)
    if BASE not in sweep.variants:
        raise ValueError(f"{out / 'sweep.json'}: no {BASE!r} variant to compare against")
    for alias, target in sweep.aliases.items():
        if target not in sweep.variants:
            raise ValueError(f"{out / 'sweep.json'}: alias {alias} names no variant ({target})")
    try:
        axes = spec_axes(sweep.spec)
    except (ValueError, msgspec.ValidationError) as e:
        raise ValueError(f"{out / 'sweep.json'}: {e}") from e
    seen: set[tuple[str, int]] = set()
    loaded: dict[tuple[str, int], rp.Report] = {}
    missing: list[Missing] = []
    for p in sweep.points:
        if p.variant not in sweep.variants:
            raise ValueError(f"point {p.variant}/{p.length}: not a variant of sweep.json")
        if (p.variant, p.length) in seen:
            raise ValueError(f"point {p.variant}/{p.length}: listed twice in sweep.json")
        seen.add((p.variant, p.length))
        if p.status != "done":
            missing.append(Missing(p.variant, p.length, p.status, f"status {p.status}"))
        elif not (out / p.report).is_file():
            missing.append(Missing(p.variant, p.length, p.status, f"no report at {p.report}"))
        else:
            loaded[p.variant, p.length] = read_point(out, p)
    lengths = sorted({p.length for p in sweep.points})
    seeds = sorted({s for p in sweep.points for s in p.seeds})
    if not loaded:
        return SweepReport(
            SCHEMA, sweep.ref, sweep.template_sha256, BASE, axes, sweep.variants, seeds, [],
            [Length(n, [], [], []) for n in lengths], missing, sweep.aliases,
        )  # fmt: skip
    first = next(iter(loaded.values()))
    protocol = msgspec.structs.replace(first.protocol, train_steps=0)
    for (variant, length), r in loaded.items():
        if msgspec.structs.replace(r.protocol, train_steps=0) != protocol:
            raise ValueError(
                f"{variant}/{length}: its protocol differs from the other points' beyond the "
                "training length"
            )
    states = {key: r.states[0] for key, r in loaded.items()}
    some = next(iter(states.values()))
    try:
        keys = table_measures(some.measures[next(iter(some.measures))])
    except ValueError as e:
        raise ValueError(f"the sweep's reports: {e}") from e
    proj = protocol.projection_seeds
    out_lengths: list[Length] = []
    for length in lengths:
        base: tuple[rp.Report, rp.State, Row] | None = None
        if (BASE, length) in states:
            x = states[BASE, length]
            base = (loaded[BASE, length], x, _row(BASE, x, keys, proj, None))
        rows: list[Row] = []
        for variant in sweep.variants:
            if (variant, length) not in states:
                continue
            x = states[variant, length]
            if any(k not in m for m in x.measures.values() for k in keys):
                raise ValueError(f"{variant}/{length}: lacks some of the measures {keys}")
            try:
                rows.append(_row(variant, x, keys, proj, base))
            except ValueError as e:
                e.add_note(f"refused: sweep point {variant}/{length}")
                raise
        ranked = [k for k in RANKED if k in keys]
        out_lengths.append(
            Length(length, rows, [ranking(rows, k) for k in ranked], front(rows, ranked))
        )
    measures = [*keys, RANDOM] if MEAN in keys and proj else keys
    return SweepReport(
        SCHEMA, sweep.ref, sweep.template_sha256, BASE, axes, sweep.variants, seeds,
        measures, out_lengths, missing, sweep.aliases,
    )  # fmt: skip


def to_json(r: SweepReport) -> bytes:
    """Canonical JSON: sorted keys, as `report.to_json` writes."""
    return (json.dumps(msgspec.to_builtins(r), sort_keys=True, indent=1) + "\n").encode()


def decode(data: bytes) -> SweepReport:
    try:
        return msgspec.json.decode(data, type=SweepReport)
    except msgspec.ValidationError as e:
        raise ValueError(f"not a {SCHEMA} report: {e}") from e


def steps_text(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def _mark(direction: str) -> str:
    if direction.endswith("increased"):
        return " ↑"
    if direction.endswith("decreased"):
        return " ↓"
    return ""


def _delta(row: Row, key: str) -> str:
    if key not in row.deltas:
        return ""
    return f"{row.deltas[key]:+.4f}{_mark(row.directions[key])}"


def _columns(r: SweepReport) -> list[tuple[str, str, bool]]:
    """(key, heading, with a delta column) of the Markdown table."""
    cols = [("kendall_tau_a", "τ-a", True)]
    if MEAN in r.measures:
        cols += [(WORST, "MNIST-C worst", True), (MEAN, "MNIST-C mean", True)]
        if RANDOM in r.measures:
            cols.append((RANDOM, "random MNIST-C mean", False))
    cols += [(CODE_SIZE, "code size", True), (NEVER_WINS, "never wins", True)]
    if "separation_chance" in r.measures:
        cols.append(("separation", "separation (chance)", False))
    return cols


def _cell(row: Row, key: str) -> str:
    v = row.medians[key]
    if key == CODE_SIZE:
        return f"{v:.1f}"
    if key == "separation" and "separation_chance" in row.medians:
        return f"{v:.3f} ({row.medians['separation_chance']:.3f})"
    return f"{v:.4f}"


def _delta_cell(row: Row, key: str) -> str:
    if key == CODE_SIZE and key in row.deltas:
        return f"{row.deltas[key]:+.1f}{_mark(row.directions[key])}"
    return _delta(row, key)


def to_markdown(r: SweepReport) -> str:
    n_variants = len(r.variants)
    lines = [
        f"# Sweep: {n_variants} variants at " + ", ".join(steps_text(x.length) for x in r.lengths),
        "",
        f"Built from git `{r.ref}`, template SHA256 `{r.template_sha256}`; training seeds "
        f"{r.seeds}. Each value is the median over the training seeds.",
    ]
    if r.axes:
        lines += ["", "Grid axes:", ""]
        for a in r.axes:
            of = f" times `{a.ratio_of}`" if a.ratio_of else ""
            values = ", ".join(axis_text(a, v) for v in a.values)
            lines.append(f"- `{a.pointer}`: {values}{of}")
    if r.aliases:
        lines += [
            "",
            "Grid points that are another variant (every value the template's), not run again: "
            + ", ".join(f"`{a}` is `{t}`" for a, t in r.aliases.items())
            + ".",
        ]
    lines += [
        "",
        f"Δ is the median over seeds of each seed's value minus `{r.base}`'s, paired by seed. "
        "↑ or ↓: every seed moved that way (the reading rule); no mark: not every seed did. "
        f"Guards: `code_size` fires outside {1 / CODE_SIZE_FACTOR:g} to {CODE_SIZE_FACTOR:g} "
        f"times `{r.base}`'s code size, `never_wins` when never-wins rose in every seed; a "
        "guarded row may have changed how many units win rather than what they code. No "
        "thresholds on the measures, no significance claim.",
    ]
    if r.missing:
        lines += ["", "## Not in this report", ""]
        lines += [
            f"- `{m.variant}` at {steps_text(m.length)}: {m.reason}"
            for m in sorted(r.missing, key=lambda m: (m.length, m.variant))
        ]
    cols = _columns(r)
    head = ["Variant"]
    for _, name, delta in cols:
        head += [name, "Δ"] if delta else [name]
    head.append("guards")
    for x in r.lengths:
        lines += ["", f"## {steps_text(x.length)} training steps", ""]
        if not x.rows:
            lines.append("No point at this length is done.")
            continue
        if not any(row.variant == r.base for row in x.rows):
            lines += [f"`{r.base}` is missing at this length, so there are no Δ.", ""]
        lines += ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 2) + "---|"]
        for row in x.rows:
            cells = [f"`{row.variant}`"]
            for key, _, delta in cols:
                cells.append(_cell(row, key))
                if delta:
                    cells.append(_delta_cell(row, key))
            cells.append("; ".join(f"{k}: {v}" for k, v in row.guards.items()))
            lines.append("| " + " | ".join(cells) + " |")
        lines += _ranking_lines(x)
    return "\n".join([*lines, ""])


def _ranking_lines(x: Length) -> list[str]:
    if not x.rankings:
        return []
    by = {row.variant: row for row in x.rows}
    names = {"kendall_tau_a": "τ-a", WORST: "MNIST-C worst"}
    head = ["Rank", *(f"by {names[k.measure]}" for k in x.rankings)]
    lines = [
        "",
        "### Ranking",
        "",
        "Highest median first; a guarded variant is marked (g).",
        "",
        "| " + " | ".join(head) + " |",
        "|---:|" + "---|" * (len(head) - 1),
    ]
    for i in range(len(x.rows)):
        cells = [str(i + 1)]
        for k in x.rankings:
            v = by[k.variants[i]]
            mark = " (g)" if v.guards else ""
            cells.append(f"`{v.variant}` {v.medians[k.measure]:.4f}{mark}")
        lines.append("| " + " | ".join(cells) + " |")
    if x.front:
        lines += [
            "",
            "Beaten on both by no other variant: " + ", ".join(f"`{v}`" for v in x.front) + ".",
        ]
    return lines


def write(out: Path) -> list[Path]:
    """Aggregate the sweep in `out`; write `sweep_report.json` and `sweep_report.md` there."""
    r = aggregate(out)
    js, md = out / "sweep_report.json", out / "sweep_report.md"
    js.write_bytes(to_json(r))
    md.write_text(to_markdown(r))
    return [js, md]
