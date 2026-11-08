#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "coast-eval",
#   "matplotlib",
# ]
#
# [tool.uv.sources]
# coast-eval = { path = "../eval", editable = true }
# ///
"""Draw the figures of one frozen-evaluation report, the curves of a length sweep (#82), or one
measure over a hyperparameter sweep's grid.

Usage:
    scripts/eval_plot.py <report.json> --out-dir <dir>
    scripts/eval_plot.py --curves [--by template|variant] <report.json>... --out-dir <dir>
    scripts/eval_plot.py --heatmap <measure> <sweep_report.json> --out-dir <dir>

The input is a `report.json` of schema `coast-eval-report/2` that `scripts/eval_ab.py` wrote,
of one or two code states. Anything else is refused: another JSON, or a report of an older
schema (convert a /1 report with `scripts/eval_report_upgrade.py`). A /1 report written before
5f0e17d8 has no chance levels for separation and domination, and neither has its conversion;
they are then drawn without one, and the panel and caption say so. A /1 report written after
5f0e17d8 keeps them through conversion.

Writes into --out-dir, and nowhere else:
- `measures.png`: every summary measure per training seed. With two states, a slope chart
  per measure whose title is the reading rule's result; with one state, the seeds' values.
- `corruptions.png`: the 15 MNIST-C AUCs. With two states, a dumbbell per corruption sorted
  by its median change; with one state, each corruption against its references.
- `controls.png`: where each state's MNIST-C mean sits between its references.
- `figures.md`: what each measure represents, and each figure with its caption.
The last two figures need the MNIST-C conditions; without them only `measures.png` is drawn.

Figures are generated output and are never committed (`.gitignore`). Every title is a
computed finding: the reading rule's result for two states, the measured values for one.
Separation and domination rise with code size alone, so they are drawn and judged against
their chance levels (`separation_chance`, `domination_chance`), not on their own.

With --curves, the inputs are one-state reports of the same protocol at different training
lengths (`protocol.train_steps`), one or more templates (grouped by the state's `template`).
Writes `curves.png` and `curves.md`: each measure over training steps, one colour per
template, with its references. Reports that differ in anything but the training length, or
two reports of one template at the same length, are refused. A hyperparameter sweep's points
all run one template, so `--by variant` groups them by the state's label, the variant, instead:
`--curves --by variant <out>/<a>/*/report.json <out>/<b>/*/report.json` draws up to four
variants.

With --heatmap MEASURE, the input is one `sweep_report.json` (`scripts/eval_sweep_report.py`).
Writes `heatmap_<measure>[_<axis>=<value>...].png` and `.md`: the measure's median per grid
variant, rows the first axis, columns the second, a panel per value of a third and a row of
panels per training length; guarded variants are marked, and a grid point that is `base` (an
alias in `sweep.json`) shows base's value. The grid must have two or three axes after
`--fix AXIS=VALUE` (repeatable) has sliced it to the variants with that value on that axis;
AXIS is the pointer or its last key. Stage 1's four axes take one fix, e.g.
`--heatmap model/mnistc_worst --fix common_learning_rate=0.0001`.
"""

# matplotlib's own stubs leave every **kwargs untyped, so strict pyright reports each plotting
# call as partially unknown; these three rules are off for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import math
import statistics
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # files only; never a window

import matplotlib.pyplot as plt
import msgspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from matplotlib.typing import RcKeyType

from coast_eval import report as rp
from coast_eval import sweep
from coast_eval import sweep_report as sr
from coast_eval.datasets import MNIST_C

# Colour: one meaning per figure. Blues are the states under test, in a fixed order; greys
# are references only; text is always ink, never a series colour.
STATE = ("#8fb8ea", "#2a78d6", "#0f4c9c", "#e8710a")
REF_DARK, REF, REF_LIGHT = "#52514e", "#9a9995", "#c9c8c4"
INK, INK2 = "#0b0b0b", "#52514e"
DOWN, UP = "#c0392b", "#2aa876"
SEED_ALPHA = 0.35
GRID = "#e6e5e1"

STYLE: dict[RcKeyType, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
}


class Measure(msgspec.Struct, frozen=True):
    """One measure's wording, used in the panel subtitle, the axis label and the table."""

    name: str
    question: str
    statistic: str
    reading: str
    axis: str


MEASURES: dict[str, Measure] = {
    "kendall_tau_a": Measure(
        "τ-a",
        "Does the code order images as the pixels do?",
        "Kendall τ-a of pixel-cosine against code-cosine distances over all image pairs; "
        "ties count against",
        "higher: the code keeps the pixels' similarity order",
        "τ-a",
    ),
    "model/mnistc_mean": Measure(
        "MNIST-C mean",
        "Does a corrupted code still identify its image?",
        "cosine AUC of each clean code against its own corrupted code versus other images' "
        "corrupted codes, mean over the 15 corruptions; 0.5 is chance",
        "higher: identity survives corruption",
        "AUC (mean of 15)",
    ),
    "model/mnistc_worst": Measure(
        "MNIST-C worst",
        "How much identity survives the hardest corruption?",
        "the lowest of the 15 per-corruption AUCs, per seed",
        "higher; 0.5: that corruption erases identity",
        "AUC (worst of 15)",
    ),
    "separation": Measure(
        "separation",
        "How much do the codes of different images overlap?",
        "mean winner-union cosine over pairs of different clean images",
        "read against chance (grey), which rises with code size alone",
        "cosine (mean)",
    ),
    "domination": Measure(
        "domination",
        "Does one unit win most images?",
        "the largest share of presentations won by a single unit",
        "read against chance (grey), which rises with code size alone",
        "share of presentations",
    ),
    "never_wins": Measure(
        "never wins",
        "Which units does the code never use?",
        "fraction of units that win on no clean presentation",
        "lower: more of the population is used",
        "fraction of units",
    ),
    "code_size/clean": Measure(
        "code size",
        "How many units win per image?",
        "mean number of winners per clean presentation",
        "context: separation and domination rise with it",
        "units (mean)",
    ),
}
CHANCE = {"separation": "separation_chance", "domination": "domination_chance"}
GEOMETRY = ("kendall_tau_a", "model/mnistc_mean", "model/mnistc_worst")
USAGE = ("separation", "domination", "never_wins", "code_size/clean")
SHARED_AUC = ("model/mnistc_mean", "model/mnistc_worst")
MIN_SPAN = 0.1  # a panel is never zoomed into less than this
NOISE = 0.01  # a median change below this is said to be one
SMALL = 9  # points: the smallest text a reader has to read (the plotting spec's minimum)
LINE = 1.2 * SMALL  # points: a line of it
READING_WIDTH = 34  # characters of a panel's reading per line, at SMALL in the narrowest panel
END_GAP = 1.25 * SMALL  # points between the centres of two end-of-line labels


# --- the report ---------------------------------------------------------------------------


def load(path: Path) -> rp.Report:
    """The report at `path`; raise unless it is a toolkit report of the current schema with one
    or two states, every seed carrying the summary measures, and either every seed or none
    carrying the chance levels (none: a report converted from a /1 written before 5f0e17d8)."""
    try:
        r = msgspec.json.decode(path.read_bytes(), type=rp.Report)
    except msgspec.ValidationError as e:
        raise ValueError(
            f"{path}: not a coast-eval report of the current schema ({rp.SCHEMA}; convert a /1 "
            f"report with scripts/eval_report_upgrade.py): {e}"
        ) from e
    if len(r.states) not in (1, 2):
        raise ValueError(f"{path}: {len(r.states)} states, a report has one or two")
    needed = {*GEOMETRY, *USAGE} - {"model/mnistc_mean", "model/mnistc_worst"}
    carried: set[bool] = set()
    for x in r.states:
        for seed, m in x.measures.items():
            missing = sorted(needed - m.keys())
            if missing:
                raise ValueError(f"{path}: state {x.label} seed {seed} lacks {missing}")
            carried |= {k in m for k in CHANCE.values()}
    if len(carried) > 1:
        raise ValueError(f"{path}: some seeds carry the chance levels and others do not")
    return r


def has_chance(r: rp.Report) -> bool:
    """Whether the report carries the chance levels (all seeds do, or none: see `load`)."""
    return CHANCE["separation"] in r.states[0].measures[seeds(r)[0]]


def judged(r: rp.Report, key: str) -> bool:
    """Whether `key` is judged against its chance level in `r`."""
    return key in CHANCE and has_chance(r)


def has_mnistc(r: rp.Report) -> bool:
    return all(c in r.protocol.conditions for c in MNIST_C)


def seeds(r: rp.Report) -> list[int]:
    return list(r.states[0].measures)


def values(x: rp.State, key: str) -> list[float]:
    return [x.measures[s][key] for s in x.measures]


def median(x: rp.State, key: str) -> float:
    return statistics.median(values(x, key))


def random_values(r: rp.Report, x: rp.State, condition: str) -> list[float]:
    """A state's random-projection AUCs for `condition` over seeds and projection seeds."""
    return [
        m[f"random/{q}/{condition}"]
        for m in x.measures.values()
        for q in r.protocol.projection_seeds
    ]


def short(direction: str) -> str:
    """`report.direction`'s wording, shortened for a title."""
    if direction.endswith("increased"):
        return direction.replace("increased", "up")
    if direction.endswith("decreased"):
        return direction.replace("decreased", "down")
    return "mixed"


def finding(r: rp.Report, key: str) -> str:
    """The reading rule's result for `key` (two states): the report's own for a primary, the
    same rule over the deltas otherwise; for separation and domination over the deltas of
    their excess over chance, since both rise with code size alone."""
    if all(m[key] == 0 for m in r.deltas.values()) and (
        not judged(r, key) or all(m[CHANCE[key]] == 0 for m in r.deltas.values())
    ):
        return "unchanged"
    if key in r.directions:
        return short(r.directions[key])
    if judged(r, key):
        d = [m[key] - m[CHANCE[key]] for m in r.deltas.values()]
    else:
        d = [m[key] for m in r.deltas.values()]
    return short(rp.direction(d))


def panel_title(r: rp.Report, key: str) -> str:
    name = MEASURES[key].name
    if len(r.states) == 1:
        x = r.states[0]
        if judged(r, key):
            return f"{name}: {median(x, key) / median(x, CHANCE[key]):.2f} times chance"
        return (
            f"{name}: {median(x, key):.2f}"
            if key != "code_size/clean"
            else (f"{name}: {median(x, key):.1f}")
        )
    head = f"{name} above chance" if judged(r, key) else name
    title = f"{head}: {finding(r, key)}"
    a, b = r.states
    change = median(b, key) - median(a, key)
    if judged(r, key):
        change -= median(b, CHANCE[key]) - median(a, CHANCE[key])
    if abs(change) < NOISE and key != "code_size/clean" and title.endswith(("up", "down", "mixed")):
        title += f" (change < {NOISE})"
    return title


def measures_title(r: rp.Report) -> str:
    keys = [k for k in GEOMETRY if k in r.states[0].measures[seeds(r)[0]]]
    if len(r.states) == 2:
        return " · ".join(f"{MEASURES[k].name}: {finding(r, k)}" for k in keys)
    x = r.states[0]
    parts = [f"{MEASURES[k].name} {median(x, k):.2f}" for k in keys]
    if not has_chance(r):
        return f"{x.label}: " + ", ".join([*parts, f"code size {median(x, 'code_size/clean'):.1f}"])
    sep = median(x, "separation") / median(x, "separation_chance")
    return f"{x.label}: " + ", ".join([*parts, f"separation {sep:.2f} times its chance level"])


def corruptions_title(r: rp.Report) -> str:
    if len(r.states) == 2:
        words = [finding(r, f"model/{c}").split(" ")[-1] for c in MNIST_C]
        counts = sorted(
            ((words.count(w), w) for w in ("down", "up", "mixed", "unchanged") if words.count(w)),
            reverse=True,
        )
        (n, w), rest = counts[0], counts[1:]
        head = f"{n} of {len(MNIST_C)} corruptions " + (
            "mixed across seeds" if w == "mixed" else f"{w} in every seed"
        )
        return ", ".join([head, *(f"{k} {v}" for k, v in rest)])
    x = r.states[0]
    worst = min(MNIST_C, key=lambda c: median(x, f"model/{c}"))
    above = sum(
        median(x, f"model/{c}") > statistics.median(random_values(r, x, c)) for c in MNIST_C
    )
    return (
        f"worst: {label(worst)} {median(x, f'model/{worst}'):.2f}; above the random "
        f"projection on {above} of {len(MNIST_C)} corruptions"
    )


def controls_title(r: rp.Report) -> str:
    parts: list[str] = []
    for x in r.states:
        bits: list[str] = []
        for summary, word in (("mnistc_mean", "mean"), ("mnistc_worst", "worst")):
            m = median(x, f"model/{summary}")
            q = statistics.median(random_values(r, x, summary))
            side = "above" if m > q else "below" if m < q else "level with"
            bits.append(f"{word} {m:.2f} {side} random {q:.2f}")
        parts.append(f"{x.label}: " + ", ".join(bits))
    return "\n".join(parts)  # one line per state


def label(condition: str) -> str:
    return condition.replace("_", " ")


def provenance(label: str, x: rp.State) -> str:
    return f"{label}: git {x.git_sha[:8]}, {x.template}"


def subtitle(r: rp.Report) -> str:
    return "\n".join(provenance(x.label, x) for x in r.states)  # one line per state


def setup(r: rp.Report) -> str:
    p = r.protocol
    leads = ", ".join(f"{x.lead} ({x.label})" for x in r.states)
    n_c = sum(c in p.conditions for c in MNIST_C)
    return (
        f"Training: {p.train_steps} steps at {p.train_steps_per_image} per image over "
        f"{p.train_images} images; {len(seeds(r))} training seeds. Frozen trials: "
        f"{len(p.test_indices)} test images, {p.window} steps each, settle lead {leads}, "
        f"winners at ≥ {p.frac} of the peak; {n_c} MNIST-C corruptions."
    )


def limit(r: rp.Report) -> str:
    if len(r.states) == 1:
        return "One state: the values of one model, compared with nothing but its references."
    n = len(seeds(r))
    return (
        f"A measure moved only if all {n} seeds moved the same strict way; no thresholds, "
        "no significance."
    )


def states_drawn(r: rp.Report) -> str:
    if len(r.states) == 1:
        return f"{r.states[0].label} in blue"
    return ", ".join(
        f"{x.label} in {'light' if i == 0 else 'dark'} blue" for i, x in enumerate(r.states)
    )


def caption(r: rp.Report, figure: str) -> str:
    """Three parts: what is drawn, the setup in numbers, the limit of the claim."""
    if figure == "measures":
        marks = (
            "Each thin line is one training seed, the thick line the median over seeds"
            if len(r.states) == 2
            else "Each dot is one training seed, the bar the median over seeds"
        )
        drawn = f"{marks}; {states_drawn(r)}. " + (
            "Grey is the chance level of separation and domination: the same codes with each "
            "presentation's units shuffled."
            if has_chance(r)
            else "This report has no chance levels for separation and domination (it was "
            "converted from schema /1), so they are drawn without one; read them with code size."
        )
    elif figure == "corruptions":
        drawn = (
            f"One row per MNIST-C corruption, the medians over seeds; {states_drawn(r)}; rows "
            "sorted by median change, printed at the right. References in grey on each row: "
            "each state's random projection at its own code sizes (tick, median over seeds and "
            "projection seeds) and raw pixels (diamond); the line is chance (0.5)."
            if len(r.states) == 2
            else f"One row per MNIST-C corruption, sorted by the model's median: small dots "
            f"are the training seeds, the line spans them, the large dot is their median "
            f"({states_drawn(r)}). References in grey on each row: the random projection at the "
            "same code sizes (median over seeds and projection seeds) and raw pixels (diamond); "
            "the line is chance (0.5)."
        )
    else:
        drawn = (
            "MNIST-C mean (left) and worst (right). Each line spans the training seeds (and, "
            "for a random projection, its projection seeds); the dot is the median. References "
            "are grey, states blue. A random projection is matched to one state's code sizes, "
            "so each state has its own; the identity tick is the clean codes against themselves."
        )
    return f"{drawn} {setup(r)} {limit(r)}"


# --- drawing ------------------------------------------------------------------------------


def wrap_lines(text: str, width: int) -> str:
    """`text` wrapped to `width` characters line by line: each of its lines starts a new one,
    and the continuation of a long line is indented, so each line reads as its own block."""
    return "\n".join(
        textwrap.fill(line, width=width, subsequent_indent="    ") for line in text.splitlines()
    )


def frame(width: float, body: float, title: str, sub: str, cap: str) -> tuple[Figure, SubFigure]:
    """A figure of `body` inches of plots, its title and subtitle above and its caption below,
    each space sized to its wrapped lines. A line of the title or the subtitle (one per
    template or state) starts a new line."""
    wrapped = textwrap.fill(cap, width=int(width * 15))
    foot = 0.2 * len(wrapped.splitlines()) + 0.25
    heading = wrap_lines(title, int(width * 8.5))
    below = wrap_lines(sub, int(width * 15))
    # inches, as rendered: margin, title lines (12 pt bold at 1.25), gap, subtitle lines (9 pt
    # at 1.3), margin
    sub_h = 0.19 * len(below.splitlines()) + 0.1
    top = 0.1 + 0.25 * len(heading.splitlines()) + 0.1 + sub_h
    fig = plt.figure(figsize=(width, top + body + foot), layout="constrained")
    head, main, tail = stack(fig, [top, body, foot])
    head.text(
        0.0,
        1 - 0.1 / top,
        heading,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=INK,
        linespacing=1.25,
    )
    head.text(0.0, sub_h / top, below, ha="left", va="top", fontsize=9, color=INK2, linespacing=1.3)
    tail.text(0.0, 0.97, wrapped, ha="left", va="top", fontsize=9, color=INK2, linespacing=1.3)
    return fig, main


def stack(fig: Figure | SubFigure, ratios: Sequence[float], space: float = 0.0) -> list[SubFigure]:
    """Subfigures stacked top to bottom with the given height ratios."""
    subs = fig.subfigures(len(ratios), 1, height_ratios=list(ratios), hspace=space, squeeze=False)
    return [cast(SubFigure, s) for s in subs[:, 0]]


def span(lo: float, hi: float) -> tuple[float, float]:
    """A y-range around [lo, hi], never narrower than MIN_SPAN."""
    mid, half = (lo + hi) / 2, max(hi - lo, MIN_SPAN) / 2 * 1.15
    return mid - half, mid + half


def panel_values(r: rp.Report, key: str) -> list[float]:
    out = [v for x in r.states for v in values(x, key)]
    if judged(r, key):
        out += [v for x in r.states for v in values(x, CHANCE[key])]
    return out


def ylim(r: rp.Report, key: str) -> tuple[float, float]:
    group = SHARED_AUC if key in SHARED_AUC else (key,)
    vs = [v for k in group for v in panel_values(r, k)]
    if key == "code_size/clean":  # whole units, so that its ticks are whole too
        lo, hi = min(vs), max(vs)
        pad = max(hi - lo, 1.0) * 0.2
        return max(math.floor(lo - pad), 0.0), math.ceil(hi + pad)
    lo, hi = span(min(vs), max(vs))
    if key != "kendall_tau_a" and lo < 0:  # every other measure is non-negative
        lo, hi = 0.0, hi - lo
    if key != "kendall_tau_a" and lo == 0.0:  # keep points at 0 off the axis line
        lo = -0.06 * (hi - lo)
    return lo, hi


def spread(ys: Sequence[float], gap: float) -> list[float]:
    """Positions for labels wanted at `ys`, each at least `gap` from its neighbours and in the
    order of `ys` (ties in the order given). Labels that would collide are pushed apart as a
    block centred on the mean of their wanted positions; a block that then reaches its
    neighbour takes it in. Labels with room stay where they are."""
    order = sorted(range(len(ys)), key=lambda i: (ys[i], i))

    def bottom(block: list[int]) -> float:
        return statistics.mean(ys[i] for i in block) - gap * (len(block) - 1) / 2

    blocks: list[list[int]] = []
    for i in order:
        blocks.append([i])
        while len(blocks) > 1 and bottom(blocks[-1]) - bottom(blocks[-2]) < gap * len(blocks[-2]):
            blocks[-2:] = [blocks[-2] + blocks[-1]]
    out = [0.0] * len(ys)
    for block in blocks:
        for k, i in enumerate(block):
            out[i] = bottom(block) + k * gap
    return out


def separate(fig: Figure, labels: Sequence[Annotation]) -> None:
    """Move each axes' end-of-line labels apart vertically (`spread`, END_GAP points). A label
    is an annotation offset in points from its line's end; only that offset's height changes.
    The figure is laid out first, since the gap is in points and the axes' scale is final only
    then."""
    fig.draw_without_rendering()
    per_points = 72 / fig.dpi
    for ax in {id(a.axes): a.axes for a in labels if a.axes is not None}.values():
        group = [a for a in labels if a.axes is ax]
        ends = [ax.transData.transform(a.xy)[1] * per_points for a in group]
        for a, end, y in zip(group, ends, spread(ends, END_GAP), strict=True):
            a.xyann = (a.xyann[0], y - end)


def fmt(key: str, v: float) -> str:
    return f"{v:.1f}" if key == "code_size/clean" else f"{v:.2f}"


def slope(ax: Axes, r: rp.Report, key: str) -> list[Annotation]:
    """A slope chart of `key`; return its labels at the right end, to be kept apart."""
    a, b = r.states
    for s in seeds(r):
        ax.plot(
            [0, 1],
            [a.measures[s][key], b.measures[s][key]],
            color=STATE[1],
            alpha=SEED_ALPHA,
            lw=0.9,
            solid_capstyle="round",
        )
    ma, mb = median(a, key), median(b, key)
    ax.plot([0, 1], [ma, mb], color=STATE[1], lw=2.6, solid_capstyle="round", zorder=3)
    ax.scatter([0, 1], [ma, mb], color=[STATE[0], STATE[1]], s=34, zorder=4, edgecolors="white")
    ax.annotate(
        fmt(key, ma),
        (0, ma),
        xytext=(-7, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=INK,
    )
    ends = [
        ax.annotate(
            fmt(key, mb),
            (1, mb),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
        )
    ]
    if judged(r, key):
        ca, cb = median(a, CHANCE[key]), median(b, CHANCE[key])
        ax.plot([0, 1], [ca, cb], color=REF, lw=1.6, zorder=2)
        ends.insert(  # first: a tie puts it below the median, as when it is lower
            0,
            ax.annotate(
                "chance",
                (1, cb),
                xytext=(7, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=SMALL,
                color=REF_DARK,
            ),
        )
    ax.set_xticks([0, 1], [a.label, b.label])
    ax.set_xlim(-0.55, 1.55)
    return ends


def strip(ax: Axes, r: rp.Report, key: str) -> None:
    x = r.states[0]
    vs = values(x, key)
    n = len(vs)
    offsets = [(i - (n - 1) / 2) * (0.12 / max(n - 1, 1)) for i in range(n)]
    ax.scatter(
        offsets, vs, color=STATE[0], alpha=0.9, s=16, zorder=3, edgecolors=STATE[1], linewidths=0.6
    )
    m = median(x, key)
    ax.plot([-0.2, 0.2], [m, m], color=STATE[1], lw=2.6, zorder=4, solid_capstyle="round")
    ax.annotate(
        fmt(key, m),
        (0.2, m),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        color=INK,
    )
    if judged(r, key):
        c = median(x, CHANCE[key])
        ax.plot([-0.5, 0.5], [c, c], color=REF, lw=1.6, zorder=2)
        ax.annotate(
            "chance",
            (-0.5, c),
            xytext=(0, 4 if c >= m else -4),
            textcoords="offset points",
            ha="left",
            va="bottom" if c >= m else "top",
            fontsize=SMALL,
            color=REF_DARK,
        )
    ax.set_xticks([0], [x.label])
    ax.set_xlim(-0.6, 0.8)


def panel_heading(ax: Axes, title: str, reading: str, lines: int) -> None:
    """A panel's title over its reading in small type, then the axes. `lines` is the room
    left for the reading: the most any panel of the row needs, so that a row's titles align."""
    ax.annotate(
        textwrap.fill(reading, READING_WIDTH),
        (0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=SMALL,
        color=INK2,
    )
    ax.set_title(title, fontsize=10, pad=8 + lines * LINE)


def reading_lines(readings: Sequence[str]) -> int:
    """The room for the longest of `readings`, and never less than two lines."""
    return max(2, *(len(textwrap.wrap(x, READING_WIDTH)) for x in readings))


def measure_reading(r: rp.Report, key: str) -> str:
    if key not in CHANCE or has_chance(r):
        return MEASURES[key].reading
    return "no chance level in this report (converted from /1): read with code size"


def measure_panels(
    r: rp.Report, sub: SubFigure, group: str, keys: Sequence[str]
) -> list[Annotation]:
    """One row of measure panels; return their end-of-line labels, to be kept apart."""
    axes = sub.subplots(1, len(keys), squeeze=False)[0]
    ends: list[Annotation] = []
    sub.suptitle(group, x=0.0, ha="left", fontsize=9, color=REF_DARK, fontweight="normal")
    lines = reading_lines([measure_reading(r, k) for k in keys])
    for ax, key in zip(axes, keys, strict=True):
        if len(r.states) == 2:
            ends += slope(ax, r, key)
        else:
            strip(ax, r, key)
        m = MEASURES[key]
        # name over finding, then the reading in small type, then the axes
        title = panel_title(r, key).replace(": ", "\n", 1)
        panel_heading(ax, title, measure_reading(r, key), lines)
        ax.set_ylabel(m.axis)
        ax.set_ylim(*ylim(r, key))
        size = key == "code_size/clean"  # whole units: a tick between two would repeat a label
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10], integer=size))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}" if size else "{x:.2f}"))
        ax.tick_params(axis="x", length=0)
    return ends


def plot_measures(r: rp.Report, out: Path) -> Path:
    keys = [k for k in GEOMETRY if k in r.states[0].measures[seeds(r)[0]]]
    fig, body = frame(10.0, 6.8, measures_title(r), subtitle(r), caption(r, "measures"))
    top, bottom = stack(body, [1.0, 1.0], space=0.06)
    ends = measure_panels(r, top, "Geometry and robustness", keys)
    ends += measure_panels(r, bottom, "Population usage", USAGE)
    separate(fig, ends)
    path = out / "measures.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_corruptions(r: rp.Report, out: Path) -> Path:
    """Each corruption on one row, with its references on the same row: chance, raw pixels and
    each state's random projection at its own code sizes."""
    two = len(r.states) == 2
    fig, body = frame(7.6, 6.2, corruptions_title(r), subtitle(r), caption(r, "corruptions"))
    ax = body.subplots()
    a, b = r.states[0], r.states[-1]
    change = {c: median(b, f"model/{c}") - median(a, f"model/{c}") for c in MNIST_C}
    order = sorted(
        MNIST_C, key=lambda c: change[c] if two else median(a, f"model/{c}"), reverse=True
    )
    for y, c in enumerate(order):
        key = f"model/{c}"
        ax.scatter(
            [r.pixel[f"pixel/{c}"]],
            [y],
            marker="D",
            facecolors="none",
            edgecolors=REF_DARK,
            s=26,
            zorder=2,
            linewidths=0.9,
        )
        for i, x in enumerate(r.states):
            shade = REF if i == len(r.states) - 1 else REF_LIGHT
            ax.scatter(
                [statistics.median(random_values(r, x, c))],
                [y],
                marker="|",
                color=shade,
                s=90,
                linewidths=1.8,
                zorder=2,
            )
        if two:
            va, vb = median(a, key), median(b, key)
            ax.plot([va, vb], [y, y], color=STATE[1], alpha=SEED_ALPHA, lw=2.2, zorder=3)
            ax.scatter([va], [y], color=STATE[0], s=38, zorder=4, edgecolors="white")
            ax.scatter([vb], [y], color=STATE[1], s=38, zorder=5, edgecolors="white")
            d = change[c]
            ax.text(
                1.02,
                y,
                f"{d:+.3f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=9,
                color=DOWN if d < 0 else UP if d > 0 else INK2,
            )
        else:
            vs = values(a, key)
            ax.plot(
                [min(vs), max(vs)], [y, y], color=STATE[0], lw=2.2, zorder=3, solid_capstyle="round"
            )
            ax.scatter(
                vs, [y] * len(vs), color=STATE[1], alpha=SEED_ALPHA, s=14, zorder=4, linewidths=0
            )
            ax.scatter([median(a, key)], [y], color=STATE[1], s=38, zorder=5, edgecolors="white")
    ax.axvline(0.5, color=REF, lw=1.0, zorder=1)
    ax.text(0.5, -0.75, " chance", ha="left", va="center", fontsize=SMALL, color=REF_DARK)
    if two:
        ax.text(
            1.02,
            -0.75,
            "median change",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=SMALL,
            color=INK2,
        )
    handles = [
        *(
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                color=STATE[i] if two else STATE[1],
                label=f"{x.label}" if two else f"{x.label}: seeds, median",
            )
            for i, x in enumerate(r.states)
        ),
        *(
            Line2D(
                [],
                [],
                ls="",
                marker="|",
                ms=10,
                mew=1.8,
                color=REF if i == len(r.states) - 1 else REF_LIGHT,
                label=f"random, {x.label}" if two else "random projection",
            )
            for i, x in enumerate(r.states)
        ),
        Line2D([], [], ls="", marker="D", mfc="none", color=REF_DARK, label="raw pixels"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=len(handles),
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.4,
    )
    ax.set_yticks(range(len(order)), [label(c) for c in order])
    ax.set_ylim(len(order) - 0.4, -1.2)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("AUC, median over seeds (matched vs mismatched cosine)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.tick_params(axis="y", length=0)
    path = out / "corruptions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def control_rows(r: rp.Report, summary: str) -> list[tuple[str, list[float], str] | None]:
    """The rows of one controls panel for `summary` ("mnistc_mean" or "mnistc_worst"):
    references, a gap (None), then the states."""
    rows: list[tuple[str, list[float], str] | None] = [
        ("raw pixels", [r.pixel[f"pixel/{summary}"]], REF_DARK)
    ]
    rows += [(f"random projection, {x.label}", random_values(r, x, summary), REF) for x in r.states]
    if "zeros" in r.protocol.conditions:
        rows += [(f"zero input, {x.label}", values(x, "model/zeros"), REF) for x in r.states]
    rows.append(None)
    one = len(r.states) == 1  # a single state is drawn in the dark blue, as in every figure
    rows += [
        (x.label, values(x, f"model/{summary}"), STATE[1] if one else STATE[i])
        for i, x in enumerate(r.states)
    ]
    return rows


def controls_panel(ax: Axes, r: rp.Report, summary: str, first: bool, xlo: float) -> None:
    rows = control_rows(r, summary)
    for y, row in enumerate(rows):
        if row is None:
            continue
        _, vs, colour = row
        ax.plot([min(vs), max(vs)], [y, y], color=colour, lw=2.2, solid_capstyle="round")
        ax.scatter([statistics.median(vs)], [y], color=colour, s=34, zorder=3, edgecolors="white")
    ax.set_yticks(range(len(rows)), [row[0] if row else "" for row in rows])
    ax.tick_params(axis="y", labelleft=first)
    gap = rows.index(None)
    if first:
        ax.text(
            -0.02,
            -0.75,
            "references",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=SMALL,
            color=REF_DARK,
        )
        ax.text(
            -0.02,
            gap,
            "models under test",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=SMALL,
            color=REF_DARK,
        )
        ident = statistics.median(v for x in r.states for v in values(x, "model/clean"))
        ax.plot([ident, ident], [-1.15, -0.85], color=REF_DARK, lw=1.0, clip_on=False)
        ax.annotate(
            f"identity {ident:.2f}",
            (ident, -1.15),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=SMALL,
            color=REF_DARK,
        )
    ax.axvline(0.5, color=REF_LIGHT, lw=1.0, zorder=1)
    ax.set_ylim(len(rows) - 0.4, -1.3)
    ax.set_xlim(xlo, 1.02)  # both panels are AUC: one range
    ax.set_title("MNIST-C mean" if summary == "mnistc_mean" else "MNIST-C worst", fontsize=10)
    ax.set_xlabel("AUC, mean of 15" if summary == "mnistc_mean" else "AUC, worst of 15")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="y", length=0)


def plot_controls(r: rp.Report, out: Path) -> Path:
    n = len(control_rows(r, "mnistc_mean"))
    height = 1.4 + 0.3 * n
    fig, body = frame(10.0, height, controls_title(r), subtitle(r), caption(r, "controls"))
    left, right = body.subplots(1, 2, sharey=True, width_ratios=[1, 1])
    lowest = min(
        v
        for summary in ("mnistc_mean", "mnistc_worst")
        for row in control_rows(r, summary)
        if row is not None
        for v in row[1]
    )
    xlo = min(0.45, lowest - 0.02)
    controls_panel(left, r, "mnistc_mean", True, xlo)
    controls_panel(right, r, "mnistc_worst", False, xlo)
    path = out / "controls.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def markdown(r: rp.Report, figures: Sequence[tuple[Path, str, str]]) -> str:
    lines = [
        "# Figures: " + " vs ".join(x.label for x in r.states),
        "",
        " · ".join(provenance(x.label, x) for x in r.states),
        "",
        "| Measure | Question | Statistic | Reading |",
        "|---|---|---|---|",
        *(f"| {m.name} | {m.question} | {m.statistic} | {m.reading} |" for m in MEASURES.values()),
    ]
    for path, title, cap in figures:
        flat = "; ".join(title.splitlines())  # a heading is one line
        lines += ["", f"## {flat}", "", f"![{flat}]({path.name})", "", cap]
    return "\n".join([*lines, ""])


def draw(r: rp.Report, out: Path) -> list[Path]:
    """Every figure of `r` and `figures.md`, written into `out`; return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        figures = [(plot_measures(r, out), measures_title(r), caption(r, "measures"))]
        if has_mnistc(r):
            figures.append(
                (plot_corruptions(r, out), corruptions_title(r), caption(r, "corruptions"))
            )
            figures.append((plot_controls(r, out), controls_title(r), caption(r, "controls")))
    md = out / "figures.md"
    md.write_text(markdown(r, figures))
    return [*(p for p, _, _ in figures), md]


# --- curves over training length ----------------------------------------------------------

CURVE_KEYS = (
    "kendall_tau_a",
    "model/mnistc_mean",
    "model/mnistc_worst",
    "code_size/clean",
    "separation",
    "domination",
)
BASE_TEMPLATE = "mnist_ngm25_frozen_eval"


def template_label(template: str) -> str:
    """A short name for a template file: `pixel` for the base template, its suffix for a
    variant of it (`retina`, `retina divnorm`), else the file's stem."""
    stem = Path(template).stem
    if stem == BASE_TEMPLATE:
        return "pixel"
    if stem.startswith(BASE_TEMPLATE + "_"):
        return stem.removeprefix(BASE_TEMPLATE + "_").replace("_", " ")
    return stem


class Curve(msgspec.Struct, frozen=True):
    """One group's reports (a template's, or a sweep variant's), ordered by training length."""

    label: str
    reports: list[rp.Report]

    @property
    def steps(self) -> list[int]:
        return [r.protocol.train_steps for r in self.reports]


GROUPINGS = ("template", "variant")


def curves(reports: Sequence[rp.Report], by: str = "template") -> list[Curve]:
    """The reports grouped `by` template (the state's `template`) or by variant (the state's
    label: a sweep's points of one variant, which all share one template), each group ordered
    by training length; raise unless every report has one state and they all share a protocol
    apart from the training length, and each group has one template, one build, one seed set,
    one length per report and either chance levels in every report or in none."""
    if by not in GROUPINGS:
        raise ValueError(f"group by {by!r}: one of {GROUPINGS}")
    if not reports:
        raise ValueError("no reports")
    base = msgspec.structs.replace(reports[0].protocol, train_steps=0)
    groups: dict[str, list[rp.Report]] = {}
    for r in reports:
        if len(r.states) != 1:
            raise ValueError(f"a curve point is a one-state report, got {len(r.states)} states")
        if msgspec.structs.replace(r.protocol, train_steps=0) != base:
            raise ValueError("the reports differ in their protocol beyond the training length")
        x = r.states[0]
        groups.setdefault(x.template if by == "template" else x.label, []).append(r)
    if len(groups) > len(STATE):
        raise ValueError(f"{len(groups)} {by}s; a figure tells at most {len(STATE)} apart")
    out: list[Curve] = []
    for key, rs in groups.items():
        rs.sort(key=lambda r: r.protocol.train_steps)
        steps = [r.protocol.train_steps for r in rs]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{key}: two reports at the same training length")
        if len({r.states[0].template for r in rs}) != 1:
            raise ValueError(f"{key}: the reports run different templates")
        if len({(r.states[0].git_sha, r.states[0].binary_sha256) for r in rs}) != 1:
            raise ValueError(f"{key}: the reports come from different builds")
        if len({tuple(seeds(r)) for r in rs}) != 1:
            raise ValueError(f"{key}: the reports have different training seeds")
        if len({has_chance(r) for r in rs}) != 1:
            raise ValueError(f"{key}: some reports carry chance levels and others do not")
        out.append(Curve(template_label(key) if by == "template" else key, rs))
    if len({c.label for c in out}) != len(out):
        raise ValueError("two templates have the same short name")
    return out


def steps_text(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def labelled(steps: Sequence[int], gap: float = 0.15) -> list[int]:
    """The steps that get a tick label: from the longest down, each at least `gap` of the
    axis from the last one labelled, so labels never collide; the rest keep a bare tick."""
    lo, hi = min(steps), max(steps)
    out: list[int] = []
    for n in sorted(steps, reverse=True):
        if not out or (out[-1] - n) >= gap * max(hi - lo, 1):
            out.append(n)
    return sorted(out)


def curve_medians(c: Curve, key: str) -> list[float]:
    return [median(r.states[0], key) for r in c.reports]


def curves_title(cs: Sequence[Curve]) -> str:
    """Per template: τ-a and MNIST-C mean at the first and the last length, and their change
    over the last interval, the part that says whether the curve still rises."""
    parts: list[str] = []
    for c in cs:
        s = c.steps
        keys = [k for k in ("kendall_tau_a", "model/mnistc_mean") if k in seeds_measures(c)]
        moves = []
        for k in keys:
            m = curve_medians(c, k)
            moves.append(f"{MEASURES[k].name} {m[0]:.2f} to {m[-1]:.2f}")
        head = f"{c.label}, {steps_text(s[0])} to {steps_text(s[-1])}: " + ", ".join(moves)
        if len(s) > 1:
            last = [
                f"{MEASURES[k].name} {curve_medians(c, k)[-1] - curve_medians(c, k)[-2]:+.3f}"
                for k in keys
            ]
            head += f"; over the last {steps_text(s[-1] - s[-2])}: " + ", ".join(last)
        parts.append(head)
    return "\n".join(parts)  # one line per template


def seeds_measures(c: Curve) -> rp.Measures:
    r = c.reports[0]
    return r.states[0].measures[seeds(r)[0]]


def curves_caption(cs: Sequence[Curve]) -> str:
    r = cs[0].reports[0]
    p = r.protocol
    colours = ", ".join(
        f"{c.label} {name}" for c, (_, name) in zip(cs, curve_colours(cs), strict=True)
    )
    lengths = "; ".join(f"{c.label}: " + ", ".join(steps_text(n) for n in c.steps) for c in cs)
    no_chance = [c.label for c in cs if not has_chance(c.reports[0])]
    chance = (
        "Separation and domination are drawn with their chance levels (grey), the same codes "
        "with each presentation's units shuffled."
        if not no_chance
        else f"{', '.join(no_chance)}: no chance levels (converted from schema /1), so "
        "separation and domination are drawn without one; read them with code size."
    )
    leads = ", ".join(f"{c.reports[0].states[0].lead} ({c.label})" for c in cs)
    n_seeds = ", ".join(f"{len(seeds(c.reports[0]))} ({c.label})" for c in cs)
    return (
        "Each panel is one measure over training steps: thin lines are the training seeds, "
        f"the thick line with dots their median; {colours}. Grey: the random projection at the "
        "same code sizes (median over seeds and projection seeds) and raw pixels (horizontal). "
        f"{chance} Each point is a separate frozen evaluation of training at that length "
        f"({lengths}); training at {p.train_steps_per_image} step per image over "
        f"{p.train_images} images, training seeds {n_seeds}; {len(p.test_indices)} test "
        f"images, {p.window} steps each, settle lead {leads}, winners at ≥ {p.frac} of the "
        "peak. The curves show where the medians move; no thresholds, no significance, and a "
        "flat stretch is not shown to be a plateau."
    )


COLOUR_NAMES = ("in light blue", "in blue", "in dark blue", "in orange")


def curve_colours(cs: Sequence[Curve]) -> list[tuple[str, str]]:
    """(colour, its name) per template: the full blue alone, else the fixed state order."""
    if len(cs) == 1:
        return [(STATE[1], "in blue")]
    return [(STATE[i], COLOUR_NAMES[i]) for i in range(len(cs))]


def curve_reading(cs: Sequence[Curve], key: str) -> str:
    if key not in CHANCE:
        return MEASURES[key].reading
    if all(has_chance(c.reports[0]) for c in cs):
        return "read against chance (grey)"
    return "no chance level (converted from /1): read with code size"


def curve_panel(
    ax: Axes, cs: Sequence[Curve], key: str, pixel: rp.Measures, lines: int
) -> list[Annotation]:
    """One measure's curves; return their end-of-line labels, to be kept apart."""
    summary = key.removeprefix("model/")
    ends: list[Annotation] = []
    for c, (colour, _) in zip(cs, curve_colours(cs), strict=True):
        s = c.steps
        for seed in seeds(c.reports[0]):
            ax.plot(
                s,
                [r.states[0].measures[seed][key] for r in c.reports],
                color=colour,
                alpha=SEED_ALPHA,
                lw=0.9,
            )
        ax.plot(s, curve_medians(c, key), color=colour, lw=2.4, marker="o", ms=4, zorder=3)
        ref: list[float] | None = None
        name = ""
        if key in SHARED_AUC:
            ref = [statistics.median(random_values(r, r.states[0], summary)) for r in c.reports]
            name = "random"
        elif key in CHANCE and has_chance(c.reports[0]):
            ref, name = curve_medians(c, CHANCE[key]), "chance"
        if ref is not None:
            ax.plot(s, ref, color=REF, lw=1.4, zorder=2)
            tag = name if len(cs) == 1 else f"{name}, {c.label}"
            ends.append(
                ax.annotate(
                    tag,
                    (s[-1], ref[-1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=SMALL,
                    color=REF_DARK,
                )
            )
    if key in SHARED_AUC and f"pixel/{summary}" in pixel:
        v = pixel[f"pixel/{summary}"]
        ax.axhline(v, color=REF_DARK, lw=0.9, zorder=1)
        ax.annotate(
            "raw pixels",
            (0.0, v),
            xycoords=("axes fraction", "data"),
            xytext=(2, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=SMALL,
            color=REF_DARK,
        )
    m = MEASURES[key]
    panel_heading(ax, m.name, curve_reading(cs, key), lines)
    ax.set_ylabel(m.axis)
    every = sorted({n for c in cs for n in c.steps})
    lo, hi = min(every), max(every)
    ax.set_xticks(every, ["" for _ in every], minor=True)
    ax.set_xticks(labelled(every), [steps_text(n) for n in labelled(every)], fontsize=SMALL)
    ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.16 * (hi - lo))
    if key == "code_size/clean":
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10], integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.grid(axis="x", visible=False)
    return ends


def plot_curves(cs: Sequence[Curve], out: Path) -> Path:
    keys = [k for k in CURVE_KEYS if k in seeds_measures(cs[0])]
    title, cap = curves_title(cs), curves_caption(cs)
    sub = "\n".join(
        provenance(c.label, c.reports[0].states[0]) for c in cs
    )  # one line per template
    rows = (len(keys) + 2) // 3
    fig, body = frame(10.0, 3.2 * rows, title, sub, cap)
    axes = body.subplots(rows, 3, squeeze=False).ravel()
    pixel = cs[0].reports[0].pixel
    lines = reading_lines([curve_reading(cs, k) for k in keys])
    ends = [
        a
        for ax, key in zip(axes, keys, strict=False)
        for a in curve_panel(ax, cs, key, pixel, lines)
    ]
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    separate(fig, ends)
    path = out / "curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def draw_curves(reports: Sequence[rp.Report], out: Path, by: str = "template") -> list[Path]:
    """`curves.png` and `curves.md` for a length sweep, written into `out`."""
    cs = curves(reports, by)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        png = plot_curves(cs, out)
    md = out / "curves.md"
    md.write_text(curves_markdown(cs, png))
    return [png, md]


def curves_markdown(cs: Sequence[Curve], png: Path) -> str:
    """`curves.md`: the title's finding per template as a list under a heading naming the
    templates, the figure with the findings as its alt text, and the caption."""
    findings = curves_title(cs).splitlines()
    return "\n".join(
        [
            "# Curves over training length: " + ", ".join(c.label for c in cs),
            "",
            *(f"- {f}" for f in findings),
            "",
            f"![{'; '.join(findings)}]({png.name})",
            "",
            curves_caption(cs),
            "",
        ]
    )


# --- one measure over a sweep's grid ------------------------------------------------------

type Coord = tuple[Any, ...]


def load_sweep(path: Path) -> sr.SweepReport:
    try:
        return sr.decode(path.read_bytes())
    except ValueError as e:
        raise ValueError(f"{path}: {e} (write it with scripts/eval_sweep_report.py)") from e


def sweep_measure_name(key: str) -> str:
    """A sweep measure's short name."""
    if key == sr.RANDOM:
        return "random MNIST-C mean"
    return MEASURES[key].name if key in MEASURES else key


def axis_name(pointer: str) -> str:
    """A grid axis by its pointer's last segment, as the sweep names its variants."""
    return pointer.rsplit("/", 1)[-1]


def axis_value(a: sr.Axis, text: str) -> Any:
    """The value of `a` that `text` names: as a variant's name writes it (`0.0001`, `x0.01`)
    or as JSON of the same value (`sweep.same_value`: `1e-4`, `x1` for `x1.0`); raise if it
    names none."""
    for v in a.values:
        if text == sr.axis_text(a, v):
            return v
    try:
        given = json.loads(text.removeprefix("x") if a.ratio_of else text)
    except json.JSONDecodeError:
        given = None
    for v in a.values:
        if given is not None and sweep.same_value(given, v):
            return v
    values = ", ".join(sr.axis_text(a, v) for v in a.values)
    raise ValueError(f"{axis_name(a.pointer)}={text}: not one of its values ({values})")


def parse_fix(r: sr.SweepReport, items: Sequence[str]) -> dict[str, Any]:
    """`--fix AXIS=VALUE` items as {pointer: axis value}; AXIS is the pointer or its last key."""
    fix: dict[str, Any] = {}
    for item in items:
        k, sep, text = item.partition("=")
        hits = [a for a in r.axes if k in (a.pointer, axis_name(a.pointer))]
        if not sep or len(hits) != 1:
            names = [axis_name(a.pointer) for a in r.axes]
            raise ValueError(f"--fix {item!r}: give AXIS=VALUE with AXIS one of {names}")
        if hits[0].pointer in fix:
            raise ValueError(f"--fix {item!r}: {k} is fixed twice")
        fix[hits[0].pointer] = axis_value(hits[0], text)
    return fix


def fix_text(r: sr.SweepReport, fix: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{axis_name(a.pointer)}={sr.axis_text(a, fix[a.pointer])}"
        for a in r.axes
        if a.pointer in fix
    )


def grid(
    r: sr.SweepReport, fix: Mapping[str, Any] | None = None
) -> tuple[list[sr.Axis], dict[Coord, str], list[str]]:
    """(the axes left after `fix`, the variant at each cell by its coordinates on them in axis
    order, the variants off the grid). The first axis left runs down the rows, the second
    across the columns, a third over the panels; a derived axis' coordinate is its factor.

    A variant is on the grid if it overrides exactly the axes, each to one of the axis'
    values; an alias (a grid name whose values are all the template's) places its target,
    `base`, at its point. Named variants, and `base` unless an alias places it, are off the
    grid. `fix` ({pointer: value}) keeps the cells with that value on that axis and drops
    the axis. Raise unless two or three axes are left and no two variants share a cell."""
    fix = fix or {}
    left = [a for a in r.axes if a.pointer not in fix]
    if len(left) not in (2, 3):
        fixed = f", {len(fix)} of them fixed" if fix else ""
        more = "; fix more with --fix AXIS=VALUE" if len(left) > 3 else ""
        raise ValueError(
            f"a heatmap needs two or three grid axes, the sweep has {len(r.axes)}{fixed}{more}"
        )
    pointers = {a.pointer for a in r.axes}
    placed: dict[Coord, str] = {}
    for name, over in r.variants.items():
        coord = tuple(sr.coordinate(a, over) for a in r.axes)
        if over.keys() == pointers and None not in coord:
            if coord in placed:
                raise ValueError(f"variants {placed[coord]} and {name} share a grid cell")
            placed[coord] = name
    for alias, target in r.aliases.items():
        coord = sr.name_coordinates(r.axes, alias)
        if coord is None:
            raise ValueError(f"alias {alias}: not the name of a grid point")
        if coord in placed:
            raise ValueError(f"alias {alias} and variant {placed[coord]} share a grid cell")
        placed[coord] = target
    off = [name for name in r.variants if name not in placed.values()]
    keep = [i for i, a in enumerate(r.axes) if a.pointer not in fix]
    cells = {
        tuple(coord[i] for i in keep): name
        for coord, name in placed.items()
        if all(coord[i] == fix[a.pointer] for i, a in enumerate(r.axes) if a.pointer in fix)
    }
    return left, cells, off


def axis_label(a: sr.Axis) -> str:
    return axis_name(a.pointer) + (f" (times {axis_name(a.ratio_of)})" if a.ratio_of else "")


def heatmap_title(
    r: sr.SweepReport, key: str, cells: dict[Coord, str], fix: Mapping[str, Any] | None = None
) -> str:
    """Per length: the highest grid point, base's value, and how many grid points are done."""
    at = f" at {fix_text(r, fix)}" if fix else ""
    lines = [f"{sweep_measure_name(key)} over the grid{at}"]
    for x in r.lengths:
        by = {w.variant: w for w in x.rows}
        on = sorted((v for v in cells.values() if v in by), key=lambda v: (-by[v].medians[key], v))
        if not on:
            lines.append(f"{sr.steps_text(x.length)}: no grid point is done")
            continue
        text = f"{sr.steps_text(x.length)}: highest {on[0]} ({by[on[0]].medians[key]:.3f})"
        if r.base in by:
            text += f"; {r.base} {by[r.base].medians[key]:.3f}"
        lines.append(text + f"; {len(on)} of {len(cells)} grid points done")
    return "\n".join(lines)  # one line per length


def heatmap_caption(
    r: sr.SweepReport, key: str, axes: Sequence[sr.Axis], off: Sequence[str]
) -> str:
    panels = f"; panels: {axis_label(axes[2])}" if len(axes) == 3 else ""
    aliased = [a for a, t in r.aliases.items() if t == r.base]
    at = (
        f" The cell of {', '.join(aliased)} is {r.base}: every value there is the template's."
        if aliased
        else ""
    )
    return (
        f"Each cell is one variant's {sweep_measure_name(key)}, the median over training "
        f"seeds {r.seeds}; rows: {axis_label(axes[0])}, columns: {axis_label(axes[1])}{panels}; "
        "one row of panels per training length. One colour scale for all panels. Grey: the "
        "point is not done. g: a collapse guard fired (code size far from "
        f"{r.base}'s, or never-wins up in every seed).{at} Off the grid, not drawn: "
        + (", ".join(off) if off else "none")
        + ". Medians only: whether every seed moved the same way is in sweep_report.md."
    )


def heatmap_path(r: sr.SweepReport, key: str, fix: Mapping[str, Any], out: Path) -> Path:
    tail = "".join(f"_{part}" for part in fix_text(r, fix).split(", ") if part)
    return out / f"heatmap_{key.replace('/', '_')}{tail}.png"


def plot_heatmap(
    r: sr.SweepReport, key: str, out: Path, fix: Mapping[str, Any] | None = None
) -> Path:
    if key not in r.measures:
        raise ValueError(f"{key!r} is not a measure of this sweep report: {r.measures}")
    fix = fix or {}
    axes, cells, off = grid(r, fix)
    ys, xs = axes[0].values, axes[1].values
    panels: list[Any] = axes[2].values if len(axes) == 3 else [None]
    rows = [{w.variant: w for w in x.rows} for x in r.lengths]
    done = [by[v].medians[key] for by in rows for v in cells.values() if v in by]
    if not done:
        raise ValueError("no grid point is done")
    lo, hi = min(done), max(done)
    cmap = plt.get_cmap("Blues").with_extremes(bad=GRID)
    panel_w, panel_h = 0.75 * len(xs) + 1.0, 0.42 * len(ys) + 0.9
    fig, body = frame(
        max(6.0, panel_w * len(panels) + 1.2),
        panel_h * len(rows),
        heatmap_title(r, key, cells, fix),
        f"git {r.ref[:8]}, template SHA256 {r.template_sha256[:12]}",
        heatmap_caption(r, key, axes, off),
    )
    grid_axes = body.subplots(len(rows), len(panels), squeeze=False)
    image = None
    for i, by in enumerate(rows):
        for j, pv in enumerate(panels):
            ax: Axes = grid_axes[i, j]
            m = [[math.nan] * len(xs) for _ in ys]
            for a, y in enumerate(ys):
                for b, x in enumerate(xs):
                    name = cells.get((y, x, pv) if len(axes) == 3 else (y, x))
                    if name is None or name not in by:
                        continue
                    w = by[name]
                    v = m[a][b] = w.medians[key]
                    light = hi == lo or (v - lo) / (hi - lo) < 0.6
                    text = f"{v:.1f}" if key == sr.CODE_SIZE else f"{v:.3f}"
                    ax.text(b, a, text + (" g" if w.guards else ""), ha="center", va="center",
                            fontsize=SMALL, color=INK if light else "white")  # fmt: skip
            image = ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            ax.set_xticks(range(len(xs)), [sr.axis_text(axes[1], v) for v in xs], fontsize=SMALL)
            ax.set_yticks(range(len(ys)), [sr.axis_text(axes[0], v) for v in ys], fontsize=SMALL)
            if i == len(rows) - 1:  # the axes' names once, on the outer panels
                ax.set_xlabel(axis_label(axes[1]))
            if j == 0:
                ax.set_ylabel(axis_label(axes[0]))
            ax.grid(visible=False)
            ax.tick_params(length=0)
            where = (
                f"{axis_name(axes[2].pointer)} {sr.axis_text(axes[2], pv)}, "
                if len(axes) == 3
                else ""
            )
            ax.set_title(f"{where}{sr.steps_text(r.lengths[i].length)}", fontsize=10)
    assert image is not None
    body.colorbar(image, ax=grid_axes.ravel().tolist(), label=sweep_measure_name(key))
    path = heatmap_path(r, key, fix, out)
    fig.savefig(path)
    plt.close(fig)
    return path


def draw_heatmap(
    r: sr.SweepReport, key: str, out: Path, fix: Mapping[str, Any] | None = None
) -> list[Path]:
    """The heatmap of one measure of a sweep and its `.md`, written into `out`."""
    fix = fix or {}
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        png = plot_heatmap(r, key, out, fix)
    axes, cells, off = grid(r, fix)
    findings = heatmap_title(r, key, cells, fix).splitlines()
    md = png.with_suffix(".md")
    md.write_text(
        "\n".join(
            [
                f"# {findings[0]}",
                "",
                *(f"- {f}" for f in findings[1:]),
                "",
                f"![{'; '.join(findings)}]({png.name})",
                "",
                heatmap_caption(r, key, axes, off),
                "",
            ]
        )
    )
    return [png, md]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="a report.json written by scripts/eval_ab.py; with --curves, the reports of a "
        "length sweep; with --heatmap, one sweep_report.json",
    )
    ap.add_argument(
        "--curves", action="store_true", help="draw the reports as curves over training length"
    )
    ap.add_argument(
        "--by",
        choices=GROUPINGS,
        help="with --curves: one curve per template (the default) or per sweep variant",
    )
    ap.add_argument(
        "--heatmap",
        metavar="MEASURE",
        help="draw MEASURE over a sweep's grid from the sweep_report.json given",
    )
    ap.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help="with --heatmap: only the grid variants with VALUE on AXIS (pointer or last key); "
        "repeatable",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="where the figures go")
    a = ap.parse_args(argv)
    if a.fix and not a.heatmap:
        ap.error("--fix needs --heatmap")
    if a.by and not a.curves:
        ap.error("--by needs --curves")
    if a.heatmap and a.curves:
        ap.error("--heatmap and --curves draw different inputs; give one")
    if a.curves:
        paths = draw_curves([load(p) for p in a.reports], a.out_dir, a.by or "template")
    elif a.heatmap:
        if len(a.reports) != 1:
            ap.error("--heatmap draws one sweep_report.json")
        r = load_sweep(a.reports[0])
        paths = draw_heatmap(r, a.heatmap, a.out_dir, parse_fix(r, a.fix))
    else:
        if len(a.reports) != 1:
            ap.error("give one report, or several with --curves")
        paths = draw(load(a.reports[0]), a.out_dir)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
