"""`coast_eval.sweep_report` on synthetic sweeps written as the driver writes them
(`sweep.json` plus one-state `report.json` files from `report.to_json`), in the shipped shape:
the 17 conditions (clean, the 15 MNIST-C corruptions, zeros) and 2 training seeds. Covered:
the medians, the paired deltas and the reading rule against base, the collapse guards, the
rankings, points that are not done (listed, never dropped), and the plots' grouping and grid."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import msgspec
import pytest

from coast_eval import report as rp
from coast_eval import sweep
from coast_eval import sweep_report as sr
from coast_eval.datasets import MNIST_C

SEEDS = (0, 1)
PROJ = (1, 2)
PROTOCOL = rp.Protocol(
    train_steps=0,
    train_steps_per_image=1,
    train_images=59_000,
    test_indices=list(range(20)),
    conditions=["clean", *MNIST_C, "zeros"],
    projection_seeds=list(PROJ),
    frac=0.3,
)
SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_sweep_report.py"
X = "/ngm25/x"
LR = "/ngm25/neuron_groups/L4-mnist/primary_learning_rate"
CLR = "/ngm25/neuron_groups/L4-mnist/common_learning_rate"
SLR = "/ngm25/neuron_groups/L4-mnist/secondary_learning_rate"
STRENGTH = "/ngm25/inhibitions/ab_pinned/strength"
# What the synthetic sweeps override: every pointer of the tests' axes and named variants, at
# a value on none of their grids, so no grid point is base unless a test moves it there.
TEMPLATE: dict[str, Any] = {
    "ngm25": {
        "x": 0,
        "neuron_groups": {
            "L4-mnist": {
                "primary_learning_rate": 0.05,
                "common_learning_rate": 0.05,
                "secondary_learning_rate": 0.05,
            }
        },
        "inhibitions": {"ab_pinned": {"strength": 7}},
    }
}


def measures(
    seed: int, tau: float = 0.3, auc: float = 0.6, size: float = 8.0, never: float = 0.1
) -> rp.Measures:
    """A seed's measures as `report.seed_measures` writes them for the 17 conditions. The
    worst corruption is the first, at `auc` + 0.01 seed; the mean is `auc` + 0.14 + 0.01 seed."""
    m: rp.Measures = {
        "kendall_tau_a": tau + 0.01 * seed,
        "separation": 0.4,
        "separation_chance": 0.35,
        "domination": 0.3,
        "domination_chance": 0.2,
        "never_wins": never + 0.01 * seed,
        "model/clean": 1.0,
        "model/zeros": 0.5,
        "code_size/clean": size + seed,
        "code_size/zeros": 50.0,
    }
    for i, c in enumerate(MNIST_C):
        m[f"model/{c}"] = auc + 0.02 * i + 0.01 * seed
        m[f"code_size/{c}"] = size
        for q in PROJ:
            m[f"random/{q}/{c}"] = 0.55 + 0.01 * i + 0.001 * q
    for prefix in ("model", *(f"random/{q}" for q in PROJ)):
        rp._summarise(m, prefix)  # pyright: ignore[reportPrivateUsage]
    return m


def point_report(variant: str, length: int, by_seed: dict[int, rp.Measures]) -> rp.Report:
    x = rp.State(
        variant, "a" * 40, None, 1, by_seed, {s: [] for s in by_seed},
        "mnist_ngm25_frozen_eval.json",
    )  # fmt: skip
    return rp.build(msgspec.structs.replace(PROTOCOL, train_steps=length), {}, [x])


def driven(
    axes: dict[str, Any],
    named: dict[str, dict[str, Any]],
    lengths: list[int],
    template: dict[str, Any] = TEMPLATE,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    """(the spec as submitted, the variants, the aliases) as the driver makes them: its own
    `sweep.load_spec` and `sweep.expand`, so a fixture cannot copy the reader's reading."""
    spec: dict[str, Any] = {
        "schema": sweep.SPEC_SCHEMA,
        "template": "t.json",
        "seeds": list(SEEDS),
        "lengths": lengths,
        "conditions": ["clean"],
        "tuning_images": 1,
        "axes": axes,
        "variants": named,
    }
    variants, aliases = sweep.expand(sweep.load_spec(json.dumps(spec).encode()), template)
    return spec, variants, aliases


def write_sweep(
    out: Path,
    points: dict[tuple[str, int], rp.Report | str],
    named: dict[str, dict[str, Any]] | None = None,
    axes: dict[str, Any] | None = None,
    template: dict[str, Any] = TEMPLATE,
    **raw: Any,
) -> Path:
    """`sweep.json` and the reports in `out`. The spec, variants and aliases are the driver's
    (`driven`): the grid of `axes`, then the `named` variants, by default one per variant of
    `points` other than base, each setting X. `raw` replaces fields of sweep.json, to write one
    the driver would not. A point given as a string is not done: its status, or `done` without
    a report file. A point's seeds are its report's, else SEEDS."""
    if named is None:
        others = dict.fromkeys(v for v, _ in points if v != "base")
        named = {v: {X: i} for i, v in enumerate(others, 1)}
    spec, variants, aliases = driven(axes or {}, named, sorted({n for _, n in points}), template)
    listed: list[dict[str, Any]] = []
    for (variant, length), r in points.items():
        rel = f"{variant}/{length}/report.json"
        status = r if isinstance(r, str) else "done"
        seeds = list(SEEDS) if isinstance(r, str) else sorted(r.states[0].measures)
        if not isinstance(r, str):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(rp.to_json(r))
        listed.append(
            {"variant": variant, "length": length, "report": rel, "seeds": seeds,
             "status": status}
        )  # fmt: skip
    doc: dict[str, Any] = {
        "schema": "coast-eval-sweep/1",
        "spec": spec,
        "template_sha256": "f" * 64,
        "ref": "b" * 40,
        "daemons": ["host:1"],
        "variants": variants,
        "points": listed,
        "aliases": aliases,
        **raw,
    }
    (out / "sweep.json").write_text(json.dumps(doc))
    return out


def seeds_of(**kw: float) -> dict[int, rp.Measures]:
    return {s: measures(s, **kw) for s in SEEDS}


def row(r: sr.SweepReport, length: int, variant: str) -> sr.Row:
    (x,) = [x for x in r.lengths if x.length == length]
    (out,) = [w for w in x.rows if w.variant == variant]
    return out


def test_the_table_is_the_median_over_seeds_of_the_reports_own_measures(tmp_path: Path) -> None:
    write_sweep(tmp_path, {("base", 500_000): point_report("base", 500_000, seeds_of())})
    r = sr.aggregate(tmp_path)
    assert r.seeds == list(SEEDS) and r.aliases == {}
    b = row(r, 500_000, "base")
    assert b.medians["kendall_tau_a"] == pytest.approx(0.305)  # two seeds: their mean
    assert b.medians["code_size/clean"] == pytest.approx(8.5)
    assert b.medians["model/mnistc_worst"] == pytest.approx(0.605)
    assert b.medians["model/mnistc_mean"] == pytest.approx(0.745)
    # random: the median over seeds and projection seeds of random/<q>/mnistc_mean
    assert b.medians["random/mnistc_mean"] == pytest.approx(0.62 + 0.0015)
    assert b.deltas == {} and b.directions == {} and b.guards == {}
    assert r.measures[:3] == ["kendall_tau_a", "model/mnistc_mean", "model/mnistc_worst"]
    assert all(f"model/{c}" in r.measures for c in MNIST_C)
    md = sr.to_markdown(r)
    assert (
        "| Variant | τ-a | Δ | MNIST-C worst | Δ | MNIST-C mean | Δ | random MNIST-C mean |" in md
    )


def test_the_delta_is_paired_by_seed_and_moved_only_if_every_seed_did(tmp_path: Path) -> None:
    mixed = seeds_of()
    mixed[0]["kendall_tau_a"] += 0.10
    mixed[1]["kendall_tau_a"] -= 0.02
    write_sweep(
        tmp_path,
        {
            ("base", 500_000): point_report("base", 500_000, seeds_of()),
            ("up", 500_000): point_report("up", 500_000, seeds_of(tau=0.35)),
            ("mixed", 500_000): point_report("mixed", 500_000, mixed),
        },
    )
    r = sr.aggregate(tmp_path)
    u = row(r, 500_000, "up")
    assert u.deltas["kendall_tau_a"] == pytest.approx(0.05)
    assert u.directions["kendall_tau_a"] == "all 2 increased"
    assert u.directions["model/mnistc_worst"] == "not all 2 moved in the same direction"
    m = row(r, 500_000, "mixed")
    assert m.deltas["kendall_tau_a"] == pytest.approx(0.04)
    assert m.directions["kendall_tau_a"] == "not all 2 moved in the same direction"
    table = sr.to_markdown(r)
    assert "| `up` | 0.3550 | +0.0500 ↑ |" in table
    assert "| `mixed` | 0.3450 | +0.0400 |" in table


def test_with_three_seeds_the_delta_is_the_median_of_paired_differences(tmp_path: Path) -> None:
    # With two seeds the median is the mean, so pairing cannot change Δ; with three it can.
    # Per seed base 0.30, 0.31, 0.32 against 0.40, 0.29, 0.33: the paired differences
    # +0.10, -0.02, +0.01 have median +0.01; the difference of the medians would be +0.02.
    three = (0, 1, 2)
    v = {s: measures(s) for s in three}
    for s, tau in zip(three, (0.40, 0.29, 0.33), strict=True):
        v[s]["kendall_tau_a"] = tau
    write_sweep(
        tmp_path,
        {
            ("base", 1000): point_report("base", 1000, {s: measures(s) for s in three}),
            ("v", 1000): point_report("v", 1000, v),
        },
    )
    assert row(sr.aggregate(tmp_path), 1000, "v").deltas["kendall_tau_a"] == pytest.approx(0.01)


def test_collapse_guards_flag_code_size_far_from_base_and_never_wins_up(tmp_path: Path) -> None:
    never_one = seeds_of()
    never_one[0]["never_wins"] += 0.2  # seed 1 unchanged: not every seed rose
    write_sweep(
        tmp_path,
        {
            ("base", 250_000): point_report("base", 250_000, seeds_of(size=8.0)),
            ("wide", 250_000): point_report("wide", 250_000, seeds_of(size=26.0)),
            ("narrow", 250_000): point_report("narrow", 250_000, seeds_of(size=3.0)),
            ("larger", 250_000): point_report("larger", 250_000, seeds_of(size=13.0)),
            ("unused", 250_000): point_report("unused", 250_000, seeds_of(never=0.4)),
            ("one", 250_000): point_report("one", 250_000, never_one),
        },
    )
    r = sr.aggregate(tmp_path)
    # base 8.5 units (median over seeds): 26.5 is 3.12 times, 3.5 is 0.41 times, 13.5 is 1.59
    assert row(r, 250_000, "wide").guards == {"code_size": "3.12 times base (26.5 vs 8.5 units)"}
    assert row(r, 250_000, "narrow").guards.keys() == {"code_size"}
    assert row(r, 250_000, "larger").guards == {}
    assert row(r, 250_000, "unused").guards == {
        "never_wins": "all 2 increased (0.405 vs base 0.105)"
    }
    assert row(r, 250_000, "one").guards == {}
    assert row(r, 250_000, "base").guards == {}
    # a guarded variant still ranks, marked
    md = sr.to_markdown(r)
    assert "code_size: 3.12 times base" in md and "`wide` " in md and "(g)" in md


def test_points_not_done_are_listed_and_a_missing_base_gives_no_deltas(tmp_path: Path) -> None:
    write_sweep(
        tmp_path,
        {
            ("base", 250_000): point_report("base", 250_000, seeds_of()),
            ("a", 250_000): point_report("a", 250_000, seeds_of(tau=0.4)),
            ("base", 500_000): "pending",
            ("a", 500_000): point_report("a", 500_000, seeds_of(tau=0.4)),
            ("b", 250_000): "failed",
            ("b", 500_000): "done",  # done, but its report was never written
        },
    )
    r = sr.aggregate(tmp_path)
    assert [(m.variant, m.length, m.status, m.reason) for m in r.missing] == [
        ("base", 500_000, "pending", "status pending"),
        ("b", 250_000, "failed", "status failed"),
        ("b", 500_000, "done", "no report at b/500000/report.json"),
    ]
    assert [w.variant for w in r.lengths[0].rows] == ["base", "a"]
    (a500,) = r.lengths[1].rows
    assert a500.variant == "a" and a500.deltas == {} and a500.guards == {}
    md = sr.to_markdown(r)
    assert "- `b` at 250k: status failed" in md
    assert "- `b` at 500k: no report at b/500000/report.json" in md
    assert "`base` is missing at this length, so there are no Δ." in md
    # Nothing done at all still writes a report that says so.
    empty = tmp_path / "empty"
    empty.mkdir()
    write_sweep(empty, {("base", 250_000): "pending"})
    nothing = sr.aggregate(empty)
    assert nothing.lengths == [sr.Length(250_000, [], [], [])] and len(nothing.missing) == 1
    assert "No point at this length is done." in sr.to_markdown(nothing)


def test_the_ranking_orders_by_each_primary_and_the_front_is_the_unbeaten(tmp_path: Path) -> None:
    write_sweep(
        tmp_path,
        {
            ("base", 500_000): point_report("base", 500_000, seeds_of()),
            ("tau", 500_000): point_report("tau", 500_000, seeds_of(tau=0.5)),
            ("auc", 500_000): point_report("auc", 500_000, seeds_of(auc=0.8)),
            ("both", 500_000): point_report("both", 500_000, seeds_of(tau=0.4, auc=0.7)),
            ("tie", 500_000): point_report("tie", 500_000, seeds_of(tau=0.4, auc=0.7)),
            ("worse", 500_000): point_report("worse", 500_000, seeds_of(tau=0.2, auc=0.5)),
        },
    )
    (x,) = sr.aggregate(tmp_path).lengths
    assert [k.measure for k in x.rankings] == ["kendall_tau_a", "model/mnistc_worst"]
    assert x.rankings[0].variants == ["tau", "both", "tie", "auc", "base", "worse"]
    assert x.rankings[1].variants == ["auc", "both", "tie", "base", "tau", "worse"]
    # equal on both measures, neither beats the other: both stay on the front
    assert x.front == ["tau", "auc", "both", "tie"]


def test_a_sweep_without_corruptions_ranks_by_tau_only(tmp_path: Path) -> None:
    clean = msgspec.structs.replace(PROTOCOL, conditions=["clean"])
    keep = ("kendall_tau_a", "separation", "domination", "never_wins", "code_size/clean")

    def rep(variant: str, tau: float) -> rp.Report:
        by = {s: {k: measures(s, tau=tau)[k] for k in keep} for s in SEEDS}
        x = rp.State(variant, "a" * 40, None, 1, by, {s: [] for s in by}, "t.json")
        return rp.build(msgspec.structs.replace(clean, train_steps=1000), {}, [x])

    write_sweep(tmp_path, {("base", 1000): rep("base", 0.3), ("a", 1000): rep("a", 0.2)})
    r = sr.aggregate(tmp_path)
    (x,) = r.lengths
    assert sr.WORST not in r.measures and sr.RANDOM not in r.measures
    assert [k.measure for k in x.rankings] == ["kendall_tau_a"] and x.front == []
    assert "MNIST-C" not in sr.to_markdown(r)


def test_points_that_do_not_match_their_sweep_are_refused(tmp_path: Path) -> None:
    def sweep_with(name: str, points: dict[tuple[str, int], rp.Report | str], **kw: Any) -> Path:
        out = tmp_path / name
        out.mkdir()
        return write_sweep(out, points, **kw)

    base = point_report("base", 1000, seeds_of())
    with pytest.raises(ValueError, match="one state 'a'"):
        sr.aggregate(sweep_with("label", {("base", 1000): base, ("a", 1000): base}))
    with pytest.raises(ValueError, match=r"trained 1000 steps, sweep\.json says 2000"):
        sr.aggregate(sweep_with("length", {("base", 2000): base}))
    seeds = sweep_with("seeds", {("base", 1000): base})
    data = json.loads((seeds / "sweep.json").read_text())
    data["points"][0]["seeds"] = [0, 1, 2]
    (seeds / "sweep.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"seeds \[0, 1\], sweep\.json says \[0, 1, 2\]"):
        sr.aggregate(seeds)
    other = rp.build(
        msgspec.structs.replace(base.protocol, frac=0.5),
        {},
        [msgspec.structs.replace(base.states[0], label="a")],
    )
    with pytest.raises(ValueError, match="beyond the training length"):
        sr.aggregate(sweep_with("protocol", {("base", 1000): base, ("a", 1000): other}))
    with pytest.raises(ValueError, match=r"not a variant of sweep\.json"):
        sr.aggregate(
            sweep_with(
                "unknown",
                {("base", 1000): base, ("a", 1000): point_report("a", 1000, seeds_of())},
                variants={"base": {}},
            )
        )
    with pytest.raises(ValueError, match="no 'base' variant"):  # a sweep.json without base
        a = point_report("a", 1000, seeds_of())
        sr.aggregate(sweep_with("nobase", {("a", 1000): a}, variants={"a": {X: 1}}))
    with pytest.raises(ValueError, match="alias x=1 names no variant"):
        sr.aggregate(sweep_with("alias", {("base", 1000): base}, aliases={"x=1": "gone"}))
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "sweep.json").write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(ValueError, match="not a coast-eval-sweep/1 file"):
        sr.aggregate(tmp_path / "junk")
    dup = sweep_with("dup", {("base", 1000): base})
    data = json.loads((dup / "sweep.json").read_text())
    data["points"].append(data["points"][0])
    (dup / "sweep.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="listed twice"):
        sr.aggregate(dup)


def test_a_variant_on_another_toolchain_than_base_is_refused_with_its_point(
    tmp_path: Path,
) -> None:
    def hosted(variant: str, compiler: str) -> rp.Report:
        r = point_report(variant, 1000, seeds_of())
        hosts = {s: rp.Host("h", compiler, "znver5") for s in SEEDS}
        return msgspec.structs.replace(
            r, states=[msgspec.structs.replace(r.states[0], hosts=hosts)]
        )

    write_sweep(
        tmp_path, {("base", 1000): hosted("base", "gcc 15"), ("a", 1000): hosted("a", "clang 21")}
    )
    with pytest.raises(ValueError, match="toolchain change") as e:
        sr.aggregate(tmp_path)
    assert "refused: sweep point a/1000" in "".join(e.value.__notes__)


def test_the_script_writes_the_two_files_into_the_sweep_directory(tmp_path: Path) -> None:
    write_sweep(
        tmp_path,
        {
            ("base", 1000): point_report("base", 1000, seeds_of()),
            ("a", 1000): point_report("a", 1000, seeds_of(tau=0.4)),
        },
    )
    before = {p.name for p in tmp_path.iterdir()}
    spec = importlib.util.spec_from_file_location("eval_sweep_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main([str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} - before == {
        "sweep_report.json",
        "sweep_report.md",
    }
    r = sr.decode((tmp_path / "sweep_report.json").read_bytes())
    assert r == sr.aggregate(tmp_path)  # the JSON holds the whole report
    assert r.schema == "coast-eval-sweep-report/1" and r.variants["a"] == {"/ngm25/x": 1}
    assert (tmp_path / "sweep_report.md").read_text() == sr.to_markdown(r)


# --- scripts/eval_plot.py over a sweep: grouping and grid only, no pixels -----------------


def load_plot() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "eval_plot.py"
    spec = importlib.util.spec_from_file_location("eval_plot_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ep = load_plot()


def grid_sweep(
    out: Path, axes: dict[str, Any], lengths: tuple[int, ...], alias: tuple[Any, ...] = ()
) -> Path:
    """A full grid over the spec `axes` (a derived axis as plan D1 writes it), plus base and
    one named variant, every point done; τ-a rises with the first axis' index, and base's is
    the highest of all. Variants, names and aliases are the driver's (`sweep.expand`). At the
    coordinates `alias` (a derived axis by its factor) the template holds the grid point's
    values, so the driver makes that point base's alias."""
    template = TEMPLATE
    if alias:
        at = dict(zip(axes, alias, strict=True))
        template = sweep.apply(
            TEMPLATE,
            {p: at[g["ratio_of"]] * at[p] if isinstance(g, dict) else at[p]
             for p, g in axes.items()},
        )  # fmt: skip
    named = {"named": {LR: 0.5}}
    _, variants, _ = driven(axes, named, list(lengths), template)
    points: dict[tuple[str, int], rp.Report | str] = {}
    first = next(iter(axes.values()))
    for name, over in variants.items():
        rise = first.index(over[LR]) if len(over) == len(axes) else 0
        tau = 0.9 if name == "base" else 0.3 + 0.01 * rise
        for n in lengths:
            points[name, n] = point_report(name, n, seeds_of(tau=tau))
    write_sweep(out, points, named, axes, template)
    return out


def test_spec_axes_keep_their_order_and_read_a_derived_axis() -> None:
    spec = {"axes": {STRENGTH: [5, 10], LR: [0.1], SLR: {"ratio_of": LR, "factors": [0.01]}}}
    a = sr.spec_axes(spec)
    assert [x.pointer for x in a] == [STRENGTH, LR, SLR]
    assert a[2] == sr.Axis(SLR, [0.01], LR)
    assert sr.coordinate(a[2], {LR: 0.003, SLR: 0.003 * 0.001}) is None  # 0.001 not a factor
    # 0.003 * 0.001 is not exactly 3e-6 in floating point; the factor is still found
    assert sr.coordinate(sr.Axis(SLR, [0.001], LR), {LR: 0.003, SLR: 3e-6}) == 0.001
    assert sr.coordinate(a[0], {STRENGTH: 7}) is None
    assert [sr.axis_text(a[1], 1e-4), sr.axis_text(a[2], 0.01)] == ["0.0001", "x0.01"]
    named = "strength=10,primary_learning_rate=0.1,secondary_learning_rate=x0.01"
    assert sr.name_coordinates(a, named) == (10, 0.1, 0.01)
    assert sr.name_coordinates(a, "strength=10,primary_learning_rate=0.1") is None
    assert sr.name_coordinates(a, named.replace("x0.01", "x0.5")) is None
    for bad in ({"ratio_of": LR}, {"ratio_of": LR, "factors": [1], "of": 2}, "0.1"):
        with pytest.raises(msgspec.ValidationError):  # the driver's own spec type refuses it
            sr.spec_axes({"axes": {SLR: bad}})


def test_curves_by_variant_group_a_sweeps_points_that_one_template_cannot(tmp_path: Path) -> None:
    reports = [
        point_report(v, n, seeds_of(tau=t))
        for v, t in (("base", 0.3), ("a", 0.4))
        for n in (500_000, 250_000)
    ]
    with pytest.raises(ValueError, match="same training length"):
        ep.curves(reports)  # one template: four reports, two at each length
    cs = ep.curves(reports, "variant")
    assert [c.label for c in cs] == ["base", "a"]
    assert all(c.steps == [250_000, 500_000] for c in cs)
    assert ep.curve_medians(cs[1], "kendall_tau_a") == pytest.approx([0.405, 0.405])
    other = msgspec.structs.replace(
        reports[1], states=[msgspec.structs.replace(reports[1].states[0], template="x.json")]
    )
    with pytest.raises(ValueError, match="different templates"):
        ep.curves([reports[0], other], "variant")
    with pytest.raises(ValueError, match="group by"):
        ep.curves(reports, "seed")
    for i, r in enumerate(reports):
        (tmp_path / f"{i}.json").write_bytes(rp.to_json(r))
    files = [str(tmp_path / f"{i}.json") for i in range(len(reports))]
    ep.main(["--curves", "--by", "variant", *files, "--out-dir", str(tmp_path / "c")])
    assert {p.name for p in (tmp_path / "c").iterdir()} == {"curves.png", "curves.md"}
    assert (
        (tmp_path / "c" / "curves.md")
        .read_text()
        .startswith("# Curves over training length: base, a\n")
    )
    with pytest.raises(SystemExit):
        ep.main(["--by", "variant", files[0], "--out-dir", str(tmp_path / "x")])


def test_the_grid_places_each_grid_variant_and_keeps_the_others_off_it(tmp_path: Path) -> None:
    axes = {LR: [0.01, 0.1], SLR: {"ratio_of": LR, "factors": [0.001, 0.01]}, STRENGTH: [5, 10]}
    sr.write(grid_sweep(tmp_path, axes, (1000,)))
    r = ep.load_sweep(tmp_path / "sweep_report.json")
    got, cells, off = ep.grid(r)
    assert [a.pointer for a in got] == [LR, SLR, STRENGTH] and len(cells) == 8
    top = "primary_learning_rate=0.1,secondary_learning_rate=x0.01,strength=10"
    assert cells[0.1, 0.01, 10] == top
    assert off == ["base", "named"]  # no alias: base is on no grid point
    assert ep.heatmap_title(r, "kendall_tau_a", cells).splitlines() == [
        "τ-a over the grid",
        # four variants tie at 0.315; the first by name is taken
        "1k: highest primary_learning_rate=0.1,secondary_learning_rate=x0.001,strength=10 "
        "(0.315); base 0.905; 8 of 8 grid points done",
    ]
    assert f"- `{SLR}`: x0.001, x0.01 times `{LR}`" in (tmp_path / "sweep_report.md").read_text()
    with pytest.raises(ValueError, match="two or three grid axes, the sweep has 3, 2 of them"):
        ep.grid(r, {LR: 0.1, SLR: 0.01})
    twice = msgspec.structs.replace(r, variants={**r.variants, "copy": r.variants[top]})
    with pytest.raises(ValueError, match="share a grid cell"):
        ep.grid(twice)


def test_an_aliased_grid_point_shows_base(tmp_path: Path) -> None:
    # The driver runs a grid point whose every value is the template's once, as base, and
    # lists its grid name as an alias. Its cell must show base, not "not done".
    axes = {LR: [0.01, 0.1], STRENGTH: [5, 10]}
    sr.write(grid_sweep(tmp_path, axes, (1000,), alias=(0.1, 10)))
    r = ep.load_sweep(tmp_path / "sweep_report.json")
    assert r.aliases == {"primary_learning_rate=0.1,strength=10": "base"}
    assert "primary_learning_rate=0.1,strength=10" not in r.variants
    _, cells, off = ep.grid(r)
    assert cells[0.1, 10] == "base" and len(cells) == 4
    assert off == ["named"]  # base is on the grid now
    (x,) = r.lengths
    by = {w.variant: w for w in x.rows}
    assert by[cells[0.1, 10]].medians["kendall_tau_a"] == pytest.approx(0.905)
    assert ep.heatmap_title(r, "kendall_tau_a", cells).splitlines()[1] == (
        "1k: highest base (0.905); base 0.905; 4 of 4 grid points done"
    )
    assert "`primary_learning_rate=0.1,strength=10` is `base`" in sr.to_markdown(r)
    clash = msgspec.structs.replace(r, aliases={"primary_learning_rate=0.01,strength=5": "base"})
    with pytest.raises(ValueError, match="share a grid cell"):
        ep.grid(clash)
    bad = msgspec.structs.replace(r, aliases={"primary_learning_rate=0.5,strength=10": "base"})
    with pytest.raises(ValueError, match="not the name of a grid point"):
        ep.grid(bad)


def test_an_integer_factor_is_named_as_the_driver_names_it(tmp_path: Path) -> None:
    # The driver reads a derived axis' factors as floats (`sweep.Ratio`): the spec's factor 1
    # names its grid points `x1.0`. An alias there must still place base, and every cell's
    # tick labels must spell its variant's name.
    axes = {LR: [0.01, 0.1], SLR: {"ratio_of": LR, "factors": [0.1, 1]}}
    sr.write(grid_sweep(tmp_path, axes, (1000,), alias=(0.1, 1)))
    r = ep.load_sweep(tmp_path / "sweep_report.json")
    alias = "primary_learning_rate=0.1,secondary_learning_rate=x1.0"
    assert r.aliases == {alias: "base"}
    left, cells, off = ep.grid(r)
    assert cells[0.1, 1.0] == "base" and len(cells) == 4 and off == ["named"]
    for coord, name in cells.items():
        label = ",".join(
            f"{ep.axis_name(a.pointer)}={sr.axis_text(a, v)}"
            for a, v in zip(left, coord, strict=True)
        )
        assert label == (alias if name == "base" else name)
    assert ep.parse_fix(r, ["secondary_learning_rate=x1"]) == {SLR: 1.0}


def test_fix_slices_stage_ones_four_axes_to_three(tmp_path: Path) -> None:
    axes = {
        LR: [0.01, 0.1],
        CLR: [1e-4, 1e-3],
        SLR: {"ratio_of": LR, "factors": [0.01, 0.1]},
        STRENGTH: [5, 10],
    }
    out = grid_sweep(tmp_path / "sweep", axes, (1000,), alias=(0.01, 1e-4, 0.01, 5))
    sr.write(out)
    r = ep.load_sweep(out / "sweep_report.json")
    with pytest.raises(ValueError, match="the sweep has 4; fix more with --fix"):
        ep.grid(r)
    fix = ep.parse_fix(r, ["common_learning_rate=0.0001"])
    assert fix == {CLR: 1e-4}
    assert ep.parse_fix(r, [f"{CLR}=1e-4"]) == fix  # the pointer, and the value as JSON
    assert ep.parse_fix(r, ["secondary_learning_rate=x0.1"]) == {SLR: 0.1}
    left, cells, off = ep.grid(r, fix)
    assert [a.pointer for a in left] == [LR, SLR, STRENGTH] and len(cells) == 8
    assert all(r.variants[v][CLR] == 1e-4 for v in cells.values() if v != "base")
    assert cells[0.01, 0.01, 5] == "base" and off == ["named"]
    _, other, _ = ep.grid(r, {CLR: 1e-3})
    assert "base" not in other.values() and len(other) == 8
    for bad, why in (
        (["common_learning_rate=0.5"], "not one of its values"),
        (["rate=0.0001"], "AXIS one of"),
        (["common_learning_rate"], "AXIS one of"),
        (["common_learning_rate=0.0001", f"{CLR}=0.001"], "fixed twice"),
    ):
        with pytest.raises(ValueError, match=why):
            ep.parse_fix(r, bad)
    figs = tmp_path / "figs"
    ep.main(["--heatmap", "model/mnistc_worst", "--fix", "common_learning_rate=0.0001",
             str(out / "sweep_report.json"), "--out-dir", str(figs)])  # fmt: skip
    stem = "heatmap_model_mnistc_worst_common_learning_rate=0.0001"
    assert {p.name for p in figs.iterdir()} == {f"{stem}.png", f"{stem}.md"}
    md = (figs / f"{stem}.md").read_text().splitlines()
    assert md[0] == "# MNIST-C worst over the grid at common_learning_rate=0.0001"
    with pytest.raises(SystemExit):  # --fix belongs to --heatmap
        ep.main(["--fix", "common_learning_rate=0.0001", str(out / "sweep_report.json"),
                 "--out-dir", str(figs)])  # fmt: skip


def test_the_heatmap_writes_its_two_files_and_refuses_an_unknown_measure(tmp_path: Path) -> None:
    out = grid_sweep(tmp_path / "sweep", {LR: [0.01, 0.1, 0.3], STRENGTH: [5, 10]}, (1000, 2000))
    sr.write(out)
    figs = tmp_path / "figs"
    ep.main(["--heatmap", sr.WORST, str(out / "sweep_report.json"), "--out-dir", str(figs)])
    names = {"heatmap_model_mnistc_worst.png", "heatmap_model_mnistc_worst.md"}
    assert {p.name for p in figs.iterdir()} == names
    assert (figs / "heatmap_model_mnistc_worst.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    md = (figs / "heatmap_model_mnistc_worst.md").read_text().splitlines()
    assert md[0] == "# MNIST-C worst over the grid"
    assert md[2].startswith("- 1k: highest ") and md[3].startswith("- 2k: highest ")
    assert "Off the grid, not drawn: base, named." in md[-1]
    with pytest.raises(ValueError, match="not a measure of this sweep report"):
        ep.main(["--heatmap", "model/mnistc_present_worst", str(out / "sweep_report.json"),
                 "--out-dir", str(figs)])  # fmt: skip
    with pytest.raises(ValueError, match=r"eval_sweep_report\.py"):
        ep.main(["--heatmap", sr.WORST, str(out / "sweep.json"), "--out-dir", str(figs)])
    with pytest.raises(SystemExit):
        ep.main(["--heatmap", sr.WORST, "--curves", str(out / "sweep_report.json"),
                 "--out-dir", str(figs)])  # fmt: skip
