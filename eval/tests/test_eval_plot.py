"""`scripts/eval_plot.py` on synthetic reports with known findings: the titles say what the
reading rule gives, separation and domination are judged against chance, a report it cannot
draw is refused, and it writes into its output directory and nowhere else."""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

import msgspec
import pytest

from coast_eval import report as rp
from coast_eval.datasets import MNIST_C

SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_plot.py"
SEEDS = (0, 1, 2)
PROJ = (1, 2)


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("eval_plot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ep = load_script()


def measures(seed: int, shift: float = 0.0, **over: float) -> rp.Measures:
    """Every key a report carries, with values that differ by seed and by `shift`."""
    m: rp.Measures = {
        "kendall_tau_a": 0.3 + 0.01 * seed + shift,
        "separation": 0.4 + 0.01 * seed,
        "separation_chance": 0.35 + 0.01 * seed,
        "domination": 0.3,
        "domination_chance": 0.2,
        "never_wins": 0.0,
        "code_size/clean": 6.0 + seed,
        "code_size/zeros": 50.0,
        "model/clean": 1.0,
        "model/zeros": 0.5,
    }
    for i, c in enumerate(MNIST_C):
        m[f"model/{c}"] = 0.6 + 0.02 * i + 0.01 * seed + shift
        for q in PROJ:
            m[f"random/{q}/{c}"] = 0.55 + 0.02 * i + 0.001 * q
    for prefix in ("model", *(f"random/{q}" for q in PROJ)):
        rp._summarise(m, prefix)  # pyright: ignore[reportPrivateUsage]
    m.update(over)
    return m


def state(label: str, by_seed: dict[int, rp.Measures]) -> rp.State:
    return rp.State(
        label, "796f973f" + "0" * 32, None, 1, by_seed, {s: [] for s in by_seed}, "t.json"
    )


PROTOCOL = rp.Protocol(
    train_steps=1_000_000,
    train_steps_per_image=1,
    train_images=60_000,
    test_indices=list(range(20)),
    conditions=["clean", *MNIST_C, "zeros"],
    projection_seeds=list(PROJ),
    frac=0.3,
)
PIXEL = {f"pixel/{c}": 0.9 for c in MNIST_C} | {"pixel/mnistc_mean": 0.9, "pixel/mnistc_worst": 0.9}


def two(after: dict[int, rp.Measures]) -> rp.Report:
    before = state("before", {s: measures(s) for s in SEEDS})
    return rp.build(PROTOCOL, PIXEL, [before, state("after", after)])


def test_two_state_titles_give_the_reading_rule() -> None:
    down = two({s: measures(s, shift=-0.05) for s in SEEDS})
    assert ep.finding(down, "kendall_tau_a") == "all 3 down"
    assert ep.measures_title(down).startswith("τ-a: all 3 down · MNIST-C mean: all 3 down")
    assert ep.corruptions_title(down).startswith("15 of 15 corruptions down in every seed")
    # one seed against the others: mixed, and a median change below 0.01 says so
    mixed = two({s: measures(s, shift=0.002 if s else -0.004) for s in SEEDS})
    assert ep.finding(mixed, "kendall_tau_a") == "mixed"
    assert ep.panel_title(mixed, "kendall_tau_a") == "τ-a: mixed (change < 0.01)"
    up = two({s: measures(s, shift=0.05) for s in SEEDS})
    assert ep.panel_title(up, "kendall_tau_a") == "τ-a: all 3 up"


def test_two_identical_states_read_unchanged_in_every_title() -> None:
    # An A/A report: no seed moved, so the reading rule's "not all moved the same way" does not
    # apply, and the corruptions must say what the measures say.
    same = two({s: measures(s) for s in SEEDS})
    assert all(p.endswith(": unchanged") for p in ep.measures_title(same).split(" · "))
    assert ep.corruptions_title(same) == "15 of 15 corruptions unchanged in every seed"


def test_separation_and_domination_are_judged_against_chance() -> None:
    # Separation rises in every seed, but its chance level rises more: the code got larger,
    # not more overlapping, so the excess over chance went down.
    after = {
        s: measures(s, separation=0.5 + 0.01 * s, separation_chance=0.5 + 0.01 * s) for s in SEEDS
    }
    r = two(after)
    assert all(d["separation"] > 0 for d in r.deltas.values())
    assert ep.finding(r, "separation") == "all 3 down"
    assert ep.panel_title(r, "separation").startswith("separation above chance: all 3 down")
    # the excess is what is judged: the same rise with an unchanged chance level reads up
    same_chance = {s: measures(s, separation=0.5 + 0.01 * s) for s in SEEDS}
    assert ep.finding(two(same_chance), "separation") == "all 3 up"
    one = rp.build(PROTOCOL, PIXEL, [state("only", {s: measures(s) for s in SEEDS})])
    assert ep.panel_title(one, "separation") == f"separation: {0.41 / 0.36:.2f} times chance"


def test_one_state_titles_give_the_values() -> None:
    r = rp.build(PROTOCOL, PIXEL, [state("retina", {s: measures(s) for s in SEEDS})])
    assert ep.measures_title(r) == (
        "retina: τ-a 0.31, MNIST-C mean 0.75, MNIST-C worst 0.61, "
        f"separation {0.41 / 0.36:.2f} times its chance level"
    )
    assert ep.corruptions_title(r) == (
        f"worst: {ep.label(MNIST_C[0])} 0.61; above the random projection on 15 of 15 corruptions"
    )
    assert ep.controls_title(r).startswith("retina: mean 0.75 above random 0.69")


def test_code_size_is_drawn_on_whole_units() -> None:
    # One seed of 5.465 units: a range of 5.27 to 5.67 holds no whole unit, and its ticks at
    # 5.3, 5.4, ... printed as whole units read "5, 5, 5, 6, 6".
    one = {0: measures(0, **{"code_size/clean": 5.465})}
    r = rp.build(PROTOCOL, PIXEL, [state("pixel", one)])
    assert ep.ylim(r, "code_size/clean") == (5, 6)


def test_a_report_it_cannot_draw_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": "something-else", "rows": []}))
    with pytest.raises(ValueError, match="not a coast-eval report"):
        ep.load(other)
    # A report of the older schema still carries extra_lead.
    r = rp.build(PROTOCOL, PIXEL, [state("a", {s: measures(s) for s in SEEDS})])
    old = msgspec.to_builtins(r)
    old["protocol"]["extra_lead"] = 0
    (tmp_path / "old.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="current schema"):
        ep.load(tmp_path / "old.json")
    # Three states: `report.build` never writes one, but a hand-edited file could.
    three = msgspec.to_builtins(two({s: measures(s) for s in SEEDS}))
    three["states"].append(three["states"][0])
    (tmp_path / "three.json").write_text(json.dumps(three))
    with pytest.raises(ValueError, match="3 states"):
        ep.load(tmp_path / "three.json")
    # A /1 report, as written before the schema was bumped, names the converter.
    old["schema"] = "coast-eval-report/1"
    (tmp_path / "v1.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="eval_report_upgrade"):
        ep.load(tmp_path / "v1.json")
    # Chance levels on some seeds only: neither a current report nor a converted one.
    part = {s: measures(s) for s in SEEDS}
    del part[1]["separation_chance"]
    path = tmp_path / "part.json"
    path.write_bytes(rp.to_json(rp.build(PROTOCOL, PIXEL, [state("a", part)])))
    with pytest.raises(ValueError, match="some seeds carry the chance levels"):
        ep.load(path)


def bare(seed: int, shift: float = 0.0) -> rp.Measures:
    """A seed's measures as a report converted from /1 has them: no chance levels."""
    return {k: v for k, v in measures(seed, shift).items() if not k.endswith("_chance")}


def test_a_converted_report_is_drawn_without_chance_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / "converted.json"
    one = rp.build(PROTOCOL, PIXEL, [state("pixel", {s: bare(s) for s in SEEDS})])
    path.write_bytes(rp.to_json(one))
    r = ep.load(path)
    assert not ep.has_chance(r)
    assert ep.panel_title(r, "separation") == "separation: 0.41"
    assert ep.measures_title(r).endswith("code size 7.0")
    assert "no chance levels" in ep.caption(r, "measures")
    # Two states: separation is judged on its own deltas, and the title does not claim chance.
    after = {s: bare(s) | {"separation": 0.3 + 0.01 * s} for s in SEEDS}
    two_bare = rp.build(
        PROTOCOL, PIXEL, [state("a", {s: bare(s) for s in SEEDS}), state("b", after)]
    )
    assert ep.panel_title(two_bare, "separation") == "separation: all 3 down"
    paths = ep.draw(two_bare, tmp_path / "figs")
    assert {p.name for p in paths} == {
        "measures.png",
        "corruptions.png",
        "controls.png",
        "figures.md",
    }


def point(
    steps: int, shift: float, template: str = "mnist_ngm25_frozen_eval.json", **kw: Any
) -> rp.Report:
    """A one-state report of a length sweep at `steps` training steps."""
    p = msgspec.structs.replace(PROTOCOL, train_steps=steps, **kw)
    x = msgspec.structs.replace(
        state("now", {s: measures(s, shift) for s in SEEDS}), template=template
    )
    return rp.build(p, PIXEL, [x])


def test_curves_group_by_template_and_order_by_length() -> None:
    reports = [
        point(1_000_000, 0.05),
        point(50_000, 0.0, "mnist_ngm25_frozen_eval_retina.json"),
        point(50_000, 0.0),
        point(1_000_000, 0.02, "mnist_ngm25_frozen_eval_retina.json"),
        point(500_000, 0.04),
    ]
    cs = ep.curves(reports)
    assert [c.label for c in cs] == ["pixel", "retina"]
    assert cs[0].steps == [50_000, 500_000, 1_000_000] and cs[1].steps == [50_000, 1_000_000]
    assert ep.curve_medians(cs[0], "kendall_tau_a") == pytest.approx([0.31, 0.35, 0.36])
    assert ep.curves_title(cs[:1]) == (
        "pixel, 50k to 1M: τ-a 0.31 to 0.36, MNIST-C mean 0.75 to 0.80; over the last 500k: "
        "τ-a +0.010, MNIST-C mean +0.010"
    )
    # Several templates: one line each, so each template's finding reads on its own.
    lines = ep.curves_title(cs).splitlines()
    assert [line.split(",")[0] for line in lines] == ["pixel", "retina"]
    assert lines[0] == ep.curves_title(cs[:1])
    assert ep.subtitle(reports[1]).count("\n") == 0
    assert ep.template_label("my_template.json") == "my_template"
    assert ep.labelled([50_000, 100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000]) == [
        100_000, 300_000, 500_000, 700_000, 1_000_000,
    ]  # fmt: skip


def one_seed(r: rp.Report) -> rp.Report:
    """`r` with its first training seed only."""
    x = r.states[0]
    return msgspec.structs.replace(
        r, states=[msgspec.structs.replace(x, measures={0: x.measures[0]}, worst={0: x.worst[0]})]
    )


def test_curves_give_each_template_its_own_seed_count() -> None:
    retina = "mnist_ngm25_frozen_eval_retina.json"
    cs = ep.curves([point(50_000, 0.0), one_seed(point(50_000, 0.0, retina))])
    assert "training seeds 3 (pixel), 1 (retina);" in ep.curves_caption(cs)


def test_end_labels_closer_than_the_gap_are_spread_apart_in_order() -> None:
    # The curves' grey labels: "random, retina" and "random, retina divnorm" end 0.5 pt apart.
    wanted = [40.0, 100.0, 100.5, 101.0, 300.0]
    placed = ep.spread(wanted, 10.0)
    assert all(b - a >= 10.0 - 1e-9 for a, b in itertools.pairwise(placed))
    assert placed[0] == 40.0 and placed[-1] == 300.0  # labels with room stay where they are
    assert statistics.mean(placed[1:4]) == pytest.approx(100.5)  # a block keeps its centre
    # A block that grows into a neighbour takes it along; the order given is kept, ties too.
    # 0 and 0 split to -5, 5; that block meets 5 and becomes -8.3, 1.7, 11.7; that one meets 12,
    # and the four centre on their mean 4.25.
    assert ep.spread([5.0, 0.0, 0.0, 12.0], 10.0) == pytest.approx([9.25, -10.75, -0.75, 19.25])
    assert ep.spread([], 10.0) == []


def test_curves_refuse_points_that_are_not_one_sweep() -> None:
    with pytest.raises(ValueError, match="one-state report"):
        ep.curves([two({s: measures(s) for s in SEEDS})])
    with pytest.raises(ValueError, match="beyond the training length"):
        ep.curves([point(50_000, 0.0), point(100_000, 0.0, frac=0.5)])
    with pytest.raises(ValueError, match="same training length"):
        ep.curves([point(50_000, 0.0), point(50_000, 0.01)])
    other = point(100_000, 0.0)
    other = msgspec.structs.replace(
        other, states=[msgspec.structs.replace(other.states[0], git_sha="0" * 40)]
    )
    with pytest.raises(ValueError, match="different builds"):
        ep.curves([point(50_000, 0.0), other])
    with pytest.raises(ValueError, match="different training seeds"):
        ep.curves([point(50_000, 0.0), one_seed(point(100_000, 0.0))])
    converted = point(100_000, 0.0)
    converted = msgspec.structs.replace(
        converted,
        states=[msgspec.structs.replace(converted.states[0], measures={s: bare(s) for s in SEEDS})],
    )
    with pytest.raises(ValueError, match="carry chance levels and others do not"):
        ep.curves([point(50_000, 0.0), converted])


def test_curves_write_into_their_output_directory_only(tmp_path: Path) -> None:
    for n, shift in ((50_000, 0.0), (500_000, 0.03), (1_000_000, 0.04)):
        (tmp_path / f"{n}.json").write_bytes(rp.to_json(point(n, shift)))
    ep.main(["--curves", *(str(tmp_path / f"{n}.json") for n in (50_000, 500_000, 1_000_000)),
             "--out-dir", str(tmp_path / "curves")])  # fmt: skip
    assert {p.name for p in (tmp_path / "curves").iterdir()} == {"curves.png", "curves.md"}
    md = (tmp_path / "curves" / "curves.md").read_text().splitlines()
    assert md[0] == "# Curves over training length: pixel"
    assert md[2] == "- pixel, 50k to 1M: τ-a 0.31 to 0.35, MNIST-C mean 0.75 to 0.79; over the " \
        "last 500k: τ-a +0.010, MNIST-C mean +0.010"  # fmt: skip
    assert md[4] == f"![{md[2].removeprefix('- ')}](curves.png)"
    assert (tmp_path / "curves" / "curves.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(SystemExit):  # several reports need --curves
        ep.main([str(tmp_path / "50000.json"), str(tmp_path / "500000.json"),
                 "--out-dir", str(tmp_path / "x")])  # fmt: skip


def test_it_writes_into_its_output_directory_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "in" / "report.json"
    src.parent.mkdir()
    src.write_bytes(rp.to_json(two({s: measures(s, shift=-0.05) for s in SEEDS})))
    ep.main([str(src), "--out-dir", str(tmp_path / "figs")])
    want = {"measures.png", "corruptions.png", "controls.png", "figures.md"}
    assert {p.name for p in (tmp_path / "figs").iterdir()} == want
    assert {p.name for p in tmp_path.iterdir()} == {"in", "figs"}
    assert {p.name for p in src.parent.iterdir()} == {"report.json"}
    for name in want - {"figures.md"}:
        assert (tmp_path / "figs" / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    md = (tmp_path / "figs" / "figures.md").read_text()
    assert "![" in md and "| separation |" in md and "A measure moved only if all 3" in md
    # The controls title has a line per state; its heading joins them into one.
    assert "\n## before: mean 0.75 above random 0.69, worst 0.61 above random 0.55; after: " in md
    # Without the MNIST-C conditions only the measures are drawn.
    clean_only = rp.Protocol(
        train_steps=1000, train_steps_per_image=1, train_images=60_000,
        test_indices=list(range(20)), conditions=["clean"], projection_seeds=list(PROJ),
        frac=0.3,
    )  # fmt: skip
    keep = ("kendall_tau_a", "separation", "separation_chance", "domination",
            "domination_chance", "never_wins", "code_size/clean", "model/clean")  # fmt: skip
    m = {s: {k: measures(s)[k] for k in keep} for s in SEEDS}
    src.write_bytes(rp.to_json(rp.build(clean_only, {}, [state("a", m)])))
    ep.main([str(src), "--out-dir", str(tmp_path / "clean")])
    assert {p.name for p in (tmp_path / "clean").iterdir()} == {"measures.png", "figures.md"}
