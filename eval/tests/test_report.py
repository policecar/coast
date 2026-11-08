"""The report: random WTA reference, per-seed measures and summaries, the reading rule, and
reading a state from run directories shaped like coastd's."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import msgspec
import numpy as np
import pytest
from numpy.typing import NDArray

from coast_eval import datasets as ds
from coast_eval import metrics, protocol, runs
from coast_eval import report as rp
from coast_eval.observe import Codes, relative_winners

SHA = "796f973ffc4fa3f4ddcf1ca563398d224244928f"
TEMPLATE = Path(__file__).parents[1] / "configs" / "mnist_ngm25_frozen_eval.json"
CONDITIONS = ["clean", *ds.MNIST_C, "zeros"]


def codes(amps: NDArray[np.floating]) -> Codes:
    a = amps.astype(np.float32)
    return Codes(np.arange(a.shape[0], dtype=np.int64), a, relative_winners(a, 0.3))


def test_random_codes_pick_the_k_largest_projections() -> None:
    rng = np.random.default_rng(0)
    px = rng.random((6, 20))
    sizes = np.array([0, 1, 3, 8, 8, 2])
    amps, masks = rp.random_codes(px, sizes, 8, seed=4)
    assert amps.shape == masks.shape == (6, 8)
    assert masks.sum(axis=1).tolist() == sizes.tolist()
    for p, k in enumerate(sizes):
        if 0 < k < 8:  # every winner beats every loser
            assert amps[p][masks[p]].min() > amps[p][~masks[p]].max()
    again, _ = rp.random_codes(px, sizes, 8, seed=4)
    other, _ = rp.random_codes(px, sizes, 8, seed=5)
    assert np.array_equal(amps, again) and not np.allclose(amps, other)
    # the projection is linear in the pixels and scaled by 1/sqrt(width)
    double, _ = rp.random_codes(2 * px, sizes, 8, seed=4)
    assert np.allclose(double, 2 * amps)


def test_random_codes_break_ties_to_the_lower_unit() -> None:
    _, masks = rp.random_codes(np.zeros((1, 5)), np.array([2]), 4, seed=1)
    assert masks.tolist() == [[True, True, False, False]]


def state_codes(n: int = 12) -> dict[str, Codes]:
    """Clean codes plus 15 corruptions mixed with more noise each (distinct AUCs) and zeros."""
    rng = np.random.default_rng(3)
    clean = rng.random((n, 10)) ** 4
    out = {"clean": codes(clean), "zeros": codes(np.zeros_like(clean))}
    for i, c in enumerate(ds.MNIST_C):
        w = (i + 8) / 24
        out[c] = codes((1 - w) * clean + w * rng.random(clean.shape) ** 4)
    out["zigzag"] = out["stripe"]  # a tie at the worst: both are reported
    return out


def test_seed_measures_summarise_the_15_corruptions_only() -> None:
    cs = state_codes()
    px = {c: np.random.default_rng(i).random((12, 784)) for i, c in enumerate(cs)}
    m, worst = rp.seed_measures(cs, px, [1, 2])
    clean = cs["clean"]
    for c, x in cs.items():
        want = metrics.matched_mismatched_auc(
            clean.amps,
            clean.masks,
            x.amps,
            x.masks,
            clean_ids=clean.stim_ids,
            corrupt_ids=x.stim_ids,
        )
        assert m[f"model/{c}"] == want
        assert m[f"code_size/{c}"] == x.masks.sum(axis=1).mean()
    aucs = [m[f"model/{c}"] for c in ds.MNIST_C]
    assert len(set(aucs)) == 14  # varied, except the planted tie
    # zeros (chance) and clean (self) would move the worst and the mean if they counted
    assert m["model/zeros"] == 0.5 < min(aucs) and m["model/clean"] > max(aucs)
    assert m["model/mnistc_mean"] == pytest.approx(np.mean(aucs))
    assert m["model/mnistc_worst"] == min(aucs)
    assert worst == ["stripe", "zigzag"] and m["model/stripe"] == min(aucs)
    for s in (1, 2):
        r = [m[f"random/{s}/{c}"] for c in ds.MNIST_C]
        assert m[f"random/{s}/mnistc_worst"] == min(r)
    full = np.ones((12, 784), dtype=np.bool_)
    assert m["kendall_tau_a"] == metrics.similarity_preservation(
        px["clean"], clean.amps, clean.masks
    )
    assert m["never_wins"] == metrics.never_wins_fraction(clean.masks)
    assert m["domination"] == metrics.domination(clean.masks)
    shuffled = [metrics.shuffled_units(clean.amps, clean.masks, s) for s in (1, 2)]
    assert m["domination_chance"] == pytest.approx(
        np.mean([metrics.domination(w) for _, w in shuffled])
    )
    assert m["separation_chance"] == pytest.approx(
        np.mean([metrics.separation(a, w, stim_ids=clean.stim_ids) for a, w in shuffled])
    )
    assert m["separation"] == metrics.separation(clean.amps, clean.masks, stim_ids=clean.stim_ids)
    pix = rp.pixel_measures(px, clean.stim_ids)
    assert pix["pixel/fog"] == metrics.matched_mismatched_auc(
        px["clean"], full, px["fog"], full, clean_ids=clean.stim_ids, corrupt_ids=clean.stim_ids
    )
    assert pix["pixel/mnistc_worst"] == min(pix[f"pixel/{c}"] for c in ds.MNIST_C)


def test_seed_measures_refuse_a_partial_corruption_set_or_no_clean() -> None:
    cs = state_codes()
    del cs["fog"]
    px = {c: np.ones((12, 784)) for c in cs}
    with pytest.raises(ValueError, match="14 of the 15"):
        rp.seed_measures(cs, px, [1])
    with pytest.raises(ValueError, match="need clean"):
        rp.seed_measures({"zeros": cs["zeros"]}, {"zeros": px["zeros"]}, [1])


def state(label: str, measures: dict[int, dict[str, float]]) -> rp.State:
    return rp.State(label, SHA, None, 2, measures, {s: [] for s in measures}, "t.json")


PROTOCOL = rp.Protocol(
    train_steps=50_000,
    train_steps_per_image=5,
    train_images=60_000,
    test_indices=list(range(200)),
    conditions=CONDITIONS,
    projection_seeds=[1, 2],
    frac=0.3,
)


def test_build_deltas_and_reading_rule() -> None:
    a = state(
        "before",
        {
            0: {"kendall_tau_a": 0.3, "model/fog": 0.5, "model/mnistc_mean": 0.7},
            1: {"kendall_tau_a": 0.2, "model/fog": 0.6, "model/mnistc_mean": 0.8},
            2: {"kendall_tau_a": 0.4, "model/fog": 0.7, "model/mnistc_mean": 0.9},
        },
    )
    b = state(
        "after",
        {
            0: {"kendall_tau_a": 0.1, "model/fog": 0.6, "model/mnistc_mean": 0.6},
            1: {"kendall_tau_a": 0.15, "model/fog": 0.7, "model/mnistc_mean": 0.8},
            2: {"kendall_tau_a": 0.3, "model/fog": 0.75, "model/mnistc_mean": 1.0},
        },
    )
    r = rp.build(PROTOCOL, {}, [a, b])
    assert r.deltas[1] == pytest.approx(
        {"kendall_tau_a": -0.05, "model/fog": 0.1, "model/mnistc_mean": 0.0}
    )
    assert r.deltas[2]["model/mnistc_mean"] == pytest.approx(0.1)
    assert r.directions == {
        "kendall_tau_a": "all 3 decreased",
        "model/mnistc_mean": "not all 3 moved in the same direction",  # one zero delta
        "model/fog": "all 3 increased",
    }
    assert rp.build(PROTOCOL, {}, [a]).directions == {}
    with pytest.raises(ValueError, match="seeds differ"):
        rp.build(PROTOCOL, {}, [a, state("x", {0: a.measures[0]})])
    with pytest.raises(ValueError, match="different keys"):
        rp.build(PROTOCOL, {}, [a, state("x", {**a.measures, 2: {"kendall_tau_a": 0.0}})])


def test_report_json_round_trips_and_markdown_has_every_row() -> None:
    measures = {
        k: 0.25
        for k in [
            *rp.PRIMARY,
            *rp.SECONDARY,
            "model/clean",
            "model/zeros",
            "code_size/clean",
            "code_size/zeros",
            "random/1/mnistc_mean",
            "random/2/mnistc_mean",
        ]
    }
    a = state("before", {0: measures, 4: measures})
    b = state("after", {0: measures, 4: {**measures, "kendall_tau_a": 0.5}})
    r = rp.build(PROTOCOL, {"pixel/mnistc_mean": 0.9}, [a, b])
    decoded = msgspec.json.decode(rp.to_json(r), type=rp.Report)
    md = rp.to_markdown(r)
    assert decoded == r and rp.to_markdown(decoded) == md  # the JSON alone regenerates the Markdown
    assert "| 4 | kendall_tau_a | 0.2500 | 0.5000 | 0.2500 |" in md
    assert md.count("| kendall_tau_a |") == 2 and "10000 images, 0.2 passes over the 60000" in md
    assert "(a sample of indices" not in md
    spread = msgspec.structs.replace(PROTOCOL, test_indices=[9, 200, 8984])
    assert "(a sample of indices 9..8984)" in rp.to_markdown(rp.build(spread, r.pixel, [a, b]))
    short = msgspec.structs.replace(PROTOCOL, train_steps=2000, train_images=24)
    assert "present 400 images, 16.7 passes over the 24 training images" in rp.to_markdown(
        rp.build(short, {"pixel/mnistc_mean": 0.9}, [a])
    )
    # same build, same template: neither difference is claimed
    assert "every commit between" not in md and "different templates" not in md
    other = msgspec.structs.replace(b, git_sha="0" * 40)
    md = rp.to_markdown(rp.build(PROTOCOL, {"pixel/mnistc_mean": 0.9}, [a, other]))
    assert "every commit between `796f973f` and `00000000`" in md
    assert "different templates" not in md
    retina = msgspec.structs.replace(b, template="retina.json")
    md = rp.to_markdown(rp.build(PROTOCOL, {"pixel/mnistc_mean": 0.9}, [a, retina]))
    assert "run different templates, `t.json` and `retina.json`." in md
    assert "every commit between" not in md
    assert "- **after**: git `796f973f" in md and "template `retina.json`, settle lead 2." in md


# --- reading a state from run dirs ------------------------------------------------------

STEPS = 15


def manifest(sha: str = SHA, dirty: bool = False) -> dict[str, Any]:
    window = {"ranges": [[100, 100 + 3 * STEPS - 1]], "combine": "or"}
    return {
        "schema_version": "1",
        "coast_git_sha": sha,
        "coast_git_dirty": dirty,
        "coast_build_type": "Release",
        "seed_offset": 0,
        "n_steps_requested": 3 * STEPS,
        "unlimited": False,
        "n_steps_completed": 100 + 3 * STEPS,
        "exit_reason": "completed_n_steps",
        "wall_clock_seconds": 0.1,
        "hostname": "h",
        "gpu_adapter": None,
        "started_at_utc": "t",
        "completed_at_utc": "t",
        "observe": {
            "output_dir": "./observe_out",
            "groups": {"population": window, "input": window},
        },
        "datasets": {},
        "retina": {},
        "node_seeds": [],
        "savestate_outputs": [],
    }


def write_site(path: Path, name: str, values: NDArray[np.generic]) -> None:
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["type"] = np.bytes_(b"observe_log")
        f[name] = values
        f[name + "__step_index"] = np.arange(100, 100 + values.shape[0], dtype=np.uint64)


def write_run(
    d: Path,
    cfg: dict[str, Any],
    ids: list[int],
    m: dict[str, Any] | None = None,
    step_values: NDArray[np.float32] | None = None,
    steps: int = STEPS,
) -> runs.RunDir:
    """A frozen run of `steps` steps per presentation. Row t of `step_values` is the output at
    step t + 1 of every presentation; by default unit u reads (t + 1) * (u + 1), so the
    settled mean depends on how many leading steps are dropped."""
    (d / "observe_out").mkdir(parents=True)
    if step_values is None:
        step_values = ((np.arange(STEPS)[:, None] + 1.0) * np.arange(1, 4)).astype(np.float32)
    write_site(
        d / "observe_out/population.observe.h5",
        "g[1]__out",
        np.tile(step_values, (len(ids), 1)),
    )
    write_site(
        d / "observe_out/input.observe.h5",
        "m[0]__index",
        np.repeat(np.array(ids, dtype=np.uint64), steps),
    )
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "coast_runtime.json").write_text(json.dumps(m or manifest()))
    return runs.RunDir(d)


def with_retina(cfg: dict[str, Any]) -> dict[str, Any]:
    nodes = cfg["sim"]["nodes"]
    nodes.insert(1, {"id": 10, "entity_type": "retina", "inputs": [[0, "in"]]})
    nodes[-1]["inputs"] = [[10, "on"]]
    return cfg


def test_read_state_takes_the_lead_from_each_run_config(tmp_path: Path) -> None:
    base = msgspec.structs.replace(PROTOCOL, test_indices=[4, 1, 7], conditions=["clean", "fog"])
    for name, cfg, lead in (
        ("pixel", json.loads(TEMPLATE.read_text()), 1),
        ("retina", with_retina(json.loads(TEMPLATE.read_text())), 2),
    ):
        dirs = {
            (s, c): write_run(tmp_path / name / f"{s}{c}", cfg, [4, 1, 7])
            for s in (0, 1)
            for c in ("clean", "fog")
        }
        dirs[0, "train"] = runs.RunDir(tmp_path / name / "train")
        dirs[0, "train"].root.mkdir()
        dirs[0, "train"].manifest.write_text(json.dumps(manifest()))
        sha, got, by_seed, _ = rp.read_state(dirs, base)
        assert (sha, got) == (SHA, lead)
        # steps lead+1..15 averaged: mean of t = (lead + 1 + 15) / 2, times (u + 1)
        want = (lead + 1 + STEPS) / 2 * np.arange(1, 4)
        assert sorted(by_seed) == [0, 1]
        assert np.allclose(by_seed[1]["fog"].amps, want)
        assert by_seed[1]["fog"].stim_ids.tolist() == [4, 1, 7]


def test_read_state_averages_the_steps_after_the_lead(tmp_path: Path) -> None:
    # One-hot steps: unit t - 1 reads 1 at step t, so a unit's mean is nonzero exactly when
    # its step entered the average, and 1 / (number of steps averaged) then.
    one_hot = np.eye(STEPS, dtype=np.float32)
    p = msgspec.structs.replace(PROTOCOL, test_indices=[4, 1, 7], conditions=["clean"])
    for name, cfg, first in (
        ("pixel", json.loads(TEMPLATE.read_text()), 2),
        ("retina", with_retina(json.loads(TEMPLATE.read_text())), 3),
    ):
        run = write_run(tmp_path / name, cfg, [4, 1, 7], step_values=one_hot)
        _, lead, by_seed, _ = rp.read_state({(0, "clean"): run}, p)
        assert lead == first - 1
        want = np.zeros(STEPS)
        want[first - 1 :] = 1 / (STEPS - first + 1)  # steps first..15
        for row in by_seed[0]["clean"].amps:
            assert np.allclose(row, want)
    # A longer presentation would average its last 15 steps, not lead + 1 .. 15: refused.
    longer = json.loads(TEMPLATE.read_text())
    longer["datasets"]["mnist"]["default"]["change_interval"] = 2 * STEPS
    run = write_run(tmp_path / "longer", longer, [4, 1, 7], step_values=one_hot)
    with pytest.raises(ValueError, match="30 steps per image"):
        rp.read_state({(0, "clean"): run}, p)


def test_read_state_refuses_a_lead_that_leaves_no_step(tmp_path: Path) -> None:
    # The lead must leave a step to average: with two hops, a 3-step window averages step 3
    # alone, and a 2-step window would be clamped to its last step, so it raises.
    p = msgspec.structs.replace(PROTOCOL, test_indices=[4, 1, 7], conditions=["clean"])
    retina = with_retina(json.loads(TEMPLATE.read_text()))
    retina["datasets"]["mnist"]["default"]["change_interval"] = 3
    one_hot = np.eye(3, dtype=np.float32)
    run = write_run(tmp_path / "three", retina, [4, 1, 7], step_values=one_hot, steps=3)
    _, lead, by_seed, _ = rp.read_state({(0, "clean"): run}, msgspec.structs.replace(p, window=3))
    assert lead == 2 and np.allclose(by_seed[0]["clean"].amps, one_hot[-1])
    retina["datasets"]["mnist"]["default"]["change_interval"] = 2
    run = write_run(tmp_path / "two", retina, [4, 1, 7], step_values=one_hot[:2], steps=2)
    with pytest.raises(ValueError, match=r"settle lead 2 \(hop count\) leaves no step"):
        rp.read_state({(0, "clean"): run}, msgspec.structs.replace(p, window=2))


def test_read_state_refuses_mixed_builds_leads_and_misaligned_runs(tmp_path: Path) -> None:
    tpl = json.loads(TEMPLATE.read_text())
    p = msgspec.structs.replace(PROTOCOL, test_indices=[4, 1, 7], conditions=["clean"])
    cases = {
        "one clean build": [(tpl, [4, 1, 7], manifest()), (tpl, [4, 1, 7], manifest("0" * 40))],
        "clean build": [(tpl, [4, 1, 7], manifest(dirty=True))],
        "one settle lead": [
            (tpl, [4, 1, 7], None),
            (with_retina(json.loads(TEMPLATE.read_text())), [4, 1, 7], None),
        ],
        "first difference": [(tpl, [4, 7, 1], None)],
    }
    for i, (match, specs) in enumerate(cases.items()):
        dirs = {
            (s, "clean"): write_run(tmp_path / f"{i}-{s}", cfg, ids, m)
            for s, (cfg, ids, m) in enumerate(specs)
        }
        with pytest.raises(ValueError, match=match):
            rp.read_state(dirs, p)
    trained_only = {(1, "train"): runs.RunDir(tmp_path / "c")}
    with pytest.raises(ValueError, match="exactly the conditions"):
        rp.read_state({(0, "clean"): write_run(tmp_path / "c", tpl, [4, 1, 7])} | trained_only, p)
    with pytest.raises(ValueError, match="exactly the conditions"):
        rp.read_state(
            {(0, "clean"): runs.RunDir(tmp_path / "c")},
            msgspec.structs.replace(p, conditions=["clean", "fog"]),
        )


def test_a_run_the_report_refuses_is_named_for_redo(tmp_path: Path) -> None:
    """As after a crash that left a frozen trial's observe files short: the refusal names
    the key to pass to --redo."""
    tpl = json.loads(TEMPLATE.read_text())
    p = msgspec.structs.replace(PROTOCOL, test_indices=[4, 1, 7], conditions=["clean"])
    short = write_run(tmp_path / "short", tpl, [1, 7])  # the first presentation is missing
    with pytest.raises(ValueError, match="observed 2 presentations, expected 3") as e:
        rp.read_state({(2, "clean"): short}, p, "after")
    assert "run the same evaluation with --redo after/2/clean" in "\n".join(e.value.__notes__)


def test_read_state_records_each_seeds_host_and_refuses_a_split_seed(tmp_path: Path) -> None:
    tpl = json.loads(TEMPLATE.read_text())
    p = msgspec.structs.replace(PROTOCOL, test_indices=[4, 1, 7], conditions=["clean"])
    new = manifest() | {"coast_compiler": "GNU 16.2.1", "coast_target_cpu": "znver5"}
    dirs = {
        (0, "clean"): write_run(tmp_path / "0", tpl, [4, 1, 7], new | {"hostname": "a"}),
        (3, "clean"): write_run(tmp_path / "3", tpl, [4, 1, 7]),  # an older binary
    }
    assert rp.read_state(dirs, p)[3] == {
        0: rp.Host("a", "GNU 16.2.1", "znver5"),
        3: rp.Host("h", None, None),
    }
    trained = write_run(tmp_path / "t", tpl, [4, 1, 7], new | {"hostname": "b"})
    with pytest.raises(ValueError, match="must name one host"):
        rp.read_state(dirs | {(0, "train"): trained}, p)


def hosted(x: rp.State, hosts: dict[int, rp.Host]) -> rp.State:
    return msgspec.structs.replace(x, hosts=hosts)


def test_hosts_appear_only_when_there_are_several_and_must_match_per_seed() -> None:
    m = {"kendall_tau_a": 0.25}
    zen = (rp.Host("a", "GNU 16.2.1", "znver5"), rp.Host("b", "GNU 16.2.1", "znver5"))
    one = [hosted(state(x, {0: m, 1: m}), {0: zen[0], 1: zen[0]}) for x in ("before", "after")]
    assert "Hosts" not in rp.to_markdown(rp.build(PROTOCOL, {}, one))
    two = [hosted(state(x, {0: m, 1: m}), {0: zen[0], 1: zen[1]}) for x in ("before", "after")]
    r = rp.build(PROTOCOL, {}, two)
    md = rp.to_markdown(r)
    assert "Hosts (both states of a seed ran on the same one): seed 0 on a, seed 1 on b." in md
    assert "toolchains differ" not in md
    assert msgspec.json.decode(rp.to_json(r), type=rp.Report) == r
    mac = rp.Host("b", "AppleClang 21.0.0", "apple-m4")
    mixed = [hosted(state(x, {0: m, 1: m}), {0: zen[0], 1: mac}) for x in ("before", "after")]
    assert ("toolchains differ (AppleClang 21.0.0 for apple-m4; GNU 16.2.1 for znver5)"
            in rp.to_markdown(rp.build(PROTOCOL, {}, mixed)))  # fmt: skip
    old = rp.Host("b", None, None)  # a build from before the toolchain was recorded
    unrecorded = rp.to_markdown(
        rp.build(PROTOCOL, {}, [hosted(two[0], {0: zen[0], 1: old}), two[1]])
    )
    assert "toolchains differ" not in unrecorded and "do not record their toolchain" in unrecorded
    assert "do not record" not in md
    assert "built with GNU 16.2.1 for znver5, template" in md  # one toolchain per state
    assert (
        "built with GNU 16.2.1 for znver5 (seeds [0]); AppleClang 21.0.0 for apple-m4 "
        "(seeds [1])" in rp.to_markdown(rp.build(PROTOCOL, {}, mixed))
    )
    crossed = [two[0], hosted(two[1], {0: zen[1], 1: zen[0]})]
    with pytest.raises(ValueError, match="seed 0 ran on a for before but on b for after"):
        rp.build(PROTOCOL, {}, crossed)


def test_build_refuses_a_seed_whose_two_states_were_built_by_different_toolchains() -> None:
    m = {"kendall_tau_a": 0.25}
    old_gcc, new_gcc = rp.Host("a", "GNU 15.2.1", "znver5"), rp.Host("a", "GNU 16.2.1", "znver5")
    before = hosted(state("before", {0: m}), {0: old_gcc})  # a cached build, same host
    after = hosted(state("after", {0: m}), {0: new_gcc})
    with pytest.raises(ValueError, match=r"seed 0: before was built with GNU 15\.2\.1 for "
                       r"znver5, after with GNU 16\.2\.1 for znver5"):  # fmt: skip
        rp.build(PROTOCOL, {}, [before, after])
    other_cpu = hosted(after, {0: rp.Host("a", "GNU 15.2.1", "znver4")})
    with pytest.raises(ValueError, match="mix the toolchain change"):
        rp.build(PROTOCOL, {}, [before, other_cpu])
    unrecorded = hosted(after, {0: rp.Host("a", None, None)})  # cannot be checked: said so
    md = rp.to_markdown(rp.build(PROTOCOL, {}, [before, unrecorded]))
    assert "built with an unrecorded toolchain" in md
    assert "is not checked for them" in md
    assert "not checked" not in rp.to_markdown(rp.build(PROTOCOL, {}, [before, before]))


def test_condition_pixels(tmp_path: Path) -> None:
    images = np.arange(5 * 28 * 28, dtype=np.int64).reshape(5, 28, 28) % 256
    (tmp_path / "mnist").mkdir()
    (tmp_path / "mnist_c/fog").mkdir(parents=True)
    ds.write_idx(tmp_path / ds.TEST_IMAGES, images.astype(np.uint8))
    ds.write_idx(tmp_path / ds.corruption_files("fog")[0], (255 - images).astype(np.uint8))
    clean = rp.condition_pixels(tmp_path, "clean", [3, 0])
    assert clean.shape == (2, 784) and np.allclose(clean[0], images[3].ravel() / 255)
    assert np.allclose(rp.condition_pixels(tmp_path, "fog", [3, 0]), 1 - clean)
    assert not rp.condition_pixels(tmp_path, "zeros", [3, 0, 1]).any()
    assert protocol.CONDITIONS[0] == "clean" and protocol.CONDITIONS[-1] == "zeros"


def measured(label: str, conditions: list[str]) -> tuple[rp.State, dict[str, NDArray[np.float64]]]:
    """A state measured by seed_measures on the given conditions (two seeds), and its pixels."""
    cs = {c: x for c, x in state_codes().items() if c in conditions}
    px = {c: np.random.default_rng(i).random((12, 784)) for i, c in enumerate(cs)}
    m, worst = rp.seed_measures(cs, px, [1, 2])
    state = rp.State(label, SHA, None, 2, {0: m, 3: m}, {0: worst, 3: worst}, "t.json")
    return state, px


@pytest.mark.parametrize(
    "conditions",
    [["clean", "zeros"], ["clean", *ds.MNIST_C], ["clean"], CONDITIONS],
)
def test_markdown_shows_the_controls_that_ran(conditions: list[str]) -> None:
    a, px = measured("before", conditions)
    b, _ = measured("after", conditions)
    pixel = rp.pixel_measures(px, np.arange(12, dtype=np.int64))
    r = rp.build(msgspec.structs.replace(PROTOCOL, conditions=conditions), pixel, [a, b])
    md = rp.to_markdown(r)
    corruptions, zeros = "fog" in conditions, "zeros" in conditions
    assert ("## Controls" in md) == (corruptions or zeros)
    assert ("| random mnistc_mean | pixel mnistc_mean |" in md) == corruptions
    assert ("only statistically" in md) == corruptions
    assert ("| model/zeros | code_size/zeros |" in md) == zeros
    assert ("The zeros control is full codes" in md) == zeros
    if corruptions or zeros:  # one header cell per value in each state's row
        table = md.split("## Controls")[1].splitlines()
        (row,) = (line for line in table if line.startswith("| before |"))
        assert row.count("|") == 2 + 3 * corruptions + 2 * zeros
    assert rp.to_markdown(msgspec.json.decode(rp.to_json(r), type=rp.Report)) == md


def test_check_conditions() -> None:
    for ok in (["clean"], ["clean", "zeros"], ["zeros", *ds.MNIST_C, "clean"], CONDITIONS):
        rp.check_conditions(ok)
    with pytest.raises(ValueError, match="1 of the 15 MNIST-C"):
        rp.check_conditions(["clean", "fog"])
    with pytest.raises(ValueError, match="14 of the 15 MNIST-C"):
        rp.check_conditions([c for c in CONDITIONS if c != "fog"])
    for bad in (["zeros"], ["clean", "clean"], ["clean", "blur"], []):
        with pytest.raises(ValueError, match="unique, known and include clean"):
            rp.check_conditions(bad)
