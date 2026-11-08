"""`coast_eval.sweep`: the spec and its grid, the chains a sweep submits through the runner
against an in-process fake coastd (the external boundary, faked at `Client.request`), the
`sweep.json` it writes, resume and redo, and `scripts/eval_sweep.py`'s refusals."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest

from coast_eval import datasets as ds
from coast_eval import protocol as pr
from coast_eval import report as rp
from coast_eval import runs, sweep
from coast_eval.daemon import Client, DaemonError, MachineInfo
from coast_eval.runner import Daemon

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "eval" / "configs" / "mnist_ngm25_frozen_eval.json"
STAGE1 = ROOT / "eval" / "sweeps" / "stage1_ngm25_lr_inhib.json"
SHA = "796f973ffc4fa3f4ddcf1ca563398d224244928f"
G = "/ngm25/neuron_groups/L4-mnist/"
STRENGTH = "/ngm25/inhibitions/ab_pinned/strength"
EXPECTED = {"mnist/train-images-idx3-ubyte": "a" * 64}
INPUTS = pr.TuningInputs((5, 9), *(f"./data/eval/t/{n}" for n in
                                   ("train.txt", "rows.txt", "img", "lab", "zeros")))  # fmt: skip


def spec_doc(**changes: Any) -> dict[str, Any]:
    return {"schema": "coast-eval-sweep-spec/1", "template": str(TEMPLATE),
            "axes": {STRENGTH: [5]}, "seeds": [0, 1], "lengths": [100, 300],
            "conditions": ["clean", "zeros"], "tuning_images": 2} | changes  # fmt: skip


def spec(**changes: Any) -> sweep.Spec:
    return sweep.load_spec(json.dumps(spec_doc(**changes)).encode())


def template() -> dict[str, Any]:
    return json.loads(TEMPLATE.read_text())


# --- the spec and its grid --------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema": "coast-eval-sweep-spec/2"}, "not a coast-eval-sweep-spec/1"),
        ({"extra": 1}, "unknown field"),
        ({"conditions": []}, "conditions must be non-empty"),
        ({"conditions": ["clean", "blur"]}, "conditions must be"),
        ({"conditions": ["clean", "clean"]}, "conditions must be"),
        ({"seeds": [1, 1]}, "seeds must be"),
        ({"seeds": [-1]}, "seeds must be"),
        ({"lengths": [300, 100]}, "increasing"),
        ({"lengths": [0, 100]}, "increasing"),
        ({"tuning_images": 0}, "tuning images"),
        ({"tuning_images": pr.TUNING_POOL + 1}, "tuning images"),
        ({"variants": {"base": {STRENGTH: 1}}}, "'base' is the template unchanged"),
        ({"variants": {"v": {}}}, "overrides nothing"),
        ({"variants": {"a/b": {STRENGTH: 1}}}, "must be made of"),
    ],
)
def test_load_spec_refuses(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        spec(**changes)


def test_the_grid_is_named_by_last_keys_in_axis_order_after_base() -> None:
    s = spec(axes={G + "primary_learning_rate": [0.1, 3e-5], STRENGTH: [5, 10]},
             variants={"wide": {G + "neuron_cnt": 80}})  # fmt: skip
    v, aliases = sweep.expand(s, template())
    assert aliases == {}
    assert list(v) == [
        "base",
        "primary_learning_rate=0.1,strength=5",
        "primary_learning_rate=0.1,strength=10",
        "primary_learning_rate=3e-05,strength=5",
        "primary_learning_rate=3e-05,strength=10",
        "wide",
    ]
    assert v["base"] == {}
    assert v["primary_learning_rate=3e-05,strength=10"] == {
        G + "primary_learning_rate": 3e-5, STRENGTH: 10}  # fmt: skip
    applied = sweep.apply(template(), v["primary_learning_rate=3e-05,strength=10"])
    assert applied["ngm25"]["neuron_groups"]["L4-mnist"]["primary_learning_rate"] == 3e-5
    assert applied["ngm25"]["inhibitions"]["ab_pinned"]["strength"] == 10
    assert template()["ngm25"]["inhibitions"]["ab_pinned"]["strength"] == 10.0  # a copy


def test_a_derived_axis_is_its_reference_times_each_factor_named_by_the_factor() -> None:
    s = spec(axes={G + "primary_learning_rate": [0.1, 0.3],
                   G + "secondary_learning_rate": {"ratio_of": G + "primary_learning_rate",
                                                   "factors": [0.01, 0.5]}})  # fmt: skip
    v, aliases = sweep.expand(s, template())
    assert aliases == {}
    assert list(v)[1:] == [
        "primary_learning_rate=0.1,secondary_learning_rate=x0.01",
        "primary_learning_rate=0.1,secondary_learning_rate=x0.5",
        "primary_learning_rate=0.3,secondary_learning_rate=x0.01",
        "primary_learning_rate=0.3,secondary_learning_rate=x0.5",
    ]
    got = v["primary_learning_rate=0.3,secondary_learning_rate=x0.5"]
    assert got == {G + "primary_learning_rate": 0.3, G + "secondary_learning_rate": 0.3 * 0.5}
    # secondary <= primary at every primary, which an absolute list could not keep
    for overrides in list(v.values())[1:]:
        assert overrides[G + "secondary_learning_rate"] < overrides[G + "primary_learning_rate"]


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        ({G + "secondary_learning_rate": {"ratio_of": G + "primary_learning_rate",
                                          "factors": [0.1]}}, "not a plain axis"),
        ({G + "primary_learning_rate": {"ratio_of": STRENGTH, "factors": [2]},
          G + "secondary_learning_rate": {"ratio_of": G + "primary_learning_rate",
                                          "factors": [0.1]},
          STRENGTH: [5]}, "not a plain axis"),  # a chain
        ({G + "secondary_learning_rate": {"ratio_of": G + "secondary_learning_rate",
                                          "factors": [0.1]}}, "not a plain axis"),  # itself
        ({G + "primary_learning_rate": [0.1],
          G + "secondary_learning_rate": {"ratio_of": G + "primary_learning_rate",
                                          "factors": list[float]()}}, "non-empty and unique"),
        ({G + "penalty_policy": ["a", "b"],
          G + "secondary_learning_rate": {"ratio_of": G + "penalty_policy",
                                          "factors": [0.1]}}, "non-numbers"),
        ({G + "primary_learning_rate": list[float]()}, "non-empty and unique"),
        ({G + "primary_learning_rate": [0.1, 0.1]}, "non-empty and unique"),
        ({STRENGTH: [1, 1.0]}, "non-empty and unique"),  # one value to coast
        ({G + "primary_learning_rate": [0.1, 0.1 + 1e-12]}, "non-empty and unique"),
        ({G + "primary_learning_rate": [0.1],
          G + "secondary_learning_rate": {"ratio_of": G + "primary_learning_rate",
                                          "factors": [0.5, 0.5 + 1e-12]}}, "non-empty and unique"),
        ({G + "neuron_noise_range": [[0.0, 0.1]]}, "not a scalar"),
        ({G + "primary_lr": [0.1]}, "not in the template"),
        ({"/ngm25/neuron_groups/L4-mnist/inputs/3/bank": ["x"]}, "not in the template"),
        ({"ngm25": [1]}, "not a JSON pointer"),
        ({"/frozen": [False]}, "the protocol writes this key"),
        ({"/datasets/mnist/default/change_interval": [5]}, "the protocol writes"),
        ({"/datasets/mnist": [dict[str, Any]()]}, "the protocol writes"),  # contains its keys
        ({"/sim/nodes/0/random_seed_offset": [3]}, "the protocol writes"),
        ({G + "strength": [1], STRENGTH: [5]}, "distinct keys"),
    ],
)  # fmt: skip
def test_expand_refuses(axes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sweep.expand(spec(axes=axes), template())


def test_same_value_is_isclose_for_numbers_and_exact_for_the_rest() -> None:
    assert sweep.same_value(0.1 * 0.1, 0.01) and 0.1 * 0.1 != 0.01
    assert sweep.same_value(1, 1.0) and sweep.same_value(0.1, 0.1 + 1e-12)
    assert sweep.same_value(0, 0.0) and not sweep.same_value(0.0, 1e-300)  # no abs_tol
    assert not sweep.same_value(0.1, 0.1 * (1 + 1e-8))
    assert not sweep.same_value(True, 1) and not sweep.same_value(1.0, True)  # bools aren't
    assert sweep.same_value(True, True) and not sweep.same_value(True, False)
    assert sweep.same_value("a", "a") and not sweep.same_value("1", 1)


def test_axis_duplicates_are_named_with_their_axis() -> None:
    with pytest.raises(ValueError) as e:
        sweep.expand(spec(axes={STRENGTH: [5, 1, 1.0]}), template())
    assert f"axis {STRENGTH}" in str(e.value) and "1 and 1.0 are the same value" in str(e.value)


def test_a_derived_value_off_by_rounding_from_the_templates_is_still_base() -> None:
    """The reviewer's case: p=0.1, s=0.01 in the template and s = x0.1 of p. The product is
    0.010000000000000002, not 0.01; exact equality would train base again under a grid name."""
    tpl = sweep.apply(template(), {G + "primary_learning_rate": 0.1,
                                   G + "secondary_learning_rate": 0.01})  # fmt: skip
    s = spec(axes={G + "primary_learning_rate": [0.1, 0.3],
                   G + "secondary_learning_rate": {"ratio_of": G + "primary_learning_rate",
                                                   "factors": [0.1, 0.5]}})  # fmt: skip
    v, aliases = sweep.expand(s, tpl)
    is_base = "primary_learning_rate=0.1,secondary_learning_rate=x0.1"
    assert aliases == {is_base: "base"} and is_base not in v
    assert list(v) == [
        "base",
        "primary_learning_rate=0.1,secondary_learning_rate=x0.5",
        "primary_learning_rate=0.3,secondary_learning_rate=x0.1",
        "primary_learning_rate=0.3,secondary_learning_rate=x0.5",
    ]


def test_a_grid_point_with_the_templates_values_is_base_not_a_second_variant() -> None:
    s = spec(axes={STRENGTH: [5, 10]})  # the template's strength is 10.0: 10 is the same
    v, aliases = sweep.expand(s, template())
    assert list(v) == ["base", "strength=5"] and aliases == {"strength=10": "base"}
    with pytest.raises(ValueError, match="also a grid point"):  # an alias's name is taken
        sweep.expand(spec(axes={STRENGTH: [5, 10]}, variants={"strength=10": {STRENGTH: 7}}),
                     template())  # fmt: skip


def test_named_variants_are_checked_too() -> None:
    with pytest.raises(ValueError, match="not in the template"):
        sweep.expand(spec(variants={"typo": {G + "primry_learning_rate": 1}}), template())
    with pytest.raises(ValueError, match="also a grid point"):
        sweep.expand(spec(variants={"strength=5": {STRENGTH: 7}}), template())


def test_the_stage1_spec_is_the_planned_grid() -> None:
    raw = STAGE1.read_bytes()
    s = sweep.load_spec(raw)
    rp.check_conditions(s.conditions)  # a report can summarise it
    tpl_path = ROOT / s.template
    v, aliases = sweep.expand(s, json.loads(tpl_path.read_text()))
    # base + primary x secondary ratio x strength x common, less the grid point that is base
    assert len(v) == 1 + 4 * 3 * 4 * 2 - 1
    is_base = ("primary_learning_rate=0.01,secondary_learning_rate=x0.01,strength=10,"
               "common_learning_rate=0.0001")  # fmt: skip
    assert aliases == {is_base: "base"} and is_base not in v
    assert (s.seeds, s.lengths, s.tuning_images) == ([0, 1], [250_000, 500_000], 1000)
    for overrides in list(v.values())[1:]:
        assert overrides[G + "secondary_learning_rate"] <= overrides[G + "primary_learning_rate"]


# --- the tuning inputs, pins and pixels -------------------------------------------------


def test_input_pins_name_every_file_a_sweeps_runs_read() -> None:
    assert sweep.input_pins(["clean", "zeros"]).keys() == {ds.TRAIN_IMAGES, ds.TRAIN_LABELS}
    rel = ds.corruption_tuning_images("fog")
    assert sweep.input_pins(["clean", "fog"])[rel] == ds.TUNING_PINS[rel]
    assert len(ds.TUNING_PINS) == len(ds.MNIST_C)


def test_tuning_pixels_are_the_first_rows_of_each_tuning_file(tmp_path: Path) -> None:
    n = 3
    images = np.arange(pr.TUNING_POOL * 4, dtype=np.uint64).reshape(-1, 2, 2) % 256
    ds.write_idx(tmp_path / "img", images.astype(np.uint8))
    rel = ds.corruption_tuning_images("fog")
    (tmp_path / rel).parent.mkdir(parents=True)
    ds.write_idx(tmp_path / rel, (255 - images).astype(np.uint8))
    inputs = pr.TuningInputs((1, 2, 3), "t", "r", str(tmp_path / "img"), "l", "z")
    clean = sweep.tuning_pixels(tmp_path, inputs, "clean")
    assert clean.shape == (n, 4) and np.allclose(clean * 255, images[:n].reshape(n, 4))
    assert np.allclose(sweep.tuning_pixels(tmp_path, inputs, "fog"), 1 - clean)
    assert not sweep.tuning_pixels(tmp_path, inputs, "zeros").any()


# --- the chains through a fake coastd ---------------------------------------------------


@dataclass
class Jobs:
    """What the fake coastd holds: a job is done on the second `list` after its submit. As in
    coastd, a `load_state_from` job starts at its source's completed step, and "auto" becomes a
    run_id of the job's own when its config.json is written; a concrete one is kept. `fail`
    names a job id that ends `failed`."""

    runs_root: Path
    data_link: Path
    fail: str = ""
    jobs: dict[str, dict[str, Any]] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class FakeClient(Client):
    state: Jobs = field(default_factory=lambda: Jobs(Path(), Path()))

    def request(self, obj: dict[str, Any]) -> dict[str, Any]:
        st = self.state
        reply: dict[str, Any] = {"ok": True, "proto_version": 1}
        if obj["cmd"] == "hash_inputs":
            return reply | {"sha256": {p: EXPECTED.get(p, "0" * 64) for p in obj["paths"]}}
        if obj["cmd"] == "submit":
            src = st.jobs.get(obj.get("load_state_from", ""), {"bucket": "done", "completed": 0})
            if src["bucket"] != "done":
                raise DaemonError("load_state_from: not done")
            n = len(st.jobs)
            st.jobs[f"job{n}"] = {"req": obj, "bucket": "active", "polls": 0,
                                  "base": src["completed"],
                                  "at": f"2026-11-08T10:{n // 60:02d}:{n % 60:02d}Z"}  # fmt: skip
            return reply | {"id": f"job{n}"}
        assert obj["cmd"] == "list", obj
        for job_id, job in st.jobs.items():
            job["polls"] += 1
            if job["bucket"] == "active" and job["polls"] == 2:
                self._finish(job_id, job)
        listed = [{"id": i, "bucket": j["bucket"], "submitted_at": j["at"],
                   "failure_reason": "", "user_message": j["req"].get("message", "")}
                  for i, j in st.jobs.items()]  # fmt: skip
        return reply | {"jobs": listed}

    def _finish(self, job_id: str, job: dict[str, Any]) -> None:
        st, req = self.state, job["req"]
        if job_id == st.fail:
            job["bucket"] = "failed"
            return
        cfg = json.loads(req["config_text"])
        run_id = cfg["run_id"] if cfg.get("run_id", "auto") != "auto" else f"run-{job_id}"
        job["completed"] = job["base"] + req["steps_target"]
        job["bucket"], job["run_id"] = "done", run_id
        run = st.runs_root / job_id
        run.mkdir()
        (run / "config.submitted.json").write_text(req["config_text"])
        (run / "config.json").write_text(json.dumps(cfg | {"run_id": run_id}))
        (run / "data").symlink_to(st.data_link)
        (run / "coast_runtime.json").write_text(json.dumps({
            "schema_version": "1", "coast_git_sha": req["ref"], "coast_git_dirty": False,
            "coast_build_type": "Release", "seed_offset": 0,
            "n_steps_requested": req["steps_target"], "unlimited": False,
            "n_steps_completed": job["completed"], "exit_reason": "completed_n_steps",
            "wall_clock_seconds": 1.0, "hostname": "fakehost", "gpu_adapter": None,
            "started_at_utc": "t", "completed_at_utc": "t",
            "observe": {"output_dir": "./observe", "groups": cfg["observe"]["groups"]},
            "datasets": {}, "retina": {}, "node_seeds": [], "savestate_outputs": [],
            "frozen": cfg["frozen"], "run_id": run_id,
        }))  # fmt: skip


@pytest.fixture
def jobs(tmp_path: Path) -> Jobs:
    (tmp_path / "runs").mkdir()
    (tmp_path / "base").mkdir()
    return Jobs(tmp_path / "runs", tmp_path / "base")


def daemon(st: Jobs) -> Daemon:
    info = MachineInfo(hostname="fakehost", runs_root=str(st.runs_root), data_link_ok=True,
                       dataset_base=str(st.data_link), tcp_port=51900)  # fmt: skip
    return Daemon(FakeClient("fake", 1, state=st), info, st.runs_root)


def make(ref: str = SHA, **changes: Any) -> sweep.Sweep:
    doc = spec_doc(**changes)
    s = sweep.load_spec(json.dumps(doc).encode())
    return sweep.make_sweep(s, doc, ref=ref, template_path=TEMPLATE, inputs=INPUTS)


def stub_report(path: Path, variant: str, ref: str) -> None:
    """What resume reads of a report: its one state's label and git SHA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"states": [{"label": variant, "git_sha": ref}]}))


@dataclass
class Points:
    """Stands in for `write_point`: records each point's run dirs and writes a report file,
    the one thing resume reads."""

    written: dict[tuple[str, int], dict[tuple[int, str], runs.RunDir]] = field(
        default_factory=lambda: {}
    )

    def __call__(self, out: Path, sw: sweep.Sweep, variant: str, length: int,
                 dirs: dict[tuple[int, str], runs.RunDir], _pixels: object) -> None:  # fmt: skip
        self.written[variant, length] = dict(dirs)
        stub_report(out / sw.report_path(variant, length), variant, sw.ref)


@pytest.fixture
def points(monkeypatch: pytest.MonkeyPatch) -> Points:
    p = Points()
    monkeypatch.setattr(sweep, "write_point", p)
    return p


def run(sw: sweep.Sweep, st: Jobs, out: Path, redo: tuple[str, ...] = ()) -> sweep.Doc:
    return sweep.run_sweep([daemon(st)], sw, out=out, pixels={}, expected_inputs=EXPECTED,
                           poll_s=0.0, timeout_s=60.0, retry_s=1.0, redo=redo)  # fmt: skip


def req(st: Jobs, job_id: str) -> dict[str, Any]:
    return st.jobs[job_id]["req"]


def described(st: Jobs) -> list[tuple[str, str | None, int]]:
    """(ledger key, load_state_from, steps_target) of every job, in submit order."""
    return [(r["message"].rsplit(" ", 1)[1], r.get("load_state_from"), r["steps_target"])
            for r in (j["req"] for j in st.jobs.values())]  # fmt: skip


def test_a_chain_trains_continues_under_its_run_id_and_runs_trials_off_each_length(
    jobs: Jobs, points: Points, tmp_path: Path
) -> None:
    sw = make()
    doc = run(sw, jobs, tmp_path / "out")
    by_key = {k: (i, src, n) for i, (k, src, n) in zip(jobs.jobs, described(jobs), strict=True)}
    # every chain's first training job is submitted before anything else
    assert [k for k, _, _ in described(jobs)][:4] == [
        "base/100/0/train", "base/100/1/train", "strength=5/100/0/train", "strength=5/100/1/train"
    ]  # fmt: skip
    assert len(jobs.jobs) == 2 * 2 * (2 + 2 * 2)  # per chain: 2 training jobs, 2 x 2 trials
    for v in sw.variants:
        for s in (0, 1):
            train, src, steps = by_key[f"{v}/100/{s}/train"]
            assert (src, steps) == (None, 100)
            assert req(jobs, train)["priority"] == 0  # first training jobs: the default
            cont, src, steps = by_key[f"{v}/300/{s}/train"]
            assert (src, steps) == (train, 200)  # steps_target counts from the loaded state
            assert req(jobs, cont)["priority"] == sweep.FOLLOW_UP_PRIORITY > 0  # runs first
            first, then = (json.loads(req(jobs, j)["config_text"]) for j in (train, cont))
            assert "run_id" not in first and then["run_id"] == jobs.jobs[train]["run_id"]
            assert jobs.jobs[cont]["run_id"] == jobs.jobs[train]["run_id"]  # one run
            assert then == first | {"run_id": then["run_id"]} and then["frozen"] is False
            for length, source in ((100, train), (300, cont)):
                for c in ("clean", "zeros"):
                    trial, src, steps = by_key[f"{v}/{length}/{s}/{c}"]
                    assert (src, steps) == (source, 2 * pr.FROZEN_STEPS_PER_IMAGE)
                    assert req(jobs, trial)["priority"] == sweep.FOLLOW_UP_PRIORITY
                    cfg = json.loads(req(jobs, trial)["config_text"])
                    window = {"ranges": [[length, length + 29]], "combine": "or"}
                    assert cfg["observe"]["groups"] == {"population": window, "input": window}
                    d = cfg["datasets"]["mnist"]["default"]
                    assert d["index_manifest"] == INPUTS.tuning_manifest
                    want = INPUTS.images if c == "clean" else INPUTS.zeros_images
                    assert (d["image_file"], d["label_file"]) == (want, INPUTS.labels)
                    assert cfg["frozen"] is True and "run_id" not in cfg
        strength = json.loads(req(jobs, by_key[f"{v}/100/0/train"][0])["config_text"])
        assert strength["ngm25"]["inhibitions"]["ab_pinned"]["strength"] == (
            10.0 if v == "base" else 5)  # fmt: skip
    # each point's report reads its seeds' training run at that length and their trials
    assert set(points.written) == {(v, n) for v in sw.variants for n in (100, 300)}
    dirs = points.written["base", 300]
    assert set(dirs) == {(s, c) for s in (0, 1) for c in ("train", "clean", "zeros")}
    assert dirs[0, "train"].root.name == by_key["base/300/0/train"][0]
    on_disk = msgspec.json.decode((tmp_path / "out" / "sweep.json").read_bytes(), type=sweep.Doc)
    assert on_disk == doc
    assert (doc.schema, doc.ref, doc.daemons) == ("coast-eval-sweep/1", SHA, ["fakehost:51900"])
    assert doc.spec == spec_doc() and doc.variants == {"base": {}, "strength=5": {STRENGTH: 5}}
    assert doc.aliases == {}
    assert doc.template_sha256 == ds.sha256_file(TEMPLATE)
    assert [(p.variant, p.length, p.report, p.seeds, p.status) for p in doc.points] == [
        ("base", 100, "base/100/report.json", [0, 1], "done"),
        ("base", 300, "base/300/report.json", [0, 1], "done"),
        ("strength=5", 100, "strength=5/100/report.json", [0, 1], "done"),
        ("strength=5", 300, "strength=5/300/report.json", [0, 1], "done"),
    ]


def test_a_rerun_skips_points_with_a_report_and_reattaches_to_the_rest(
    jobs: Jobs, points: Points, tmp_path: Path
) -> None:
    sw, out = make(), tmp_path / "out"
    run(sw, jobs, out)
    submitted = len(jobs.jobs)
    run(sw, jobs, out)  # every report exists: nothing to do
    assert len(jobs.jobs) == submitted
    (out / "strength=5" / "300" / "report.json").unlink()
    points.written.clear()
    doc = run(sw, jobs, out)
    assert len(jobs.jobs) == submitted  # its jobs are adopted from the ledger, not resubmitted
    assert list(points.written) == [("strength=5", 300)]
    assert all(p.status == "done" for p in doc.points)


def test_an_out_dir_of_another_sweep_is_refused_not_taken_as_done(
    jobs: Jobs, points: Points, tmp_path: Path
) -> None:
    out, other = tmp_path / "out", "1" * 40
    run(make(), jobs, out)
    first = json.loads((out / "sweep.json").read_text())["fingerprint"]
    (out / "ledger.json").unlink()  # as a refused ledger's message suggests
    submitted = len(jobs.jobs)
    with pytest.raises(ValueError, match=r"another sweep's output \(fingerprint " + first):
        run(make(ref=other), jobs, out)
    (out / "sweep.json").unlink()  # without the index, each report still names its ref
    with pytest.raises(ValueError, match=rf"base/100/report\.json reports .*, not base at {other}"):
        run(make(ref=other), jobs, out)
    assert len(jobs.jobs) == submitted  # nothing was submitted for the other sweep
    doc = run(make(), jobs, out)  # the same sweep resumes: every point is done
    assert len(jobs.jobs) == submitted and all(p.status == "done" for p in doc.points)
    assert doc.fingerprint == first


def test_without_a_ledger_a_point_with_a_report_runs_no_trials(
    jobs: Jobs, points: Points, tmp_path: Path
) -> None:
    sw, out = make(), tmp_path / "out"
    for v in sw.variants:
        stub_report(out / v / "100" / "report.json", v, sw.ref)
    run(sw, jobs, out)
    keys = [k for k, _, _ in described(jobs)]
    assert not [k for k in keys if "/100/" in k and not k.endswith("/train")]
    assert len(keys) == 2 * 2 * (2 + 2)  # the training chain is still needed to reach 300
    assert set(points.written) == {(v, 300) for v in sw.variants}


def test_a_failed_job_marks_its_point_failed_and_a_rerun_submits_it_afresh(
    jobs: Jobs, points: Points, tmp_path: Path
) -> None:
    sw, out = make(axes={}), tmp_path / "out"
    jobs.fail = "job1"  # base/100/1/train
    with pytest.raises(DaemonError, match="ended in failed") as e:
        run(sw, jobs, out)
    assert "re-attach" in "\n".join(e.value.__notes__)
    doc = msgspec.json.decode((out / "sweep.json").read_bytes(), type=sweep.Doc)
    assert [(p.length, p.status) for p in doc.points] == [(100, "failed"), (300, "pending")]
    jobs.fail = ""
    doc = run(sw, jobs, out)
    assert [p.status for p in doc.points] == ["done", "done"]
    retrained = [i for i, (k, _, _) in zip(jobs.jobs, described(jobs), strict=True)
                 if k == "base/100/1/train"]  # fmt: skip
    assert len(retrained) == 2


def test_redo_expands_a_training_key_over_the_rest_of_its_chain() -> None:
    sw = make(axes={})
    got = sweep.redo_keys(sw, ["base/100/1/train", "base/300/0/clean"], set())
    assert got == {"base/300/0/clean"} | {f"base/{n}/1/{c}" for n in (100, 300)
                                          for c in ("train", "clean", "zeros")}  # fmt: skip
    assert sweep.redo_keys(sw, ["base/300/1/train"], set()) == {
        "base/300/1/train", "base/300/1/clean", "base/300/1/zeros"}  # fmt: skip
    with pytest.raises(ValueError, match="names no job of this sweep"):
        sweep.redo_keys(sw, ["base/200/1/train"], set())
    with pytest.raises(ValueError, match=r"base/300/report\.json exists"):
        sweep.redo_keys(sw, ["base/300/1/clean"], {("base", 300)})


def test_redo_of_a_training_key_is_refused_if_a_later_length_has_a_report() -> None:
    """The later point is skipped, so its report would stay from the old chain while the
    earlier length trains anew: the output would mix two training runs of one chain."""
    sw = make(axes={})
    with pytest.raises(ValueError, match=r"base/300/report\.json exists; remove it"):
        sweep.redo_keys(sw, ["base/100/1/train"], {("base", 300)})
    with pytest.raises(ValueError, match=r"base/100/report\.json, base/300/report\.json exist"):
        sweep.redo_keys(sw, ["base/100/1/train"], {("base", 100), ("base", 300)})
    # a trial redoes only its own point, which has no report
    assert sweep.redo_keys(sw, ["base/100/1/clean"], {("base", 300)}) == {"base/100/1/clean"}


def test_the_fingerprint_follows_the_config_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    d = [daemon(Jobs(Path("r"), Path("b")))]
    a = sweep.fingerprint(d, make(), EXPECTED)
    assert a == sweep.fingerprint(d, make(), EXPECTED)
    assert a != sweep.fingerprint(d, make(lengths=[100, 400]), EXPECTED)
    monkeypatch.setattr(pr, "FROZEN_STEPS_PER_IMAGE", 14)  # an edited protocol
    assert a != sweep.fingerprint(d, make(), EXPECTED)


def test_make_sweep_needs_a_full_sha_and_a_template_without_a_fixed_run_id(
    tmp_path: Path,
) -> None:
    s = spec()
    with pytest.raises(ValueError, match="full commit SHA"):
        sweep.make_sweep(s, {}, ref="main", template_path=TEMPLATE, inputs=INPUTS)
    fixed = tmp_path / "t.json"
    fixed.write_text(json.dumps(template() | {"run_id": "r1"}))
    with pytest.raises(ValueError, match="run_id must be"):
        sweep.make_sweep(s, {}, ref=SHA, template_path=fixed, inputs=INPUTS)


def test_write_point_labels_the_state_by_variant_and_redo_keys_by_point(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sw = make(axes={})
    seen: list[str] = []

    def measure_state(label: str, *_: object, **__: object) -> rp.State:
        seen.append(label)
        return rp.State(label, SHA, None, 1, {0: {}}, {0: []}, "t.json")

    monkeypatch.setattr(rp, "measure_state", measure_state)
    pixels = {"clean": np.ones((2, 4)), "zeros": np.zeros((2, 4))}
    sweep.write_point(tmp_path, sw, "base", 300, {}, pixels)
    assert seen == ["base/300"]  # a refusal's note names base/300/<seed>/<condition>
    r = msgspec.json.decode((tmp_path / "base" / "300" / "report.json").read_bytes(),
                            type=rp.Report)  # fmt: skip
    assert [s.label for s in r.states] == ["base"]
    assert (r.protocol.train_steps, r.protocol.train_images) == (300, pr.TRAIN_COUNT - 2)
    assert r.protocol.test_indices == [0, 1]  # rows of the tuning files
    assert "2 tuning images held out of the train set" in (
        tmp_path / "base" / "300" / "report.md").read_text()  # fmt: skip


# --- the command line -------------------------------------------------------------------


def load_script() -> Any:
    path = ROOT / "scripts" / "eval_sweep.py"
    spec_ = importlib.util.spec_from_file_location("eval_sweep", path)
    assert spec_ is not None and spec_.loader is not None
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"conditions": ["clean", "fog"]}, "1 of the 15 MNIST-C"),  # no report could sum it
        ({"seeds": []}, "seeds must be"),
    ],
)
def test_the_command_line_refuses_a_sweep_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    changes: dict[str, Any],
    match: str,
) -> None:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec_doc(**changes)))
    monkeypatch.setattr("sys.argv", ["eval_sweep.py", "--spec", str(path), "--ref", "x", SHA,
                                     "--daemon", "h", "--dataset-base", "b", "--data-root", "d",
                                     "--out", str(tmp_path / "out")])  # fmt: skip
    with pytest.raises(SystemExit) as e:
        load_script().main()
    assert e.value.code == 2 and match in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
