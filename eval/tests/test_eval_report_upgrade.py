"""`scripts/eval_report_upgrade.py`: a schema /1 report becomes /2 beside it, or is refused."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import msgspec
import pytest

from coast_eval import report as rp

SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_report_upgrade.py"
SHA = "796f973ffc4fa3f4ddcf1ca563398d224244928f"


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("eval_report_upgrade", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


up = load_script()

MEASURES = {
    "kendall_tau_a": 0.39,
    "model/mnistc_mean": 0.83,
    "model/mnistc_worst": 0.52,
    "separation": 0.23,
    "never_wins": 0.0,
    "domination": 0.19,
    "code_size/clean": 5.5,
    "random/1/mnistc_mean": 0.80,
    "random/2/mnistc_mean": 0.81,
}


def v1(extra_lead: int = 0) -> dict[str, Any]:
    """A report as the /1 code before 5f0e17d8 wrote it: protocol.extra_lead, no chance
    levels, string seeds."""
    state = rp.State("now", SHA, None, 1, {0: MEASURES, 2: MEASURES}, {0: [], 2: []}, "t.json")
    p = rp.Protocol(
        train_steps=1_000_000,
        train_steps_per_image=1,
        train_images=60_000,
        test_indices=[9, 14, 20],
        conditions=["clean"],
        projection_seeds=[1, 2],
        frac=0.3,
    )
    data = json.loads(
        rp.to_json(rp.build(p, {"pixel/clean": 1.0, "pixel/mnistc_mean": 0.89}, [state]))
    )
    data["schema"] = "coast-eval-report/1"
    data["protocol"]["extra_lead"] = extra_lead
    return data


def write(tmp_path: Path, data: dict[str, Any], name: str = "report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n")
    return path


def test_a_v1_report_converts_beside_itself_and_decodes_as_v2(tmp_path: Path) -> None:
    path = write(tmp_path, v1())
    before = path.read_bytes()
    out = up.convert(path)
    assert out == tmp_path / "report.v2.json" and path.read_bytes() == before
    r = msgspec.json.decode(out.read_bytes(), type=rp.Report)
    assert r.schema == rp.SCHEMA == "coast-eval-report/2"
    assert r.states[0].measures == {0: MEASURES, 2: MEASURES}  # seeds back to ints, values kept
    assert r.protocol.train_steps == 1_000_000 and r.pixel["pixel/clean"] == 1.0
    md = rp.to_markdown(r)  # without chance levels the Markdown says so, not the chance note
    assert "no chance levels" in md and "`separation_chance` and" not in md
    assert up.output_path(tmp_path / "x.json") == tmp_path / "x.v2.json"


def test_refused_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="extra_lead 1"):
        up.upgrade(v1(extra_lead=1))
    with pytest.raises(ValueError, match="already coast-eval-report/2"):
        up.upgrade({**v1(), "schema": rp.SCHEMA})
    with pytest.raises(ValueError, match="not a coast-eval-report/1"):
        up.upgrade({**v1(), "schema": "coast-eval-settling/1"})
    stray = v1()
    stray["protocol"]["lead"] = 2  # a field neither schema has: the strict decoder refuses it
    with pytest.raises(msgspec.ValidationError, match="lead"):
        up.upgrade(stray)
    path = write(tmp_path, v1())
    up.convert(path)
    first = up.output_path(path).read_bytes()
    with pytest.raises(ValueError, match="exists"):  # never overwrites, not even its own output
        up.convert(path)
    assert up.output_path(path).read_bytes() == first


def test_a_v1_report_without_extra_lead_converts_too(tmp_path: Path) -> None:
    data = v1()
    del data["protocol"]["extra_lead"]  # /1 written before extra_lead existed
    assert up.upgrade(data).schema == rp.SCHEMA


def test_a_v1_report_written_after_5f0e17d8_keeps_its_chance_levels() -> None:
    data = v1()
    del data["protocol"]["extra_lead"]  # dropped in 1657fa3c, before the label changed
    chance = {"0": (0.11, 0.07), "2": (0.13, 0.05)}
    for seed, (sep, dom) in chance.items():
        m = data["states"][0]["measures"][seed]
        m["separation_chance"], m["domination_chance"] = sep, dom
    r = up.upgrade(data)
    assert r.schema == rp.SCHEMA
    got = {
        s: (m["separation_chance"], m["domination_chance"]) for s, m in r.states[0].measures.items()
    }
    assert got == {0: chance["0"], 2: chance["2"]}
    assert "`separation_chance` and" in rp.to_markdown(r)  # the chance note, not "no chance levels"
