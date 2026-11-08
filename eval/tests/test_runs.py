"""Run-directory manifest: parsing, the provenance check and the observe location, on
manifests shaped like a real `coast --emit-run-manifest` output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgspec
import pytest

from coast_eval import runs

SHA = "796f973ffc4fa3f4ddcf1ca563398d224244928f"


def manifest(**changes: Any) -> dict[str, Any]:
    """A frozen trial of 360 steps loaded from a state at step 2000."""
    window = {"ranges": [[2000, 2359]], "combine": "or"}
    m: dict[str, Any] = {
        "schema_version": "1",
        "coast_git_sha": SHA,
        "coast_git_dirty": False,
        "coast_build_type": "Release",
        "seed_offset": 0,
        "n_steps_requested": 360,
        "frozen": True,
        "unlimited": False,
        "n_steps_completed": 2360,
        "exit_reason": "completed_n_steps",
        "wall_clock_seconds": 0.21,
        "hostname": "host",
        "gpu_adapter": None,
        "started_at_utc": "2026-11-03T15:43:38Z",
        "completed_at_utc": "2026-11-03T15:43:38Z",
        "observe": {
            "output_dir": "./observe_out",
            "groups": {"population": window, "input": window},
        },
        "datasets": {"mnist": {"default": {"change_interval": 15, "shuffle": False}}},
        "retina": {},
        "node_seeds": [{"id": 0, "label": "mnist", "effective_seed": 2}],
        "savestate_outputs": [{"path": "./final_state.h5", "step": 2360, "bytes": 1024}],
    }
    return m | changes


def write(tmp_path: Path, m: dict[str, Any]) -> runs.RunDir:
    run = runs.RunDir(tmp_path)
    run.manifest.write_text(json.dumps(m))
    return run


def test_reads_a_real_shaped_manifest(tmp_path: Path) -> None:
    m = runs.read_manifest(write(tmp_path, manifest()))
    assert (m.n_steps_requested, m.n_steps_completed) == (360, 2360)
    assert m.observe.groups is not None
    assert m.observe.groups["input"].ranges == [(2000, 2359)]


@pytest.mark.parametrize(
    "changes",
    [
        {"new_field": 1},  # unknown field
        {"coast_git_dirty": "false"},  # wrong type, not coerced
        {"coast_compiler": 16},
        {"n_steps_completed": 2360.0},
        {"schema_version": "2"},
        {"exit_reason": "done"},
    ],
)
def test_malformed_manifest_raises(tmp_path: Path, changes: dict[str, Any]) -> None:
    with pytest.raises(msgspec.ValidationError):
        runs.read_manifest(write(tmp_path, manifest(**changes)))


def test_toolchain_fields_are_read_and_optional(tmp_path: Path) -> None:
    """A binary built before the fields existed writes none; a newer one writes both."""
    old = runs.read_manifest(write(tmp_path, manifest()))
    assert (old.coast_compiler, old.coast_target_cpu) == (None, None)
    new = runs.read_manifest(
        write(tmp_path, manifest(coast_compiler="GNU 16.2.1", coast_target_cpu="znver5"))
    )
    assert (new.coast_compiler, new.coast_target_cpu) == ("GNU 16.2.1", "znver5")


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not exit cleanly"):
        runs.read_manifest(runs.RunDir(tmp_path))


def check(
    m: dict[str, Any], *, base_step: int, steps: int, sha: str = SHA, frozen: bool = True
) -> None:
    decoded = msgspec.json.decode(json.dumps(m), type=runs.Manifest)
    runs.check_provenance(decoded, git_sha=sha, base_step=base_step, steps=steps, frozen=frozen)


def test_steps_are_counted_from_the_loaded_state() -> None:
    check(manifest(), base_step=2000, steps=360)
    fresh = manifest(n_steps_requested=2000, n_steps_completed=2000)
    check(fresh, base_step=0, steps=2000)
    with pytest.raises(ValueError, match="completed step 2360, expected 0 \\+ 360"):
        check(manifest(), base_step=0, steps=360)
    with pytest.raises(ValueError, match="completed step 2000, expected 2000 \\+ 2000"):
        check(fresh, base_step=2000, steps=2000)


def test_a_resumed_launch_is_judged_by_the_absolute_count() -> None:
    # coastd resumed a frozen trial from its checkpoint at step 2200 with -n 160: still the
    # same 360 steps, and its sequence continued (same run_id, SPEC.md S.5.14). Training alike.
    check(manifest(n_steps_requested=160), base_step=2000, steps=360)
    with pytest.raises(ValueError, match="completed step 2359"):
        check(manifest(n_steps_completed=2359), base_step=2000, steps=360)
    trained = manifest(n_steps_requested=800, n_steps_completed=2000, frozen=False)
    check(trained, base_step=0, steps=2000, frozen=False)


def test_a_run_must_record_the_frozen_its_job_asked_for() -> None:
    with pytest.raises(ValueError, match="run was frozen, expected the opposite"):
        check(manifest(), base_step=2000, steps=360, frozen=False)
    with pytest.raises(ValueError, match="run was not frozen"):
        check(manifest(frozen=False), base_step=2000, steps=360)
    old = manifest()
    del old["frozen"]  # a coast from before the top-level frozen
    with pytest.raises(ValueError, match="records no `frozen`"):
        check(old, base_step=2000, steps=360)


@pytest.mark.parametrize(
    ("changes", "sha", "match"),
    [
        ({}, "0" * 40, "expected 0000"),
        ({"coast_git_dirty": True}, SHA, "dirty"),
        ({"exit_reason": "user_signal"}, SHA, "user_signal"),
        ({"unlimited": True}, SHA, "unlimited=True"),
    ],
)
def test_provenance_violations_raise(changes: dict[str, Any], sha: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check(manifest(**changes), base_step=2000, steps=360, sha=sha)


def test_observe_dir_follows_the_manifest(tmp_path: Path) -> None:
    run = write(tmp_path, manifest())
    m = runs.read_manifest(run)
    assert runs.observe_dir(run, m, ["population", "input"]) == tmp_path / "observe_out"
    elsewhere = tmp_path.parent / "elsewhere"
    run = write(tmp_path, manifest(observe=manifest()["observe"] | {"output_dir": str(elsewhere)}))
    assert runs.observe_dir(run, runs.read_manifest(run), ["input"]) == elsewhere


def test_observe_dir_raises_on_what_it_cannot_locate(tmp_path: Path) -> None:
    run = write(tmp_path, manifest())
    with pytest.raises(ValueError, match=r"\['apical'\] not in"):
        runs.observe_dir(run, runs.read_manifest(run), ["population", "apical"])
    run = write(tmp_path, manifest(observe={"output_dir": "./observe_out", "groups": None}))
    with pytest.raises(ValueError, match=r"\['input'\] not in \[\]"):
        runs.observe_dir(run, runs.read_manifest(run), ["input"])
    bare = manifest(observe=manifest()["observe"] | {"output_dir": "observe_out"})
    run = write(tmp_path, bare)
    with pytest.raises(ValueError, match=r"neither \./ nor absolute"):
        runs.observe_dir(run, runs.read_manifest(run), ["input"])


def test_a_run_is_reused_only_with_the_same_config_and_inputs(tmp_path: Path) -> None:
    """A frozen run of the --binary driver: it loads a training state and reads its test
    manifest through the run dir's `inputs` link."""
    inputs, run = tmp_path / "inputs", runs.RunDir(tmp_path / "frozen")
    inputs.mkdir()
    run.root.mkdir()
    (run.root / "inputs").symlink_to(inputs)
    (inputs / "test_indices.txt").write_text("7\n3\n")
    (inputs / "unused.txt").write_text("not named by the config\n")
    state = tmp_path / "train_final_state.h5"
    state.write_bytes(b"state trained at 5 steps per image")
    coast = tmp_path / "coast"
    coast.write_bytes(b"coast 796f973f built with clang")
    text = json.dumps({"datasets": {"m": {"index_manifest": "./inputs/test_indices.txt",
                                          "image_file": "./data/mnist/t10k"}}})  # fmt: skip

    def finish() -> bytes:  # what spawn does once coast has run
        record = runs.inputs_record(run, text, state, coast, 360)
        run.config.write_text(text)
        run.inputs_record.write_bytes(record)
        run.manifest.write_text(json.dumps(manifest()))
        return record

    record = json.loads(finish())
    assert set(record["inputs"]) == {"./inputs/test_indices.txt"}  # named, not just present
    assert runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))
    other = text.replace("t10k", "t10k-other")  # another config text
    assert not runs.reusable(run, other, runs.inputs_record(run, other, state, coast, 360))
    # The training run was redone (10 steps per image): its final state changed.
    state.write_bytes(b"state trained at 10 steps per image")
    assert not runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))
    finish()
    (inputs / "test_indices.txt").write_text("3\n7\n")  # the manifest's content changed
    assert not runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))
    finish()
    coast.write_bytes(b"coast 796f973f built with gcc")  # same git SHA, another binary
    assert not runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))
    finish()
    assert runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))
    # no state now
    assert not runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 360))
    run.inputs_record.unlink()  # a run dir from before the record existed
    assert not runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))
    finish()
    run.manifest.unlink()  # coast did not finish
    assert not runs.reusable(run, text, runs.inputs_record(run, text, state, coast, 360))


def test_a_finished_run_is_reused_at_its_own_length_only(tmp_path: Path) -> None:
    """--out reused with another --steps. A training config does not hold its length, so the
    text alone would reuse the shorter run, and the provenance check would then abort with a
    false report of a resume. Same length: reused, a rerun costs nothing."""
    coast = tmp_path / "coast"
    coast.write_bytes(b"coast")
    run = runs.RunDir(tmp_path / "train")
    run.root.mkdir()
    text = '{"image_file": "./data/mnist/train"}'
    run.config.write_text(text)
    run.inputs_record.write_bytes(runs.inputs_record(run, text, None, coast, 2000))
    run.manifest.write_text(json.dumps(manifest()))
    assert runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 2000))
    assert not runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 2100))
    assert not runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 1900))


def test_link_re_points_a_stale_link_and_another_data_root_redoes_the_run(tmp_path: Path) -> None:
    """--out reused with another --data-root: the run must read the new, verified root."""
    first, second, coast = tmp_path / "nas", tmp_path / "local", tmp_path / "coast"
    coast.write_bytes(b"coast")
    first.mkdir()
    second.mkdir()
    run = runs.RunDir(tmp_path / "run")
    run.root.mkdir()
    runs.link(run, "data", first)
    text = '{"image_file": "./data/mnist/t10k"}'
    record = runs.inputs_record(run, text, None, coast, 360)
    run.config.write_text(text)
    run.inputs_record.write_bytes(record)
    run.manifest.write_text(json.dumps(manifest()))
    runs.link(run, "data", first)  # same target: kept
    assert runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 360))
    runs.link(run, "data", second)
    assert (run.root / "data").resolve() == second.resolve()
    assert json.loads(runs.inputs_record(run, text, None, coast, 360))["data"] == str(
        second.resolve()
    )
    assert not runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 360))
    run.inputs_record.write_bytes(runs.inputs_record(run, text, None, coast, 360))  # the redone run
    (tmp_path / "alias").symlink_to(second)  # the same directory under another name: kept
    runs.link(run, "data", tmp_path / "alias")
    assert runs.reusable(run, text, runs.inputs_record(run, text, None, coast, 360))
    (run.root / "inputs").mkdir()  # not a link: refused, never replaced
    with pytest.raises(ValueError, match="not a symlink"):
        runs.link(run, "inputs", first)
