"""`scripts/eval_ab.py`'s `spawn`, against a stand-in coast: what it launches, where, and when
it reuses a finished run. The stand-in records its launches, so a test can tell a reused run
from a redone one; a marker file makes it fail before it writes a manifest, as coast would.
And its command line: what it refuses, `--discover`, and how `--ref` reaches the runner."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from coast_eval import datasets, protocol, runner, runs
from coast_eval.daemon import Client, DaemonSpec, Discovered, MachineInfo

SCRIPT = Path(__file__).parents[2] / "scripts" / "eval_ab.py"
CONFIG = json.dumps({"ngm25": {"neuron_groups": {"g": {"max_branch_level": 3}}}})
COAST = """#!/bin/sh
echo "$PWD" >> {launches}
printf '%s\\n' "$@" > argv.txt
echo "coast says hello"
test ! -e {fail} || exit 3
echo '{{}}' > coast_runtime.json
"""


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("eval_ab", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ab = load_script()


class Rig:
    """A stand-in coast, one run dir and a `data` directory to link in."""

    def __init__(self, tmp_path: Path) -> None:
        self.launches, self.fail = tmp_path / "launches", tmp_path / "fail"
        self.coast = tmp_path / "coast"
        self.coast.write_text(COAST.format(launches=self.launches, fail=self.fail))
        self.coast.chmod(0o755)
        self.run = runs.RunDir(tmp_path / "out" / "seed0" / "train")
        self.data = tmp_path / "data"
        self.data.mkdir()

    def spawn(
        self,
        steps: int = 100,
        text: str = CONFIG,
        state: Path | None = None,
        coast: Path | None = None,
    ) -> None:
        job = protocol.Job(seed=0, condition="train", steps=steps, config_text=text)
        links = {"data": self.data}
        assert ab.spawn(coast or self.coast, self.run, job, links, state) == self.run

    def count(self) -> int:
        return len(self.launches.read_text().splitlines()) if self.launches.exists() else 0

    def argv(self) -> list[str]:
        return (self.run.root / "argv.txt").read_text().splitlines()


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_spawn_runs_coast_in_the_run_dir(rig: Rig) -> None:
    rig.spawn()
    assert rig.launches.read_text().strip() == str(rig.run.root.resolve())
    assert rig.argv() == [
        *("-c", "config.json", "-n", "100"),
        *("--save-state", "final_state.h5", "--emit-run-manifest", "coast_runtime.json"),
    ]
    assert rig.run.config.read_text() == CONFIG
    assert (rig.run.root / "data").resolve() == rig.data.resolve()
    assert "coast says hello" in (rig.run.root / "coast.log").read_text()


def test_spawn_resolves_paths_that_are_relative_to_the_caller(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # coast runs in the run dir, where a path relative to the caller would not resolve.
    state = tmp_path / "state.h5"
    state.write_bytes(b"trained")
    monkeypatch.chdir(tmp_path)
    rig.spawn(state=Path("state.h5"), coast=Path("coast"))
    assert rig.argv()[-2:] == ["--load-state", str(state.resolve())]


def test_spawn_reuses_a_finished_run_only_while_nothing_that_decides_it_changed(
    rig: Rig, tmp_path: Path
) -> None:
    state = tmp_path / "state.h5"
    state.write_bytes(b"trained")
    rig.spawn(state=state)
    rig.spawn(state=state)
    assert rig.count() == 1
    rig.spawn(steps=200, state=state)  # no config holds the run length
    assert rig.count() == 2
    rig.spawn(steps=200, state=state, text=CONFIG.replace("3", "4"))
    assert rig.count() == 3
    state.write_bytes(b"retrained")
    rig.spawn(steps=200, state=state, text=CONFIG.replace("3", "4"))
    assert rig.count() == 4
    with rig.coast.open("a") as f:
        f.write("# rebuilt\n")
    rig.spawn(steps=200, state=state, text=CONFIG.replace("3", "4"))
    assert rig.count() == 5


def test_a_failed_run_raises_and_leaves_nothing_to_reuse(rig: Rig) -> None:
    rig.spawn()
    rig.fail.touch()
    with pytest.raises(subprocess.CalledProcessError):
        rig.spawn(steps=200)
    assert not rig.run.manifest.exists()  # the finished run's manifest does not outlive it
    rig.fail.unlink()
    rig.spawn(steps=200)  # same config and record as the failed launch, so only the
    assert rig.count() == 3  # missing manifest makes this a new run
    assert rig.run.manifest.exists()


# --- the command line: --daemon, --ports-file, --discover -------------------------------

SHA_A, SHA_B = "a" * 40, "b" * 40


def run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["eval_ab.py", *argv])
    ab.main()


RUN_ARGS = ("--data-root", "d", "--out", "o", "--report-dir", "r")


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (("--ref", "x", SHA_A, *RUN_ARGS), "--dataset-base and either --ports-file or --daemon"),
        (
            (
                "--ref",
                "x",
                SHA_A,
                "--dataset-base",
                "b",
                "--ports-file",
                "p",
                "--daemon",
                "h",
                *RUN_ARGS,
            ),
            "either --ports-file or --daemon",
        ),
        (("--ref", "x", SHA_A, "--daemon", "h", *RUN_ARGS), "--dataset-base"),
        (
            ("--binary", "x", "c", SHA_A, "--daemon", "h", *RUN_ARGS),
            "all --ref or all --binary|--ref only",
        ),
        (("--ref", "x", SHA_A, "--dataset-base", "b", "--daemon", "h:0", *RUN_ARGS), "port 0"),
        (
            ("--ref", "x", SHA_A, "--dataset-base", "b", "--daemon", "h"),
            "required: --data-root, --out, --report-dir",
        ),
    ],
)
def test_the_command_line_refuses_what_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, *argv)
    assert e.value.code == 2
    assert re.search(match, capsys.readouterr().err)


def test_daemon_without_ref_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--binary", "x", "c", SHA_A, "--ports-file", "p", *RUN_ARGS)
    assert "--ports-file, --daemon and --cancel are --ref only" in capsys.readouterr().err


def test_discover_lists_the_daemons_and_needs_no_run_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = [Discovered("alpha", "10.0.0.2", 52000, 1), Discovered("zeta", "10.0.0.9", 51900, 1)]
    monkeypatch.setattr(ab, "discover", lambda: list(found))
    run_main(monkeypatch, "--discover")
    out = capsys.readouterr().out.splitlines()
    assert [line.split()[:3] for line in out] == [
        ["alpha", "--daemon", "10.0.0.2:52000"],
        ["zeta", "--daemon", "10.0.0.9"],  # the default port is left out
    ]
    monkeypatch.setattr(ab, "discover", lambda: list[Discovered]())
    run_main(monkeypatch, "--discover")
    assert "no coastd answered" in capsys.readouterr().out


class Stop(Exception):
    """Ends main() once the runner has been called: what follows reads real runs."""


def test_ref_runs_on_every_named_daemon_in_canonical_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    files = [tmp_path / n for n in ("train.txt", "test.txt", "zeros")]
    for f in files:
        f.write_text(f.name)
    inputs = protocol.Inputs((0,), *map(str, files))

    def nothing(*_: object) -> None:
        return None

    def the_inputs(*_: object) -> protocol.Inputs:
        return inputs

    def shipped(i: protocol.Inputs, *_: object) -> protocol.Inputs:
        return i

    monkeypatch.setattr(ab.datasets, "verify_pins", nothing)
    monkeypatch.setattr(ab.protocol, "write_inputs", the_inputs)
    monkeypatch.setattr(ab.report, "condition_pixels", nothing)
    monkeypatch.setattr(ab.runner, "ship_inputs", shipped)
    connected: list[DaemonSpec] = []

    def connect(spec: DaemonSpec) -> runner.Daemon:
        connected.append(spec)
        info = MachineInfo(hostname=spec.host.upper(), runs_root=str(tmp_path),
                           data_link_ok=True, dataset_base="/data", tcp_port=spec.port)  # fmt: skip
        return runner.Daemon(Client(spec.host, spec.port), info, tmp_path)

    called: dict[str, Any] = {}

    def run_milestone1(daemons: list[runner.Daemon], states: list[runner.State], *_: Any,
                       **kw: Any) -> None:  # fmt: skip
        called.update(daemons=daemons, states=states, **kw)
        raise Stop

    monkeypatch.setattr(ab.runner, "connect", connect)
    monkeypatch.setattr(ab.runner, "run_milestone1", run_milestone1)
    with pytest.raises(Stop):
        run_main(monkeypatch, "--ref", "before", SHA_A, "--ref", "after", SHA_B,
                 "--daemon", "zz", "--daemon", "aa:52000", "--dataset-base", str(tmp_path),
                 "--seeds", "0", "1", "2", *RUN_ARGS)  # fmt: skip
    assert connected == [DaemonSpec("zz", 51900), DaemonSpec("aa", 52000)]
    assert [d.info.hostname for d in called["daemons"]] == ["AA", "ZZ"]  # not naming order
    assert [(s.label, s.ref) for s in called["states"]] == [("before", SHA_A), ("after", SHA_B)]
    assert called["seeds"] == [0, 1, 2]
    assert called["ledger"] == Path("o") / "ledger.json"  # in --out: rerun to resume
    expected = called["expected_inputs"]
    assert expected.keys() >= datasets.PINS.keys()
    assert {k for k in expected if k.startswith("eval/")} == {
        f"eval/m1-60000-1000/{f.name}" for f in files
    }
    out = capsys.readouterr().out
    assert "AA (aa:52000): seeds [0, 2]" in out and "ZZ (zz:51900): seeds [1]" in out

    cancelled: list[str] = []

    def cancel_evaluation(daemons: list[runner.Daemon], fp: str) -> list[Any]:
        cancelled.append(fp)
        return [(daemons[0], "j1", "before/0/train")]

    monkeypatch.setattr(ab.runner, "cancel_evaluation", cancel_evaluation)
    run_main(monkeypatch, "--ref", "before", SHA_A, "--ref", "after", SHA_B,
             "--daemon", "zz", "--daemon", "aa:52000", "--dataset-base", str(tmp_path),
             "--seeds", "0", "1", "2", "--cancel", *RUN_ARGS)  # fmt: skip
    expected_fp = runner.fingerprint(called["daemons"], called["states"], inputs,
                                     called["expected_inputs"], [0, 1, 2], 1_000_000, 1,
                                     list(protocol.CONDITIONS))  # fmt: skip
    assert cancelled == [expected_fp]  # the same evaluation as the run: same fingerprint
    out = capsys.readouterr().out
    assert "cancelled AA (aa:52000): j1 (before/0/train)" in out and "1 active job(s)" in out
