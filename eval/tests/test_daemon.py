"""The coastd client and the milestone-1 sequencing, against a fake JSON-lines daemon.

coastd is the external boundary, so it is the one thing faked: a TCP server on port 0,
started per test, whose replies each test scripts.
"""

from __future__ import annotations

import fcntl
import itertools
import json
import socket
import socketserver
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import msgspec
import numpy as np
import pytest

from coast_eval import datasets as ds
from coast_eval import protocol as pr
from coast_eval import runner
from coast_eval.daemon import (
    Client,
    DaemonError,
    DaemonSpec,
    DaemonUnreachable,
    JobSummary,
    _keepalive,  # pyright: ignore[reportPrivateUsage]
    discover,
    parse_daemon_spec,
    run_dir,
    verify_daemon_inputs,
)

Respond = Callable[[dict[str, Any]], bytes]  # a reply without a final newline: then hang up
SHA = "796f973ffc4fa3f4ddcf1ca563398d224244928f"
# The fake daemon runs in this process and answers at once: a budget on it is a
# stall limit for a frozen test, not a performance bound (CLAUDE.md "Testing").
STALL_S = 60.0
TEMPLATE = Path(__file__).parents[1] / "configs" / "mnist_ngm25_frozen_eval.json"


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, respond: Respond) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.respond = respond
        self.received: list[bytes] = []


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = cast(_Server, self.server)
        for raw in self.rfile:
            server.received.append(raw)
            out = server.respond(json.loads(raw))
            self.wfile.write(out)
            if not out.endswith(b"\n"):
                return


Serve = Callable[[Respond], tuple[Client, list[bytes]]]


@pytest.fixture
def serve() -> Iterator[Serve]:
    servers: list[_Server] = []

    def start(respond: Respond) -> tuple[Client, list[bytes]]:
        server = _Server(respond)
        servers.append(server)
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        return Client("127.0.0.1", server.server_address[1]), server.received

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def ok(**data: Any) -> bytes:
    return json.dumps({"ok": True, "proto_version": 1, **data}).encode() + b"\n"


def summary(job_id: str, bucket: str, reason: str = "", **changes: Any) -> dict[str, Any]:
    return {
        "id": job_id, "ref": SHA, "submitted_at": "2026-11-04T09:00:00Z", "user_message": "",
        "status": bucket if bucket != "active" else "running", "bucket": bucket,
        "steps_done": 0, "live_step": 0, "steps_target": 2000, "checkpoint": "",
        "priority": 0, "failure_reason": reason, "cancel_requested": False,
    } | changes  # fmt: skip


def test_submit_sends_one_line_with_exactly_the_given_fields(serve: Serve) -> None:
    client, received = serve(lambda req: ok(id=f"job-{len(req)}"))
    text = '{\n  "sim": {"nodes": []}\n}\n'  # layout must survive byte-exact
    assert client.submit("main", text, 360, priority=2) == "job-5"
    assert client.submit(SHA, text, 360, priority=0, load_state_from="train-0") == "job-6"
    first, second = (json.loads(r) for r in received)
    assert all(r.endswith(b"}\n") and r.count(b"\n") == 1 for r in received)
    assert first == {"cmd": "submit", "ref": "main", "config_text": text,
                     "steps_target": 360, "priority": 2}  # fmt: skip
    assert second == first | {"ref": SHA, "priority": 0, "load_state_from": "train-0"}
    with pytest.raises(ValueError, match="0 is unbounded"):
        client.submit(SHA, text, 0, priority=0)


def test_daemon_rejection_raises_with_its_message(serve: Serve) -> None:
    error = "submit: load_state_from 'x' is not a job in done/"
    client, _ = serve(lambda _: json.dumps({"ok": False, "error": error, "proto_version": 1})
                      .encode() + b"\n")  # fmt: skip
    with pytest.raises(DaemonError, match="is not a job in done/"):
        client.submit(SHA, "{}", 10, priority=0, load_state_from="x")


def test_requests_wait_on_tcp_keepalive_not_on_a_deadline() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        _keepalive(s, 10, 5, 3)
        assert s.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) != 0  # BSD: the bit, 8
        idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", 0))
        got = [s.getsockopt(socket.IPPROTO_TCP, o) for o in (idle, socket.TCP_KEEPINTVL,
                                                             socket.TCP_KEEPCNT)]  # fmt: skip
        assert got == [10, 5, 3]
        assert s.gettimeout() is None


def test_a_connection_closed_before_any_reply_counts_as_unreachable(serve: Serve) -> None:
    """A daemon that stops or restarts mid-request closes the connection: `list` retries
    that (`runner._list`); a reply cut off midway is still a protocol error."""
    client, _ = serve(lambda _: b"")
    with pytest.raises(DaemonUnreachable, match="closed before any reply"):
        client.list_jobs()
    cut, _ = serve(lambda _: b'{"ok": true')
    with pytest.raises(DaemonError, match="before a complete line") as e:
        cut.list_jobs()
    assert not isinstance(e.value, DaemonUnreachable)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"not json\n", "not JSON"),
        (b"[1, 2]\n", "boolean 'ok'"),
        (b'{"jobs": []}\n', "boolean 'ok'"),
        (b'{"ok": "true", "proto_version": 1}\n', "boolean 'ok'"),
        (b'{"ok": true, "proto_version": 1, "jobs": []', "before a complete line"),
        (b"", "before any reply"),
        (b'{"ok": true, "jobs": []}\n', "proto_version None"),
        (b'{"ok": true, "proto_version": 2, "jobs": []}\n', "proto_version 2"),
        (b'{"ok": true, "proto_version": 1}\n', "missing required field `jobs`"),
        (ok(jobs=[summary("a", "done", failure_reason=5)]), r"Expected `str`.*failure_reason"),
        (ok(jobs=[summary("a", "finished")]), "bucket"),
    ],
)
def test_malformed_replies_raise(serve: Serve, raw: bytes, match: str) -> None:
    client, _ = serve(lambda _: raw)
    with pytest.raises(DaemonError, match=match):
        client.list_jobs()


def test_unknown_reply_fields_are_ignored(serve: Serve) -> None:
    # protocol.h: added fields do not bump proto_version, so a compatible daemon sends them
    client, _ = serve(lambda _: ok(jobs=[summary("a", "done", load_state_step=2000)], new=1))
    (job,) = client.list_jobs()
    assert (job.id, job.bucket) == ("a", "done")


def test_connection_refused_raises() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again before the client connects
    with pytest.raises(DaemonError, match=f"127.0.0.1:{port}"):
        Client("127.0.0.1", port).list_jobs()


def test_from_ports_file(tmp_path: Path) -> None:
    path = tmp_path / "coastd.ports"
    path.write_text('{"pid": 41, "tcp_port": 50123, "udp_port": 0}')
    assert Client.from_ports_file(path) == Client("127.0.0.1", 50123)
    for bad, match in [
        ('{"pid": 41}', "missing required field `tcp_port`"),
        ('{"pid": 41, "tcp_p', "truncated"),  # torn write
        ("", "truncated"),
        ('{"tcp_port": "50123"}', "Expected `int`"),
        ("[50123]", "Expected `object`"),
    ]:
        path.write_text(bad)
        with pytest.raises(DaemonError, match=match):
            Client.from_ports_file(path)


H_A, H_B = "a" * 64, "b" * 64


def refuse(error: str) -> bytes:
    return json.dumps({"ok": False, "error": error, "proto_version": 1}).encode() + b"\n"


def test_machine_info_reads_dataset_base_and_names_an_old_daemon(serve: Serve) -> None:
    info = {"hostname": "h", "runs_root": "/r", "data_link_ok": True, "tcp_port": 51900}
    client, _ = serve(lambda _: ok(**info, dataset_base="/mnt/NAS/datasets"))
    assert client.machine_info().dataset_base == "/mnt/NAS/datasets"
    old, _ = serve(lambda _: ok(**info))
    with pytest.raises(DaemonError, match="daemon too old"):
        old.machine_info()


def test_hash_inputs_sends_the_paths_and_returns_the_hashes(serve: Serve) -> None:
    client, received = serve(lambda req: ok(sha256=dict.fromkeys(req["paths"], H_A)))
    assert client.hash_inputs(["mnist/x", "eval/m1/y"]) == {"mnist/x": H_A, "eval/m1/y": H_A}
    assert json.loads(received[0]) == {"cmd": "hash_inputs", "paths": ["mnist/x", "eval/m1/y"]}
    with pytest.raises(ValueError, match="at least one path"):
        client.hash_inputs([])


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (ok(sha256={"x": H_A}), r"hashes for \['x'\], asked for \['x', 'y'\]"),  # one missing
        (ok(sha256={"x": H_A, "y": H_B, "z": H_A}), "asked for"),  # one extra
        (ok(sha256={"x": H_A, "y": H_A.upper()}), r"not a SHA256 hex digest for \['y'\]"),
        (ok(sha256={"x": H_A, "y": H_A[:63]}), "not a SHA256 hex digest"),
        (ok(sha256={"x": H_A, "y": 5}), "Expected `str`"),
        (ok(hashes={}), "missing required field `sha256`"),
        (refuse("unknown command 'hash_inputs'"), "daemon too old"),
        (refuse("hash_inputs: 'y' contains '..'"), "contains '..'"),
    ],
)
def test_hash_inputs_raises_on_a_reply_it_cannot_trust(
    serve: Serve, reply: bytes, match: str
) -> None:
    client, _ = serve(lambda _: reply)
    with pytest.raises(DaemonError, match=match):
        client.hash_inputs(["x", "y"])


def test_verify_daemon_inputs_names_every_mismatch(serve: Serve) -> None:
    client, _ = serve(lambda _: ok(sha256={"x": H_A, "y": H_A, "z": H_B}))
    verify_daemon_inputs(client, {"x": H_A, "y": H_A, "z": H_B})
    with pytest.raises(DaemonError) as e:
        verify_daemon_inputs(client, {"x": H_B, "y": H_A, "z": H_A})
    assert f"x: daemon {H_A}, expected {H_B}" in str(e.value)
    assert f"z: daemon {H_B}, expected {H_A}" in str(e.value)
    assert "y:" not in str(e.value)


@pytest.mark.parametrize(
    ("spec", "want"),
    [
        ("majorcrunch.local:51900", DaemonSpec("majorcrunch.local", 51900)),
        ("192.168.100.81:1", DaemonSpec("192.168.100.81", 1)),
        ("[::1]:51900", DaemonSpec("::1", 51900)),
        ("h:51900=/mnt/runs", DaemonSpec("h", 51900, Path("/mnt/runs"))),
        ("h:51900=rel/a=b", DaemonSpec("h", 51900, Path("rel/a=b"))),
        ("majorcrunch.local", DaemonSpec("majorcrunch.local", 51900)),  # the default port
        ("h=/mnt/runs", DaemonSpec("h", 51900, Path("/mnt/runs"))),
        ("[::1]", DaemonSpec("::1", 51900)),
    ],
)
def test_parse_daemon_spec(spec: str, want: DaemonSpec) -> None:
    assert parse_daemon_spec(spec) == want


@pytest.mark.parametrize(
    "spec",
    ["h:", ":51900", "h:0", "h:65536", "h:port", "::1", "::1:51900", "h=", "h:51900=", "h p", ""],
)
def test_parse_daemon_spec_refuses(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_daemon_spec(spec)


def test_run_dir(tmp_path: Path) -> None:
    assert run_dir("20261104-mnist-1", tmp_path).root == tmp_path / "20261104-mnist-1"
    for bad in ("", ".", "..", "../other", "a/b"):
        with pytest.raises(DaemonError, match="single path component"):
            run_dir(bad, tmp_path)


def test_ship_inputs(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "mnist").mkdir(parents=True)
    ds.write_idx(root / ds.TRAIN_LABELS, np.zeros(12, dtype=np.uint8))
    ds.write_idx(root / ds.TEST_IMAGES, np.full((20, 28, 28), 9, dtype=np.uint8))
    local = pr.write_inputs(tmp_path / "inputs", root, range(12), [7, 3])
    base = tmp_path / "dataset_base"
    shipped = runner.ship_inputs(local, base, "m1")
    assert shipped == pr.Inputs(
        (7, 3),
        "./data/eval/m1/train_indices.txt",
        "./data/eval/m1/test_indices.txt",
        "./data/eval/m1/zeros-images-idx3",
    )
    for path in (local.train_manifest, local.test_manifest, local.zeros_images):
        assert (base / "eval" / "m1" / Path(path).name).read_bytes() == Path(path).read_bytes()
    assert runner.ship_inputs(local, base, "m1") == shipped  # same content: a no-op
    other = pr.write_inputs(tmp_path / "other", root, range(12), [3, 7])
    with pytest.raises(ValueError, match=r"test_indices\.txt: exists with content different"):
        runner.ship_inputs(other, base, "m1")
    with pytest.raises(ValueError, match="single path component"):
        runner.ship_inputs(local, base, "../m1")


class FakeDaemon:
    """What run_milestone1 relies on: `load_state_from` must name a done job, a job is
    done on the second `list` after its submit, and a done job's run dir holds the
    submitted config, a manifest naming the daemon's host, and the `data` link to the
    daemon's dataset_base (`data_link`; `linked_elsewhere` names a job id whose link points
    to another directory). A job in `resumed` finishes as if coastd had resumed it halfway:
    its last launch requested half of its steps. A manifest records the config's top-level
    `frozen`, unless `frozen_as` maps the job id to another value (None: not recorded, as by
    a coast from before it). Its `config.json` names the job's run_id (`run_ids`, default
    `run-<id>`, as coastd fixes "auto"), and its manifest records it (`ran_as` overrides).
    `hash_inputs` reports `hashes` while a job is active or none was submitted, then
    `hashes_after` (default: the same).
    As in coastd, a frozen job starts at its source's `n_steps_completed`. As in coast, a
    training manifest lists a checkpoint before the final state, so a base step read from
    `savestate_outputs` (or a one-entry assumption) shows."""

    def __init__(
        self,
        runs_root: Path,
        data_link: Path,
        fail: str = "",
        hostname: str = "fakehost",
        linked_elsewhere: str = "",
        resumed: tuple[str, ...] = (),
        hashes: dict[str, str] | None = None,
        hashes_after: dict[str, str] | None = None,
        manifest_host: str = "",
        submitted_text: str | None = None,
        tcp_port: int = 51900,
        fail_once: bool = False,
        polls_to_finish: int = 2,
        frozen_as: dict[str, bool | None] | None = None,
        run_ids: dict[str, str] | None = None,
        ran_as: dict[str, str] | None = None,
    ) -> None:
        self.polls_to_finish = polls_to_finish
        self.frozen_as = frozen_as or {}
        self.run_ids, self.ran_as = run_ids or {}, ran_as or {}
        self.tcp_port, self.fail_once, self.failed = tcp_port, fail_once, False
        self.runs_root, self.data_link, self.fail = runs_root, data_link, fail
        self.hostname, self.manifest_host = hostname, manifest_host or hostname
        self.linked_elsewhere, self.resumed = linked_elsewhere, resumed
        self.hashes = EXPECTED if hashes is None else hashes
        self.hashes_after = self.hashes if hashes_after is None else hashes_after
        self.submitted_text = submitted_text
        self.jobs: dict[str, dict[str, Any]] = {}

    def __call__(self, req: dict[str, Any]) -> bytes:
        if req["cmd"] == "machine_info":
            return ok(hostname=self.hostname, runs_root=str(self.runs_root), data_link_ok=True,
                      dataset_base=str(self.data_link), tcp_port=self.tcp_port)  # fmt: skip
        if req["cmd"] == "hash_inputs":
            settled = self.jobs and all(j["bucket"] != "active" for j in self.jobs.values())
            known = self.hashes_after if settled else self.hashes
            return ok(sha256={p: known.get(p, "0" * 64) for p in req["paths"]})
        if req["cmd"] == "cancel":
            job = self.jobs.get(req["id"])
            if job is None or job["bucket"] != "active":
                return refuse(f"cancel: unknown or already-terminal job '{req['id']}'")
            job["bucket"] = "cancelled"
            return ok()
        if req["cmd"] == "submit":
            src = self.jobs.get(req.get("load_state_from", ""), {"bucket": "done", "completed": 0})
            if src["bucket"] != "done":
                return refuse("not done")
            n = len(self.jobs)
            job_id = f"job{n}"
            at = f"2026-11-04T09:{n // 60:02d}:{n % 60:02d}Z"  # one second apart: ordered
            self.jobs[job_id] = {"req": req, "bucket": "active", "polls": 0,
                                 "base": src["completed"], "submitted_at": at}  # fmt: skip
            return ok(id=job_id)
        for job_id, job in self.jobs.items():
            job["polls"] += 1
            if job["bucket"] == "active" and job["polls"] == self.polls_to_finish:
                self._finish(job_id, job)
        return ok(jobs=[summary(i, j["bucket"], j.get("reason", ""),
                                user_message=j["req"].get("message", ""),
                                submitted_at=j["submitted_at"])
                        for i, j in self.jobs.items()])  # fmt: skip

    def seeds(self) -> list[tuple[str, int, bool]]:
        """(ref, seed, frozen) of every job, in submit order."""
        return [(j["req"]["ref"], _seed(j["req"]["config_text"]), "load_state_from" in j["req"])
                for j in self.jobs.values()]  # fmt: skip

    def _finish(self, job_id: str, job: dict[str, Any]) -> None:
        cfg = json.loads(job["req"]["config_text"])
        if f"seed{_seed(job['req']['config_text'])}" == self.fail and not (
            self.fail_once and self.failed
        ):
            job["bucket"], job["reason"] = "failed", "coast exited 1"
            self.failed = True
            return
        n, base = job["req"]["steps_target"], job["base"]
        job["completed"] = base + n
        job["bucket"], job["config"] = "done", cfg
        saved = [{"path": "./final_state.h5", "step": base + n, "bytes": 1}]
        if base == 0:  # training: one --checkpoint write before the --save-state one
            saved.insert(0, {"path": f"./snapshots/s_{n // 2}.h5", "step": n // 2, "bytes": 1})
        run = self.runs_root / job_id
        run.mkdir()
        text = job["req"]["config_text"] if self.submitted_text is None else self.submitted_text
        (run / "config.submitted.json").write_text(text)
        run_id = self.run_ids.get(job_id, f"run-{job_id}")  # as coastd fixes "auto"
        (run / "config.json").write_text(json.dumps(json.loads(text) | {"run_id": run_id}))
        elsewhere = job_id == self.linked_elsewhere
        target = self.data_link.parent / "elsewhere" if elsewhere else self.data_link
        (run / "data").symlink_to(target)
        (run / "coast_runtime.json").write_text(json.dumps({
            "schema_version": "1", "coast_git_sha": job["req"]["ref"], "coast_git_dirty": False,
            "coast_build_type": "Release", "seed_offset": 0,
            "n_steps_requested": n // 2 if job_id in self.resumed else n,
            "unlimited": False, "n_steps_completed": base + n,
            "exit_reason": "completed_n_steps", "wall_clock_seconds": 1.0,
            "hostname": self.manifest_host,
            "gpu_adapter": None, "started_at_utc": "t", "completed_at_utc": "t",
            "observe": {"output_dir": "./observe", "groups": cfg["observe"]["groups"]},
            "datasets": {}, "retina": {}, "node_seeds": [],
            "savestate_outputs": saved,
            "run_id": self.ran_as.get(job_id, run_id),
        } | ({} if self.frozen_as.get(job_id, cfg["frozen"]) is None
             else {"frozen": self.frozen_as.get(job_id, cfg["frozen"])})))  # fmt: skip


def _seed(config_text: str) -> int:
    return json.loads(config_text)["sim"]["nodes"][0]["random_seed_offset"]


INPUTS = pr.Inputs((7, 3), "./data/eval/m1/train.txt", "./data/eval/m1/test.txt", "./zeros")
EXPECTED = {"mnist/t10k-images-idx3-ubyte": H_A, "eval/m1/test.txt": H_B}
SHA2 = "0" * 39 + "1"
STATE = runner.State("before", SHA, json.loads(TEMPLATE.read_text()))


def daemon_on(serve: Serve, fake: FakeDaemon) -> runner.Daemon:
    client, _ = serve(fake)
    return runner.connect(DaemonSpec(client.host, client.port))


_LEDGERS = itertools.count()


def milestone1(
    daemons: list[runner.Daemon],
    states: tuple[runner.State, ...] = (STATE,),
    seeds: tuple[int, ...] = (4, 1),
    retry_s: float = 1.0,
    ledger: Path | None = None,
    timeout_s: float = STALL_S,
    redo: tuple[str, ...] = (),
) -> dict[str, dict[tuple[int, str], Any]]:
    """A fresh ledger per call unless one is given (a resume)."""
    if ledger is None:
        where = daemons[0].runs_root.parent if daemons else Path(tempfile.mkdtemp())
        ledger = where / f"ledger{next(_LEDGERS)}.json"
    return runner.run_milestone1(
        daemons, states, INPUTS, expected_inputs=EXPECTED, seeds=seeds, steps=2000,
        change_interval=5, conditions=["clean", "zeros"], poll_s=0.0, timeout_s=timeout_s,
        ledger=ledger, retry_s=retry_s, redo=redo,
    )  # fmt: skip


@pytest.fixture
def roots(tmp_path: Path) -> Callable[[str], tuple[Path, Path]]:
    """(runs_root, dataset_base) of a fake daemon; one runs_root per daemon name."""
    (tmp_path / "dataset_base").mkdir()

    def make(name: str = "d0") -> tuple[Path, Path]:
        (tmp_path / name).mkdir()
        return tmp_path / name, tmp_path / "dataset_base"

    return make


Roots = Callable[..., tuple[Path, Path]]


def test_run_milestone1_trains_then_runs_frozen_from_each_seeds_state(
    serve: Serve, roots: Roots
) -> None:
    fake = FakeDaemon(*roots())
    dirs = milestone1([daemon_on(serve, fake)])["before"]
    assert {k: d.root.name for k, d in dirs.items()} == {
        (4, "train"): "job0", (1, "train"): "job1",
        (4, "clean"): "job2", (4, "zeros"): "job3", (1, "clean"): "job4", (1, "zeros"): "job5",
    }  # fmt: skip
    for job_id, src in {"job2": "job0", "job3": "job0", "job4": "job1", "job5": "job1"}.items():
        job = fake.jobs[job_id]
        assert (job["req"]["load_state_from"], job["req"]["steps_target"]) == (src, 30)
        window = {"ranges": [[2000, 2029]], "combine": "or"}  # not the checkpoint's 1000
        assert job["config"]["observe"]["groups"] == {"population": window, "input": window}
        dataset = job["config"]["datasets"]["mnist"]["default"]
        assert dataset["index_manifest"] == INPUTS.test_manifest
    assert all("load_state_from" not in fake.jobs[i]["req"] for i in ("job0", "job1"))


@pytest.mark.parametrize(
    ("seeds", "n", "want"),
    [
        ([0, 1, 2, 3, 4], 1, [0, 0, 0, 0, 0]),
        ([0, 1, 2, 3, 4], 2, [0, 1, 0, 1, 0]),
        ([0, 1, 2, 3, 4], 3, [0, 1, 2, 0, 1]),
        ([4, 1], 3, [1, 1]),  # by value, not by position
        ([0], 2, [0]),  # fewer seeds than daemons: daemon 1 idles
    ],
)
def test_assign_places_seed_s_on_daemon_s_mod_n(seeds: list[int], n: int, want: list[int]) -> None:
    assert list(runner.assign(seeds, n).values()) == want


def test_assign_refuses_no_daemon_and_repeated_seeds() -> None:
    with pytest.raises(ValueError, match="at least one daemon"):
        runner.assign([0], 0)
    with pytest.raises(ValueError, match="unique"):
        runner.assign([1, 1], 2)


def test_two_daemons_two_states_each_chain_on_its_seeds_daemon(serve: Serve, roots: Roots) -> None:
    fakes = [FakeDaemon(*roots("d0"), hostname="h0"), FakeDaemon(*roots("d1"), hostname="h1")]
    states = (STATE, runner.State("after", SHA2, STATE.template))
    # named in reverse: daemon i is the i-th by (hostname, tcp_port), not by naming order
    out = milestone1([daemon_on(serve, f) for f in reversed(fakes)], states, seeds=(0, 1, 2))
    # every job of seed s, both states, training and frozen, ran on daemon s mod 2
    for i, fake in enumerate(fakes):
        assert {s for _, s, _ in fake.seeds()} == {s for s in (0, 1, 2) if s % 2 == i}
        assert {(r, s, f) for r, s, f in fake.seeds()} == {
            (r, s, f) for r in (SHA, SHA2) for s in (0, 1, 2) if s % 2 == i for f in (False, True)
        }
    for label in ("before", "after"):
        for (seed, _), run in out[label].items():
            assert run.root.parent == fakes[seed % 2].runs_root
    # a frozen job names its training job on the same daemon
    for fake in fakes:
        for job in fake.jobs.values():
            src = job["req"].get("load_state_from")
            if src is not None:
                source = fake.jobs[src]["req"]
                assert _seed(source["config_text"]) == _seed(job["req"]["config_text"])
                assert source["ref"] == job["req"]["ref"]


def test_canonical_orders_by_the_daemons_identity_and_refuses_one_named_twice(
    serve: Serve, roots: Roots
) -> None:
    a1 = daemon_on(serve, FakeDaemon(*roots("d0"), hostname="a", tcp_port=2))
    a0 = daemon_on(serve, FakeDaemon(*roots("d1"), hostname="a", tcp_port=1))  # same host
    b = daemon_on(serve, FakeDaemon(*roots("d2"), hostname="b", tcp_port=1))
    for named in ([b, a1, a0], [a0, b, a1], [a1, a0, b]):
        assert runner.canonical(named) == [a0, a1, b]
    again = daemon_on(serve, FakeDaemon(*roots("d3"), hostname="a", tcp_port=2))
    with pytest.raises(ValueError, match=r"named more than once: \[\('a', 2\)\]"):
        runner.canonical([a1, b, again])
    with pytest.raises(ValueError, match="named more than once"):
        milestone1([a1, again])


def test_discover_collects_the_replies_sorted_and_skips_noise() -> None:
    replies = [
        b"not json",
        json.dumps({"hostname": "zeta", "tcp_port": 51900, "proto_version": 1}).encode(),
        json.dumps({"hostname": "alpha", "tcp_port": 52000, "proto_version": 1}).encode(),
        json.dumps({"hostname": "zeta"}).encode(),  # not a reply: no port
    ]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as responder:
        responder.bind(("127.0.0.1", 0))
        probes: list[bytes] = []

        def answer() -> None:
            data, addr = responder.recvfrom(1024)
            probes.append(data)
            for r in replies:
                responder.sendto(r, addr)

        t = threading.Thread(target=answer)
        t.start()
        found = discover(0.5, responder.getsockname()[1], broadcast="127.0.0.1")
        t.join()
    assert probes == [b"coastd-probe-v1"]
    assert [(d.hostname, d.tcp_port) for d in found] == [("alpha", 52000), ("zeta", 51900)]
    assert [d.spec for d in found] == ["127.0.0.1:52000", "127.0.0.1"]  # default port omitted


def test_run_milestone1_stops_at_a_failed_training_job_and_names_the_active_ones(
    serve: Serve, roots: Roots
) -> None:
    fake = FakeDaemon(*roots(), fail="seed1")
    with pytest.raises(DaemonError, match=r"job job1 on fakehost \(.*\) ended in failed: "
                       "'coast exited 1'") as e:  # fmt: skip
        milestone1([daemon_on(serve, fake)])
    # seed 4 trained in the same poll, so its frozen jobs were submitted and are active
    assert list(fake.jobs) == ["job0", "job1", "job2", "job3"]
    note = "\n".join(e.value.__notes__)
    assert '"cmd":"cancel"' in note
    assert "job2 (before, seed 4, clean)" in note and "job3 (before, seed 4, zeros)" in note
    assert "job1 " not in note  # it is settled


def test_run_milestone1_needs_full_shas_and_unique_labels(serve: Serve, roots: Roots) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    with pytest.raises(ValueError, match="full commit SHA"):
        milestone1([d], (runner.State("x", "feat/eval-python-runs", STATE.template),))
    with pytest.raises(ValueError, match="unique labels"):
        milestone1([d], (STATE, STATE))
    with pytest.raises(ValueError, match="at least one daemon"):
        milestone1([])
    assert not fake.jobs


def test_run_milestone1_accepts_resumed_training_and_frozen_jobs(
    serve: Serve, roots: Roots
) -> None:
    """coast continues a checkpoint's sequence when the same run wrote it (`run_id`,
    SPEC.md S.5.14): a resumed frozen trial is as valid as training."""
    runs_root, base = roots()
    fake = FakeDaemon(runs_root, base, resumed=("job0", "job3"))
    assert len(milestone1([daemon_on(serve, fake)])["before"]) == 6
    assert fake.jobs["job3"]["req"].get("load_state_from")  # job3 is a frozen trial


@pytest.mark.parametrize(
    ("frozen_as", "match"),
    [
        ({"job0": True}, "run was frozen, expected the opposite"),
        ({"job3": False}, "run was not frozen, expected the opposite"),
        ({"job3": None}, "records no `frozen`: its coast predates"),
    ],
)
def test_a_run_frozen_other_than_its_job_is_refused(
    serve: Serve, roots: Roots, frozen_as: dict[str, bool | None], match: str
) -> None:
    runs_root, base = roots()
    with pytest.raises(ValueError, match=match):
        milestone1([daemon_on(serve, FakeDaemon(runs_root, base, frozen_as=frozen_as))])


@pytest.mark.parametrize(
    ("run_ids", "ran_as", "match"),
    [
        ({"job3": "auto"}, {}, 'run_id is still "auto": the daemon\'s coastd'),
        ({}, {"job3": "another"}, "ran as run 'another', its config names 'run-job3'"),
    ],
)
def test_a_run_whose_run_id_is_not_fixed_is_refused(
    serve: Serve, roots: Roots, run_ids: dict[str, str], ran_as: dict[str, str], match: str
) -> None:
    """A coastd that leaves "auto" lets a resumed frozen trial restart its sequence while
    its steps count right; `check_provenance` cannot see that, `check_run_id` can."""
    runs_root, base = roots()
    fake = FakeDaemon(runs_root, base, run_ids=run_ids, ran_as=ran_as)
    with pytest.raises(ValueError, match=match):
        milestone1([daemon_on(serve, fake)])


def test_a_refused_job_is_named_and_redone_with_redo(serve: Serve, roots: Roots) -> None:
    """A crash can leave a run dir the checks refuse (here: its config kept "auto"). The
    refusal names the key; rerun as is, the same job is adopted and refused again; with
    --redo KEY that job is submitted afresh and the evaluation completes on the new one."""
    fake = FakeDaemon(*roots(), run_ids={"job3": "auto"})
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    key = "before/4/zeros"  # job3: seed 4's second frozen trial
    with pytest.raises(ValueError, match="still") as e:
        milestone1([d], ledger=ledger)
    assert f"run the same evaluation with --redo {key}" in "\n".join(e.value.__notes__)
    with pytest.raises(ValueError, match="still"):  # as is: adopted again, refused again
        milestone1([d], ledger=ledger)
    before = len(fake.jobs)
    dirs = milestone1([d], ledger=ledger, redo=(key,))["before"]
    assert len(fake.jobs) == before + 1  # only the named job again
    assert dirs[4, "zeros"].root.name == f"job{before}"
    again = milestone1([d], ledger=ledger)["before"]  # later runs adopt the newest job
    assert again == dirs and len(fake.jobs) == before + 1


def test_redoing_a_training_job_redoes_its_frozen_trials(serve: Serve, roots: Roots) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    first = milestone1([d], ledger=ledger)["before"]
    n = len(fake.jobs)
    dirs = milestone1([d], ledger=ledger, redo=("before/4/train",))["before"]
    assert len(fake.jobs) == n + 3  # seed 4: its training job and both frozen trials
    for c in ("train", "clean", "zeros"):
        assert dirs[4, c] != first[4, c]
        assert dirs[1, c] == first[1, c]  # seed 1 untouched
    with pytest.raises(ValueError, match="names no job of this evaluation"):
        milestone1([d], ledger=ledger, redo=("before/9/clean",))


def test_two_newest_jobs_of_one_key_are_refused(serve: Serve, roots: Roots) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    milestone1([d], ledger=ledger)
    fake.jobs["job1"]["submitted_at"] = fake.jobs["job0"]["submitted_at"] = "2026-11-04T10:00:00Z"
    fake.jobs["job1"]["req"]["message"] = fake.jobs["job0"]["req"]["message"]
    with pytest.raises(DaemonError, match="same second"):
        milestone1([d], ledger=ledger)


def test_inputs_are_verified_on_every_daemon_before_any_submit_and_after_the_last(
    serve: Serve, roots: Roots
) -> None:
    wrong = EXPECTED | {"eval/m1/test.txt": H_A}
    good = FakeDaemon(*roots("d0"), hostname="good")
    bad = FakeDaemon(*roots("d1"), hostname="bad", hashes=wrong)
    with pytest.raises(DaemonError, match=f"eval/m1/test.txt: daemon {H_A}, expected {H_B}"):
        milestone1([daemon_on(serve, good), daemon_on(serve, bad)])
    assert not good.jobs and not bad.jobs
    drifted = FakeDaemon(*roots("d2"), hashes_after=wrong)  # changed while the jobs ran
    with pytest.raises(DaemonError, match="inputs differ"):
        milestone1([daemon_on(serve, drifted)])
    assert len(drifted.jobs) == 6


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"linked_elsewhere": "job1"}, "job1/data points to .*elsewhere, not to the daemon's"),
        ({"linked_elsewhere": "job4"}, "job4/data points to .*elsewhere, not to the daemon's"),
        ({"manifest_host": "other"}, "written on 'other', but fakehost .* reports host"),
        ({"submitted_text": "{}"}, "config.submitted.json: not the config submitted as job"),
    ],
)
def test_run_milestone1_refuses_a_run_dir_that_is_not_the_jobs(
    serve: Serve, roots: Roots, change: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        milestone1([daemon_on(serve, FakeDaemon(*roots(), **change))])


def interrupted(fake_daemon: runner.Daemon, ledger: Path) -> None:
    """Submit the training jobs, then stop before any is done, as a killed eval would."""
    with pytest.raises(DaemonError, match="timed out") as e:
        milestone1([fake_daemon], ledger=ledger, timeout_s=0.0)
    assert "run the same evaluation again to re-attach" in "\n".join(e.value.__notes__)


def test_an_interrupted_run_resumes_from_its_ledger_without_submitting_twice(
    serve: Serve, roots: Roots
) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    interrupted(d, ledger)
    assert list(fake.jobs) == ["job0", "job1"]  # the two training jobs, still active
    dirs = milestone1([d], ledger=ledger)["before"]
    assert len(dirs) == 6 and len(fake.jobs) == 6  # re-attached: no training job again
    assert {dirs[4, "train"].root.name, dirs[1, "train"].root.name} == {"job0", "job1"}
    again = milestone1([d], ledger=ledger)["before"]  # a finished run: everything re-used
    assert again == dirs and len(fake.jobs) == 6


def test_a_ledger_whose_config_texts_changed_is_refused(
    serve: Serve, roots: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit of the toolkit between two invocations: without the texts in the
    fingerprint, the rerun adopted the done frozen trials and refused each on its
    `config.submitted.json`, on this and every later rerun."""
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    milestone1([d], ledger=ledger)
    made = pr.frozen_jobs

    def edited(*args: Any, **kwargs: Any) -> list[pr.Job]:
        return [msgspec.structs.replace(j, config_text=j.config_text + "\n")
                for j in made(*args, **kwargs)]  # fmt: skip

    monkeypatch.setattr(pr, "frozen_jobs", edited)
    with pytest.raises(ValueError, match="the ledger of another evaluation"):
        milestone1([d], ledger=ledger)
    assert len(fake.jobs) == 6  # nothing submitted


def test_a_ledger_of_another_evaluation_is_refused(serve: Serve, roots: Roots) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    interrupted(d, ledger)
    with pytest.raises(ValueError, match="the ledger of another evaluation"):
        milestone1([d], ledger=ledger, seeds=(4, 2))
    assert len(fake.jobs) == 2


def test_a_job_that_failed_in_an_earlier_invocation_is_submitted_afresh(
    serve: Serve, roots: Roots
) -> None:
    fake = FakeDaemon(*roots(), fail="seed1", fail_once=True)
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    with pytest.raises(DaemonError, match=r"job1 .* ended in failed"):  # this invocation: fatal
        milestone1([d], ledger=ledger)
    dirs = milestone1([d], ledger=ledger)["before"]  # the next one retries it
    assert dirs[1, "train"].root.name not in ("job0", "job1")
    assert fake.jobs["job1"]["bucket"] == "failed" and len(dirs) == 6


def test_a_job_still_active_at_resume_that_then_fails_stops_the_run(
    serve: Serve, roots: Roots
) -> None:
    """Only a job that had already failed when this invocation started is retried; one
    that fails during it is fatal, as in a fresh run (review 2, finding 1)."""
    # jobs end on their third `list`: still active in the resumed run's start-up listing
    fake = FakeDaemon(*roots(), fail="seed1", polls_to_finish=3)
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    interrupted(d, ledger)  # seed 1's training job is still active
    with pytest.raises(DaemonError, match=r"job job1 .* ended in failed: 'coast exited 1'"):
        milestone1([d], ledger=ledger)
    assert all(_seed(j["req"]["config_text"]) != 1 for j in list(fake.jobs.values())[2:])


def test_without_its_ledger_an_evaluation_starts_afresh(serve: Serve, roots: Roots) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    first = milestone1([d], ledger=ledger)["before"]
    ledger.unlink()  # the same evaluation, but its jobs carry the old run id
    second = milestone1([d], ledger=ledger)["before"]
    assert len(fake.jobs) == 12
    assert {r.root.name for r in first.values()}.isdisjoint(r.root.name for r in second.values())


def test_a_second_evaluation_on_a_ledger_in_use_is_refused(serve: Serve, roots: Roots) -> None:
    """Seen for real: an eval thought killed kept running, the rerun raced it, both submitted
    the same frozen jobs and one crashed replacing the other's ledger write."""
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    ledger = d.runs_root.parent / "ledger.json"
    with ledger.with_name("ledger.json.lock").open("a") as held:  # the running evaluation
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="another evaluation is running on this ledger"):
            milestone1([d], ledger=ledger)
    assert not fake.jobs
    assert len(milestone1([d], ledger=ledger)["before"]) == 6  # released: runs


def test_the_ledger_stays_locked_for_the_whole_run(serve: Serve, roots: Roots) -> None:
    """Not only at start: a second evaluation must be refused while jobs are polled."""
    fake = FakeDaemon(*roots())
    ledger = fake.runs_root.parent / "ledger.json"
    attempts: list[bool] = []

    def probing(req: dict[str, Any]) -> bytes:
        if req["cmd"] == "list" and fake.jobs:  # mid-run: try to take the lock ourselves
            with ledger.with_name("ledger.json.lock").open("a") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    attempts.append(True)
                except BlockingIOError:
                    attempts.append(False)
        return fake(req)

    client, _ = serve(probing)
    milestone1([runner.connect(DaemonSpec(client.host, client.port))], ledger=ledger)
    assert attempts and not any(attempts)


def test_cancel_evaluation_stops_this_evaluations_active_jobs_of_any_run(
    serve: Serve, roots: Roots
) -> None:
    fake = FakeDaemon(*roots())
    d = daemon_on(serve, fake)
    a, b = d.runs_root.parent / "a.json", d.runs_root.parent / "b.json"
    milestone1([d], ledger=a)  # run 1 of evaluation A: 6 done jobs
    fake.polls_to_finish = 10**6  # from here on, jobs stay active
    a.unlink()
    with pytest.raises(DaemonError, match="timed out"):  # run 2 of A, interrupted
        milestone1([d], ledger=a, timeout_s=0.0)
    with pytest.raises(DaemonError, match="timed out"):  # another evaluation, interrupted
        milestone1([d], ledger=b, seeds=(4, 2), timeout_s=0.0)
    a.unlink()  # no ledger needed
    fp = runner.fingerprint([d], [STATE], INPUTS, EXPECTED, (4, 1), 2000, 5, ["clean", "zeros"])
    gone = runner.cancel_evaluation([d], fp)
    assert sorted(k for _, _, k in gone) == ["before/1/train", "before/4/train"]
    assert sorted(i for _, i, _ in gone) == ["job6", "job7"]  # A's second run only
    buckets = [fake.jobs[f"job{n}"]["bucket"] for n in range(10)]
    assert buckets[:6] == ["done"] * 6  # A's first run: done jobs are left alone
    assert buckets[6:8] == ["cancelled"] * 2
    assert buckets[8:] == ["active"] * 2  # the other evaluation's jobs: untouched
    assert runner.cancel_evaluation([d], fp) == []  # nothing left to cancel


def test_a_damaged_ledger_is_named(serve: Serve, roots: Roots) -> None:
    d = daemon_on(serve, FakeDaemon(*roots()))
    ledger = d.runs_root.parent / "ledger.json"
    ledger.write_text("{not json")
    with pytest.raises(ValueError, match=r"ledger\.json: not a coast-eval ledger"):
        milestone1([d], ledger=ledger)


@dataclass(frozen=True)
class FlakyClient(Client):
    """`list` answers `fine[0]` times, then fails to connect while `failures` still holds
    entries."""

    failures: list[int] = field(default_factory=lambda: list[int]())
    fine: list[int] = field(default_factory=lambda: [0])

    def list_jobs(self) -> list[JobSummary]:
        if self.fine[0] > 0:
            self.fine[0] -= 1
        elif self.failures:
            self.failures.pop()
            raise DaemonUnreachable(f"coastd at {self.host}:{self.port}: refused")
        return super().list_jobs()


def flaky(serve: Serve, fake: FakeDaemon, failures: int, fine: int = 0) -> runner.Daemon:
    d = daemon_on(serve, fake)
    c = FlakyClient(d.client.host, d.client.port, failures=[0] * failures, fine=[fine])
    return runner.Daemon(c, d.info, d.runs_root)


def test_a_list_that_cannot_connect_is_retried_within_the_budget(
    serve: Serve, roots: Roots
) -> None:
    assert len(milestone1([flaky(serve, FakeDaemon(*roots("d0")), 3)])["before"]) == 6
    fake = FakeDaemon(*roots("d1"))
    with pytest.raises(DaemonUnreachable, match="refused") as e:  # down while jobs run
        milestone1([flaky(serve, fake, 10**6, fine=1)], retry_s=0.05)
    assert "job0 (before, seed 4, train)" in "\n".join(e.value.__notes__)
    with pytest.raises(DaemonUnreachable):  # down from the start: nothing submitted
        milestone1([flaky(serve, FakeDaemon(*roots("d2")), 10**6)], retry_s=0.05)


def test_connect_checks_the_data_link_and_the_runs_root(serve: Serve, roots: Roots) -> None:
    runs_root, base = roots()
    client, _ = serve(FakeDaemon(runs_root, base))
    d = runner.connect(DaemonSpec(client.host, client.port))
    assert (d.runs_root, d.info.dataset_base) == (runs_root, str(base))
    mapped = runner.connect(DaemonSpec(client.host, client.port, base))  # any visible dir
    assert mapped.runs_root == base
    with pytest.raises(DaemonError, match="map it with HOST:PORT=LOCAL_RUNS_ROOT"):
        runner.connect(DaemonSpec(client.host, client.port, runs_root / "nope"))
    unlinked, _ = serve(lambda _: ok(hostname="h", runs_root=str(runs_root), data_link_ok=False,
                                     dataset_base="", tcp_port=51900))  # fmt: skip
    with pytest.raises(DaemonError, match="no dataset_base link"):
        runner.connect(DaemonSpec(unlinked.host, unlinked.port))


def test_check_data_link(tmp_path: Path) -> None:
    run = run_dir("j", tmp_path / "runs")
    run.root.mkdir(parents=True)
    with pytest.raises(ValueError, match="not a symlink"):
        runner.check_data_link(run, "/mnt/NAS/datasets")
    (run.root / "data").symlink_to("/mnt/NAS/datasets")  # need not exist on this host
    runner.check_data_link(run, "/mnt/NAS/datasets")
    with pytest.raises(ValueError, match="points to /mnt/NAS/datasets, not to"):
        runner.check_data_link(run, "/mnt/NAS/datasets/")  # the daemon's string, exactly
    (run.root / "data").unlink()
    (run.root / "data").symlink_to("../../datasets")  # coastd writes an absolute target
    with pytest.raises(ValueError, match=r"points to \.\./\.\./datasets"):
        runner.check_data_link(run, "/mnt/NAS/datasets")
    (run.root / "data").unlink()
    (run.root / "data").mkdir()  # a copy is not the verified directory
    with pytest.raises(ValueError, match="not a symlink"):
        runner.check_data_link(run, "/mnt/NAS/datasets")
