"""Milestone 1 through coastd, on one or more daemons: ship the inputs, train, run the
frozen trials, and prove every run read the pinned inputs and wrote the dir we read.

coastd has no job dependencies, so the order lives here: a chain is one (state, seed)'s
training job and its frozen trials, and the frozen jobs are submitted only once the
training job is `done`, on the same daemon, because `load_state_from` names a job id
known only to that daemon. Seed `s` runs on daemon `s mod N` for every state
(`design/eval_multi_daemon_plan.md` D4), so the two states of an A/B comparison share a
host within each seed; daemon `i` is the `i`-th in `canonical` order, which depends on
the daemons only, never on the order they were named in. This module sequences and checks
provenance; it computes no metrics.

A run can be resumed: the job ids go to a ledger as they are submitted, and running the
same evaluation again with the same ledger re-attaches to them (`run_milestone1`).

`run_jobs` is the loop under `run_milestone1`, for drivers whose chains have another shape
(`sweep.py`: training continued over several lengths): the caller plans the first jobs and,
per done job, the jobs that follow it; placement, ledger, adoption and checks are the loop's.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import msgspec

from coast_eval import protocol, runs
from coast_eval.daemon import (
    Client,
    DaemonError,
    DaemonSpec,
    DaemonUnreachable,
    JobSummary,
    MachineInfo,
    run_dir,
    verify_daemon_inputs,
)

DATA_REF = "./data"  # coastd links its dataset_base into every run dir as `data`
TRAIN = "train"


def ship_inputs(inputs: protocol.Inputs, dataset_base: Path, name: str) -> protocol.Inputs:
    """Copy the files of `write_inputs` into `<dataset_base>/eval/<name>/` and return the
    `Inputs` as a run dir sees them (`./data/eval/<name>/...`).

    `dataset_base` is a daemon's dataset_base as this host sees it. Shipping proves
    nothing about what a daemon reads: `run_milestone1` checks the shipped files on every
    daemon with `hash_inputs`. A file already there with different content raises,
    because jobs that read it earlier would then have run other inputs than the name says.
    """
    files = (inputs.train_manifest, inputs.test_manifest, inputs.zeros_images)
    return protocol.Inputs(inputs.test_indices, *ship_files(map(Path, files), dataset_base, name))


def ship_files(files: Iterable[Path], dataset_base: Path, name: str) -> list[str]:
    """`ship_inputs` for any files: copy each into `<dataset_base>/eval/<name>/` and return
    the paths a run dir sees (`./data/eval/<name>/<file>`), under the same rules."""
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"inputs name {name!r} is not a single path component")
    dest = dataset_base / "eval" / name
    dest.mkdir(parents=True, exist_ok=True)
    shipped: list[str] = []
    for src in files:
        target = dest / src.name
        if not target.exists():
            shutil.copyfile(src, target)
        elif target.read_bytes() != src.read_bytes():
            raise ValueError(f"{target}: exists with content different from {src}")
        shipped.append(f"{DATA_REF}/eval/{name}/{src.name}")
    return shipped


def check_data_link(run: runs.RunDir, dataset_base: str) -> None:
    """The run read the daemon's `dataset_base`: coastd links it into every run dir as
    `data` (`coastd/src/child_cmd.cpp`), with the target `machine_info` reports. Compared
    as the daemon's own path string, since that path need not exist on this host."""
    link = run.root / "data"
    if not link.is_symlink():
        raise ValueError(f"{link}: not a symlink to the daemon's dataset_base")
    target = os.readlink(link)
    if target != dataset_base:
        raise ValueError(
            f"{link} points to {target}, not to the daemon's dataset_base {dataset_base}"
        )


@dataclass(frozen=True)
class Daemon:
    """A daemon to run on, as checked once by `connect`. `runs_root` is where this host
    reads the daemon's run dirs."""

    client: Client
    info: MachineInfo
    runs_root: Path

    @property
    def name(self) -> str:
        return f"{self.info.hostname} ({self.client.host}:{self.client.port})"


def connect(spec: DaemonSpec) -> Daemon:
    """Ask the daemon who it is; raise unless its runs can be checked from this host."""
    client = spec.client()
    info = client.machine_info()
    where = f"coastd at {client.host}:{client.port}"
    if not info.data_link_ok or not info.dataset_base:
        raise DaemonError(f"{where} has no dataset_base link: ./data would not resolve")
    runs_root = spec.local_runs_root or Path(info.runs_root)
    if not runs_root.is_dir():
        raise DaemonError(
            f"{where}: its runs_root {info.runs_root} is not a directory on this host at "
            f"{runs_root}; map it with HOST:PORT=LOCAL_RUNS_ROOT (a daemon that never ran a "
            "job has not created it yet)"
        )
    return Daemon(client, info, runs_root)


def canonical(daemons: Sequence[Daemon]) -> list[Daemon]:
    """The daemons sorted by what each reports about itself, (hostname, tcp_port): the
    order `assign`'s indices refer to. The same daemon named twice (by IP and by name,
    say) would run its seeds' chains twice over, so it is refused."""
    order = sorted(daemons, key=lambda d: (d.info.hostname, d.info.tcp_port))
    ids = [(d.info.hostname, d.info.tcp_port) for d in order]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"the same daemon is named more than once: {dup}")
    return order


def assign(seeds: Sequence[int], n_daemons: int) -> dict[int, int]:
    """Seed -> index into the `canonical` daemons: `seed mod n_daemons`, the same for every
    state (plan D4)."""
    if n_daemons < 1:
        raise ValueError(f"need at least one daemon, got {n_daemons}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be unique, got {list(seeds)}")
    return {s: s % n_daemons for s in seeds}


@dataclass(frozen=True)
class State:
    """One code state. `ref` must be a full commit SHA: coastd resolves a branch name at
    each submit, and the frozen jobs are submitted long after the training jobs."""

    label: str
    ref: str
    template: dict[str, Any]


@dataclass(frozen=True)
class Planned:
    """A job to submit. `label` names its chain (a state, a sweep point), `base_step` is the
    step of the state it loads (0: none), `source` the id of the done job it loads that
    state from, on the daemon of its seed (None: a fresh run). `priority` is added to the
    run's (coastd runs a higher one first)."""

    label: str
    ref: str
    job: protocol.Job
    base_step: int
    source: str | None = None
    priority: int = 0

    @property
    def key(self) -> str:
        return _key(self.label, self.job.seed, self.job.condition)


@dataclass(frozen=True)
class Done:
    """A planned job whose run dir passed every check."""

    planned: Planned
    job_id: str
    run: runs.RunDir


@dataclass(frozen=True)
class _Job:
    """A submitted job and what its run dir must show."""

    planned: Planned
    daemon: int  # index into the daemons
    job_id: str


def _key(label: str, seed: int, condition: str) -> str:
    return f"{label}/{seed}/{condition}"


class Ledger(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Which run an evaluation is, so that running it again re-attaches to its jobs.
    `run_id` is drawn and written once, before any submit, and every job carries it in its
    tag on its daemon (the daemons hold the job ids; coastd keeps a job's message across
    restarts). `fingerprint` hashes everything that decides the jobs (`fingerprint`), so a
    ledger is only resumed by the same evaluation. Removing the ledger starts the
    evaluation afresh: no job carries the next run's id."""

    schema: Literal["coast-eval-ledger/1"]
    fingerprint: str
    run_id: str


def fingerprint(
    daemons: Sequence[Daemon],
    states: Sequence[State],
    inputs: protocol.Inputs,
    expected_inputs: Mapping[str, str],
    seeds: Sequence[int],
    steps: int,
    change_interval: int,
    conditions: Sequence[str],
) -> str:
    """SHA256 of what decides an evaluation's jobs and where they run. The priority is not
    in it: it changes only the order the daemons run them in.

    It hashes the config texts the jobs are submitted with, not only what they are made
    from: an edit of `protocol` between two invocations changes the texts, and a ledger that
    still matched would adopt done jobs whose `config.submitted.json` no longer equals the
    text, refused on every rerun. The frozen texts are those for `base_step = steps`, the
    only base a training run passes `check_provenance` with."""
    texts = {s.label: _config_texts(s, inputs, seeds, steps, change_interval, conditions)
             for s in states}  # fmt: skip
    doc = {
        "daemons": [_identity(d) for d in canonical(daemons)],
        "states": [[s.label, s.ref, s.template] for s in states],
        "inputs": msgspec.to_builtins(inputs),
        "expected_inputs": dict(sorted(expected_inputs.items())),
        "seeds": list(seeds),
        "steps": steps,
        "change_interval": change_interval,
        "conditions": list(conditions),
        "config_texts": texts,
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def _config_texts(
    s: State,
    inputs: protocol.Inputs,
    seeds: Sequence[int],
    steps: int,
    change_interval: int,
    conditions: Sequence[str],
) -> list[str]:
    train = protocol.train_jobs(s.template, inputs, data_ref=DATA_REF, seeds=seeds,
                                steps=steps, change_interval=change_interval)  # fmt: skip
    frozen = [j for seed in seeds
              for j in protocol.frozen_jobs(s.template, inputs, data_ref=DATA_REF, seed=seed,
                                            conditions=conditions, base_step=steps)]  # fmt: skip
    return [j.config_text for j in (*train, *frozen)]


def _evaluation_tag(fp: str) -> str:
    """The start of every tag of one evaluation's jobs, whatever the run: `cancel_evaluation`
    finds them by it without a ledger."""
    return f"coast-eval {fp[:16]} "


def cancel_evaluation(daemons: Sequence[Daemon], fp: str) -> list[tuple[Daemon, str, str]]:
    """Cancel every active job of the evaluation with fingerprint `fp` on `daemons`, from any
    of its runs; done jobs and other evaluations' jobs are left alone. Returns (daemon, job
    id, label/seed/condition) of each job cancelled."""
    prefix = _evaluation_tag(fp)
    cancelled: list[tuple[Daemon, str, str]] = []
    for d in canonical(daemons):
        for j in d.client.list_jobs():
            if j.bucket == "active" and j.user_message.startswith(prefix):
                d.client.cancel(j.id)
                cancelled.append((d, j.id, j.user_message.removeprefix(prefix).partition(" ")[2]))
    return cancelled


def _identity(d: Daemon) -> str:
    return f"{d.info.hostname}:{d.info.tcp_port}"


@contextmanager
def _locked(ledger: Path) -> Generator[None]:
    """Hold an exclusive lock on the ledger for the whole run: two evaluations on one ledger
    would submit the same jobs twice. The kernel drops
    the lock when the process ends, SIGKILL included, so a crash never leaves it stale."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_name(ledger.name + ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"{ledger}: another evaluation is running on this ledger") from None
        yield


def _open_ledger(path: Path, fp: str) -> Ledger:
    """The ledger at `path`, or a new one with a fresh run id, written before any job is
    submitted under that id."""
    if path.exists():
        try:
            ledger = msgspec.json.decode(path.read_bytes(), type=Ledger)
        except (msgspec.DecodeError, msgspec.ValidationError) as e:
            raise ValueError(f"{path}: not a coast-eval ledger: {e}") from e
        if ledger.fingerprint != fp:
            raise ValueError(
                f"{path}: the ledger of another evaluation (other states, seeds, inputs or "
                "daemons); remove it to start afresh, or write this one elsewhere"
            )
        return ledger
    ledger = Ledger("coast-eval-ledger/1", fp, secrets.token_hex(8))
    part = path.with_name(path.name + ".part")  # atomic: no torn ledger after a crash
    part.write_bytes(msgspec.json.encode(ledger))
    os.replace(part, path)
    return ledger


Key = tuple[int, str]  # (seed, condition): the key `report.measure_state` reads


def run_milestone1(
    daemons: Sequence[Daemon],
    states: Sequence[State],
    inputs: protocol.Inputs,
    *,
    expected_inputs: Mapping[str, str],
    seeds: Sequence[int],
    steps: int,
    change_interval: int,
    conditions: Sequence[str],
    poll_s: float,
    timeout_s: float,
    ledger: Path,
    retry_s: float = 60.0,
    priority: int = 0,
    redo: Collection[str] = (),
) -> dict[str, dict[Key, runs.RunDir]]:
    """Train one state per (state, seed), then run every condition frozen from it. Returns,
    per state label, the run dirs keyed (seed, "train") and (seed, condition).

    `expected_inputs` maps every file a run reads, relative to dataset_base, to its SHA256.
    Each daemon must report those hashes (`hash_inputs`) before the first submit and again
    after the last job is done, which bounds the window in which a file could have changed
    under a run. Each run dir must be the job's (its `config.submitted.json` is the text
    submitted), pass the provenance check, name the daemon's host and link `data` to the
    daemon's dataset_base.

    Fails fast. A `list` that cannot connect is retried for `retry_s` before it is fatal;
    any other error, and `timeout_s` passing, raises at once, with a note naming the jobs
    still active on each daemon and how to cancel them.

    Resumable. `ledger` names the run (`Ledger`), and each job is tagged `coast-eval
    <fingerprint> <run id> <label/seed/condition>` on its daemon. Run with an existing
    ledger, the same evaluation re-attaches: for each key the newest job a daemon lists with
    its tag is adopted if it is active or done when this invocation starts (a done one is
    checked and used, an active one waited for); one that had failed or been cancelled is
    submitted afresh; a job that fails from then on stops the run.

    `redo` names keys (`label/seed/condition`) to submit afresh although a job of theirs is
    adopted: the way back from a run dir the checks refused (a crash can leave a run's
    observe files incomplete). A training key redoes its seed's frozen trials too. Every
    refusal names its key; with `redo` the evaluation continues where it stopped.

    A ledger of another evaluation
    (`fingerprint`) is refused. Without the ledger the evaluation starts afresh; jobs of the
    old run that are still active keep running on their daemons (`cancel_evaluation` stops
    them).
    """
    if not daemons:
        raise ValueError("need at least one daemon")
    daemons = canonical(daemons)
    labels = [s.label for s in states]
    if not states or len(set(labels)) != len(labels):
        raise ValueError(f"need one or more states with unique labels, got {labels}")
    for s in states:
        if not re.fullmatch(r"[0-9a-f]{40}", s.ref):
            raise ValueError(f"state {s.label!r}: ref must be a full commit SHA, got {s.ref!r}")
    placement = assign(seeds, len(daemons))
    keys = {_key(s.label, seed, c) for s in states for seed in seeds for c in (TRAIN, *conditions)}
    if unknown := sorted(set(redo) - keys):
        raise ValueError(f"--redo names no job of this evaluation: {unknown}")
    redone = set(redo) | {k.rsplit("/", 1)[0] + "/" + c for k in redo if k.endswith("/" + TRAIN)
                          for c in conditions}  # fmt: skip
    fp = fingerprint(daemons, states, inputs, expected_inputs, seeds, steps, change_interval,
                     conditions)  # fmt: skip
    by_label = {s.label: s for s in states}
    first: list[Planned] = []
    for s in states:
        train = protocol.train_jobs(s.template, inputs, data_ref=DATA_REF, seeds=seeds,
                                    steps=steps, change_interval=change_interval)  # fmt: skip
        first += [Planned(s.label, s.ref, job, 0) for job in train]

    def follow(done: Done) -> list[Planned]:
        if done.planned.job.condition != TRAIN:
            return []
        s = by_label[done.planned.label]
        base = runs.read_manifest(done.run).n_steps_completed  # load_state_step
        frozen_jobs = protocol.frozen_jobs(
            s.template, inputs, data_ref=DATA_REF, seed=done.planned.job.seed,
            conditions=conditions, base_step=base,
        )  # fmt: skip
        return [Planned(s.label, s.ref, f, base, done.job_id) for f in frozen_jobs]

    finished = run_jobs(daemons, placement, first, follow, fp=fp,
                        expected_inputs=expected_inputs, poll_s=poll_s, timeout_s=timeout_s,
                        ledger=ledger, retry_s=retry_s, priority=priority,
                        redone=redone)  # fmt: skip
    out: dict[str, dict[Key, runs.RunDir]] = {s.label: {} for s in states}
    for d in finished:
        out[d.planned.label][d.planned.job.seed, d.planned.job.condition] = d.run
    return out


def run_jobs(
    daemons: Sequence[Daemon],
    placement: Mapping[int, int],
    first: Iterable[Planned],
    follow: Callable[[Done], Iterable[Planned]],
    *,
    fp: str,
    expected_inputs: Mapping[str, str],
    poll_s: float,
    timeout_s: float,
    ledger: Path,
    retry_s: float,
    priority: int,
    redone: Collection[str],
    started: Callable[[], None] = lambda: None,
    failed: Callable[[Planned], None] = lambda _: None,
) -> list[Done]:
    """Submit `first`, then, as each job is done and checked, the jobs `follow` plans after
    it, until none is pending; returns every job's `Done` in the order they settled.
    `run_milestone1` documents the rules (inputs verified before and after, the checks of a
    run dir, fail-fast, the ledger, adoption, `redone` keys submitted afresh).

    `daemons` are in `canonical` order, `placement` maps each seed to one of them, `fp` is the
    `fingerprint` of everything that decides the jobs. `started` runs once the ledger is held
    and the inputs are verified, before any submit: a driver writes its own state there,
    never while another invocation holds the ledger. `failed` is told the job whose failure
    or refused run dir is about to stop the run."""
    with _locked(ledger):
        book = _open_ledger(ledger, fp)
        for d in daemons:
            verify_daemon_inputs(d.client, expected_inputs)
        started()
        tag = f"{_evaluation_tag(fp)}{book.run_id} "
        # this run's jobs on each daemon, by key: the newest is adopted if active or done
        tagged = [_adoptable(_list(d, retry_s, poll_s), tag, redone) for d in daemons]

        pending: dict[tuple[int, str], _Job] = {}  # by (daemon, job id): ids are per daemon
        finished: list[Done] = []

        def submit(p: Planned) -> None:
            i = placement[p.job.seed]
            job_id = tagged[i].get(p.key)
            if job_id is None:  # never submitted, or it had failed or been cancelled
                job_id = daemons[i].client.submit(p.ref, p.job.config_text, p.job.steps,
                                                  priority=priority + p.priority,
                                                  load_state_from=p.source,
                                                  message=tag + p.key)  # fmt: skip
            pending[i, job_id] = _Job(p, i, job_id)

        try:
            for p in first:
                submit(p)
            deadline = time.monotonic() + timeout_s
            while pending:
                for i in sorted({j.daemon for j in pending.values()}):
                    listed = {j.id: j for j in _list(daemons[i], retry_s, poll_s)}
                    for key, job in sorted(pending.items()):
                        if job.daemon != i:
                            continue
                        summary = listed.get(job.job_id)
                        if summary is not None and summary.bucket != "active":
                            del pending[key]  # settled: never "still active", even if it fails
                        try:
                            run = _settled(job, summary, daemons[i])
                        except ValueError as e:  # the run dir was refused: redo it
                            k = job.planned.key
                            e.add_note(
                                f"refused: {k} ({job.job_id} on {daemons[i].name}); to "
                                f"redo it, run the same evaluation with --redo {k}"
                            )
                            failed(job.planned)
                            raise
                        except DaemonError:
                            failed(job.planned)
                            raise
                        if run is None:
                            continue
                        done = Done(job.planned, job.job_id, run)
                        finished.append(done)
                        for p in follow(done):
                            submit(p)
                if pending and time.monotonic() >= deadline:
                    raise DaemonError(f"timed out after {timeout_s} s")
                if pending:
                    time.sleep(poll_s)
        except BaseException as e:  # Ctrl-C too: the jobs keep running on the daemons
            e.add_note(_still_active(daemons, pending.values()))
            e.add_note(f"run the same evaluation again to re-attach to them (ledger {ledger})")
            raise
        for d in daemons:
            verify_daemon_inputs(d.client, expected_inputs)
        return finished


def _adoptable(listed: Iterable[JobSummary], tag: str, redone: Collection[str]) -> dict[str, str]:
    """Job id by key of this run's jobs to adopt: per key the newest tagged job, if it is
    active or done and its key is not to be redone. Two newest jobs of one key (submitted
    in the same second) are refused: which one counts would be a guess."""
    newest: dict[str, list[JobSummary]] = {}
    for j in listed:
        if not j.user_message.startswith(tag):
            continue
        key = j.user_message.removeprefix(tag)
        if key not in newest or j.submitted_at > newest[key][0].submitted_at:
            newest[key] = [j]
        elif j.submitted_at == newest[key][0].submitted_at:
            newest[key].append(j)
    if tied := sorted(k for k, js in newest.items() if len(js) > 1):
        raise DaemonError(f"jobs submitted in the same second for {tied}: cannot tell the newest")
    return {key: js[0].id for key, js in newest.items()
            if js[0].bucket in ("active", "done") and key not in redone}  # fmt: skip


def _list(daemon: Daemon, retry_s: float, poll_s: float) -> list[JobSummary]:
    """`list`, retrying a failed connection for `retry_s`: a poll lasting hours meets
    transient network errors. Never used for `submit`, which is not idempotent."""
    deadline = time.monotonic() + retry_s
    while True:
        try:
            return daemon.client.list_jobs()
        except DaemonUnreachable:
            if time.monotonic() >= deadline:
                raise
            time.sleep(max(poll_s, 0.01))


def _settled(job: _Job, listed: JobSummary | None, daemon: Daemon) -> runs.RunDir | None:
    """The job's checked run dir once it is `done`, None while it is active; raises on any
    other bucket and on a run dir that is not this job's."""
    if listed is None:
        raise DaemonError(f"job {job.job_id} is not known to {daemon.name}")
    if listed.bucket == "active":
        return None
    if listed.bucket != "done":
        raise DaemonError(
            f"job {job.job_id} on {daemon.name} ended in {listed.bucket}: {listed.failure_reason!r}"
        )
    p = job.planned
    run = run_dir(job.job_id, daemon.runs_root)
    submitted = run.root / "config.submitted.json"
    if not submitted.is_file() or submitted.read_text() != p.job.config_text:
        raise ValueError(
            f"{submitted}: not the config submitted as job {job.job_id}; is "
            f"{daemon.runs_root} the runs_root of {daemon.name}?"
        )
    m = runs.read_manifest(run)
    runs.check_provenance(m, git_sha=p.ref, base_step=p.base_step, steps=p.job.steps,
                          frozen=p.job.condition != TRAIN)  # fmt: skip
    runs.check_run_id(run, m)
    if m.hostname != daemon.info.hostname:
        raise ValueError(
            f"{run.manifest}: written on {m.hostname!r}, but {daemon.name} reports host "
            f"{daemon.info.hostname!r}"
        )
    check_data_link(run, daemon.info.dataset_base)
    return run


def _still_active(daemons: Sequence[Daemon], jobs: Iterable[_Job]) -> str:
    lines = [
        f"  {daemons[j.daemon].name}: {j.job_id} "
        f"({j.planned.label}, seed {j.planned.job.seed}, {j.planned.job.condition})"
        for j in sorted(jobs, key=lambda j: (j.daemon, j.job_id))
    ]
    if not lines:
        return "no eval job was still active"
    cancel = """printf '{"cmd":"cancel","id":"<id>"}\\n' | nc HOST PORT"""
    return f"eval jobs still active (stop each with {cancel}):\n" + "\n".join(lines)
