"""One coastd run directory and the `coast_runtime.json` manifest inside it.

Layout (`docs/content/coastd/persistence.md`): `config.json` (what coast ran),
`config.submitted.json` (the client's bytes), `coast_runtime.json` (written on a clean
exit only), `snapshots/`, `final_state.h5`, and the observe files wherever the manifest's
`observe.output_dir` says. A run that `scripts/eval_ab.py --binary` spawns also holds
`inputs.sha256.json`, the record of what it read beyond its config (`inputs_record`).

The manifest is the provenance: the binary's git SHA and dirty flag, and the step count.
Every reader here raises ValueError (or msgspec.ValidationError) instead of coercing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, cast

import msgspec

from coast_eval.datasets import sha256_file


class RunDir(msgspec.Struct, frozen=True):
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def manifest(self) -> Path:
        return self.root / "coast_runtime.json"

    @property
    def final_state(self) -> Path:
        return self.root / "final_state.h5"

    @property
    def inputs_record(self) -> Path:
        return self.root / "inputs.sha256.json"


class ObserveGroup(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    ranges: list[tuple[int, int]]  # inclusive absolute step ranges
    combine: Literal["or", "and"]


class ObserveBlock(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    output_dir: str  # "./…" = relative to the manifest's directory, else absolute
    groups: dict[str, ObserveGroup] | None  # null when the config declares none


class NodeSeed(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    id: int
    label: str
    effective_seed: int
    inhibition_seed: int | None = None


class SavedState(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    path: str
    step: int
    bytes: int


class Manifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """`coast_runtime.json`, schema "1" (`include/sim_core/run_manifest.h`).

    `n_steps_requested` is the `-n` of the launch that wrote the manifest; `n_steps_completed`
    is absolute, because coast continues a loaded state's step count.
    """

    schema_version: Literal["1"]
    coast_git_sha: str
    coast_git_dirty: bool
    coast_build_type: str
    seed_offset: int
    n_steps_requested: int
    unlimited: bool
    n_steps_completed: int
    exit_reason: Literal["completed_n_steps", "fatal_error", "user_signal", "step_budget_exceeded"]
    wall_clock_seconds: float
    hostname: str
    gpu_adapter: str | None
    started_at_utc: str
    completed_at_utc: str
    observe: ObserveBlock
    datasets: msgspec.Raw  # audit mirror of the resolved dataset configs; not read here
    retina: msgspec.Raw
    node_seeds: list[NodeSeed]
    savestate_outputs: list[SavedState]
    # The toolchain that built the binary (`cmake/coast_build_info.cmake`): results are
    # bit-identical across hosts only for the same compiler and target CPU. Absent from
    # manifests of binaries built before these fields existed.
    coast_compiler: str | None = None
    coast_target_cpu: str | None = None
    # The run's top-level `frozen` (SPEC.md S.5.14). Absent from manifests of binaries
    # built before it existed, whose frozen trials did not keep learning off by it.
    frozen: bool | None = None
    # The run's id as it ran (SPEC.md S.5.14); absent before it existed.
    run_id: str | None = None


def read_manifest(run: RunDir) -> Manifest:
    """Decode `coast_runtime.json`; a missing file means the run did not exit cleanly."""
    if not run.manifest.is_file():
        raise ValueError(f"{run.manifest}: missing (the run did not exit cleanly)")
    return msgspec.json.decode(run.manifest.read_bytes(), type=Manifest)


def check_run_id(run: RunDir, m: Manifest) -> None:
    """The run dir's `config.json` names one concrete `run_id`, and the run ran under it.

    coastd replaces "auto" when it writes a job's `config.json`, so every launch of the job
    is the same run and a resume continues its image sequence (SPEC.md S.5.14). A coastd
    from before that leaves "auto", each launch draws its own id, and a resumed frozen
    trial would restart its sequence while counting its steps right: refused here."""
    configured = json.loads(run.config.read_text()).get("run_id", "auto")
    if configured == "auto":
        raise ValueError(
            f'{run.config}: run_id is still "auto": the daemon\'s coastd does not fix a '
            "job's run_id, so a resume would restart the job's image sequence"
        )
    if m.run_id != configured:
        raise ValueError(
            f"{run.manifest}: ran as run {m.run_id!r}, its config names {configured!r}"
        )


def check_provenance(
    m: Manifest, *, git_sha: str, base_step: int, steps: int, frozen: bool
) -> None:
    """The run came from a clean build of `git_sha`, was frozen exactly when `frozen`, and
    ran its `steps` from `base_step`.

    `base_step` is the loaded state's step (0 for a fresh run; a coastd job's
    `load_state_step`). The check uses the job's `steps`, not `n_steps_requested`: a coastd
    resume relaunches with `-n <remaining>`, so only the absolute count is comparable. A
    resume is valid for every run: coast continues a checkpoint's image sequence when the
    same run wrote it (`run_id`, SPEC.md S.5.14), which a coastd resume's own checkpoint
    always was.
    """
    if m.coast_git_sha != git_sha:
        raise ValueError(f"run built from {m.coast_git_sha}, expected {git_sha}")
    if m.coast_git_dirty:
        raise ValueError(f"run built from a dirty tree at {m.coast_git_sha}")
    if m.unlimited or m.exit_reason != "completed_n_steps":
        raise ValueError(f"run ended with {m.exit_reason} (unlimited={m.unlimited})")
    if m.frozen is None:
        raise ValueError(
            "run records no `frozen`: its coast predates the top-level frozen, so its "
            "frozen trials may have learned"
        )
    if m.frozen != frozen:
        raise ValueError(f"run was {'' if m.frozen else 'not '}frozen, expected the opposite")
    if m.n_steps_completed != base_step + steps:
        raise ValueError(
            f"run completed step {m.n_steps_completed}, expected {base_step} + {steps}"
        )


def observe_dir(run: RunDir, m: Manifest, groups: Iterable[str]) -> Path:
    """The directory holding `<group>.observe.h5`, as the manifest records it (never
    recomputed from the config), after checking that each of `groups` was observed."""
    declared = m.observe.groups or {}
    missing = sorted(set(groups) - declared.keys())
    if missing:
        raise ValueError(f"{run.manifest}: observe groups {missing} not in {sorted(declared)}")
    out = m.observe.output_dir
    if out.startswith("./"):
        return run.manifest.parent / out
    if not Path(out).is_absolute():
        raise ValueError(f"{run.manifest}: observe.output_dir {out!r} is neither ./ nor absolute")
    return Path(out)


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in cast(dict[str, Any], value).values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in cast(list[Any], value):
            yield from _strings(v)


def link(run: RunDir, name: str, target: Path) -> None:
    """Make `<run>/<name>` a symlink to `target`, replacing a link that resolves elsewhere
    (left by a run with another --data-root or --out); anything else at that name raises."""
    path, want = run.root / name, target.resolve()
    if path.is_symlink() and path.resolve() != want:
        path.unlink()
    elif path.exists() and not path.is_symlink():
        raise ValueError(f"{path}: exists and is not a symlink")
    if not path.is_symlink():
        path.symlink_to(want)


def inputs_record(
    run: RunDir, config_text: str, load_state: Path | None, binary: Path, steps: int
) -> bytes:
    """What decides a run beyond its config text, as canonical JSON: the `steps` it is
    launched for (`-n`, which no config holds: a training config is the same text at any
    length), the SHA256 of the coast `binary` that runs it, of the state file it loads (null
    for none) and of every `./inputs/...` file the config names, read
    through the run dir (the `--binary` driver links its inputs directory there), and the
    directory its `data` link resolves to. The dataset files themselves are not hashed here:
    the caller verifies that directory against the pins before any run."""
    names = sorted({v for v in _strings(json.loads(config_text)) if v.startswith("./inputs/")})
    data = run.root / "data"
    record = {
        "steps": steps,
        "binary": sha256_file(binary),
        "load_state": None if load_state is None else sha256_file(load_state),
        "inputs": {name: sha256_file(run.root / name) for name in names},
        "data": str(data.resolve()) if data.is_symlink() else None,
    }
    return (json.dumps(record, sort_keys=True, indent=1) + "\n").encode()


def reusable(run: RunDir, config_text: str, record: bytes) -> bool:
    """The run finished (its manifest exists) with exactly this config text and this
    `inputs_record`. Anything else, a missing record included, means the run is redone."""
    return (
        run.manifest.is_file()
        and run.config.is_file()
        and run.inputs_record.is_file()
        and run.config.read_text() == config_text
        and run.inputs_record.read_bytes() == record
    )
