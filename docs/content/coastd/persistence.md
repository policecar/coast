+++
title = "Persistence & the Child Contract"
weight = 30
+++

# Persistence & the Child Contract

Two halves of one question: *what does `coastd` write down*, and *how does it
launch and re-launch the thing that does the actual work?*

## The two-tree layout

```
<state_root>/                     git-versioned, SMALL TEXT ONLY, authoritative
  jobs/active/<id>.json             the queue
  jobs/{done,failed,cancelled,quarantine}/<id>.json

<runs_root>/                      NOT versioned, large/binary
  <job_id>/
    config.json                     what coast actually runs
    config.submitted.json           what the client sent, verbatim
    data -> <dataset_base>          symlink, only when dataset_base is configured
    snapshots/*.h5                  checkpoints
    logs/run_<n>.log                one per launch attempt
    coast_runtime.json              run manifest (clean exit only)
    final_state.h5                  --save-state target

<worktrees_root>/
  <sha>/                          per-resolved-SHA build tree
    build/coast                     the binary jobs on this SHA run
    .coastd-ready                   build-completion sentinel
    coastd-build.log
```

The three roots are **independently configurable** because they have completely
different size, durability and backup characteristics. `state_root` is tiny and is
the thing you back up — it *is* the queue. `runs_root` is large and belongs on
whatever volume has room. `worktrees_root` is large and entirely regenerable.

Each defaults to `<data-root>/{state,runs,worktrees}` but is overridable by a
`coastd.json` key of the same name (and `repo_root` likewise, which `--repo` then
overrides). A literal `<data-root>` token in the value expands to the resolved
data root, so `"runs_root": "<data-root>/runs"` and `"runs_root": "/mnt/big/runs"`
are both valid. Only `data_root` itself has no key — the config file is discovered
at `<data-root>/coastd.json`, so it must be resolved before the file is read. This
is a genuine split of the roots, not a symlink workaround.

The split exists for a hard reason: **HDF5 checkpoints cannot be git-versioned**.
So the heavy artifacts live in a separate tree that the small, versionable job
file points at.

### The `data` symlink

A child chdirs to its run dir before it runs, so a submitted config's *relative*
dataset path — the `"data/<set>/..."` convention every stock config uses —
resolves inside that run dir. The optional `dataset_base` config key names the one
shared dataset directory on the machine; when set, `coastd` drops a `data` symlink
to it into each run dir at submit time, so those relative paths just work and a
submitter never has to hard-code an absolute dataset location. It is a path key
like the roots (`<data-root>` token accepted) but is absolutized before the link
is made — a relative target would point at the run dir, not the operator's data.
Leaving it unset disables the mechanism: no link is created and configs must
reference datasets by absolute path (the prior behaviour). Linking is best-effort
and idempotent — a resume finds the link already present, a stale link (after a
`dataset_base` change) is refreshed, a real `data` file/dir is never clobbered,
and any failure only costs the convenience, never the run.

## The filesystem is truth; git may lag

> A lifecycle transition is an **atomic `rename`** between bucket directories. A
> git commit follows, asynchronously, and **may lag**.

A crash between the rename and its commit leaves the filesystem correct and one
commit missing. Boot re-commits catch-up. **Git is never on the correctness
path** — it is an audit trail that buys free history (`git -C <state_root> log`
shows every transition a job ever made) and costs nothing when it fails.

The bucket *directory* is the authoritative lifecycle state. The `status` field
inside the file is **advisory** — a last-recorded phase for humans and for `list`,
never trusted over reality. That is why a transition is a pure rename with no
content change: there is nothing in the content that needs to agree.

The dispatcher does update `status` just before a terminal move, purely so a
client never sees `status: "running"` next to `bucket: "done"` — a cosmetic
consistency, not a correctness one.

### Two write paths, chosen by consequence

| Call | Behaviour | Used when |
|---|---|---|
| `put` | Atomic `tmp → fsync → rename → fsync(dir)`. Throws on failure. | Normal persistence. |
| `try_put` | Same durable write, but **never throws** and **always refreshes the in-memory cache**. Returns false on failure. | Where a lost update would *wedge* the daemon rather than merely miss a persist. |

The `try_put` cases are specific and each has a named hazard. Clearing a child's
pid on exit: if that write fails and the cache keeps a stale nonzero pid, the
spawn dispatcher's skew guard blocks the resume **forever**. Recording a pid after
a successful spawn: an exception unwinding there leaves a live child unrecorded,
so a later restart cannot detect the orphan and would spawn a second child on the
same run directory.

In both cases keeping the cache correct lets the daemon proceed; the next
successful `put` re-persists, and a crash before that is healed by boot recovery
re-deriving from disk. `try_put` **never creates** a new entry — a create with no
durable record would be a phantom job.

## Job identity

```
20260720T143005Z_mnist_9f3a1c
└── UTC timestamp ──┘ └mnemonic┘ └rand┘
```

The filename **is** the id. Time-sortable, so `ls` gives free FIFO ordering and a
human-readable queue. The mnemonic is the client's label, else the config
basename, sanitized to `[a-z0-9-]` and capped at 12 characters. Uniqueness comes
from the timestamp plus 24 bits of randomness, with a collision guard that
regenerates rather than overwrites — the store never silently loses a job.

There is deliberately **no persisted counter**. A counter is exactly the kind of
state that desyncs on a crash.

## Job schema and version tolerance

The job file is a **flat struct plus an advisory enum**, chosen over a
`variant`-of-states so that flat-JSON serialization stays trivial and `status`
stays genuinely advisory. `bucket` is intentionally *absent* from the file — it is
the directory.

Evolution is **tolerant and additive**: nlohmann ignores unknown keys and defaults
missing ones, so additive changes need no migration at all. `schema_version` is
the escape hatch for non-additive changes, and there is deliberately no migration
framework built up front.

{{% callout info %}}
**The hard invariant: never silently mangle.** A file that cannot be parsed, or
that declares a `schema_version` this daemon cannot interpret, is moved to
`jobs/quarantine/` **untouched** and skipped. It is never rewritten, and it can
never crash-loop the queue. Because the files are plain JSON in git, a quarantined
job stays fully analyzable — by a human, a later daemon, or `jq`.

Check `quarantine/` after any upgrade that touches the store.
{{% /callout %}}

## The child contract

`coastd` owns output locations **by construction** rather than by convention. This
is the part most likely to surprise: the daemon does not guess where `coast` will
write, and it does not re-derive `coast`'s path-resolution rules. It arranges
things so there is only one possible answer.

Three mechanisms stack:

**1. The config is rewritten.** `prepare_run_config` parses the submitted bytes,
sets `snapshot_dir` to an **absolute** `<run_dir>/snapshots`, and re-serializes to
`config.json`. Absolute wins verbatim under the path resolver's rules, so the
child's working directory becomes irrelevant to where snapshots land.

**2. Every path argument is absolute.** `--save-state`, `--load-state` and
`--emit-run-manifest` are passed as absolute paths. `coastd` therefore knows
exactly where to look for checkpoints without re-deriving anything.

**3. A daemon-owned app config shadows the ambient one.** At boot the daemon
writes `<data-root>/coast_app_config.json` with `config_base = "<config_dir>/"`
and passes it to every child via `--app-config`. Because each child is launched
with `-c <run_dir>/config.json`, `<config_dir>` expands to that job's run
directory, so every app-config-governed output — observe, logs, the snapshot
default — resolves *flatly* inside it.

The third mechanism is doing defensive work, not just configuration. Being an
**explicit** `--app-config` it has the highest discovery precedence, so it
**shadows** any inherited `$COAST_APP_CONFIG` or stray `./coast.json` that would
otherwise silently redirect a job's outputs out of its run directory. Do not set
`COAST_APP_CONFIG` in the service environment expecting it to take effect.

This is also why `coastd_core` links `coast_lib`: it uses `coast::paths` — the
*same resolver the child writes through* — rather than a parallel reimplementation
that could drift.

### Two config files, and why

| File | What it is |
|---|---|
| `config.json` | What `coast` actually runs: the parsed config with `snapshot_dir` rewritten, re-serialized (key order and indentation are nlohmann's, not the submitter's). |
| `config.submitted.json` | Exactly what the client sent, byte for byte. The provenance record. Nothing reads it; failing to write it does not fail the submission. |

Because `prepare_run_config` rewrites and re-dumps, `config.json` is **not
evidence of what was submitted**. Without the second file there would be no copy
of the user's actual bytes anywhere — which is why the wire carries the config as
a JSON *string* rather than a nested object (see
[Protocol]({{% relref "protocol" %}}#config-travels-as-content)).

This is a documented deviation from the design document, which specified one file.

### Launch and resume

```cpp
launch_plan plan_launch(const job& j, const daemon_config& cfg,
                        const std::filesystem::path& coast_bin);
```

A fresh run and a resume differ only in two arguments. If a checkpoint exists, the
plan becomes a resume: `--load-state <newest>` and **`-n` set to the remaining
steps**, not the original target. Relative `-n` is required because `coast`
counts steps for *this invocation*.

A job submitted with `load_state_from` ([Protocol]({{% relref "protocol" %}}#starting-from-another-jobs-final-state-load_state_from))
differs only on its first launch: with no checkpoint of its own, the plan passes
`--load-state <load_state>` (the source job's `final_state.h5`) and
`-n steps_target - load_state_step`, since coast continues that state's step count. Once the job has a checkpoint, it resumes from
that like any other job and the initial state is never loaded again.

The manifest is **deleted before every launch**, which turns "a manifest is
present" into "*this* launch exited cleanly" — a fact the exit path and boot
recovery both depend on. A SIGKILLed child writes none. `final_state.h5` is deleted with it, so a
`done` job's final state is the one its finishing run wrote — the file a
`load_state_from` job loads at that job's `steps_done`. This holds for a job run by
a daemon that deletes the file; a `done` job from an older daemon may hold an
earlier final state.

Log files are named `run_<steps_done>_<launch_failures>_<no_checkpoint_launches>.log`
— one **per launch attempt**, so a crash-loop leaves every attempt's log intact
rather than overwriting the evidence.

### Checkpoint identity is coastd's, not coast's

`coast` names checkpoints `<stem>_<YYYYMMDD_HHMMSS>_<step>.h5` — no random
component, one-second granularity. Demonstrate flagged that a resume re-emitting
the same step within the same wall-clock second could overwrite. So `coastd` owns
checkpoint identity itself: it selects the **newest by trailing `_<step>.h5`**
within the job's own snapshots directory, and never relies on `coast`'s filenames
being globally unique.

### Status: tail the log, trust `waitpid`

The child's stdout and stderr are redirected to the run-dir logfile, which is
simultaneously the human-readable log **and** the machine-readable status feed.
The supervisor's poll thread (~10 Hz) tails it through `line_tailer` into the
existing `parse_progress()` from `include/diag/progress.h`.

File-over-pipe was chosen deliberately. A pipe imposes **backpressure**: a slow or
stalled daemon would block the simulation itself. A file does not, survives a
daemon restart, and is directly inspectable with `tail -f`. The obvious objection —
text parsing is brittle — is contained by `diag/progress.h` co-locating the
emitter and the parser, so a field rename breaks *both ends at compile time*.

Demonstrate measured this end to end: 13 live updates with the first at 0.634 s, no
misparses across partial lines, and clean exit-code agreement. Polling at 200 ms–1 s
loses no fidelity, because the latency floor is `coast`'s own ~0.5 s emit cadence.

The authority split is strict:

- **`waitpid` is authoritative** for liveness and exit. `supervisor::live_ids()`
  is what `reality.child_alive` reports.
- **`phase=done` on the tail merely corroborates.** A non-zero exit without
  `done` means interrupted → relaunch.
- **The newest `state_NNN.h5` on disk is the authoritative checkpoint.** The
  `phase=checkpoint` line is only a prompt — it fires when the write is *queued*,
  not completed.

Under R1 there is no reattach, so no filesystem or PID liveness markers exist at
all.

### Run length, and why the child must say `--save-on-term`

`coast` requires an **explicit** run length: `-n <count>` with a count ≥ 1, or
`--unlimited`. There is no sentinel — `-n 0` is a parse error, not "run forever".
So `plan_launch` emits `-n remaining` for a bounded job and `--unlimited` for an
unbounded one (`steps_target == 0`), never both and never `-n 0`.

The two halves of the command line are coupled, and it is easy to break by
deleting the wrong flag. `--unlimited` normally **refuses** `--save-state`,
`--emit-run-manifest` and `--log-buffers`: a run that ends only by being killed
would never write them (and would leave the buffer log unclosed and unreadable).
`--save-on-term` is what lifts that refusal — with it, the daemon's `SIGTERM`
stops the loop and the child leaves through the *normal* exit path, closing the
buffer log and writing the final state and a manifest whose `exit_reason` is
`user_signal`. Dropping `--save-on-term` from `plan_launch` would therefore not
merely lose a checkpoint on shutdown; it would turn **every unbounded launch into
a parse error**. `test_child_cmd.cpp` asserts the pairing.

### Cancel and shutdown share one path

`SIGTERM`, then a `term_grace` window, then `SIGKILL`. The grace exists so the
child can write a final checkpoint via `coast --save-on-term`. The same escalation
serves a **job cancel** and a **daemon shutdown** — `supervisor::shutdown` SIGTERMs
every child and waits the same window.

Two consequences worth internalizing: raise `term_grace` for large models whose
final checkpoint write exceeds 10 s, or a shutdown discards that work; and keep
systemd's `TimeoutStopSec` above it, or systemd kills the daemon mid-grace. The
`--install-service` installer computes `TimeoutStopSec` as
`build_timeout + term_grace + 30` — dominated by an in-flight *build* step, not by
the child's grace, since `worker_pool::shutdown` lets a running build finish.

Graceful checkpoint-then-exit is an **optimization, not a correctness
prerequisite**: hard-kill plus resume-from-prior was already safe.

## The build side

```
git worktree add <worktrees_root>/<sha>
git -c submodule.docs/themes/hugo-theme-relearn.update=none \
    submodule update --init --recursive --depth 1                          ← mandatory
cmake -B <wt>/build  -DCMAKE_BUILD_TYPE=Release  [build_cmake_args…]
cmake --build <wt>/build -j <build_jobs> --target coast
write .coastd-ready                          ← atomically, LAST
```

{{% callout warning %}}
**`git worktree add` does not populate submodules.** Without the explicit
`submodule update`, the CMake configure fails. The step is mandatory.
{{% /callout %}}

Each worktree clones its own submodules. The step initialises every submodule
except the docs theme (`docs/themes/hugo-theme-relearn`, 1.1 GiB of history), at
depth 1: the theme is the one submodule a build never compiles. It is excluded by
name, so the step works on any tree the daemon builds, including a test fixture
with no submodules. An operator who points the submodule URLs at local mirrors
must set `protocol.file.allow=always` in the daemon's environment; `coastd` does
not set it itself, because that would disable git's CVE-2022-39253 mitigation in
every deployment. INSTALL §1.2 has the exact variables.

The **`.coastd-ready` sentinel** is what makes a SHA runnable — *not* the presence
of the worktree directory, which `git worktree add` creates seconds into a
minutes-long build. `reality.builds_ready` reports the sentinel, so a job is never
spawned against a half-built tree. It also makes a build interrupted by a daemon
restart resume correctly: the directory exists, the sentinel does not, so
`reconcile` emits `build_worktree` again and CMake's incremental logic picks up
where it left off.

Staleness is **delegated entirely to CMake**. Demonstrate confirmed the behaviour:
a no-op rebuild took 0 s, a one-file edit produced a 5 s targeted rebuild, and the
other worktree was untouched. `_deps` is deliberately **not** shared between trees
— the per-tree wgpu copy is intentional isolation. Only the downloaded zip is shared, through the
machine-wide dependency cache (`COAST_DEPS_CACHE`), so a new worktree's configure
needs no network once the cache holds a verified copy.

Concurrency is `build_concurrency = 1` with `-j = build_jobs` (default: half the
detected cores). The build gets half the box; running simulations keep the other
half.

### Ref freshness

`coastd` resolves a submitted ref against `repo_root`. Without a fetch, that
resolution sees only whatever the local clone happens to hold — so a stale clone
would **silently run old code under a current-sounding ref name**. That is worse
than a clean "could not resolve", because the job succeeds and lies.

So resolution fetches first (`git fetch --all --prune --tags`), then prefers
`<remote>/<ref>` over a same-named local branch for plain branch names. The
justification is specific: `git fetch` updates *remote-tracking* refs and does not
move your local `main`, and since nothing ever checks anything out in `repo_root`,
that local branch is frozen at whatever the clone was created with.

Explicit forms still resolve verbatim — `refs/heads/x`, `HEAD`, a raw SHA, a tag,
or anything carrying revision syntax (`^`, `~`, `@{…}`). Prefixing those would
change their meaning.

Two properties keep this from becoming a reliability regression:

- **A fetch failure is never fatal.** No network, no remote, or a transient
  `cannot lock ref` race against a concurrent `git worktree add` all produce a
  warning and fall back to the local clone. Turning a network blip into a
  terminally-failed job would trade a freshness feature for a reliability loss.
- **Fetches are throttled and coalesced.** At most one per `fetch_min_interval`
  (default 30 s) across all concurrent resolutions — so a burst of fifty
  submissions produces one fetch, not fifty. The throttle is a mutex held for the
  whole fetch: a resolution that arrives meanwhile waits for it, then looks up its
  ref in what the fetch brought in. (An atomic compare-and-swap did the claiming
  until 2026-11-07; the others then looked up their ref at once, so in a burst
  for a commit the clone had not seen, a job that looked before the fetch finished
  failed with "could not resolve ref".)

`--no-fetch` restores purely local, local-first resolution. The two are one
capability: disabling the fetch also disables the remote-first preference.

## Subprocess bounds

Every builder subprocess has a wall-clock ceiling, and the framing matters:

> These are **bounds, not hang detectors.** A build that is merely slow — a cold
> `FetchContent`, a big submodule fetch on a poor link — is externally
> indistinguishable from one that is hung. CPU and memory tell you nothing.

So total duration is capped and an overrun is treated as a build failure rather
than a wedge. The hazard being closed is specific: an unbounded builder step can
silently park a worker thread forever, and with `build_concurrency == 1` that
blocks **all** future builds for the daemon's lifetime.

| Knob | Default | Bounds |
|---|---|---|
| `build_timeout` | 3600 s | Each heavy step (submodule update, configure, build) — generous, so a healthy slow build is never killed |
| `git_op_timeout` | 60 s | Quick local git calls (resolve `rev-parse`, GC worktree remove/prune) |
| `fetch_timeout` | 300 s | `git fetch` — network-bound, so its own budget rather than sharing the 60 s local one |
| `audit_git_timeout` | 20 s | The committer thread's best-effort audit commit |

## Reading the state by hand

The store is plain JSON on a filesystem, so it is inspectable without the daemon:

```sh
ls ~/.local/share/coastd/state/jobs/*/
jq . ~/.local/share/coastd/state/jobs/active/*.json
git -C ~/.local/share/coastd/state log --oneline
```

**Do not edit those files while the daemon is running.** It is the single writer;
your write will be overwritten or lost.
