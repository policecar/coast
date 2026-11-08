+++
title = "Decisions (ADR)"
weight = 60
+++

# coastd — Architecture Decisions

The load-bearing choices behind [coastd]({{% relref "." %}}), in ADR form. These
record the forces, the decision, the consequences, and — where it matters most —
the alternatives that were weighed and rejected.

The ranked design lenses that arbitrated every one of them:

1. **Robustness / correctness under fault** — dominant; wins conflicts.
2. **Simplicity / minimal footprint** — co-primary, and the *discipline* on #1:
   every mechanism must earn itself against a **named fault**. "As simple as
   necessary and possible, but not simpler."
3. **Cross-platform** — Linux primary (the real target), macOS secondary
   (don't preclude, don't gold-plate).

Explicit non-goals: raw throughput (a supervisor is not a hot path),
extensibility (the future multi-daemon pool must not be *precluded*, not built
toward), and push notifications (deferred to v2).

A meta-principle adopted mid-design shaped several of these directly:
**prefer system defaults; fighting them is brittle.**

---

## ADR-1 — Single-threaded state core with bounded edges

**Status:** Accepted

**Context.** A daemon that builds, schedules, supervises subprocesses and serves
clients has many concurrent sources of change. The project's hard reliability
gates (no lost jobs, no duplicate runs on shared output paths, no corrupt output)
are all statements about state consistency under fault.

**Decision.** One thread owns **all** job state and serializes every mutation.
Every other thread — worker, connection, supervisor poll — is an **edge** whose
sole interaction with state is `queue.push(event)`.

**Consequences.** The correctness argument becomes "single writer", which is
*structural* rather than empirical: "can two things race on this job?" has an
answer you can read off the design instead of one you hope your tests found.
Client RPCs need a closure-plus-promise seam to feel synchronous, and shutdown
must be ordered (comms down before the core, or a blocked handler wedges it). The
cost is negligible at this workload — a handful of jobs and a few events a second.

**Alternatives rejected.** Fine-grained locking around a shared job table —
conventional, and it would have made every subsequent reliability question a
lock-ordering argument rather than a structural one. Performance was never the
motivation for the single writer, so the usual reason to prefer locks did not
apply.

---

## ADR-2 — Declarative reconcile, not imperative transitions

**Status:** Accepted

**Context.** The obvious way to build a job scheduler is to push jobs through
states: on event X, move job from A to B. That design's recovery path is
necessarily *different* code from its steady-state path — and recovery code is
exercised only by rare events, which is precisely why it rots.

**Decision.** The state machine is a **pure function**,
`reconcile(jobs, reality, config) → actions`, re-evaluated on every event and on a
~1 Hz backstop. Boot does the same thing with an unusual starting state.

**Consequences.** The single biggest robustness lever in the project. **Recovery
is the steady-state path**, so it runs roughly once a second on every healthy
daemon rather than once a year in an emergency. The whole policy is table-driven
unit-testable with zero subprocesses. Actions are unordered and reconverging, so a
partial dispatch after a crash is harmless. In exchange, `reconcile` deliberately
over-emits actions (it cannot know what is in flight), which forces the dispatcher
to be idempotent per `(kind, key)` and to own the in-flight dedup sets.

**Alternatives rejected.** An event-sourced log with replay — more machinery,
and the reconstruction step is itself rarely-exercised code. An imperative state
machine with a separate `recover()` entry point — the exact failure mode above.

---

## ADR-3 — Git-versioned directory of per-file jobs; filesystem is truth

**Status:** Accepted

**Context.** The store must survive both process crash and machine reboot, be
inspectable by a human, and never lose or corrupt a job.

**Decision.** One JSON file per job under `<state_root>/jobs/<bucket>/`. A
lifecycle transition is an **atomic `rename`** between bucket directories; a
content update is `tmp → fsync → rename → fsync(dir)`. A `git` repository in the
state root records history **asynchronously and may lag**.

**Consequences.** Per-file granularity makes atomic write trivial — there is no
shared file to tear. The bucket directory *is* the lifecycle state, so a
transition needs no content change and the in-file `status` can stay honestly
advisory. Git buys free history and observability (`git log` shows every
transition a job ever made) at zero correctness risk, because **git is never on
the correctness path**: a crash between rename and commit leaves the filesystem
right and one commit missing, which boot re-commits. The audit commit runs on its
own thread, coalescing bursts, so the core never blocks on `git` and two `git`
processes never race on `.git/index.lock`.

**Alternatives rejected.** SQLite — a real dependency, opaque to `jq` and `git`,
and its durability story is not better than `rename` for this shape. A single
JSON file for the whole queue — every write rewrites everything and a torn write
loses the queue. Making git authoritative — puts a subprocess on the correctness
path.

---

## ADR-4 — Everything a crash could desync is derived, never stored

**Status:** Accepted

**Context.** Assumption A9 from Discover named the concrete nightmare: a batch
running at `sha1` while `main` advances to `sha2` and a second batch launches, with
jobs silently running the wrong code. That is a *stale-counter* bug in disguise.

**Decision.** Slot availability, SHA refcount, worktree presence, build readiness,
worktree idle age and relaunch eligibility are all **derived on every pass** from
the filesystem plus the job set. The worktree is keyed by the SHA **frozen at
submit**.

**Consequences.** Cross-batch contamination is impossible *by construction* rather
than by careful bookkeeping — a derived count cannot desync, because there is
nothing to desync. The cost is a directory scan and a few `stat` calls per pass,
which is nothing at this workload and far cheaper than owning a cache-invalidation
problem across crashes. Notably, this is also why the `job` struct carries **no
checkpoint-integrity field** (see ADR-8).

**Alternatives rejected.** A persisted `ref → worktree → refcount` map (the
original Discover sketch) — it is exactly the state that desyncs on an unclean
exit, and reconstructing it correctly at boot is the same work as deriving it
every time, done more rarely and therefore less well.

---

## ADR-5 — R1: children die with the daemon

**Status:** Accepted

**Context.** The highest-risk assumption in the project (A3) was whether `coastd`
could distinguish *daemon-crashed-with-live-children* (reattach or kill?) from
*reboot-killed-everything* (relaunch). Getting it wrong means either a duplicate
run clobbering shared output, or a live job silently dropped.

**Decision.** Do not solve the problem — **dissolve it.** Embrace systemd's stock
`KillMode=control-group`: children die with the daemon, so there are no live
orphans, ever. Every restart relaunches each interrupted job from its last good
checkpoint.

**Consequences.** The duplicate-run nightmare becomes **impossible by
construction** — a dead child cannot be duplicated. There is no reattach protocol,
no liveness marker files, no PID handshake, and exactly one recovery path. The
cost is up to one checkpoint interval of recomputed work per interruption, which
is pure efficiency, an explicit non-goal. Verified during Demonstrate: cgroup reap
3/3, and `KillMode=process` reproduced the orphan 3/3.

This is the meta-principle earning its keep: `coastd` gets its most important
safety property by **not writing code**, and the service unit must therefore
*not* override `KillMode`.

**Alternatives rejected.** Reattach-to-orphan — requires a durable channel to a
process you no longer own, and a wrong answer corrupts output. Kill-and-relaunch
guarded by liveness markers — more mechanism for the same outcome R1 gets free.

---

## ADR-6 — The pid duplicate-guard as R1's fallback

**Status:** Accepted

**Context.** R1 rests on "systemd always reaps". During development the daemon
runs outside systemd, and macOS has no cgroup at all. *"systemd always behaves" is
an assumption bound to fail eventually.*

**Decision.** At boot, before relaunching, check the recorded `pid` against a
**start-time token** read from `/proc/<pid>/stat`; kill a genuine survivor first.

**Consequences.** Recycle-safe — a reused PID belonging to an unrelated process has
a different start time and is rejected (validated 5/5). Under systemd it is a
no-op. The survivor was reparented away, so it can only be signalled, never
`waitpid`ed.

{{% callout info %}}
**Platforms:** the token comes from `/proc/<pid>/stat` on Linux and from
`sysctl(KERN_PROC_PID)` on macOS (ADR-17). On any other platform there is no
source, the guard could not fire, and the daemon refuses to start (ADR-16). See
[Boundaries]({{% relref "boundaries" %}}#macos--supported-but-not-unattended).
{{% /callout %}}

---

## ADR-7 — Child status by tailing a logfile, not a socket

**Status:** Accepted — *supersedes the Discover-stage "hybrid channel" sketch (A4)*

**Context.** Discover proposed a hybrid child↔daemon channel: structured status
over a socket plus filesystem/PID liveness markers. Sockets "can be their own
beasts", so this was flagged for empirical validation.

**Decision.** The child's stdout goes to a run-dir **logfile**, which is
simultaneously the human log and the machine feed. The daemon tails it and runs
the existing `parse_progress()` from `include/diag/progress.h`. Liveness is
`waitpid`, full stop.

**Consequences.** File-over-pipe means **no backpressure** — a slow or stalled
daemon can never block the simulation, which a pipe would. The log survives a
daemon restart and is directly inspectable with `tail -f`. The obvious objection,
text-parsing brittleness, is contained by `diag/progress.h` co-locating emitter and
parser, so a field rename breaks **both ends at compile time**. Under R1 no
reattach is needed, so the liveness-marker half of the sketch evaporated entirely.

Demonstrate measured it: 13 live updates, first at 0.634 s, zero misparses across
partial lines, clean exit-code agreement. Polling at 200 ms–1 s loses no fidelity
because the floor is `coast`'s own ~0.5 s emit cadence.

**Alternatives rejected.** The socket channel — the spike found no telemetry gap
or robustness deficit that would justify its added complexity. A pipe — imposes
backpressure on the simulation, the one thing that must never be slowed by the
supervisor.

---

## ADR-8 — Prevent torn checkpoints; do not detect them

**Status:** Accepted — *a Demonstrate finding that changed the design*

**Context.** Auto-resume from a corrupt checkpoint would be silent corruption. The
planned mitigation was a resume-time integrity check.

**Decision.** Make `coast` write checkpoints **atomic-durably**
(`tmp → fsync → rename → fsync(dir)`), after which the newest complete checkpoint
is intact by construction — and `reconcile` needs **no integrity check at all**.

**Consequences.** The `job` struct carries no checkpoint-integrity state, and
recovery simply picks the newest checkpoint by step and sweeps stale `.tmp` files.
The correctness property now lives in `coast`, not `coastd`, so the two must not
be silently decoupled: were that change ever reverted, `coastd` would have to
adopt the structural check.

**Alternatives rejected — and this is the point of the ADR.** A bare `H5Fopen`
liveness check was the *intuitive* design and it is **wrong**: a real `SIGKILL`
mid-write produces a full-length, header-corrupt file that `H5Fopen` accepts as
valid **5 times out of 5**. Shipping it would have been worse than shipping
nothing, because it would have looked like protection. A structural
`open + visit + read` check does work (109/109 truncations caught, down to one
byte) and is retained as the documented fallback. A bare `mv` without the `fsync`
fence is *not* equivalent — atomic against concurrent readers, but not durable
against power loss (the ext4-2009 empty-file-after-rename trap).

---

## ADR-9 — TCP + JSONL, thread-per-connection, poll to resync

**Status:** Accepted

**Context.** Clients are transient controllers that attach and detach, possibly
across a VPN, possibly on localhost. Lab-scale client counts.

**Decision.** TCP on 51900, one minified JSON object per line, thread-per-connection
with blocking sockets. Clients hold **no subscriber state** and stay fresh by
polling `list()` (~2 s).

**Consequences.** No new dependency (nlohmann/json is already vendored), and the
whole surface is `printf | nc`-speakable, which makes ad-hoc operation and scripted
B-tier tests trivial. Statelessness is what makes a client genuinely *detachable*:
quit the GUI, return, and one `list()` re-syncs the world — with no server-side
per-client state to leak or desync. No async machinery is needed at this scale.
The costs are real but bounded: no push (deferred to v2), and no request-id, which
forces the client to poison a connection after a transport error rather than risk
mis-attributing the previous command's reply.

**Alternatives rejected.** A subscription/push protocol — server-side subscriber
state is exactly what makes a detachable client hard. An async event loop — no
workload justifies it.

---

## ADR-10 — Ports are not the wire; the config's *content* is

**Status:** Accepted — ***supersedes IDD §6.6/§6.7***

**Context.** The design specified `submit{ref, config, …}` where `config` was a
**path**. That design never confronted the fact that the client and daemon are, in
the general case, different machines — even though remote operation is the
daemon's entire premise.

**Decision.** `submit` carries the config's **bytes** in `config_text` (plus
`config_name`, used only for the job-id mnemonic). The path form is removed
entirely, **not** kept as a fallback.

**Consequences.** Across a machine boundary a path is meaningless — and worse than
meaningless: a path existing on both hosts with *different content* runs the wrong
config **successfully**, undetectably by either side. Two developer boxes each with
`~/work/configs/mnist.json` is the expected case, not a contrived one. A wrong
answer that looks right is worse than a failure, so *an option to reintroduce a
defect is the defect* — the server rejects a legacy `config` key with an explicit
migration message. A second, independent forcing reason: the **GUI could not submit
at all** under the path model, since its editor holds an edited buffer with no path.

The accepted cost: `submit` is no longer comfortably hand-typeable and needs
`jq --rawfile`. The other four commands stay trivially `nc`-speakable.

Bytes travel as a JSON *string*, not a nested object, so the archived
`config.submitted.json` is byte-exact — measured at ~8.4 % escape overhead on a
real 8 KB config versus ~33 % for base64.

**Alternatives rejected.** Keeping the path form as a fallback — reintroduces the
silent-wrong-config failure for anyone who uses it. Base64 — 4× the overhead for a
property JSON string escaping already provides.

**Later addition (`load_state_from`).** A job that starts from another job's final
state names that job by **id**, not by path: the daemon resolves the id to the
job's `final_state.h5` itself. So the wire still carries no paths, and the same
silent-wrong-file failure cannot arise ([Protocol]({{% relref "protocol" %}}#starting-from-another-jobs-final-state-load_state_from)).

---

## ADR-11 — Path safety is checked in the frame the child will actually use

**Status:** Accepted — ***supersedes IDD §6.7***

**Context.** A submitted config must not be able to write into the daemon's own
state, a build worktree, or another job's run directory.

**Decision.** Recursively scan the config for path-like strings, resolving
relative paths against the **run-dir frame** (a `runs_root/__pending__` sentinel,
since the job id is not minted until the core creates the job). Paths inside the
job's own run directory are legitimate; escapes into forbidden roots are rejected.
The scan is key-agnostic — it does not trust field names, it inspects values.

**Consequences.** The child runs with its working directory set to the run
directory, so this is the frame that actually determines where a relative path
lands. The design's original frame — the *config file's parent* — could never
produce a path under a forbidden root, which meant relative paths were effectively
**unchecked**. `repo_root` is deliberately *not* forbidden, since configs
legitimately reference datasets living there. Because the exact validated bytes are
carried through to `prepare_run_config`, there is no submit-config TOCTOU by
construction.

---

## ADR-12 — Two config files per run

**Status:** Accepted — ***supersedes IDD §4A ("config-as-run-dir-file")***

**Context.** The design assumed one config file in the run directory would serve
as both the thing that runs and the record of what was submitted.

**Decision.** Write **two**: `config.json` (parsed, `snapshot_dir` rewritten to an
absolute path, re-serialized — what `coast` actually runs) and
`config.submitted.json` (the submitted bytes, verbatim — provenance only).

**Consequences.** Since `prepare_run_config` injects `snapshot_dir` and re-dumps,
`config.json` has nlohmann's key order and indentation and is therefore **not
evidence of what was submitted**. Without the second file no copy of the user's
actual bytes would exist anywhere. The provenance file is advisory: nothing reads
it, and failing to write it does not fail the submission.

---

## ADR-13 — Bounds on subprocesses, not hang detection

**Status:** Accepted

**Context.** With `build_concurrency = 1`, a single wedged builder step blocks
**all** future builds for the daemon's lifetime.

**Decision.** Cap each builder subprocess by wall clock and treat an overrun as a
**build failure**, not as a wedge.

**Consequences.** The framing is the decision: a build that is merely slow — cold
`FetchContent`, a large submodule fetch on a poor link — is **externally
indistinguishable** from one that is hung, and CPU and memory tell you nothing. So
no attempt is made to detect hanging; total duration is simply bounded, generously
(`build_timeout` 1 h), with separate budgets for network-bound `git fetch` (300 s)
and quick local git calls (60 s). A build failure is retried with the same
exponential backoff as a job relaunch, and only caps out after
`max_build_failures`, so a transient network fault does not terminally fail every
job on a SHA.

**Alternatives rejected.** Progress-based hang detection — requires a definition of
progress the outside of a `cmake` process does not provide.

---

## ADR-14 — Fetch before resolve, and prefer the remote's branch

**Status:** Accepted *(added after the design stage)*

**Context.** `coastd` resolves a submitted ref against its local clone. Nothing
ever checks anything out in that clone, so its local branches are frozen at
whatever the clone was created with — possibly months old.

**Decision.** `git fetch --all --prune --tags` before resolving, then prefer
`<remote>/<ref>` over a same-named local branch **for plain branch names only**.

**Consequences.** Without this, submitting `main` would **silently run old code
under a current-sounding name** — worse than a clean "could not resolve", because
the job succeeds and lies. Explicit forms (`refs/heads/x`, `HEAD`, a raw SHA, a
tag, anything with revision syntax) still resolve verbatim, since prefixing them
would change their meaning. Two properties keep the freshness feature from becoming
a reliability regression: a **fetch failure is never fatal** (warn, fall back to
the local clone), and fetches are **throttled and coalesced** (one fetch per window, which every
resolution arriving meanwhile waits for), so fifty simultaneous submissions produce
one fetch and all fifty see its result. `--no-fetch` restores purely
local, local-first behaviour — the fetch and the remote-first preference are one
capability.

---

## ADR-15 — systemd *user* service with lingering

**Status:** Accepted

**Context.** The daemon must start at boot on a headless box, and it must be able
to build refs from a private remote.

**Decision.** A **user** service (`systemctl --user`) plus
`loginctl enable-linger`, with stock `KillMode` and a `TimeoutStopSec` computed as
`build_timeout + term_grace + 30`.

**Consequences.** A user service inherits that user's git credentials and
environment without a root-owned unit. Lingering is what lets it start at boot
without a login — and is the step most often forgotten, after which the daemon
quietly disappears at logout. The `TimeoutStopSec` budget is dominated by an
in-flight **build**, not by the child's checkpoint grace, because
`worker_pool::shutdown` lets a running build finish. Stock `KillMode` is
load-bearing (ADR-5), not a default left unexamined.

**Alternatives rejected.** A system service — needs explicit credential
provisioning for git and runs with more privilege than the job requires.

---

## ADR-16 — the daemon is Linux-only; the clients are not

**Status:** Accepted (2026-07-26) — *its macOS clause is superseded by ADR-17;
the startup gate stands*

**Context.** Discover §6 listed macOS as a **hard** platform constraint, and the
IDD later demoted it to a "secondary seam". What actually shipped was a launchd
plist template plus a pid-guard that does not work there: `read_start_token` reads
`/proc/<pid>/stat`, and off Linux `is_same_process` therefore answers *dead* for
every pid. R1 — never two coast children on one run dir — rests on that guard on
any platform without the systemd cgroup reap, i.e. exactly on macOS. So after a
daemon crash macOS would relaunch a job **alongside its still-running child**:
silent corruption, arriving precisely when things have already gone wrong.
"Unverified on macOS" understated it — the protection was absent.

**Decision.** macOS (and every other non-Linux platform) is **out of scope for the
daemon**, and the daemon **refuses to start** there rather than running without R1.
No override flag: a flag that re-enables a silent-corruption path *is* that path.
The **clients** — `coastd_client`, the `coast::coastd` console façade, the GUI
panel — are pure TCP and stay portable; a macOS workstation driving a Linux daemon
is the supported arrangement.

**Consequences.** The Discover hard constraint is retired, deliberately and on the
record, instead of persisting as a gap nobody owns. Everything platform-specific
is now one seam: `read_start_token` + `start_token_supported()` in
`daemon/child_proc.h`. Adopting macOS later means implementing one function
(`proc_pidinfo(…, PROC_PIDTBSDINFO, …)` or `sysctl(KERN_PROC_PID)` →
`kp_proc.p_starttime`) and adding the platform to `start_token_supported()`; the
startup gate then opens by itself, because it asks that function rather than
testing `__linux__` at the call site.

**Alternatives rejected.** *Keep the boot warning* — a warning does not stop the
double-child relaunch, and the failure is silent. *Rely on launchd's group reap* —
it kills the survivor only if the daemon is restarted by launchd; a manual restart,
or a daemon crash launchd does not observe as an exit, leaves the orphan. *Fail
closed instead (treat "cannot determine" as alive)* — safe against duplicates but
turns any pid-guard miss into a job that never relaunches, trading a loud hazard
for a silent stall on a platform we have chosen not to support anyway.

---

## ADR-17 — the daemon runs on macOS; unattended macOS operation does not

**Status:** Accepted (2026-11-03) — *supersedes ADR-16's macOS clause*

**Context.** ADR-16 kept the daemon off macOS for one reason: the pid-guard had no
start-token source there. The evaluation toolkit (issue #82) submits its runs
through `coastd`, and it should use the same interface on the macOS development
machine as on the Linux host, with no second local-spawn path.

**Decision.** Implement the platform seam for macOS. `read_start_token` reads
`kp_proc.p_starttime` with `sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})` and
returns it as microseconds since the epoch, a decimal string that, like the Linux
token, is only ever compared for equality. `is_zombie` reads `p_stat == SZOMB` from
the same record. `start_token_supported()` is true on `__APPLE__`, so the startup
gate opens. The gate itself, with no override flag, stays for every platform that
still has no token source.

**Consequences.** On macOS, R1 rests on the pid-guard alone, as on Linux run
outside systemd. The token is recycle-safe for the same reason as on Linux: the
kernel records the start time once, at fork, so a recycled pid has a different one.
The port also found that `shutdown()` on the listening socket does not wake a
blocked `accept()` on macOS, so SIGTERM never stopped the daemon there. The accept
loop now polls with a 200 ms timeout and re-checks its stop flag, which is how
`discovery_server` already handles `recvfrom`. The daemon tests that skipped on
macOS (`unit.coastd.child_proc`, `unit.coastd.comms`, `unit.coastd.recovery`,
`fault.coastd.matrix`, `unit.coastd.console`) now run and pass there.

Still out of scope: `--install-service` (a systemd unit; Linux only) and unattended
operation under launchd, whose plist template has never been run. On macOS, start
the daemon by hand.

**Alternatives rejected.** *`proc_pidinfo(PROC_PIDTBSDINFO)`* — it returns EPERM
for a process owned by another user and ESRCH for a zombie (measured on macOS 27.2),
so both read as "gone". For another user's process that is the costly direction:
`is_same_process` would call a live orphan dead, and its job would be relaunched
next to it. `sysctl` returns both, as `/proc` does on Linux. Its one trap is that a
pid with no process succeeds with zero bytes, so the length is checked.

---

## Relationship to the Discover assumptions

| Assumption | Outcome |
|---|---|
| A1 — resume fidelity | Non-issue; RNG state already serializes bit-exactly. The `design/` doc claiming otherwise was a stale pre-fix artefact. |
| A1′ — torn checkpoint | **Redirected** by measurement → ADR-8 (prevent, don't detect). |
| A3 — crash supervision | **Cleared** → ADR-5 + ADR-6. The highest-risk item became the simplest. |
| A4 — hybrid status channel | **Downgraded and dissolved** → ADR-7 (file tail; no socket). |
| A9 — worktree refcount | **Cleared** → ADR-4 (derived, never stored). |
| A10 — no dependency failure crashes the daemon | Design invariant throughout; external failures become job states. |
| A11 — reboot recovery needs service registration | Operator precondition → ADR-15. |
