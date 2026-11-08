# coastd — Specification-vs-Implementation Audit

**Date:** 2026-07-24 · **Tree audited:** `feat/coastd` @ `f12a7e9`
**Specification baseline:** `.dew/docs/01-discover.md`, `.dew/docs/02-design.md` (IDD),
`.dew/design-verification/DESIGN_VERIFICATION.md`, plus the recorded deviations
(`docs/content/coastd/decisions.md` ADR-1..15, `.dew/state.md` deviation table). The `.dew/`
files are local workflow notes, not in the repository.
**Method:** seven parallel read-only audit agents, one per subsystem (state core/recovery,
reconcile policy, job store/persistence, supervisor/builder/worker-pool, comms/discovery/
protocol, client surfaces/service units, coast-side changes + test gates), each given the
spec requirements as an explicit checklist and required to verify in code (never comments)
with file:line evidence. All material claims were then re-verified by hand. Findings that
match a *recorded, user-approved* deviation (ADR-8, ADR-10/11/12, the H1 deferral,
INADDR_ANY, GUI-runtime-untested policy) are classified as such, not re-reported as defects.

---

## Verdict

**The architectural spine is implemented faithfully to the IDD and in several places harder
than specified. No hollowed-out core mechanism was found.** The real shortfalls concentrate
in three areas:

1. the design's **hard acceptance gates were never demonstrated** (kill+resume bit-identity
   through the daemon, the M3 fault-injection matrix, the latency bounds);
2. a few **functional requirements were silently dropped at the edges** (cancel-UX,
   datasets location advertisement, §6.7 accessibility check, §10 poison routing/health
   signal, `disk_pressure`);
3. the **macOS half of the platform requirement is effectively absent** rather than merely
   unverified (the pid duplicate-guard — R1's carrier on macOS — is inoperative there).

Commit `f12a7e9` (2026-07-24) already fixed two more gaps of exactly this class
(editable-config submit, live step in status), both described in its commit message as
"specified but unmet in behavior".

---

## 1. Hard acceptance gates never demonstrated

Discover §5 defined these as *hard gates*. None was run to its stated standard.

### 1a. Auto-resume bit-identity through the daemon — no automated validation

The gate: a resumed run's final state exactly matches the uninterrupted run, verified like
the A-tier goldens.

What exists: `tests/unit/coastd/test_core_lifecycle.cpp:239-299` runs a real mnist job
*through coastd* and asserts `final_state.h5` is `h5diff`-identical (excluding `/metadata`)
to a direct `coast -n` run. Two problems:

- **Only the uninterrupted half is covered.** No test anywhere kills a running coastd
  child, lets the daemon relaunch from its checkpoint, and compares the final state. The
  untested path carries real semantic risk, not formality: global-vs-chunk step accounting
  across `--load-state` + `-n (T−S)` and the monotonic-max fold in
  `core.cpp` (`on_child_exited`, `if (ce.steps_done > j->steps_done) …`). Resume
  bit-identity is proven for *direct* coast (`tests/integration_ngm3/state_resume_*`),
  never through the daemon.
- **Even the uninterrupted half is opt-in.** It requires `COASTD_REAL_COAST` /
  `COASTD_REAL_CONFIG` (driver `tests/unit/coastd/run_real_gate.sh`) and silently skips
  otherwise. A normal `ctest` run — the source of the "16/16 coastd tests pass" figure —
  does **not** execute this gate.

### 1b. M3 fault-injection matrix never run as a matrix

The gate: **100%** of {kill client, kill daemon, reboot} × {building, queued, running}
recovers to correct completion.

What exists: fragments. Kill-daemon × running (orphan SIGKILL + identity clear,
`test_recovery.cpp:111`); crash-loop capping (`test_recovery.cpp:158`); kill-client
scenarios (silent-connection reap, slowloris, flood — `test_comms.cpp:414-451`). The
**building** and **queued** columns were never fault-injected. "Reboot" exists only as
boot-from-disk recovery in-process; the cgroup-reap evidence is a pre-implementation spike
(`.dew/design-verification/03a_cgroup_reap.sh`, local workflow notes, not in the repository), not a test of the shipped daemon.

### 1c. Latency gates never measured

"Status stale ≤ seconds", "child-death detection ≤ seconds", and — explicitly — "cancel
latency: *measure it*" have no measurement anywhere. The mechanisms exist and are
correctness-tested (`test_line_tailer.cpp`, `test_child_proc.cpp`, `test_supervisor.cpp`,
plus spike `04_tail_parse.cpp`), but the numeric bounds were the acceptance criteria and
were never checked.

---

## 2. Functional requirements dropped or narrowed — unrecorded

These appear in **no** ADR and not in the `state.md` deviation table.

### 2a. Cancel-UX acceptance criterion unmet on every surface

Discover §5: cancel must be near-instant *or* the client must show explicit "cancellation
in progress" feedback. Neither holds:

- `cancel_requested` is not in the wire DTO (`coastd/include/daemon/protocol.h:88-104`,
  `job_summary_from` in `comms_server.cpp:193-208` omits it);
- there is no `cancelling` status token (`job_json.cpp:33-42`);
- the GUI cancel button gives no feedback at all (`src/gui_tools/coastd_panel.cpp:278-282`).

A cancelled running job reads as plain `running` until the child dies (up to the 10 s
SIGTERM grace, `types.h` `term_grace`), then jumps straight to `cancelled`.

### 2b. macOS: duplicate-run protection does not exist — worse than "unverified"

`read_start_token` is `/proc`-only with no macOS branch (`coastd/src/child_proc.cpp:61-78`),
so `is_same_process` always answers "dead" on macOS. The design (§4, §6.4) relies on the
pid-guard to "carry R1 there" precisely because launchd has no cgroup reap. Consequence on
macOS after a daemon crash: a surviving orphan is treated as dead and **relaunched alongside
itself** — the duplicate-run nightmare R1 exists to make impossible by construction. This is
fail-*dangerous*, not fail-safe. It is documented (boot WARN on non-Linux, service README
note), and the IDD demoted macOS to "secondary seam" — but Discover §6 lists macOS as a
**hard** constraint, and what shipped is a launchd plist template plus an inoperative guard.

### 2c. Well-known-locations advertisement is partial

Discover must-have: a daemon config carrying well-known **data/results** locations
advertised to clients. `machine_info` serves `runs_root` (results) and `repo_root` only
(`coastd/src/machine_info.cpp:27-37`); there is **no datasets location key** in the daemon
config or on the wire. The nice-to-have client-side path templating has nothing to build on
for the data side.

### 2d. §6.7 accessibility check silently dropped

The spec requires rejecting config paths "inaccessible to the daemon+child uid" at submit
time. The implementation is a deliberately "minimal-structural" scan
(`comms_server.cpp:145-157`) — no filesystem/uid accessibility check. ADR-11 records the
frame change (run-dir frame) but *not* this narrowing. Consequence is soft — the job fails
later at run time instead of a clean submit-time rejection — but it is an unrecorded drop
of a specified check.

### 2e. §10 poison routing and the health signal

- Repeated-trip (crash-loop) jobs are filed to `failed/`, never `quarantine/`
  (`core.cpp` strike counters → `failure_reason`; `reconcile.cpp:65-71` → `bucket::failed`).
  Design §10 lumps "repeated trip" into `jobs/quarantine/`; the implementation reserves
  quarantine for schema/parse poison only (`job_store.cpp:189-234`). Net effect (bounded,
  no crash-loop) is achieved; the bucket differs from spec.
- The "daemon-health signal" on a serious store-write failure is a single `COAST_ERROR`
  log line. No health surface of any kind exists.

### 2f. `reality.disk_pressure` is a dead field

Declared per §5.5 (`coastd/include/daemon/types.h:146`), never populated by `rescan`,
never read by `reconcile` or anything else. No disk-pressure policy exists anywhere.

---

## 3. Known, user-approved decisions — with outstanding follow-ups

Not defects (decided by the project owner), but two carry unfinished compensating work:

- **H1 — job-file round-trip loss.** `dump_job` writes `no_checkpoint_launches` and
  `last_launch_ms` (`job_json.cpp:78-79`); `parse_job` never reads them back
  (`job_json.cpp:84-113`). Both guards reset on every daemon restart — and `last_launch_ms`
  is epoch ms (`ms_since_epoch(r.now)` / `now_epoch_ms()`), i.e. wall-clock, so persisting
  it *would* be meaningful. Deferred by user decision 2026-07-21 (rare / misconfigured-run
  only) **in favour of a startup-log plausibility check that was never implemented** — no
  trace in `coastd/`. The deferral currently has no compensator. The round-trip unit test
  (`test_job_json.cpp:35-56`) checks only the spec fields, so it cannot catch this.
- **No structural HDF5 integrity check in coastd** (ADR-8; the user retracted the
  belt-and-suspenders check on the grounds that old coast refs won't be exercised).
  Audit observation: the safety currently holds even for old refs *by accident* — any ref
  old enough to lack the atomic-write fence also lacks `--save-on-term`, which coastd
  always passes (`child_cmd.cpp:159`; both landed together in `8c8bed3`), so such a child
  rejects its argv and fails loudly rather than running with non-atomic checkpoints.
  ADR-8's stated residual risk stands: a future coast change that keeps the flags but
  breaks the fence would silently decouple the two.
- **D1** — the stale `reconcile.h` rule-table comment (cancel listed before failure) is
  still unfixed; the code is correct and deliberate (`reconcile.cpp:65` before `:72`).
- Also known and unchanged: INADDR_ANY unauthenticated bind (conscious decision, commented
  in `comms_server.cpp`); GUI panel runtime-untested (user policy); `--seed-offset` never
  wired (it was never in the design command line either).

---

## 4. Verified faithful (spot-checked, not assumed)

- **coast-side atomic checkpoint fence — all four steps present**: tmp co-located → write
  with checked flush/close → `fsync(file)` → `rename` → `fsync(dir)`
  (`src/logging/state_logger.cpp:610-693`). `--save-on-term` complete incl. the
  `--unlimited` refusal-lift (`src/cli/parse.cpp:327-347`, `run_headless.cpp:427-444`);
  `--emit-run-manifest` consumed by `read_outcome` (`child_cmd.cpp:187-204`); progress
  lines self-flushing (`diag/progress.h:128-134`, `std::unitbuf` in `run_headless.cpp:52`).
- **Pure reconcile**: exact signature, zero I/O/clock (all time via `reality.now`);
  N-slot cap (`reconcile.cpp:110-121`), priority-then-FIFO ordering, derived refcounts,
  6 h mtime-based cool-off GC (configurable), failure-before-cancel ordering — all
  unit-tested (`test_reconcile.cpp`, 19 cases).
- **Core loop / single writer**: `apply → rescan → reconcile → dispatch` with immediate
  event wake + ~1 Hz backstop (`core.cpp:159-184`); all edges reach state only via
  `queue.push(event)`; promise-based RPC.
- **Recovery == steady-state**: boot = load store → pid-guard (SIGKILL surviving orphan,
  `core.cpp:99-104`) → stale-`.tmp` sweep → git catch-up commit → normal loop.
- **Builder**: worktree-add → `git submodule update --init --recursive` (the mandatory
  Demonstrate finding, `builder.cpp:295-296`) → cmake; exactly one build in flight;
  `-j = cores/2` with `build_jobs` override; fetch-before-resolve with remote-preferred
  branch resolution.
- **Child contract**: relative `-n (T−S)` on resume; `steps_target==0` → `--unlimited`
  (never `-n 0`); CWD = run dir; stdout→logfile tailed through the real
  `diag::parse_progress`; SIGTERM→SIGKILL escalation with configurable grace.
- **Store**: five buckets, atomic same-dir tmp+fsync+rename+dir-fsync writes
  (`atomic_write.cpp`), rename-across-buckets with dir fsyncs, git audit trail off the
  core thread, schema-gate quarantine that never mangles files, both `config.json` and
  `config.submitted.json` written (ADR-12).
- **B-tier reality**: `test_comms.cpp` spawns the real `coastd` binary and drives the full
  JSONL surface over raw TCP + UDP discovery, incl. DoS bounds and ordered shutdown —
  the promised "drive the daemon over nc/JSONL" tier exists.
- The ~24 hardening commits fixed real bugs beyond spec (build-completion sentinel gating
  spawn, slowloris/oversize bounds, shutdown ordering, JSON-depth guard, thread reaping).

---

## 5. Minor drift (completeness)

- `max_request_bytes` default is 10 MiB (`types.h:244`) vs the documented 8 MiB.
- Wire key `message` where the design wrote `msg` (internally consistent).
- Liveness is a 100 ms `waitpid(WNOHANG)` poll, not SIGCHLD/signalfd (§6.5/§9); endorsed
  by the verification report; period finer than the stated 200 ms–1 s band. Benign.
- Raw `wstatus` is carried but never decoded; clean-vs-interrupted derives from
  manifest/progress. The `events.h:55` comment overstates ("the supervisor decodes").
- Checkpoints live under `snapshots/` (`child_cmd.cpp:62`) vs the spec's flat
  `state_NNN.h5` sketch; `failure_reason` serialized as `""` not `null`. Cosmetic.
- `reality.child_alive` at steady state comes from the supervisor's waitpid registry, not
  a pid+start_token recheck (stronger in practice; `types.h:143` comment overstates).
- Spawn gate uses the stricter `builds_ready` sentinel, not `worktrees_on_disk`
  (intentional improvement over §6.1's literal wording; tested).
- §6.4 attributes the pid-guard to `reconcile`; it lives in `core::recover_at_boot` plus
  a spawn-time skew guard. Behaviorally correct; doc localizes it wrong.
- `update_job` action arm defined but never emitted by reconcile.
- GUI panel: submit form has no priority field and no object-config path; reorder is
  Up-only; progress is text, not a bar. `reorder` renumbers and thereby overwrites
  submit-time priorities (documented in code).
- Test gaps: no cancel-arriving-mid-build unit case; no atomic_write-failure degrade test;
  no test covers the H1 fields' round-trip (which is why H1 survived).
- Open item from develop notes: `--cmake-arg -DCMAKE_BUILD_TYPE=Debug` silently overrides
  the hardcoded Release (last `-D` wins); no warning.

---

## 6. Suggested priority order for closing the real gaps

1. **Kill+resume-through-daemon bit-identity test** — validates the daemon's single
   headline promise and the one untested semantic path (step accounting across resume).
2. **H1**: either implement the promised startup-log plausibility check or simply add the
   two missing lines to `parse_job` (+ extend the round-trip test to all persisted fields).
3. **Decide macOS in or out.** If in: implement a `sysctl`/`proc_pidinfo` start-token
   source and a macOS smoke. If out: retire the Discover hard constraint explicitly.
4. **`cancel_requested` on the wire + a `cancelling` render** — small change, closes an
   explicit acceptance criterion.
5. Run the M3 matrix once (scripted is fine) and record the results; measure the latency
   gates once against their bounds.
6. One-liners: D1 comment fix; datasets location key in config + `machine_info`;
   8-vs-10 MiB doc/code reconciliation.
