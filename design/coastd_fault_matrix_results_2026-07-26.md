# coastd — M3 fault matrix and latency gates: results

**Date:** 2026-07-26 · **Harness:** `tests/unit/coastd/test_fault_matrix.cpp`
(ctest `fault.coastd.matrix`) · **Machine:** Linux 7.1.4, nvme/ext4 ·
**Build:** `build-tests`, Release

These are the Discover §5 acceptance criteria that the 2026-07-24 audit found
unmeasured. Everything below runs against the **real `coastd` binary**, real child
processes and a real `cmake` build; the daemon is driven only through the wire
client, exactly as an operator's client would.

## Matrix

`{kill client, kill daemon} × {building, queued, running}` — 6 of 6 cells pass.
Each cell injects the fault and then requires the job to reach **correct
completion with no operator action**.

| Fault | building | queued | running |
|---|---|---|---|
| client SIGKILLed mid-request | pass | pass | pass |
| daemon SIGKILLed | pass | pass | pass |

Notes on what each cell actually exercises:

- **Client killed** — a real client process opens a connection, sends a *partial*
  request line (so the daemon is mid-read) and is SIGKILLed. A clean close would
  exercise nothing. Asserted: the daemon still serves, and the job's trajectory is
  unchanged.
- **Daemon killed while building** — the interrupted build leaves a worktree
  *without* the readiness sentinel. The restarted daemon must read that as
  "rebuild", not as a built tree (which would spawn against a missing binary) and
  not as a hard failure.
- **Daemon killed while running** — the child is left an orphan (no cgroup reap in
  a test), so the restart exercises the boot pid-guard: SIGKILL the survivor, then
  relaunch. Additionally asserted: `steps_done == steps_target` at the end, i.e.
  one child finished the work, not two racing the same run dir.
- **Daemon killed while queued** — the queued job is pure store state; the case
  requires *both* the queued and the concurrently-running job to complete.

### The reboot column is deliberately out of scope

Decision 2026-07-26. A reboot's daemon-visible content — every child gone, daemon
restarts from the store — is the kill-daemon column plus an orphan kill, which is
covered above. What a reboot adds beyond that is **systemd's cgroup reap**, which
is systemd's behaviour, not coastd's: reproducing it in-process would measure the
harness. It stays an operator-level check against the shipped unit file.

## Latency

Medians over repeats on an otherwise-idle box. The spec said "≤ seconds"; the
assertions in the harness are deliberately looser than these numbers so a loaded
CI box cannot produce a red herring, but the numbers are printed on every run —
an acceptance bound with no recorded number cannot show a regression that stays
inside it.

| Measurement | Median | Max | Asserted bound |
|---|---|---|---|
| submit → visible in `list` | **1.3 ms** | 1.4 ms | < 2 s |
| child SIGKILL → no longer `running` | **35 ms** | 44 ms | < 5 s |
| `cancel` RPC round-trip | **0.23 ms** | — | (reported) |
| cancel → `cancel_requested` on the wire | **0.66 ms** | — | < 2 s |
| cancel → settled, child exits on SIGTERM | **55 ms** | 88 ms | < 3 s |
| cancel → settled, child ignores SIGTERM | **3191 ms** | — | < 15 s |

Reading these:

- **Child-death detection at 35 ms** is the 100 ms `waitpid` poll plus a reconcile
  pass, landing well under it because the poll and the ~1 Hz backstop are
  independent — the event wake does the work.
- **Cancel has two regimes, and both legs of the acceptance criterion hold.** A
  child that exits on SIGTERM is gone in ~55 ms: that is the "near-instant" leg,
  and in that regime `cancel_requested` is typically never even observable. A
  child that does not exit (the realistic case — a real `coast` writing a final
  checkpoint under `--save-on-term`) settles in ~3.2 s, which is `term_grace`
  (3 s here) plus the SIGKILL follow-up. That is the regime the flag exists for,
  and it reaches the wire in **0.66 ms**, i.e. the daemon's own contribution to
  cancel latency is negligible — the wait is the child's, by design.

## Method notes worth keeping

- **The build is made slow on purpose.** The throwaway repo's `CMakeLists.txt`
  (`project(… NONE)` + an `execute_process(COMMAND sleep N)`) makes "building" a
  state a fault can be injected *into*, rather than a millisecond to be raced. Its
  only target copies a fake `coast` script into `build/coast` — which is exactly
  what the builder checks for.
- **The fake `coast` does not checkpoint.** A resume after a kill therefore starts
  over, which is the harsher path for the accounting. Checkpoint-correct resume is
  the separate `gate.coastd.real_coast`.
- **`COASTD_BIN` is a path, not a build edge.** Both this harness and
  `test_comms` now carry an explicit `add_dependencies(… coastd)`. Without it a
  test runs against a stale daemon and proves whatever *that* binary does — which
  happened during this work and took several debug cycles to spot, because the
  source under investigation was already correct.
- **doctest's `--test-case` filter is comma-separated**, so a case name containing
  a comma is silently unselectable (the filter splits and matches nothing, and
  doctest reports SUCCESS with zero cases run). The case names here avoid commas.
