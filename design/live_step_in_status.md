# Live step in `list`/status — wiring the progress parser to the UI

**Status:** implemented (2026-07-24).

## Requirement

coastd must use the shared progress writer/parser (`diag/progress.h`) on a running
job's log to report its **current step**, so a running job visibly advances in the
status window / `list` — not just at checkpoint boundaries.

## What was wrong

The whole pipeline existed except the last connection:

- `coast` emits the structured line (`run_headless.cpp` → `emit_progress`, flushed).
- The supervisor tails every child's log at ~10 Hz and `parse_progress`es it into
  `child.last` (`supervisor.cpp`).
- A getter `supervisor::progress_of(id)` was built and documented *"for list"*.

…but `progress_of()` had **no caller anywhere in the repo**. `list`/`query`
reported only `job::steps_done`, the durable checkpoint step. So the live value the
daemon was already computing every 100 ms never reached a client, and a running job
sat frozen at its last checkpoint (0 before the first one). The missing piece was
one call site — and, tellingly, no acceptance test asserted live progress, which is
why the unconnected wire shipped.

## Fix

- `job_view` gains a **display-only** `live_step`, filled by `core::apply_live_step()`
  (core thread — `sup_` is in scope) from `progress_of(id)`. Used by both `list()`
  and `query()`.
- `proto::job_summary` gains `live_step` (+ (de)serialization); the console
  `job_info` too. `print_jobs` and the GUI panel show the current position:
  `live_step` when a running job is ahead of its last checkpoint, else `steps_done`.

## The correctness constraint (why a separate field)

`steps_done` is **durable/resumable** (`events.h:61`): `reconcile` marks a job done
on `steps_done >= steps_target`, and resume math reads it. Overwriting it with the
live (non-durable) tailed step would let a job be declared done, or skip a resume,
off a count that is not on disk. So `live_step` is strictly separate and
display-only; `steps_done` stays checkpoint-anchored and untouched.

## Test

`tests/unit/coastd/test_core_lifecycle.cpp` — "list/query report the live tailed
step of a running, un-checkpointed job": a fake `coast` emits one `phase=step`
line (step=1234, target=4000) and blocks without checkpointing; the test asserts
`query` reports `live_step == 1234` while `steps_done == 0` and the job is still
active. Mutation-proofed: stubbing `apply_live_step()` to a no-op fails it (the
original unconnected-wire behavior). This is the acceptance test whose absence let
the gap ship.
