# Quiesced State Access — Implementation Plan

Status: completed — all 6 steps landed on `fix/state-io-sim-pause` (2026-06-13);
outcome summary in `quiesced-state-access.md` §9
Author: COAST team
Date: 2026-06-12

Companion to `quiesced-state-access.md` (the design). This plan sequences
the work into independently-verifiable steps, each with an explicit
verification gate. Step 0 (buffer-restore bugfix) and Steps 1 and 2 are
mutually independent and may proceed in parallel; step 5 is independent of
0–4. Step 0 must precede Step 3; Steps 3→4 are the critical path.

A note on terminology used below: the **in-memory state tree** is a
navigable structure of `group → named field → typed value`. Two adapters
sit on the existing abstract interfaces (`state_io.h`): `mem_state_writer`
(a `state_writer` that *builds* the tree) and `mem_state_reader` (a
`state_reader` that *navigates* it). Entities are never modified — they
already call the abstract interface.

### Step 0 — io_buffer read-page restore + buffer-restore bugfix (own commit)

Pre-existing bug (design §5.3), folded into this branch as **its own
commit**, decided interface **Option A**: add the symmetric `snapshot` /
`restore` virtual pair to `io_buffer` (all four variants), switch
save/load to use them, and have restore write the **read page** directly
(no swap). Drop `gpu_state_roundtrip_test`'s manual `swap_buffer()`; add the
save→load→**step-once**→compare reproducibility test (CPU + GPU) that the
old test's manual swap hid. Depends only on the `snapshot`/`restore`
interface, so the commit is self-contained and cherry-pickable to `main`.

This must precede Step 3 (the `state_logger` split consumes the new
interface). **Open sub-question — its position in the commit sequence:**
land it *early*, right here against the current branch code (a clean
bug→fix+test diff, before pause_guard is retired), or *after* the
`state_logger` split, fixing `apply_state` directly? Early/isolated is the
more reviewable bug→fix story and keeps the cherry-pick option; to be
confirmed.

---

## Step 1 — In-memory state tree + HDF5 adapters (foundational, no threading)

Everything in the save/load split depends on this. Built and tested in
complete isolation from threads.

Sub-parts:

- **1a. Tree datatype.** `group → field → value`, where a value is a
  scalar or array carrying its element type (the `state_io.h` type set:
  f32/f64/i32/i64/u32/u64/bool, vectors of f32/f64/i32/u32/u16/u8, plus
  2D-f32 and string attrs). Nested groups.
- **1b. `mem_state_writer : state_writer`.** `begin_group/end_group` push
  /pop a node; each `write(name, …)` overload appends a typed field.
- **1c. `mem_state_reader : state_reader`.** `enter_group/leave_group`
  navigate; `has_field`/`read_*` look up by name, returning the requested
  type (converting where the interface implies it, matching
  `hdf5_state_reader`'s behavior).
- **1d. Serialize: tree → HDF5.** Walk the tree, driving the *existing*
  `hdf5_state_writer`. Because this replays the same sequence of
  `state_writer` calls the entities originally made, it reproduces the
  current on-disk structure by construction.
- **1e. Parse: HDF5 → tree.** Generic enumeration of an HDF5 file
  (`H5Literate` over groups, `H5Dread` per dataset) into the tree,
  preserving element type so `mem_state_reader` returns the right type.
  **This is the fiddliest sub-part** (dtype detection/mapping); budget for
  it accordingly.

**Verification gate (unit, sub-second, no threads):**
- Round-trip the tree across every supported dtype: build via
  `mem_state_writer` → serialize to a temp HDF5 → parse back → read via
  `mem_state_reader` → assert equality field-by-field, type-by-type
  (including empty arrays, nested groups, 2D-f32, string attrs).
- Equivalence with the old path: serialize a known tree and confirm it is
  **`coast_compare`-clean** against a file written by `hdf5_state_writer`
  directly from the same call sequence. (Contract is structural/logical
  identity + loadability, *not* byte-identity — HDF5 is not byte-repro.)

---

## Step 2 — `run_quiesced` in the controller (additive, parallel to step 1)

Add the generic seam without yet touching state IO. `pause_guard` stays in
place during this step so nothing regresses.

- Add to `sim_thread_controller` an MPSC queue of quiesced tasks and a
  generic primitive, mirroring the executor's `submit_sync` shape:
  `template<class F> auto run_quiesced(F&& fn) -> std::invoke_result_t<F&>`
  — enqueue {type-erased fn + promise}, wake the sim thread, block on the
  future, return the result (or rethrow).
- `thread_main` drains the queue **(a)** right after each `do_step()`
  completes (post-swap, GPU idle) and **(b)** while parked in the `paused`
  wait loop. The paused-wait predicate must also wake on "task pending."
- **Stop-escape:** a requester blocked in `run_quiesced` must unblock if
  the controller is stopping (mirrors `pause_guard`'s `stop_requested`
  escape), to avoid a teardown deadlock.

**Verification gate (unit — extend `test_thread_controller_hold.cpp`):**
- A task enqueued while free-running executes at a boundary; the count is
  frozen for its duration; mode is untouched; stepping resumes after.
- A task enqueued while `paused` still runs (wakes the parked thread).
- Return value propagates; an exception in the task rethrows to the caller.
- The task runs **on the sim thread** (assert `std::this_thread::get_id()`
  inside the task ≠ the test thread).
- Stop-escape: destroying the controller while a task is queued unblocks
  the requester rather than hanging.
- `pause_guard`'s existing cases still pass (both mechanisms coexist here).

---

## Step 3 — Split `state_logger` into extract / serialize / parse / apply (needs step 1)

Refactor the monolithic `save_state`/`load_state` into four pieces that
respect the thread split. No call sites change yet; the new functions are
exercised through the existing `logging_manager` wrappers plus targeted
tests.

- `extract_state(env, step) -> state_tree` — **sim-thread-only** work:
  iterate buffers (`cur_read_buffer`, `read_back_sync` for GPU-only),
  iterate entities (`entity.save_state(mem_writer)` — which includes GPU
  entities' own readbacks). Builds and returns the tree. No HDF5.
- `serialize_state(tree, cfg, h5_exec) -> future<log_result>` — submit to
  the executor (`submit`, async); the job replays the tree into
  `hdf5_state_writer`. HDF5 only here.
- `parse_state(path, h5_exec) -> {result, state_tree}` — `submit_sync` on
  the executor; generic HDF5→tree (1e). HDF5 only here.
- `apply_state(env, tree) -> state_load_result` — **sim-thread-only**:
  restore each io_buffer's **read page directly** (design §5.3 — fixes a
  pre-existing bug; replaces the write-page+swap crutch), `entity.load_state
  (mem_reader)`, returns `restored_step`. No HDF5, no swap.
  (`set_step_count` is done by the call site alongside, see step 4.)
  Depends on the io_buffer read-page-restore interface (§5.3 open decision A
  vs B) — settle that before this step.
- **Startup-load exemption:** keep the existing monolithic
  `load_state(env, path, h5_exec)` for `build_env` only — it parses and
  applies inline on the executor, which is safe solely at construction
  (design §5.2). Runtime sites use parse+apply.
- **Read-page-restore fix applies to BOTH load paths** (design §5.3): the
  read-page restore lives in the shared buffer-restore logic, so the
  monolithic `load_state` (startup `--resume`) is fixed too. The existing
  `gpu_state_roundtrip_test` **drops its manual `swap_buffer()`** (restored
  data is already in the read page) and updates its comment.

**Verification gate:**
- B-tier `--save-state` / `--load-state` / `--checkpoint` file
  side-effects pass unchanged.
- `gpu_state_roundtrip_test` passes (GPU extract/apply on the right
  thread).
- New: save→parse→apply into a fresh `env` reproduces state
  (`coast_compare`-clean) for a CPU and a GPU config.
- **New reproducibility test (the one the bug hid behind):** save at step
  N, load into a fresh env, **step once**, and assert the post-step state
  matches an uninterrupted run's step N+1 — for a CPU config *and* a GPU
  config. This exercises the read page being consumed after load, which
  the existing manual-swap round-trip test does not.

---

## Step 4 — Migrate runtime call sites; retire `pause_guard` (critical path)

Move the four runtime state-IO sites onto `run_quiesced` + the step-3 API,
then delete the old primitive.

- **GUI save** (`logging_window.cpp`): `tree = run_quiesced(extract)`;
  `fut = serialize_state(move(tree), cfg)`; await `fut` off the main
  thread and report on the status line (fire-and-forget durability is fine
  for the GUI).
- **GUI load** (`logging_window.cpp`) + **snapshot load**
  (`gui_manager.cpp:518`): `{res,tree} = parse_state(path)`;
  `run_quiesced([&]{ apply_state(env, tree); controller.set_step_count(
  res.restored_step); })`.
- **Headless checkpoint + final save** (`run_headless.cpp`):
  `tree = run_quiesced(extract)`; `fut = serialize_state(move(tree),cfg)`;
  **await `fut`** before `record_savestate()` / manifest / return (design
  §5.2 must-fix — otherwise the manifest records 0 bytes).
- **Delete** `pause_guard`, `hold_paused()`, `hold_count_`, `parked_`, and
  the associated plumbing once all four sites are migrated. Remove the
  `pause_guard`-specific tests (the contract now lives in the step-2
  `run_quiesced` tests).

**Verification gate:**
- Full unit suite + A-tier integration + B-tier + `gpu_state_roundtrip`.
- Headless: `--save-state` and `--checkpoint N` produce correct-size files
  **and** a manifest whose `savestate_outputs` sizes are non-zero/correct.
- Manual GUI smoke: save and load while the sim is free-running; confirm
  no stall pathology and correct restored step.
- Grep confirms zero remaining references to `pause_guard`/`hold_paused`.

---

## Step 5 — Fold in the Class C reads (independent of 1–4)

- **`buffer_browser` (Q1 — periodic publish):** while the IO Buffers
  window is open, register a `post_swap_hook` that, on the sim thread,
  computes per-buffer sum/min/max (using `read_back_sync` for GPU-only
  buffers) and publishes a snapshot the GUI reads (double-buffered or a
  small lock); deregister on window close. Remove the GUI-thread
  `cur_read_buffer()` iteration in `rebuild_cache()`.
- **`status_str` / browser scalars:** publish the handful of displayed
  values (e.g. `mnist_io::cur_idx`, `neuron_group_t` mismatch aggregates)
  into atomics updated at `post_swap`; the GUI reads the atomics.

**Verification gate:**
- A TSan build over a short sim-plus-browser exercise shows no data race on
  the buffer stats or status fields (this is the race we are removing).
- Visual check: buffer stats still update live; GPU-only buffers now show
  real numbers instead of blank.
- Pick `N` (publish interval): fixed default, or tied to the existing vis
  `update_interval` hint — decide here.

---

## Step 6 — Docs + final sweep

- Update `docs/content/threading/` to describe the single quiesced-state
  seam and the extract/serialize/parse/apply split; remove `pause_guard`
  references. `cd docs && hugo` with zero warnings.
- Update `sim_core.md` module page if it names `pause_guard`.
- Full verification: unit + A-tier + B-tier + `gpu_state_roundtrip`; TSan
  over unit + B-tier; ASan/UBSan smoke.

---

## Risks and mitigations

- **1e generic HDF5→tree dtype handling** is the highest-risk sub-part. Mitigate
  by testing every dtype in the step-1 round-trip before any threading lands.
- **Structural drift from the current save format** would break loading of
  previously-saved files. Mitigated by step 1's `coast_compare` equivalence
  gate (serialize-via-tree vs. write-inline) and by serialize being a literal
  replay of the same `state_writer` calls.
- **Deferred-write/teardown in headless** — covered by the step-4 await gate;
  called out so it is not lost.
- **Backpressure under high-frequency saves** is by-design (sim throttles to IO
  bandwidth, bounded memory); not a bug, but worth a one-line note wherever a
  checkpoint interval is configured.

## What is explicitly NOT in scope

- Merging `single_step` into the quiesced-task queue (design Q3 — deferred).
- Changing the on-disk state schema.
- The headless probe-observer drain (already SPSC-safe).
