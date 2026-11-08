# Quiesced State Access — Design Document

Status: implemented on `fix/state-io-sim-pause` (2026-06-13); §9 records the outcome
Author: COAST team
Date: 2026-06-12

Supersedes the `pause_guard` mechanism introduced in commit `cb5ca6a`
("sim_core: quiescence hold (pause_guard) for state save/load paths").

Implementation sequencing lives in `quiesced-state-access-plan.md`.

## 1. Problem

The simulation runs on a dedicated sim thread (`sim_thread_controller`).
Many other components — the GUI on the main thread, the headless driver
loop, the logging subsystem — read or write simulation state that the
sim thread is concurrently mutating. The state in question is:

- `io_buffer` read/write pages (CPU pages, or GPU-resident pages with or
  without a standing CPU mirror),
- entity-internal state (synapse permanences, dataset cursors, …),
- the step counter.

`cb5ca6a` fixed *one* class of these accesses — state save / load /
checkpoint — by adding `pause_guard`, an RAII hold that blocks the
caller until the sim thread parks between steps. A fresh, end-to-end
audit (2026-06-12) shows the situation more completely:

### 1.1 What is already synchronized correctly

Every **visualization** synchronizes through the hook system. Both
`vec_ring_buffer_instance` and the neuron-group visualizers
(`ngm_flat_vis_instance` and siblings) register a two-phase handshake:

- a `post_swap_hook` (the *signal*) that, on the sim thread, optionally
  performs a GPU→CPU readback into a CPU mirror, then publishes
  `requested`;
- a `post_swap_barrier_hook` (the *wait*) that, on the sim thread,
  **blocks** until the GUI has read the data and republished `idle`
  (100 ms timeout cancels).

The GUI reads sim state only inside the won CAS, i.e. while the sim
thread is blocked in the barrier hook. This is correct and is the
*established pattern in this codebase*. (An earlier review mislabeled the
neuron-group dendrite reads as a race; they are not — they run under this
handshake.)

`buffer_logger` is the read-only proof point: it captures a full,
consistent snapshot of every buffer's read page at `post_swap`, on the
sim thread, copies it out, and hands the copy to the HDF5 executor.

### 1.2 What `pause_guard` covers

State save, load, checkpoint, and `set_step_count` — by parking the sim
thread from the outside and doing the work on the *caller's* thread.

### 1.3 What is covered by NEITHER mechanism (the missed interactions)

- **`buffer_browser::rebuild_cache()`** (`buffer_browser.cpp:69-82`)
  reads `cur_read_buffer()` and iterates it (sum/min/max) **every frame,
  on the GUI thread, with no handshake**. This is a genuine data race
  against `swap_buffer()`. It also silently shows nothing for GPU-only
  buffers (empty read span, no readback).
- **`status_str()`** (called from the status dialog on the GUI thread)
  reads live mutable state — `mnist_io::cur_idx/cur_epoch`,
  `neuron_group_t::get_avg_mismatch()/get_max_mismatch()`. Display-only
  soft races.
- **`entity_browser` / `dataset_browser`** sample fetches read mutable
  dataset-entity state on the GUI thread (same class).

### 1.4 The structural observation

`pause_guard` and the `post_swap` hook-barrier are **two expressions of
the same idea**: make the sim thread quiescent at a step boundary so
other code can touch state safely. They even observe the *same* state:
when `pause_guard` parks the thread, it parks at the top of the loop
*after the last `do_step()` (process + swap_io) fully completed* — which
is the same coherent boundary as `post_swap_barrier`.

So the codebase carries two parallel quiescence primitives. Worse, the
one used for state IO (`pause_guard`) runs the state-touching work on a
thread that does **not** own the wgpu queue:
`state_logger::save_state` runs on the HDF5 executor thread and calls
`read_back_sync()`; `neuron_group_gpu_t::save_state/load_state` issue
`wgpuQueueWriteBuffer` / readbacks. These are safe *today only because*
the sim thread is parked, so nothing else is using the queue — the exact
"two threads happen not to collide on the queue" fragility `cb5ca6a`'s
own message called out.

## 2. Goals

- **One** quiescence seam for all non-sim-thread access to sim state,
  keyed to the `post_swap` boundary. Retire `pause_guard`.
- All wgpu-queue operations (buffer readback/upload, GPU-entity
  save/load) run on the **sim thread**, the queue's sole owner.
- Heavy, GPU-free file IO (HDF5 read/write) stays on the `hdf5::executor`
  worker, off the sim thread, so a ~1 s save does not stall stepping for
  ~1 s.
- Fold the currently-unsynchronized GUI reads (§1.3) onto the same seam.
- Work identically whether the sim is stepping (`unlimited`/`throttled`)
  or `paused`.

## 3. Non-goals

- Changing the on-disk state format or the `state_writer`/`state_reader`
  HDF5 schema (the in-memory intermediate in §5 must reproduce the same
  HDF5 *structure* — `coast_compare`-clean and loadable — not byte-for-byte
  identical, which HDF5 does not guarantee).
- Touching the per-visualization handshake. It is correct and stays. The
  new seam is for *requester-initiated, one-shot* operations
  (save/load/checkpoint) and for the §1.3 polled reads; the periodic
  per-frame vis updates keep their own throttled handshake.
- Parallel/overlapping saves. Quiesced tasks run one at a time.

## 4. The seam: a sim-thread quiesced-task queue

Add to `sim_thread_controller` a small MPSC queue of *quiesced tasks*.
A quiesced task is a closure plus a completion signal. The sim thread
runs queued tasks **at the quiescent boundary** — the same point
`pause_guard` parked at — in two situations:

1. right after a `do_step()` completes (post-swap, GPU idle), and
2. while parked in the `paused` wait loop (so save/load work even when
   the sim is not stepping; an enqueue wakes the thread).

```
run_quiesced(fn) :  // called by GUI / headless thread
    enqueue {fn, done-flag}
    wake sim thread
    block until done-flag set        // <-- replaces pause_guard's hold

thread_main loop:
    ... do_step() ...                // or park in paused wait
    drain_quiesced_tasks():          // at the coherent boundary
        for each {fn, done}: fn(); done.set(); notify
```

Because `fn` runs **on the sim thread at post-swap**, it may freely:
touch the wgpu queue, read coherent read pages, write idle write pages,
call `entity.save_state/load_state`, and read/set the step counter.

This lets us **delete `pause_guard` entirely**: the hold is replaced by
"enqueue a task and wait for it," with the work now on the correct
thread.

`single_step` stays a separate path (decision Q3, §7). It is conceptually
adjacent — both are "the sim thread does one discrete thing at a
boundary" — but `single_step` *advances* sim time while a quiesced task
does *not*. Merging them would only share wake/dispatch plumbing for no
correctness gain, and would refactor working, tested code; deferred.

### Relationship to the choice "barrier-hook signal/wait"

The agreed direction was a barrier-hook signal/wait that removes the
separate primitive. §4 *is* that, with one deliberate refinement forced
by the wgpu-ownership goal: the state-touching body runs **on the sim
thread**, not on the requester thread. A literal "requester runs the
save" cannot hold, because `neuron_group_gpu_t::save_state/load_state`
and `read_back_sync/write_back_sync` issue queue ops that must not run
off the sim thread. See §5 for how the heavy IO still leaves the sim
thread.

## 5. Keeping HDF5 off the sim thread (the save path)

If the sim-thread task called `entity.save_state(hdf5_writer)` directly,
it would run ~1 s of libhdf5 on the sim thread. We avoid that by
splitting *extraction* from *serialization*:

- **Extraction (sim thread, inside the quiesced task):** walk buffers and
  entities, pulling state — including GPU readbacks — into an **in-memory
  state tree** (a `state_writer` implementation that builds a tree of
  named groups/scalars/arrays in RAM instead of writing HDF5). Fast;
  bounded by a few memcpys and the GPU readbacks that must be here
  anyway.
- **Serialization (executor thread):** hand the in-memory tree to
  `hdf5::executor`, which replays it through the existing
  `hdf5_state_writer`. Because that is the same sequence of `state_writer`
  calls the entities originally made, it reproduces the current HDF5
  structure by construction (`coast_compare`-clean; not byte-identical).

The requester's `run_quiesced` returns once *extraction* completes (the
sim is free to resume stepping); the file write completes asynchronously
on the executor, and the requester can wait on the executor result for
error reporting.

`buffer_logger` already demonstrates the cheap half of this (copy read
pages out on the sim thread, hand to executor). §5 generalizes it to
entity state via the in-memory writer.

### 5.1 Sim ↔ executor coordination (locking, ownership, backpressure)

All libhdf5 calls run on the **single `hdf5::executor` worker thread**,
process-wide — never on the sim thread, never on the requester thread.
The handoff uses the executor's existing synchronization (a bounded queue
under `q_mtx_` with `q_not_empty_`/`q_not_full_` condition variables); no
new locking primitive is introduced.

The one invariant that makes this race-free **at any frequency, including
a write every step**:

> The extracted state is **owned by the submitted job** (moved in). The
> sim thread reads sim state *on the sim thread*, builds an owned
> in-memory tree, then **moves** that tree into the executor job. After
> handoff the sim thread never touches it; the worker only ever touches
> the moved tree, never live sim state.

Ownership is transferred per item, so the two threads share no mutable
state beyond the queue itself. That is what permits per-step writes
without a data race.

- **Backpressure:** the save path submits via `submit()` (fire-and-
  forget). When the worker falls behind, `submit()` blocks the sim thread
  on `q_not_full_` until a slot frees. The sim throttles to IO bandwidth,
  memory stays bounded (`queue_cap × payload`), and **no item is
  dropped**. This is the accepted "storing may delay the simulation"
  behavior, made explicit and bounded.
- **Error reporting without stalling the sim:** the extraction step
  creates a `std::promise`, moves it into the job, and returns the
  `std::future` to the requester via the `run_quiesced` result. The sim
  resumes stepping immediately; the **requester** (not the sim thread)
  awaits the future for the write outcome.
- **Caveat (honest):** a *full* state save every step deep-copies all
  entity+buffer state into a tree per step — large allocations, and the
  sim runs at disk speed under backpressure. Correct and bounded, but
  IO-limited. The realistic cases — periodic checkpoints, and the small
  per-step buffer frames `buffer_logger` already chunks — are
  comfortably handled.

### Load path (symmetric)

- **Parse (executor thread, before enqueue):** the HDF5 *read* runs on the
  worker (`submit_sync`), producing the same in-memory state tree. No
  libhdf5 anywhere else.
- **Apply (sim thread, inside the quiesced task):** restore each
  io_buffer's **read page directly**, `entity.load_state(in_memory_reader)`,
  and `set_step_count` — all on the sim thread, all at the quiescent
  boundary, none of it touching libhdf5. No swap (see §5.3).

### 5.3 Pre-existing buffer-restore bug this corrects, and the clean fix

The current production load path restores io_buffer contents into the
**write** page (`state_logger.cpp:588`, `cur_write_buffer()`;
`io_buffer_gpu.cpp:258`, `write_back_sync` → `gpu_[write_idx_]`) and relies
on "the next swap" to move it to the read page — but never swaps. On
resume, `do_step()` runs `process()` first, which overwrites the restored
write page and reads the un-restored (stale) read page, then swaps — so
**restored buffer state never reaches a consumer**, and the first resumed
step diverges. Entity state is restored correctly (entities hold their
state directly, not double-buffered), so outputs re-derive after one step
and the loss is easy to miss.

It stayed hidden because the only round-trip test
(`gpu_state_roundtrip_test.cpp:117-122`) inserts a *manual* `swap_buffer()`
immediately after load, with no `process()` between, then reads back — a
sequence the production path never performs. No test steps the sim *after*
a load.

**The clean fix (not a swap):** the read page is the only buffer state that
matters at the boundary (the write page is overwritten every step by its
producer), and save already reads the read page. So load should **write the
read page directly** — symmetric with save, no swap to perform or forget.
This eliminates the bug *class*, not just the instance. The "write page +
next swap" contract (`cur_write_buffer`-based restore; `write_back_sync` →
write page) is the crutch being removed.

io_buffer interface shape for read-page restore — **decided: Option A.**
A symmetric virtual pair, e.g. `snapshot(span<float>)` /
`restore(span<const float>)`, each variant reading/writing its own
read-side state (CPU page for cpu / gpu_to_cpu; GPU read-page upload for
cpu_to_gpu / gpu). This also removes the existing
`cur_read_buffer().empty() ? read_back_sync()` branching in `state_logger`,
and retires the `cur_write_buffer`-for-restore / `write_back_sync`-to-write
-page idiom entirely.

Applies to **every** load path (runtime `apply_state` and the monolithic
`load_state` `build_env` keeps for startup, §5.2), so `--resume` is fixed
too. Consequence: `gpu_state_roundtrip_test` drops its manual
`swap_buffer()` (restored data is already in the read page) and updates its
comment.

**Commit structure (decided):** this fix is folded into the redesign branch
(not a standalone PR — a separate PR would fragment the proposal and sit
awkwardly against the still-open pause_guard proposal this branch evolves),
but lands as **its own commit**. Because it depends only on the
`snapshot`/`restore` interface — not on the threading redesign — that commit
stays self-contained and cherry-pickable to `main` later if the live
`--resume` bug should reach users sooner.

### 5.2 Headless compatibility

Verified against every headless operation (`build_env` startup load,
stepping/`-n`, `--log-buffers`, `--checkpoint`, `--save-state`,
`--emit-run-manifest`, probe-observer drain, shutdown). The design holds,
with the following requirements:

- **Saves must be awaited before teardown (must-fix).** In `main.cpp` the
  `hdf5::executor` outlives `sim`, but `run_headless`'s `manifest_guard`
  records savestate file sizes *before* `run_headless` returns. If a save
  were fire-and-forget, `record_savestate()` would `file_size()` a
  not-yet-written file (recording 0 bytes) and the file would only flush
  when the executor is destroyed. Therefore **headless awaits the write
  future** (returned from the save's `run_quiesced`) before
  `record_savestate`/manifest/return. The sim thread is still only blocked
  for extraction; the headless *main* thread waits for durability — which
  "storing may delay the simulation" permits. The GUI may stay
  fire-and-forget; the future is awaited by the requester, so the policy
  is per-caller.
- **Startup load stays synchronous (decision: option a).** `build_env`
  loads at construction, on the main thread, with the sim thread parked in
  `paused` mode and **never stepped**. The exemption from the `run_quiesced`
  seam is valid because of one checkable precondition: *at this point the
  main thread is the sole user of the wgpu queue and the sole writer of sim
  state* — the sim thread is parked, has issued no `process()`/`swap_io()`,
  and so touches neither the queue nor entity/buffer state. The load is
  therefore concurrency-free and needs no boundary. Chosen over the uniform
  `run_quiesced` path as the smaller change and the easier one to reason
  about. **This exemption would become invalid** if `build_env` were ever
  changed to load *after* stepping has begun, or if any other thread could
  touch the queue during construction; a load at any other time MUST go
  through `run_quiesced`. (The runtime GUI load path is unaffected — it uses
  the seam.)
- **`run_quiesced` needs a stop-escape.** Like `pause_guard` today (whose
  wait breaks on `stop_requested`), a requester blocked on a quiesced task
  must unblock if the controller is shutting down, to avoid a teardown
  deadlock. With the await-before-teardown rule no save is pending at
  shutdown, but the escape is required for safety.
- **No deadlock with active buffer logging.** A checkpoint extraction on
  the sim thread `submit()`s its write to the same executor that buffer
  logging uses; backpressure may briefly block the sim thread, but the
  worker never waits on the sim thread, so there is no circular wait.
- **Already-safe, out of scope:** the headless probe-observer drain
  (`run_headless.cpp:210`) reads probe SPSC queues on the main thread
  (single-producer sim / single-consumer main) and is unchanged.

## 6. Folding in the §1.3 reads

- **`buffer_browser` live stats (decision Q1 → throttled periodic
  publish):** while the IO Buffers window is open, the sim thread computes
  sum/min/max every N steps in a `post_swap_hook` (the `buffer_logger`
  pattern) and publishes the result into a GUI-readable snapshot. The GUI
  reads the latest snapshot each frame and never touches a live buffer.
  This removes the real race, keeps today's live-updating feel, fixes
  GPU-only buffers (readback now on the sim thread), and costs only a
  cheap min/max sweep every N steps, paid only while the window is open.
  The hook is registered/removed with window visibility.
- **`status_str` / browser scalars:** snapshot the handful of values into
  atomics written at `post_swap`; the GUI reads the atomics. The full
  quiesced-task machinery is overkill for display scalars.

## 7. Decisions and remaining detail

Resolved (2026-06-12):

- **Q1 — `buffer_browser` stats:** throttled periodic publish on the sim
  thread (§6), not an on-demand Refresh button.
- **Q2 — in-memory state tree:** introduce a **new
  `state_writer`/`state_reader` pair backed by a value tree** (alongside
  the existing `hdf5_state_writer/reader`). Entities are untouched — they
  already target the abstract interface. The tree must reproduce the same
  HDF5 *structure* (`coast_compare`-clean and loadable, not byte-identical)
  so existing saves keep loading; this is a primary test obligation. A sim
  delay during storing is acceptable, but **all libhdf5 reads/writes stay
  on the executor worker** (§5.1).
- **Q3 — `single_step`:** stays separate (§4). Deferred merge, no
  correctness stake.

Error propagation surface (resolved by §5.2): the save's `run_quiesced`
returns the write future; **headless awaits it** before recording the
manifest / exiting (so exit status and savestate sizes are correct), while
the **GUI** may await it off the main thread and surface a failure on the
status line. Per-caller policy, not a global one.

- **Startup load (`build_env`):** stays synchronous as a justified
  exemption (§5.2) — chosen as the smallest, easiest-to-reason-about
  change. Decided.

Remaining detail to settle during implementation:

- Choice of N (publish interval) for `buffer_browser`, and whether it is
  fixed or tied to the existing vis `update_interval` hint.

## 8. Migration

- Total replacement, no shim: delete `pause_guard`, `hold_paused()`, and
  the `hold_count_/parked_` plumbing once `run_quiesced` lands and the
  four call sites (GUI save, GUI load, headless checkpoint, headless
  final save) move over.
- The existing `test_thread_controller_hold.cpp` contract (quiesce,
  mode-preservation, nesting, single-step deferral, shutdown) maps onto
  `run_quiesced` and is rewritten against it.
- Threading docs (`docs/content/threading/`) updated to describe the one
  seam.

## 9. Outcome — retiring `pause_guard` (post-implementation summary)

Written after the implementation landed (Steps 0–6, 2026-06-13), as the
concise record of *why* and *how* the retirement happened.

### What `pause_guard` was

A caller-side RAII hold on the sim thread: `hold_paused()` blocked until any
in-flight step finished and the thread parked, kept it parked while the hold
lived, then resumed the configured mode. It fixed a real bug — GUI save/load
and headless checkpoints had been touching live sim state with no quiescence
guarantee at all (`set_mode(paused)` only stores an atomic and returns).

### Why it had to go — three structural defects

1. **The state-touching work ran on the wrong thread.** The hold parked the
   sim thread, but the save/load body ran inside `executor::submit_sync` —
   on the HDF5 worker. That thread called `read_back_sync()` and
   `neuron_group_gpu_t::save_state/load_state` issued
   `wgpuQueueSubmit`/`wgpuQueueWriteBuffer` from there. The wgpu queue
   belongs to the sim thread; this was safe *only because* the sim happened
   to be parked — correctness rested on a scheduling coincidence, not on
   ownership (the fragility b7271a4's own message flagged).
2. **A second, redundant quiescence primitive.** The codebase already had a
   quiescence seam — the `post_swap` hook boundary used by every visualizer
   and `buffer_logger`. `pause_guard` parked at *exactly the same coherent
   state*, so the codebase carried two parallel mechanisms for one boundary,
   one of them a stop-the-world imposed from outside.
3. **Incomplete coverage, high cost.** The hold spanned the *entire* HDF5
   write — the sim stalled for the full file-write duration, not just the
   state copy. It guarded only 4 call sites; the Step-4 re-audit found two
   more GUI saves running with no guard at all (`save_auto_checkpoint`,
   `perform_auto_save`), and the §1.3 GUI reads (buffer-browser stats,
   `status_str`) were never covered. The load path it guarded also carried
   the latent write-page restore bug (§5.3).

### How it was replaced — inversion of control

The replacement keeps pause_guard's *boundary* but inverts who does the work:

- **`run_quiesced(fn)`** (§4): instead of "park the sim thread, do the work
  elsewhere," the caller enqueues a closure and the **sim thread itself**
  runs it at the post-swap boundary, the caller blocking on a future. wgpu
  queue ops land on the owning thread *by construction*. The hold's
  stop-escape maps onto teardown rejection; single-step deferral disappeared
  (nothing to defer when there is no hold).
- **The state-IO split** (§5) bounds the sim stall: `extract_state` (sim
  thread, fast, into an in-memory `mem_group` tree) / `serialize_state`
  (executor, async, tree moved into the job) / `parse_state` (executor) /
  `apply_state` (sim thread, `restore()` writes the read page directly — no
  swap, eliminating the §5.3 bug *class*). Under pause_guard the sim stalled
  for extraction *plus* the whole HDF5 write; now only for extraction.
- **The §1.3 reads** were folded onto the *existing* hook seam, not onto
  `run_quiesced` (one-shot/blocking is the wrong shape for per-frame
  polling): gated post_swap hooks compute buffer stats and `status_str()` on
  the sim thread and publish through `demand_published<T>`, which recomputes
  only when the GUI consumed the last value — cost tracks GUI rate, not step
  rate; GPU-only buffers now show real stats.
- **One justified exemption** (§5.2): `build_env`'s startup load keeps the
  monolithic path, valid only before the sim thread has ever stepped.

### Net result

One boundary, two complementary mechanisms — `run_quiesced` for one-shot
blocking ops, post_swap hooks for periodic publish — instead of two
competing quiescence primitives. Thread ownership correct by construction;
sim stall bounded by extraction, not IO; 7 sites covered instead of 4 (+2
that were unguarded); the GUI-read races closed; the restore bug class
eliminated; headless manifests record true sizes because the write future is
awaited. `pause_guard` was deleted without residue — its test contract lives
on, rewritten against `run_quiesced` in `test_thread_controller_hold.cpp`.
