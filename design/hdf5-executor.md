# HDF5 Executor — Design Document

Status: implemented — c3de44df (`include/hdf5/executor.h`)
Author: COAST team
Date: 2026-05-22

## 1. Problem

The linked HDF5 library is built without `--enable-threadsafe`. Its
contract is "exactly one thread inside the library at a time, process
wide" — not per-file. Concurrent calls from different threads, even
against disjoint files, race on the global ID table, the global error
stack, the global free lists, and the lazy `H5open()` init path.

The codebase currently violates this contract:

- `observe_writer_t` owns a worker thread that performs `H5Fcreate`,
  `H5Dcreate2`, `H5Dwrite`, `H5Dclose`, `H5Fclose` on its writer thread
  (one per observe group; multiple writers can exist concurrently).
- `buffer_logger` owns its own writer thread doing HDF5.
- `state_logger::save_state` / `load_state` perform HDF5 calls on the
  caller's thread (main in headless, GUI in interactive).

In headless mode with the new observe feature, `--save-state` (and
`--checkpoint`) calls `H5Fcreate` from the main thread while one or
more observe writer threads are simultaneously inside HDF5 calls.
Result: SIGSEGV inside libhdf5.

The bug is structural. Any fix that leaves the "each subsystem owns
its own HDF5 thread + we coordinate them" topology in place will
regress as soon as a new HDF5-using component appears.

## 2. Goals

- Encode HDF5's "single-threaded library" contract in the type system,
  so misuse cannot be expressed without a compile error or an
  obvious runtime assertion.
- One dedicated thread is the *only* thread that ever enters
  `libhdf5`. All other code submits work to it.
- Remove all per-subsystem HDF5 threads (`observe_writer_t`,
  `buffer_logger`'s writer).
- Keep the executor agnostic of `sim::env`, `logging`, `observe`. It
  knows about HDF5 and nothing else.

## 3. Non-goals

- Parallel I/O. HDF5 serialises all calls behind a single global
  contract; we accept that. Real parallel HDF5 (MPI-IO) is out of
  scope.
- Replacing HDF5 with another format.
- Process-wide singleton. The executor is constructed explicitly in
  `main` and passed down; there is no global accessor.
- Backwards compatibility shims for the old per-subsystem HDF5
  threading. The migration is total, not incremental.

## 4. Design overview

A single class, `hdf5::executor`, owns one worker thread. The thread
calls `H5open()` once before accepting work, then services a queue of
submissions until shutdown. All HDF5 entry points used anywhere in
COAST are reached only through submissions to this queue.

Caller-facing surface is two functions:

```cpp
namespace hdf5 {

class executor {
public:
    executor();   // starts the worker thread, calls H5open() on it
    ~executor();  // drains queue, joins worker, never blocks forever

    executor(const executor&) = delete;
    executor& operator=(const executor&) = delete;

    // Fire-and-forget. Returns immediately. Used by observe/buffer
    // writers for per-step appends.
    void submit(std::function<void()> job);

    // Blocking. Caller waits for the job to complete on the worker
    // thread and gets its return value (or rethrown exception).
    // Used by state_logger::save_state / load_state.
    template <class F>
    auto submit_sync(F&& fn) -> std::invoke_result_t<F&>;
};

} // namespace hdf5
```

Internally both go through the same SPMC queue
(single-producer-per-call-site, single-consumer = worker thread). The
sync variant is implemented as an async submit of a wrapper that
fulfils a `std::promise`.

### 4.1 Backpressure

The queue is bounded (cap configurable, default ~1024 entries — same
order as the per-writer cap today). `submit()` blocks the producer
when full; this preserves the temporal-fidelity guarantee that the
observe writer queue already provides. `submit_sync()` blocks the
caller anyway.

### 4.2 Exception safety

Worker-thread exceptions inside a `submit_sync` job are captured and
rethrown to the caller via the `std::promise`. Exceptions inside a
`submit` (async) job are logged to `stderr` and swallowed — the worker
must not die. (Open question: do we want a fatal-exit policy instead?
See §9.)

## 5. Handle lifecycle — RAII with off-thread destruction

The current `logging::hdf5::{file,group,dataset,dataspace,plist}`
wrappers (`include/logging/hdf5_utils.h`) own a `hid_t` and call the
appropriate `H5*close` in their destructor. Today those destructors
run on whatever thread destroys the wrapper — which is exactly the
bug we are eliminating.

New rule: an `hdf5::handle` is created by the executor, and its
destructor posts a close-job to the executor instead of calling
`H5*close` directly.

```cpp
namespace hdf5 {

template <auto CloseFn>   // e.g. H5Fclose, H5Dclose, H5Sclose, ...
class handle {
    hid_t   id_   = H5I_INVALID_HID;
    executor* exec_ = nullptr;

public:
    handle() = default;
    handle(hid_t id, executor& exec) : id_(id), exec_(&exec) {}

    handle(handle&& o) noexcept;             // move-only
    handle& operator=(handle&& o) noexcept;
    handle(const handle&) = delete;
    handle& operator=(const handle&) = delete;

    ~handle() {
        if (id_ >= 0 && exec_) {
            const hid_t id = id_;
            exec_->submit([id] { CloseFn(id); });
        }
    }

    [[nodiscard]] hid_t id() const { return id_; }
    [[nodiscard]] explicit operator bool() const { return id_ >= 0; }
};

using file      = handle<H5Fclose>;
using group_h   = handle<H5Gclose>;
using dataset_h = handle<H5Dclose>;
using space_h   = handle<H5Sclose>;
using plist_h   = handle<H5Pclose>;
using attr_h    = handle<H5Aclose>;
using type_h    = handle<H5Tclose>;

} // namespace hdf5
```

Properties:

- A `handle` cannot exist without an executor reference, so it cannot
  be closed on the wrong thread by construction.
- Move-only. No reference counting; if you need to share, the right
  answer is "redesign the data flow," not "add shared ownership of
  global HDF5 state."
- The destructor's close-job is fire-and-forget. The handle's owning
  thread does not block on close. This is fine because HDF5 close
  ordering is intra-file (children before parents), and the executor
  preserves submission order — so as long as you destroy children
  before parents (which RAII gives you naturally with the right
  declaration order), the close calls execute in the same order on
  the worker.

### 5.1 Creating handles

Creation goes through `submit_sync` because the caller needs the
resulting `hid_t`:

```cpp
hdf5::file create_file(executor& ex, const std::filesystem::path& path) {
    return ex.submit_sync([&]{
        hid_t id = H5Fcreate(path.string().c_str(),
                             H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (id < 0) throw std::runtime_error("H5Fcreate failed: " + path.string());
        return hdf5::file{id, ex};
    });
}
```

Open helpers (`open_dataset`, `create_dataset`, ...) follow the same
shape. They live in `include/hdf5/ops.h` and form a thin typed layer
over the C API. Callers never see raw `hid_t` lifecycle.

### 5.2 Operating on handles

Operations that read or write data are submitted as jobs whose body
calls `H5Dwrite`, `H5Dset_extent`, etc. The job captures the relevant
`hid_t` *values* (not the handle wrapper, since the wrapper owns
destruction and we don't want a job to keep the handle alive longer
than the caller intends). Caller discipline: the handle must outlive
the submitted job. For sync jobs this is automatic. For async jobs,
the calling subsystem is responsible — same rule the queue cap
already enforces in observe_writer.

## 6. Lifetime and ownership

The executor is constructed at the top of `main`, before anything
else that might touch HDF5, and destroyed last:

```cpp
int main(int argc, char** argv) {
    hdf5::executor h5_exec;     // step 1: HDF5 thread up, H5open() done

    cli_options opts = parse_args(argc, argv);
    // ... pass `h5_exec` into sim_instance / run_headless / GUI ...
    return opts.headless ? run_headless(opts, h5_exec) : run_gui(opts, h5_exec);
}
```

`sim_instance`, `logging_manager`, `observe_registry` all take a
non-owning reference to `h5_exec`. They never construct or destroy
it. Teardown order is:

1. Sim thread stops.
2. `observe_registry::detach()` — flushes writers' pending submissions
   by calling `submit_sync` on a barrier job, then closes handles
   (destruction posts close-jobs to the executor).
3. `logging_manager` reset — same.
4. Returns from `main`. `h5_exec`'s destructor drains the final close
   jobs and joins the worker.

## 7. Migration of existing subsystems

### 7.1 `observe_writer_t`

Becomes a stateful submitter, not a thread owner.

- Remove `std::thread worker_`, `q_mutex_`, `q_not_empty_`,
  `q_not_full_`, `queue_`, `stop_`, `thread_main_`.
- `start()` becomes `open()`: synchronously creates the `hdf5::file`
  on the executor, stores it.
- `enqueue(site, step)` copies the scratch bytes into an owned buffer
  (as today) and posts an async job that calls `H5Dset_extent`,
  `H5Sselect_hyperslab`, `H5Dwrite` for both value and step datasets.
- `stop_and_join()` becomes `close()`: posts a barrier `submit_sync`,
  then drops the handles (which post close jobs).

Net effect: `observe_writer.{h,cpp}` shrinks substantially. The
banner comment about "ALL HDF5 calls on the worker thread" is
obsolete (true now of every HDF5 user) and is deleted.

### 7.2 `buffer_logger`

Same pattern. Its existing writer thread is removed; appends become
submissions.

### 7.3 `state_logger::save_state` / `load_state`

Becomes a `submit_sync` job at the top level. The bulk of the
function body runs unchanged on the worker thread. No changes to the
file format or schema.

### 7.4 `observe_registry::attach()`

The `H5open()` call here is removed — the executor does it. This
fixes the existing init-order workaround in a structural way: the
library is initialised exactly once, in a known thread context,
before any other code path can race.

### 7.5 `logging::hdf5` utilities

The wrappers in `include/logging/hdf5_utils.h` are replaced by the
new `hdf5::handle` template and op helpers. The convenience
free functions (`write_attr`, etc.) are reimplemented as
`submit_sync` wrappers.

## 8. Behavioural changes the caller should be aware of

- **`save_state` pauses observe capture for its duration.** Today the
  global HDF5 lock (in the threadsafe build) or random crash (in the
  non-threadsafe build) had the same effect; the new behaviour is
  just *deterministic* serialisation. For checkpoint intervals of
  10⁴ steps and save durations of order 1s, this is negligible.
- **Observe writer queue capacity becomes shared with checkpoint
  saves.** A long-running save fills its slot in the queue; observe
  appends block behind it. This matches the underlying reality
  (HDF5 is single-threaded) — we are no longer hiding it.
- **Close ordering matters.** Today, each subsystem closes its own
  handles synchronously on its own thread. With async close
  destructors, the caller must ensure children are destroyed before
  parents (natural with RAII + correct declaration order). If we
  ever introduce a case where this is non-obvious, we add an explicit
  `submit_sync` barrier.

## 9. Open questions

1. **Async-job exception policy.** Log-and-continue, or `std::terminate`?
   Log-and-continue is more forgiving for write failures (disk
   full mid-run); terminate is safer because a partial HDF5 state
   may be unrecoverable. Proposal: log-and-continue, with a counter
   exposed via `executor::error_count()` that callers can check.

2. **Queue cap.** 1024 entries matches the current observe writer
   cap. With three subsystems sharing the queue, do we want it
   larger? Proposal: leave at 1024 for now; tune if benchmarks show
   producer-side blocking matters.

3. **`logging::hdf5` namespace.** Move to top-level `hdf5::`? The
   current location was logging-local because logging was the only
   user. With observe added, top-level is more honest. Proposal: yes,
   move; one rename in the migration.

4. **Diagnostics.** Should `submit_sync` log slow jobs (e.g. > 1s) to
   surface checkpoint-stalling-observe situations? Proposal: not yet;
   add only if it becomes a debugging pain.

## 10. Validation plan

1. **Unit test the executor.** Submit 10⁴ jobs from multiple
   threads, verify FIFO order on the worker, verify shutdown
   completes within a bounded time.
2. **Reproduce the original SIGSEGV first** (config with an observe
   group that fires during a `-n N --save-state` run) on the current
   code. Capture a core dump for the record.
3. **Run the same repro on the refactor.** Confirm no crash, and
   confirm the saved state file and observe `.h5` files are both
   well-formed (read back with `h5dump` or the existing Python
   loader).
4. **Checkpoint stress.** `-n 10⁶ --checkpoint 10⁴` with an active
   observe group. Confirm no crashes, no leaked handles
   (`H5Fis_hdf5` on shutdown, or a leak check via the executor
   tracking outstanding handles).
5. **Throughput regression.** Compare SPS for a config with observe
   active before/after the refactor. Expectation: within noise
   (±3%, per the existing perf-measurement protocol). If worse,
   investigate before merging.

## 11. Out of scope, captured for later

- A leak detector in `hdf5::executor` that tracks live handles and
  reports unclosed ones at shutdown. Probably easy to add and
  worthwhile; deferred to keep this change focused.
- A `submit_priority` variant so checkpoint saves can be deprioritised
  behind observe appends (or vice versa). Not needed for current
  workloads.
- Replacing `hid_t` capture in async jobs with a typed token that
  the executor checks against its handle registry. Belt-and-braces
  safety on top of the RAII rule; defer until we see misuse.
