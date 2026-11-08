# Sim-control console façade — drive simulations from the embedded C++ REPL

Status: **implemented** — shipped and tested on branch `feat/console-simctrl`.
Extends the h5-read console (`design/console_api_design.md`) with a second façade
that builds, steps, and persists simulations from the interpreter, by **shelling
out to the headless `coast` binary**. Covered by the `b_tier.simc_*` subprocess
tests (state chaining, async, failure, collisions, dataset-name injection, fd
hygiene).

## Goal

From the console: load a base config, mutate settings (learning rate, output
destinations, seed, …), run the sim for a chosen number of steps, inspect the
results, and iterate — for parameter search and characterization. Each run is
**file-based and self-documenting**: the exact config used and all outputs land
in one run directory.

## The decisive fact that shapes the design

Group hyperparameters (ngm2 `default_primary_learning_rate`, ngm3
`common_learning_rate`, …) are **baked into the neuron groups at build time** by
`sim_builder`; they are not re-read from the config each step. The observe
destination (`observe.output_dir`) and snapshots (`snapshot_dir`) are likewise
config fields consumed at construction, and already round-trip to JSON.

Consequence: "change a hyperparameter and keep the *same* evolving sim running"
is not achievable by mutating a live sim — it needs a fresh build, which for a
parameter search is what we *want*. The model is therefore **fresh sim per
config**, each pointed at its own run directory; "keep running" is stepping in
chunks, chained through saved state.

## Architecture decision: subprocess, not in-process

`step(n)` runs a **child `coast` process** headless, rather than building a
`sim_instance` inside the console. Rationale:

- **Reuse the whole app, not just a library.** `coast -n N config.json` already
  loads the config, builds the sim, steps, checkpoints, buffer-logs, saves state,
  and surfaces faults via exit code. The B-tier tests already exercise this exact
  CLI surface (`--save-state`/`--load-state`/`--checkpoint`/`--log-buffers`). The
  façade adds a config editor + a launcher; the simulation code is untouched.
- **Isolation closes the one hole the in-process console cannot.** A genuine
  native fault (null deref in bad JIT'd/sim code) is uncontainable in-process and
  would take down the GUI (`console_api_design.md`, "Out-of-process execution").
  In a child process it kills only the child; the console reports a nonzero exit
  and stays alive.
- **No shared-state entanglement.** The child has its own HDF5, its own threads,
  its own memory. No contention on the console's single HDF5 executor, no sim
  linked into the console, no parse-boundary growth.

Trade-off (accepted): no *live in-memory* inspection between chunks, and each
chunk reloads the dataset + (de)serializes state. Negligible for chunks of ≥10k
steps (the characterization case); wasteful only for very short chunks — document
it. Between-chunk inspection is via the written HDF5, read with the existing h5
console façade (`open("<run_dir>/observe/…")`).

## What we reuse (no reimplementation)

| Need                                   | Backed by                                                        |
|----------------------------------------|------------------------------------------------------------------|
| Load config, build sim, step, checkpoint, buffer-log, save-state, fault→exit-code | the `coast` binary itself, run headless with `-n N` |
| Resume / start from a checkpoint       | child `--load-state <path>` (or `--resume`)                      |
| Persist state at a chunk boundary      | child `--save-state <path>`                                      |
| Locate the binary to spawn             | `/proc/self/exe` — the console runs *inside* `coast`             |

## The `sim_console_api.h` surface (in `coast_gui`, no LLVM, lean header)

Parse-boundary discipline as in `console_api.h`: `nlohmann/json` is **not**
exposed — the config JSON is held by pointer in the `.cpp`, edited through a
string **JSON-Pointer** API (raw base) plus a few sugar setters. This keeps
`json.hpp` off the interpreter's parse boundary and its ~1 s JIT parse cost.

```cpp
namespace coast::simc {

// ---- editable config -----------------------------------------------------
// Parsed config held by pointer (pimpl). Edited by JSON Pointer, e.g.
// "/ngm2/groups/0/default_primary_learning_rate". Written verbatim to
// <run_dir>/config.json by run(); the child re-parses it.
class run_config {
public:
    run_config(const run_config&);              // copyable: `auto c = base;` per sweep point
    run_config& operator=(const run_config&);

    // raw base — reach any field by JSON Pointer
    run_config& set(std::string_view ptr, double v);
    run_config& set(std::string_view ptr, std::int64_t v);
    run_config& set(std::string_view ptr, bool v);
    run_config& set(std::string_view ptr, std::string_view v);
    run_config& set(std::string_view ptr, const char* v);   // see note below
    run_config& set(std::string_view ptr, int v);           // see note below
    [[nodiscard]] double      get_number(std::string_view ptr) const;
    [[nodiscard]] std::string get_string(std::string_view ptr) const;

    // sugar — the common knobs. NB: as built there is deliberately NO
    // learning_rate() helper (rates live in several places and neuron_groups is
    // keyed by group name, so a single setter would guess wrong — use set() with
    // the explicit pointer). Kept out of this sketch to match the header.
    run_config& seed(int v);                    // --seed-offset for the child
    run_config& run_dir(std::string_view d);    // observe -> d/observe,
                                                // snapshot_dir -> d/snapshots,
                                                // config.json written to d/config.json
    [[nodiscard]] std::string dump() const;     // pretty JSON, for inspection
};

// load a base config file into an editable run_config (throws on parse error)
[[nodiscard]] run_config config(const std::filesystem::path& base);

// ---- a persistent run (subprocess-backed) --------------------------------
class sim_run {
public:
    // Run n steps: spawn `coast -n n [--load-state <latest>] --save-state
    // <run_dir>/state_<NNN>.h5 --seed-offset <s> <run_dir>/config.json`,
    // redirecting the child's stdout+stderr to
    // <run_dir>/logs/step_<timestamp>_<NNN>.log (state and log share <NNN>).
    // BLOCKS the console thread until the child exits (the GUI stays responsive;
    // the console shows busy). Chains state across calls, so successive step()s
    // continue the same trajectory. Throws std::runtime_error on nonzero child
    // exit (message carries the log path). Returns cumulative steps. step(0) is a
    // no-op (NB: coast rejects a raw -n 0; it once meant UNLIMITED — guarded).
    std::uint64_t step(std::uint64_t n);

    // NON-BLOCKING: hand the run to the process pool and return at once. If a
    // pool slot is free the child spawns immediately; otherwise the run QUEUES and
    // spawns automatically when a running child exits (see the pool below) — the
    // call never blocks or fails because too many runs are going. Progress is
    // tracked in the process-wide run registry; with show_status the "Simulation
    // Runs" dialog opens (all concurrent + queued runs share its table). On
    // success the saved state chains in exactly as step() would. Throws if a step
    // is already in flight on this handle, or on an immediate-spawn failure.
    void step_async(std::uint64_t n, bool show_status = true);

    [[nodiscard]] std::uint64_t total_steps() const;   // cumulative across step()s
    [[nodiscard]] std::string   run_dir() const;
    [[nodiscard]] std::string   observe_dir() const;
    [[nodiscard]] std::string   config_path() const;   // the written config.json
    [[nodiscard]] std::string   state_path() const;    // latest saved state, "" if none
    [[nodiscard]] std::string   last_log() const;      // log of the most recent step()
    [[nodiscard]] bool          running() const;        // a step (blocking or async) is in flight
    void                        cancel();               // stop this handle's in-flight async run

    void save_state(std::filesystem::path path);       // copy latest state out
    void load_state(std::filesystem::path path);       // next step() loads this
};

// ---- async run monitoring (process-wide registry) ------------------------
// GUI thread, once per frame (mirrors coast::plot::render_all): poll children,
// then draw the status table if the dialog is open.
void render_runs_window();
// Advance async-run state without drawing (reap children, chain state, parse
// progress). render_runs_window() calls this; call it directly with no GUI.
void poll_runs();
void wait_all_runs();          // block until every async run finishes (drains the queue)
[[nodiscard]] int active_run_count();  // running + queued
void clear_finished_runs();    // drop finished rows (keeps running + queued)
void show_runs_window();       // reopen the status dialog after its X button hid it
void hide_runs_window();
void cancel_all_runs();        // SIGTERM every running child, drop every queued run

// ---- async run process pool ----------------------------------------------
// A global cap on concurrently-running children: step_async() spawns up to
// max_concurrent_runs() at once and QUEUES the rest, draining the queue inside
// poll_runs() as children exit. Each coast child is itself TBB-parallel across
// all cores, so the cap is chiefly a resource-exhaustion guard for large sweeps.
// Default 4; env COAST_SIMC_MAX_CONCURRENT overrides at startup, the setter at
// runtime (clamped to >= 1).
void set_max_concurrent_runs(int n);
[[nodiscard]] int max_concurrent_runs();
[[nodiscard]] int queued_run_count();

// Materialize a run: create the run directory, write config.json into it, return
// a handle. Does not step. Throws std::runtime_error if the dir cannot be
// created. Run directory precedence: explicit `dir` arg > config's run_dir() >
// "runs/<base-config-stem>". The resolved path is claimed atomically via
// create_directory(): the first run to claim it keeps the exact path; any later
// run resolving to the same path gets the next free "_NNN" suffix, so two runs
// never share config.json / state.h5 / logs.
[[nodiscard]] std::shared_ptr<sim_run> run(const run_config& cfg,
                                           const std::filesystem::path& dir = {});

// ---- filesystem convenience ----------------------------------------------
void                     mkdir(const std::filesystem::path& p);   // create_directories
[[nodiscard]] bool       exists(const std::filesystem::path& p);
[[nodiscard]] std::vector<std::string> ls(const std::filesystem::path& p);
[[nodiscard]] std::string cwd();

} // namespace coast::simc
```

No `set_executor` here — the child owns its own HDF5, so the façade needs nothing
from the host except the ability to find `/proc/self/exe`.

### Example session

```cpp
for (double lr : {0.01, 0.03, 0.1}) {
    auto c = simc::config("configs/mnist.json");     // fresh editable config each run
    // No learning_rate() sugar (rates live in several places keyed by group
    // name) — set the explicit JSON pointer:
    c.set("/ngm2/dendrites/default/default_primary_learning_rate", lr)
     .run_dir("runs/lr_" + std::to_string(lr));
    auto h = simc::run(c);                           // config.json written, nothing stepped
    h->step(50000);                                  // child runs 50k, writes into run_dir
    // inspect: auto f = open(h->observe_dir() + "/…");   // existing h5 console façade
    h->step(50000);                                  // continue: --load-state chains it
}
```

`runs/lr_0.03/` then holds `config.json` (exact input), `observe/`, and one
`state_<NNN>.h5` per chunk (`state_001.h5`, `state_002.h5`, …) — file-based
provenance of the run.

## step() mechanics

Each chunk saves to a fresh, per-chunk state file rather than overwriting one
`state.h5`. The next chunk `--load-state`s the newest and `--save-state`s the
next. Completed files are therefore immutable, which keeps a full rolling
history for later introspection, makes a torn write survivable (earlier files
stay intact → resumable), and removes the `save_state()` race (it copies a
stable prior file, never one a child is mid-writing). Cost: disk grows with the
number of chunks — accepted for the provenance/introspection it buys.

```
if (n == 0) return total_;                       // guard: coast rejects -n 0
idx = format("%03d", ++chunk_);                  // shared with the log name
argv = { self_exe(), "-n", to_string(n), "--seed-offset", to_string(seed_),
         [latest_state_ ? "--load-state", latest_state_ : ],
         "--save-state", run_dir_/("state_" + idx + ".h5"),
         config_path_ };
rc = spawn_and_stream(argv);                     // posix_spawn + pipe; lines -> console log
if (rc != 0) throw std::runtime_error("sim run failed (exit " + rc + ")");
latest_state_ = run_dir_/("state_" + idx + ".h5");
total_ += n;
return total_;
```

`spawn_and_stream` is the one genuinely new mechanism (justified — launching the
child). posix_spawn with a pipe, read child stdout+stderr line by line onto the
console output. Blocks the **console** thread (fine — not the GUI thread).

## Implementation notes (as-built)

- **`const char*` set() overload is required, not optional.** A string literal is
  `const char*`, and `const char* -> bool` is a *standard* conversion that
  outranks the *user-defined* `const char* -> string_view`. Without an exact-match
  `set(std::string_view, const char*)` overload, `set("/observe/output_dir",
  "runs/x")` silently stores a **boolean**. This was caught only by running a real
  child against an edited config (it failed to load with "type must be string, but
  is boolean") — a unit test on `dump()` alone would have missed it, since the
  wrong value is still valid JSON. Regression-guarded in the B-tier test.
- **`int` set() overload is required too.** A bare `int` literal converts to the
  `double`, `int64_t`, and `bool` overloads at *equal* rank, so `set(ptr, 42)` —
  the form the console help advertises — is ambiguous and fails to compile. An
  exact-match `set(std::string_view, int)` forwarding to the `int64_t` setter
  resolves it. (Same class of overload trap as the `const char*` one above.)
- **`COAST_SIMC_BINARY` overrides the spawned binary.** Normally `step()` spawns
  `/proc/self/exe` (the running `coast`). The env override targets a specific
  build and makes the façade testable from a non-`coast` host process (the B-tier
  test sets it to `$<TARGET_FILE:coast>`). The child is spawned with
  **`posix_spawnp`** (fork + `execvp`), so the value may be an absolute/relative
  path *or* a bare command name — a slash-less name is resolved via `PATH`. Plain
  `posix_spawn` (`execv`) would not search `PATH` and would read a bare `coast` as
  `./coast` in the CWD; `posix_spawnp` matches the "which command to run"
  expectation. The absolute `self_exe()` default resolves identically either way.
- **Child output** goes to a per-step log file: `spawn_child` redirects the
  child's fds 1/2 to `<run_dir>/logs/step_<ts>_<NNN>.log` via
  `posix_spawn_file_actions`, so a run is self-documenting and `tail -f` on
  `last_log()` is a live monitor. (Only if `log_path` is empty would the child
  inherit fds 1/2 and flow into the console's fd-capture; step()/step_async()
  always pass a log path, so that branch is unused by the façade.)

## Monitoring, blocking, and logs

`step(n)` is **blocking** — it returns only when the child exits. It runs on the
console thread, so the GUI stays responsive but the console cannot accept another
cell until it returns (`session::busy()` reflects this). "Finished" = the child
process exited; a nonzero exit becomes a thrown `std::runtime_error`.

Each `step()` writes the child's full stdout+stderr to
`<run_dir>/logs/step_<YYYYmmdd_HHMMSS>_<NNN>.log` (`<NNN>` = per-handle chunk
index, so two steps in the same second still get distinct files), and prints a
one-line summary (`step: ran N steps (total M) -> <log>`) to the console. Because
`coast` prints periodic progress (step counts, checkpoint lines), `tail -f` on
`h->last_log()` is a live progress monitor while a long step runs.

### Async runs and the status dialog

`step_async(n, show_status=true)` returns immediately, so a sweep fires many runs
without blocking. A **process pool** bounds how many actually run at once
(`max_concurrent_runs()`, default 4): if a slot is free the child spawns now, else
the run **queues** and spawns automatically when a running child exits. Each is
recorded as a row in a **process-wide registry**; `render_runs_window()` — called
once per frame by `gui_manager`,
right after `coast::plot::render_all()` — polls the children and draws a
**"Simulation Runs" table** (label, progress bar, `cur/target` steps, status,
elapsed) when the dialog is open. Progress is parsed from the child log:
`Starting … (from step A to B)` gives the bar's endpoints, periodic `Steps: N`
lines advance it (10k granularity, so short runs jump 0→100%).

Concurrency model:
- **Registry** = `{rows, mutex, visible}` singleton in `coast_gui`. `sim_run` is
  pimpl + `enable_shared_from_this`; its mutex guards `latest_state`/`total`/
  `async_in_flight` and keeps `<mutex>` off the interpreter parse boundary.
- **`poll_runs()`** (GUI frame, or `wait_all_runs()`, or a test) does
  `waitpid(WNOHANG)` per running row; on exit it fires that row's completion
  callback, which chains the chunk's `state_<NNN>.h5` back into the `sim_run`
  as the new `latest_state` (or, on failure,
  just clears the in-flight flag).
- **Pool drain.** After reaping, `poll_runs()` promotes queued rows to running
  (oldest first) up to `max_concurrent`. A queued row keeps its handle's
  `step_in_flight` true (so `running()` stays true and a second step is rejected);
  its argv is captured at submit time and stays valid because that same guard
  keeps the handle's `latest_state` stable until the run completes. `active_run_count()`
  = running + queued, so `wait_all_runs()` only returns once the queue has fully
  drained. A queued spawn that fails has no caller to throw to, so `poll_runs()`
  finalizes it as `failed` and fires `on_done(false)` — same outcome as a nonzero
  child exit. Cap via `set_max_concurrent_runs()` / `COAST_SIMC_MAX_CONCURRENT`.
- **Lock order is one-way.** `step_async` locks the `sim_run` *then* registers
  (sim_run→registry). `poll_runs` collects completion callbacks **under** the
  registry lock but invokes them **after releasing it** (callbacks take the
  sim_run lock) — so the two locks are never held in the reverse order.
- One in-flight async step per handle (a second `step_async`/`step` throws).
- **Cancellation.** `sim_run::cancel()` / `cancel_all_runs()` / the table's per-row
  **Cancel** + **Cancel all** buttons stop in-flight runs. A *running* row is
  SIGTERM'd (coast installs no handler → default terminate) and flagged; `poll_runs`
  reaps it and finalizes it as `cancelled` (not `failed`), so it chains nothing. A
  *queued* row is finalized immediately (no child yet) — its `on_done(false)` fires
  off the registry lock, same one-way discipline as `poll_runs`. `cancelled` counts
  as finished (dropped by "Clear finished", excluded from `active_run_count`).
- Children are independent processes: closing the GUI leaves running sweeps to
  finish on disk (a subprocess benefit). `COAST_SIMC_BINARY` still applies.

Table interaction:
- **Double-click a row** opens a **live log viewer** window (tails
  `last_log()`, re-read ~1/s, auto-follows the bottom) — the quick path to a
  child's error output. Viewers are GUI-thread-only state (no registry lock);
  the viewer render runs *after* the registry lock is released.
- **Error indication**: a failed run's progress bar turns **red** (frozen at the
  fraction it reached) and its status shows `failed (<exit>)` in red; a done run's
  bar is green.

**stdout buffering (subtle, load-bearing).** A child's stdout redirected to a
FILE is *full-buffered* by C stdio, so progress lines never reach the log until
the process exits — the bar would jump 0→100% and `tail -f` would show nothing.
Fixed in the child: `run_headless` sets `std::cout << std::unitbuf` so each line
flushes immediately. (See `memory: stdout_fullbuffered_to_file`.)

**Preamble alias.** The console preamble declares `namespace simc =
coast::simc;` (an *alias*, not `using namespace`), so `simc::run(c)` works while
`run`/`config` stay qualified (a user can still name a variable `run`). Tests that
don't `--whole-archive` must reference a `coast::simc` symbol from C++ to pull the
object in before the JIT can resolve `simc::…`.

Not built: true streaming progress finer than the child's 10k-step print cadence.
(Cancel/kill from the table IS built — see the Cancellation notes above.)

## Known couplings / footguns

- **CWD coupling (existing).** The child resolves dataset paths — and likely
  `observe.output_dir` / `snapshot_dir` — relative to **its** CWD
  (`memory: cwd_coupling_followup`). The child inherits the GUI process CWD, so
  datasets resolve as they do for the GUI; `run_dir` paths are relative to that
  CWD. Document; do not fix here.
- **`-n 0` used to mean unlimited**, not zero — `step(0)` is guarded to a no-op
  for that reason. Superseded on `refactor/headless-build`: coast now *rejects*
  `-n 0` (an unbounded run must say `--unlimited`), so the guard's purpose has
  changed from "avoid an accidental infinite run" to "avoid spawning a child that
  would exit 1". The guard stays either way.
- **Blocking.** A long `step()` blocks the console thread until the child exits;
  the console `busy()` flag reflects it, the GUI stays responsive. No cancel in
  v1 (kill the child externally if needed).

## Build / wiring

- New `include/sim_console/sim_console_api.h` + `src/sim_console/sim_console_api.cpp`
  in `coast_gui` (no LLVM). The console preamble (`repl.cpp`/`session.cpp`) gains
  `#include "sim_console/sim_console_api.h"`. No executor registration needed.
- Cheatsheet (`.? sim`) entry for `config`/`run`/`step`/`save_state`.

## Scope

**v1**: `config()` load + JSON-Pointer/sugar edit, `run()` (make dir + write
config.json), `sim_run` with `step` (subprocess + state chaining), `total_steps`,
`save_state`/`load_state`, fs utils.

**As built, the coverage lives in the B-tier suite** (`b_tier.simc_*`), not a pure
unit test: `config()` edit + JSON-Pointer set (incl. the `const char*`/`int`
overload guards), `run()` writing `config.json` with absolute output paths, a real
short `coast -n` run asserting state chains + a distinct per-chunk state file per
step + `step(0)` no-ops, run-dir collision suffixing, and concurrent `step_async`.
A standalone unit test for the pure `config()`/`dump()` round-trip (no subprocess)
was considered but not added — the editing path is exercised end-to-end by the
B-tier facade test, which also proves the edited value actually reaches the child.

**Deferred**: live in-memory chunked handle (build_env/run_headless in-process) if
between-chunk *live* inspection or load-dataset-once ever becomes a felt need;
streaming progress parsing / cancel; parallel sweep runs.
