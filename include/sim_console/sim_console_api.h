#ifndef COAST_SIM_CONSOLE_API_H
#define COAST_SIM_CONSOLE_API_H

// Drive simulations from the embedded C++ console (design/sim_console_api.md).
//
// The workflow: load a base config, edit it, run() to write it into a run
// directory, then step() to run the headless `coast` binary against it. Each
// step() spawns a CHILD `coast -n N ... config.json` process, chaining saved
// state across calls. The child owns its own HDF5/threads/memory — a native
// fault kills only the child, never the console/GUI (the one containment hole
// the in-process console cannot close).
//
// This header is the interpreter PARSE BOUNDARY, so it is kept lean:
// nlohmann/json is NOT exposed. run_config holds the parsed config by pointer
// (pimpl) and is edited through a string JSON-Pointer API. Implementations live
// in sim_console_api.cpp, compiled by the host and reached via the JIT symbol
// table (coast_lib whole-archived + -rdynamic, as the h5/plot façades are).
//
// Error model: failures (missing base config, malformed JSON pointer, run
// directory not creatable, nonzero child exit) throw std::runtime_error.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "run_config/run_config.h"

namespace coast::simc {

// The editable config type + its loader now live in coast_lib (coast::cfg) so
// the coastd console can share them (design/run_config_extraction.md). Re-export
// under coast::simc so `simc::config(...)` / `simc::run_config` keep working.
using coast::cfg::run_config;
using coast::cfg::config;

// run_config (the editable config) and config() (its loader) are declared in
// run_config/run_config.h and re-exported above — see design/run_config_extraction.md.

// ------------------------------------------------------------------- sim_run
//
// A persistent, subprocess-backed run. run() creates the run directory and
// writes config.json; nothing is stepped until step()/step_async(). Successive
// steps chain saved state, continuing the same trajectory and step count.
//
// Held only via shared_ptr (run() returns one) — step_async() keeps the handle
// alive until its child finishes via enable_shared_from_this.
class sim_run : public std::enable_shared_from_this<sim_run> {
public:
    // Passkey: only run() can mint a ctor_key, so a sim_run can be built only via
    // run() (which returns a shared_ptr). make_shared still works because it just
    // forwards the key. A direct `sim_run{...}` outside run() is now a compile
    // error rather than a runtime trap — a stack/unique_ptr instance calling
    // step_async() would hit shared_from_this() with no owning shared_ptr and
    // throw std::bad_weak_ptr. (Mirrors run_config's private-ctor discipline.)
    class ctor_key {
        ctor_key() = default;
        friend std::shared_ptr<sim_run> run(const run_config&,
                                            const std::filesystem::path&);
    };
    sim_run(ctor_key, std::string run_dir, std::string config_path, int seed);
    ~sim_run();

    sim_run(const sim_run&)            = delete;
    sim_run& operator=(const sim_run&) = delete;

    // BLOCKING: run n steps in a child `coast` process, its stdout+stderr
    // redirected to a per-step log, blocking the CONSOLE thread (never the GUI)
    // until it exits. Chains --load-state/--save-state across calls. Returns
    // cumulative steps. Throws std::runtime_error on nonzero child exit or if a
    // step (blocking or async) is already in flight on this handle. step(0) is a
    // no-op: the guard predates the CLI change that made `-n 0` an error (it
    // used to mean UNLIMITED), and is still needed — it now keeps a zero-sized
    // chunk from spawning a child that would exit 1 on the rejected argument.
    std::uint64_t step(std::uint64_t n);

    // NON-BLOCKING: spawn the child and return immediately. Progress is tracked
    // in the shared run registry; with show_status the status dialog opens (all
    // concurrent runs appear together in its table). On success the saved state
    // is chained in for the next step, exactly as step() would. Throws if a step
    // is already in flight on this handle. step_async(0) is a no-op.
    void step_async(std::uint64_t n, bool show_status = true);

    [[nodiscard]] std::uint64_t total_steps() const;
    [[nodiscard]] std::string   run_dir()     const;
    [[nodiscard]] std::string   observe_dir() const;
    [[nodiscard]] std::string   config_path() const;
    [[nodiscard]] std::string   state_path()  const;
    // Path of the log written by the most recent step()/step_async() (child
    // stdout+stderr). "" before the first step. `tail -f` it for live progress —
    // coast prints periodic step counts.
    [[nodiscard]] std::string   last_log()    const;
    // True while a step is executing on this handle — blocking step() or async
    // step_async() (its child has not yet exited).
    [[nodiscard]] bool          running()     const;

    // Cancel this handle's in-flight async run: a running child is signalled
    // (SIGTERM) and reaped as "cancelled"; a still-queued run is dropped before it
    // spawns. No-op if nothing is in flight. Afterwards running() is false and
    // total_steps()/state_path() are unchanged — a cancelled chunk chains nothing.
    // A blocking step() cannot be cancelled (the console thread is inside it).
    void cancel();

    void save_state(const std::filesystem::path& path); // copy latest state out
    void load_state(const std::filesystem::path& path); // next step() loads this state

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

// Materialize a run: create the run directory, write config.json into it, and
// return a handle. Does not step. Throws std::runtime_error if the run directory
// cannot be created.
//
// The run directory is chosen in this precedence: the explicit `dir` argument,
// else the config's run_dir() if set, else "runs/<base-config-stem>". If that
// path already exists, a running-number suffix (_001, _002, …) is claimed so two
// runs never share a directory. Whichever wins, outputs (observe/output_dir,
// snapshot_dir) are pointed into it as ABSOLUTE paths, which make config.json
// self-contained: it records exactly where outputs went, independent of the CWD
// the child is later run from. (A relative leaf would instead be placed under the
// app config's config_base — <config_dir>/<config>/… — not resolved against CWD;
// absolute is kept for the self-containment guarantee AND to bypass config_base
// entirely.) config.json is written there.
[[nodiscard]] std::shared_ptr<sim_run> run(const run_config& cfg,
                                           const std::filesystem::path& dir = {});

// -------------------------------------------------------- async run monitoring
//
// step_async() records each run in a process-wide registry. The GUI drives it
// once per frame via render_runs_window() (mirrors coast::plot::render_all):
// poll child processes, then draw the status table if visible. The other
// functions let non-GUI callers (scripts, tests) drive and query it.

// GUI thread, once per frame: advance async-run state and, if the status dialog
// is open, draw the table of all runs. No-op when nothing is running.
void render_runs_window();

// Re-open the "Simulation Runs" dialog after it was closed with its X button
// (which hides it). These only set a visibility flag — safe to call from anywhere,
// including headless (no ImGui needed); the dialog itself is drawn by
// render_runs_window(), which the GUI calls each frame and a headless run never
// does. Shows on the next GUI frame if there are any runs to list.
void show_runs_window();
void hide_runs_window();

// Advance async-run state without drawing: reap finished children, chain their
// state back into their sim_run, parse progress from logs. render_runs_window()
// calls this internally; call it directly when driving runs without a GUI.
void poll_runs();

// Block the caller until every async run has finished (polls internally). Useful
// to synchronise a sweep before analysing its outputs. Does not touch the GUI.
void wait_all_runs();

// Number of async runs not yet finished — running plus queued (see the pool
// below). wait_all_runs() returns exactly when this reaches zero.
[[nodiscard]] int active_run_count();

// Drop finished (done/failed/cancelled) rows from the status table. Running AND
// queued rows are kept.
void clear_finished_runs();

// Cancel every async run: running children are signalled (SIGTERM) and reaped as
// "cancelled", queued runs are dropped before they spawn. Each affected handle
// clears its in-flight guard. Finished rows are left as-is.
void cancel_all_runs();

// ------------------------------------------------------ async run process pool
//
// step_async() never blocks and never fails because "too many runs are going":
// at most max_concurrent_runs() children run at once; any excess is QUEUED and
// spawned automatically as running children exit (the queue drains inside
// poll_runs(), so a non-GUI sweep only needs to reach wait_all_runs()). The cap
// bounds concurrent ASYNC children across all handles — one global pool. A
// blocking step() is deliberately outside it: as a foreground call that holds the
// console thread it runs immediately (worst case cap+1), rather than stalling the
// console behind a saturated sweep.
//
// Each coast child is itself TBB-parallel across every core, so the cap is chiefly
// a guard against resource exhaustion (PIDs/fds/RSS/CPU oversubscription) on large
// sweeps rather than a throughput dial. Default 4; override at startup with the
// COAST_SIMC_MAX_CONCURRENT environment variable, or at runtime with the setter
// below. A queued run keeps its handle's running() true until it completes.

// Set the pool cap. Clamped to >= 1. Takes effect on the next poll_runs()/
// wait_all_runs()/GUI frame, which drains the queue up to the new cap.
void set_max_concurrent_runs(int n);

// Current pool cap.
[[nodiscard]] int max_concurrent_runs();

// Number of runs accepted by step_async() but not yet spawned (awaiting a slot).
[[nodiscard]] int queued_run_count();

// ------------------------------------------------------------- filesystem util
void                                   mkdir(const std::filesystem::path& p); // create_directories
[[nodiscard]] bool                     exists(const std::filesystem::path& p);
[[nodiscard]] std::vector<std::string> ls(const std::filesystem::path& p);
[[nodiscard]] std::string              cwd();

} // namespace coast::simc

#endif // COAST_SIM_CONSOLE_API_H
