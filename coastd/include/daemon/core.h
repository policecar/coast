#ifndef COAST_COASTD_CORE_H
#define COAST_COASTD_CORE_H

// -----------------------------------------------------------------------------
// coastd::core — the single-writer state machine.
//
// One thread runs the loop:  pop event -> apply (fold into the store only) ->
// rescan reality (fresh) -> reconcile -> dispatch actions. Every edge (client
// connection, worker, supervisor) reaches the core ONLY by pushing an event, so
// the store has exactly one writer and the reliability gates are provable, not
// just tested. Recovery is the same path: boot() loads the store, then run()'s
// first iteration rescans and reconciles — no separate recovery code to rot.
//
// The dispatcher lives here (not a separate class) and owns the in-flight dedup
// sets that make reconcile's deliberate over-emission safe: a resolve/build/gc
// already running is not resubmitted, and builds are capped at
// cfg.build_concurrency (1) so a build gets half the box while sims keep running.
// -----------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <future>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "daemon/builder.h"
#include "daemon/event_queue.h"
#include "daemon/job_store.h"
#include "daemon/supervisor.h"
#include "daemon/types.h"
#include "daemon/worker_pool.h"

namespace coastd {

// A submission request (the client's ask, before a job exists).
struct job_spec {
    std::string           ref_symbolic;
    // LOCAL-CALLER ONLY (tests and any in-process driver): a config file to read
    // on THIS machine. A submission arriving over the wire never sets this — a
    // remote client's path is meaningless here (see protocol.h) — it sets
    // config_text instead. Kept because an in-process caller shares the daemon's
    // filesystem by definition, so for it a path is well-defined and convenient.
    std::filesystem::path config_path;
    // The EXACT config bytes. Over the wire these are what the client sent and what
    // the edge validated; carried through to prepare_run_config so the run config is
    // written from the same bytes that were path-scanned. Nothing is ever re-read
    // between validation and use, so the submit-config TOCTOU cannot exist.
    std::string           config_text;
    // Basename of the config on the CLIENT's machine, for the job-id mnemonic only.
    // Never opened and never resolved. Falls back to config_path's stem for a local
    // caller, then to the label.
    std::string           config_name;
    std::string           label;         // mnemonic hint (else config name/stem)
    std::string           user_message;
    std::uint64_t         steps_target = 0;  // steps to run (from load_state's step, if set)
    int                   priority = 0;
    // Initial state, already resolved by the caller (comms edge: initial_state_from):
    // the source job id, its final_state.h5 + step. Empty = start fresh. See job.
    std::string           load_state_from;
    std::string           load_state;
    std::uint64_t         load_state_step = 0;
};

// A job plus its bucket, for read-only queries. `live_step` is a DISPLAY-only
// overlay: the newest step parsed live from a running child's log tail (0 unless
// a running child has emitted progress). It is deliberately NOT `job::steps_done`
// — that field is the DURABLE, resumable checkpoint step the recovery model
// depends on, so a non-durable live count must never overwrite it. Clients show
// live_step to make a running job visibly advance between checkpoints; coastd's
// own logic keeps using steps_done. Filled by apply_live_step() on the core thread.
struct job_view {
    job           j;
    bucket        b         = bucket::active;
    std::uint64_t live_step = 0;
};

class core {
public:
    explicit core(daemon_config cfg);

    // Prepare the store (create dirs, git init, load existing jobs). Call before run().
    void boot();

    // Run the loop until stop(). Starts/stops the supervisor poll thread.
    void run();

    // Ask run() to return (thread-safe; wakes the loop).
    void stop();

    // ---- thread-safe RPC seam (any thread) ----
    std::future<job_id>               submit(job_spec spec);
    std::future<bool>                 cancel(job_id id);
    std::future<bool>                 reorder(job_id id, int pos);
    // Move relative to the job's CURRENT position, resolved on the core thread.
    // Prefer this from any client that computes from a polled snapshot: an absolute
    // position is stale by construction there (see do_reorder_relative).
    std::future<bool>                 reorder_relative(job_id id, int delta);
    std::future<std::vector<job_view>> list();
    std::future<std::optional<job_view>> query(job_id id);

    // ---- invoked by client_command closures on the core thread ----
    job_id do_submit(const job_spec& spec);
    bool   do_cancel(const job_id& id);
    bool   do_reorder(const job_id& id, int pos);
    bool   do_reorder_relative(const job_id& id, int delta);

private:
    void apply(event& e);
    void dispatch(const action& a);
    // Overlay the supervisor's live tailed step onto a view (core thread only —
    // reads sup_). No-op unless a running child has emitted progress. This is the
    // load-bearing wire between the progress parser (already tailing every child)
    // and what `list`/`query` report, so a running job advances in the UI before
    // its first checkpoint. See design/live_step_in_status.md.
    void apply_live_step(job_view& v) const;
    void on_worker_result(const worker_result& wr);
    void on_child_exited(const child_exited& ce);
    // Log the onset/clearing of disk pressure once per transition, not per loop.
    void log_disk_pressure_edge(const reality& r);
    void maybe_sweep_logs();
    std::string new_job_id(const std::string& mnemonic_hint) const;

    // Boot recovery (M3): kill any orphaned child that survived a previous daemon
    // (the pid duplicate-guard), clear each active job's stale pid/start_token, and
    // sweep torn checkpoint *.tmp from its run dir. Everything else recovery needs
    // is the steady-state rescan + reconcile that run() does anyway.
    void recover_at_boot();

    // Write the daemon-owned coast app_config (cfg_.app_config_path) that every
    // child is launched with via --app-config. Pins output placement under each
    // run dir and shadows any ambient $COAST_APP_CONFIG. No-op if the path is empty.
    void write_app_config();

    daemon_config cfg_;
    job_store     store_;
    event_queue   queue_;   // declared before pool_/sup_ (they hold a reference)
    worker_pool   pool_;
    builder       builder_;
    supervisor    sup_;

    std::unordered_set<job_id> resolve_in_flight_;
    std::unordered_set<sha_t>  build_in_flight_;
    std::unordered_set<sha_t>  gc_in_flight_;

    // Log housekeeping schedule (see core.cpp maybe_sweep_logs). Core-thread only,
    // except log_sweep_in_flight_ which the worker result clears — also on the core
    // thread, since results arrive through the event queue. No synchronization
    // needed for either.
    std::chrono::steady_clock::time_point last_log_sweep_{};
    bool                                  log_sweep_in_flight_ = false;

    // Per-SHA build-retry state (in-memory; core thread only, like the in-flight
    // sets). A build failure is often TRANSIENT (a network blip during the
    // submodule fetch or cmake FetchContent), so a single failure must not
    // terminally fail every job on the SHA. Count consecutive failures and space
    // retries by the same backoff as sim relaunches; only at max_build_failures do
    // the jobs on the SHA go to failed/. Reset on a successful build. Not persisted:
    // a daemon restart re-attempts fresh, and a genuinely broken build re-fails
    // fast and re-caps (the terminal failure_reason it stamps on jobs IS durable).
    std::unordered_map<sha_t, int>           build_failures_;
    std::unordered_map<sha_t, std::uint64_t> build_last_attempt_ms_;

    // Edge-detector for the disk-pressure log line. reality is recomputed from
    // scratch every loop and reconcile is pure, so neither can remember that it
    // already complained; without this the ~1 Hz loop would emit one line per
    // second for as long as the disk stays full — which is how a real signal
    // becomes noise the operator filters out.
    bool last_disk_pressure_ = false;

    // Shutdown is LATCHED, not a mirror of "the loop is currently spinning".
    // stop() can legitimately land before run() gets scheduled (main installs the
    // signal handlers before it creates the core thread), and a flag that run()
    // unconditionally set on entry would swallow that stop: the loop would then
    // spin on an already-closed queue — pop_until returns immediately once closed,
    // so the ~1 Hz backstop is gone — burning a core, re-running rescan/reconcile
    // (spawning children) forever, and never returning to be joined.
    std::atomic<bool> stop_requested_{false};

    // Test seam. The event-ordering folds in on_child_exited (terminal-bucket guard,
    // cancel guard) are unreachable through the public RPC surface without racing a
    // real child against the loop. This friend lets a unit test set up a precise
    // store state and apply a synthetic event SYNCHRONOUSLY on its own thread — valid
    // only while the core loop is NOT running, preserving the single-writer invariant.
    // Declared here; DEFINED only by the test TU (production code never names it).
    friend struct core_test_access;
};

} // namespace coastd

#endif // COAST_COASTD_CORE_H
