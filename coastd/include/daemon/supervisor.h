#ifndef COAST_COASTD_SUPERVISOR_H
#define COAST_COASTD_SUPERVISOR_H

// -----------------------------------------------------------------------------
// coastd::supervisor — owns the one-shot `coast` children.
//
// spawn() launches a child (build the launch plan, redirect stdout+stderr to a
// run-dir logfile) and records it. A single poll thread services every live
// child ~10 Hz: it tails each logfile through line_tailer -> parse_progress to
// keep a live status, escalates a graceful SIGTERM to SIGKILL past a grace
// window, and reaps exits via waitpid — turning each exit into a child_exited
// event carrying the durable outcome read from the run dir (authoritative even
// when a SIGKILL left no manifest).
//
// waitpid is the authoritative liveness signal (R1: no reattach, no FS/PID
// liveness markers); the tailed phase=done merely corroborates. live_ids() is
// therefore the truth reality_probe uses for child_alive in steady state.
// -----------------------------------------------------------------------------

#include <sys/types.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "daemon/child_cmd.h"
#include "daemon/line_tailer.h"
#include "daemon/types.h"
#include "diag/progress.h"

namespace coastd {

class event_queue;

// Recycle-safe identity of a freshly-spawned child, persisted into the job file
// so the boot-time pid-guard can tell a survivor from a recycled pid.
struct spawn_info {
    pid_t       pid = 0;
    std::string start_token;
};

class supervisor {
public:
    supervisor(daemon_config cfg, event_queue& out);
    ~supervisor();

    supervisor(const supervisor&)            = delete;
    supervisor& operator=(const supervisor&) = delete;

    void start();      // launch the poll thread
    void shutdown();   // stop the thread; SIGKILL + reap any still-live children

    // ---- core-thread API (invoked from the dispatcher) ----
    // Spawn a child for `j` from the built binary. Returns its identity, or
    // nullopt if already supervising this job or the spawn failed.
    [[nodiscard]] std::optional<spawn_info> spawn(const job& j,
                                                  const std::filesystem::path& coast_bin);
    // Deliver a signal. SIGTERM starts a grace timer (escalates to SIGKILL);
    // SIGKILL is immediate. No-op if not supervising `id`.
    void request_kill(const job_id& id, int sig);

    // ---- reality_probe / list (core thread) ----
    [[nodiscard]] std::unordered_set<job_id> live_ids() const;
    [[nodiscard]] bool is_supervising(const job_id& id) const;
    [[nodiscard]] std::optional<coast::diag::progress> progress_of(const job_id& id) const;

    // Completed poll passes (read-only observation point; any thread). A pass
    // reaps exited children and escalates expired grace periods, so tests can
    // reason in passes ("no duplicate event in the N passes after the reap")
    // instead of wall-clock windows, and use it as a heartbeat.
    [[nodiscard]] std::uint64_t poll_passes() const noexcept {
        return poll_passes_.load(std::memory_order_acquire);
    }

private:
    struct child {
        pid_t                    pid = 0;
        std::string              start_token;
        int                      log_fd = -1;   // lazily opened in poll (race-free)
        std::filesystem::path    logfile;
        line_tailer              tail;
        job                      snapshot;      // run_dir + target for read_outcome
        coast::diag::progress    last;
        bool                     saw_progress = false;
        std::optional<std::chrono::steady_clock::time_point>
                                 term_deadline; // SIGTERM grace; escalate past it (monotonic: an
                                                // NTP step must not stretch/shrink the kill grace)
        int                      kills_sent = 0; // SIGKILLs already escalated (see poll_once)
    };

    using child_map  = std::unordered_map<job_id, child>;
    using child_iter = child_map::iterator;

    void poll_loop(std::stop_token st);
    void poll_once();
    void drain_log(child& c);   // read available bytes, feed the tailer

    // The indivisible post-reap step: emit child_exited, close the log fd, erase the
    // entry — all of it, or the registry is left describing a child that no longer
    // exists. noexcept by contract; see the comment at its definition for why a
    // deferred erase was a correctness bug and not a style choice. Returns the
    // iterator past the erased entry. Call with m_ held, only after waitpid has
    // established that the child is gone.
    [[nodiscard]] child_iter finish_child(child_iter it, int wstatus) noexcept;

    daemon_config                     cfg_;
    event_queue&                      out_;
    mutable std::mutex                m_;
    child_map                         children_;
    std::jthread                      thread_;
    std::chrono::milliseconds         poll_period_{100};
    std::atomic<std::uint64_t> poll_passes_{0};   // see poll_passes()
};

} // namespace coastd

#endif // COAST_COASTD_SUPERVISOR_H
