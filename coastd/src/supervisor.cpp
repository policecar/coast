#include "daemon/supervisor.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>   // std::strerror
#include <thread>
#include <utility>
#include <vector>

#include "daemon/child_proc.h"
#include "daemon/event_queue.h"
#include "diag/diag.h"

namespace coastd {

supervisor::supervisor(daemon_config cfg, event_queue& out)
    : cfg_(std::move(cfg)), out_(out) {}

supervisor::~supervisor() { shutdown(); }

void supervisor::start() {
    thread_ = std::jthread([this](std::stop_token st) { poll_loop(st); });
}

void supervisor::shutdown() {
    if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
    }
    std::lock_guard lk(m_);
    if (children_.empty()) return;

    // Graceful stop: SIGTERM every child so its --save-on-term handler writes a
    // final checkpoint, wait up to term_grace for them to exit, then SIGKILL any
    // straggler. The grace runs CONCURRENTLY for all children (one shared
    // deadline) — a large-model final checkpoint takes seconds, so raise
    // --term-grace (and systemd's TimeoutStopSec above it) for big jobs. Off
    // systemd this is also the only graceful-stop path; under systemd the cgroup
    // SIGTERM and this one are redundant (a child handles whichever arrives first).
    for (auto& [id, c] : children_) send_signal_group(c.pid, SIGTERM);

    const auto deadline = std::chrono::steady_clock::now() + cfg_.term_grace;
    for (auto& [id, c] : children_) {
        int st = 0;
        for (;;) {
            const pid_t w = ::waitpid(c.pid, &st, WNOHANG);
            if (w == c.pid) break;                       // exited on its own (checkpoint written)
            if (w < 0) { if (errno == EINTR) continue; break; }
            if (std::chrono::steady_clock::now() >= deadline) {
                COAST_WARN("coastd.super", "job {} (pid {}) did not exit within {}s of "
                           "shutdown SIGTERM; sending SIGKILL (final checkpoint may be "
                           "incomplete — resume falls back to the last periodic one)",
                           id, c.pid, cfg_.term_grace.count());
                send_signal_group(c.pid, SIGKILL);
                // BOUNDED reap, not a blocking waitpid: a child stuck in
                // uninterruptible (D-state) I/O cannot be reaped even after SIGKILL,
                // and a blocking wait here would wedge daemon shutdown forever. Poll
                // briefly, then move on (the kernel reaps it once the D-state clears).
                const auto kill_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
                for (;;) {
                    const pid_t w = ::waitpid(c.pid, &st, WNOHANG);
                    if (w == c.pid) break;
                    if (w < 0) { if (errno == EINTR) continue; break; }
                    if (std::chrono::steady_clock::now() >= kill_deadline) {
                        COAST_WARN("coastd.super", "job {} (pid {}) not reaped within 2s of "
                                   "shutdown SIGKILL; abandoning", id, c.pid);
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (c.log_fd >= 0) {
            ::close(c.log_fd);
            c.log_fd = -1;   // keep "log_fd is either -1 or ours" true everywhere
        }
    }
    children_.clear();
}

std::optional<spawn_info> supervisor::spawn(const job& j, const std::filesystem::path& coast_bin) {
    std::lock_guard lk(m_);
    if (children_.contains(j.id)) return std::nullopt;  // already supervising

    const launch_plan plan = plan_launch(j, cfg_, coast_bin);

    // Delete any manifest from a PREVIOUS launch before starting this one, so
    // read_outcome's "manifest present => a clean exit" holds for THIS launch. A
    // SIGKILL/segfault writes no manifest; without this delete, read_outcome would
    // find the stale one and mis-report a crashed launch as a clean exit (skewing
    // steps_done and the crash-loop guard).
    std::error_code mec;
    std::filesystem::remove(plan.manifest, mec);
    // Same for final_state.h5: present in a done job must mean "written by the run
    // that finished", since initial_state_from pairs it with that run's steps_done.
    std::filesystem::remove(std::filesystem::path(j.run_dir) / "final_state.h5", mec);

    spawn_opts o;
    o.exe     = plan.exe;
    o.args    = plan.args;
    o.logfile = plan.logfile;
    o.cwd     = plan.cwd;

    auto pid = spawn_process(o);
    if (!pid) return std::nullopt;

    child c;
    c.pid         = *pid;
    c.start_token = read_start_token(*pid).value_or("");
    if (c.start_token.empty()) {
        // No token means the pid-recycle guard (is_same_process) is OFF for this job:
        // core.cpp skips the identity check when the token is empty, so after a
        // restart a recycled pid can be mistaken for this child. Reachable without
        // anything being wrong with the child itself — an EMFILE at exactly this
        // moment is enough to fail the /proc read. Silently degrading a corruption
        // guard is not acceptable; say so, so an operator can correlate it later.
        COAST_WARN("coastd.super", "job {} pid {}: could not read a process start token — "
                   "the pid-reuse guard is disabled for this job", j.id, *pid);
    }
    c.logfile     = plan.logfile;
    c.snapshot    = j;
    children_.emplace(j.id, std::move(c));

    // The argv as launched, not j.steps_target: a resumed job's -n is the REMAINING
    // steps, so printing the absolute target misreports the command line.
    std::string cmdline = plan.exe;
    for (const auto& a : plan.args) { cmdline += ' '; cmdline += a; }
    COAST_INFO("coastd.super", "spawned job {} pid {} ({})", j.id, *pid, cmdline);
    return spawn_info{*pid, children_.at(j.id).start_token};
}

void supervisor::request_kill(const job_id& id, int sig) {
    std::lock_guard lk(m_);
    auto it = children_.find(id);
    if (it == children_.end()) return;
    child& c = it->second;
    if (sig == SIGKILL) {
        send_signal_group(c.pid, SIGKILL);
        return;
    }
    // SIGTERM (graceful): re-send each call (a repeat is a harmless no-op to the
    // child) and arm the grace timer ONCE — the deadline must not reset on repeated
    // calls, or the escalation could be deferred indefinitely. The poll loop
    // escalates to SIGKILL past the deadline.
    send_signal_group(c.pid, SIGTERM);
    if (!c.term_deadline)
        c.term_deadline = std::chrono::steady_clock::now() + cfg_.term_grace;
}

std::unordered_set<job_id> supervisor::live_ids() const {
    std::lock_guard lk(m_);
    std::unordered_set<job_id> ids;
    ids.reserve(children_.size());
    for (const auto& [id, c] : children_) ids.insert(id);
    return ids;
}

bool supervisor::is_supervising(const job_id& id) const {
    std::lock_guard lk(m_);
    return children_.contains(id);
}

std::optional<coast::diag::progress> supervisor::progress_of(const job_id& id) const {
    std::lock_guard lk(m_);
    auto it = children_.find(id);
    if (it == children_.end() || !it->second.saw_progress) return std::nullopt;
    return it->second.last;
}

void supervisor::drain_log(child& c) {
    if (c.log_fd < 0) {
        // Open lazily: by the first poll the child has created the file, so this
        // is race-free (no ENOENT window as at spawn time).
        c.log_fd = ::open(c.logfile.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (c.log_fd < 0) return;
    }
    char buf[65536];
    ssize_t n;
    while ((n = ::read(c.log_fd, buf, sizeof buf)) > 0) {
        c.tail.feed(std::string_view(buf, static_cast<std::size_t>(n)),
                    [&](std::string_view line) {
                        if (auto p = coast::diag::parse_progress(line)) {
                            c.last = *p;
                            c.saw_progress = true;
                        }
                    });
    }
}

void supervisor::poll_once() {
    std::lock_guard lk(m_);
    const auto now = std::chrono::steady_clock::now();  // monotonic: matches term_deadline

    // Everything after a successful waitpid is IRREVERSIBLE — the child is reaped,
    // its pid is free for the kernel to recycle, and this registry is the only
    // remaining record that it ever existed. So the reap, the child_exited event and
    // the erase must form ONE indivisible step, and nothing inside it may throw or
    // be deferred.
    //
    // This used to collect ids into a vector and erase them after the loop, with the
    // throwing read_outcome in between. poll_loop's guard (below) catches that throw
    // and continues — but the pending erases are then lost, leaving an
    // ALREADY-REAPED child in children_. The next pass hits the ECHILD branch and
    // emits a SECOND child_exited for one launch: on_child_exited is not idempotent,
    // so the job takes a duplicate crash-loop strike and can be filed to failed/ with
    // a fabricated reason. Worse, the stale entry's log_fd was already closed, and
    // drain_log re-opens only when it is < 0 — so the next pass reads from, and then
    // closes, a descriptor the kernel has since handed to a client socket or an
    // in-flight atomic_write. That is a wrong-fd close in a four-thread daemon.
    //
    // read_outcome is now non-throwing at the source too (child_cmd.cpp), which is
    // the real fix; the noexcept step below is the invariant that keeps this loop
    // correct for whatever throws NEXT.
    for (auto it = children_.begin(); it != children_.end(); ) {
        const job_id& id = it->first;
        child&        c  = it->second;

        drain_log(c);

        // Escalate a graceful kill that overstayed its welcome. Log it: a SIGKILL
        // here means the child did NOT finish its final --save-on-term checkpoint
        // within cfg_.term_grace, so the job will resume from its last PERIODIC
        // checkpoint instead — the operator's signal to raise --term-grace.
        // Bounded, unlike the old open-ended re-arm. A SIGKILL that does not take
        // means the child is in uninterruptible (D-state) I/O, and no amount of
        // repeating changes that — the kernel reaps it when the D-state clears.
        // Re-arming forever emitted this WARN every term_grace seconds for as long
        // as the daemon lived, which buries the one line that matters. Retry a few
        // times (a signal genuinely can be missed around exec), then say so once and
        // stop; the child is still reaped below whenever it becomes reapable.
        constexpr int max_kill_escalations = 3;
        if (c.term_deadline && now > *c.term_deadline) {
            if (c.kills_sent < max_kill_escalations) {
                COAST_WARN("coastd.super", "job {} (pid {}) did not exit within {}s of "
                           "SIGTERM; sending SIGKILL (final save-on-term checkpoint may "
                           "be incomplete — resume falls back to the last periodic one)",
                           id, c.pid, cfg_.term_grace.count());
                send_signal_group(c.pid, SIGKILL);
                ++c.kills_sent;
                c.term_deadline = now + cfg_.term_grace;  // re-arm so we don't spin SIGKILL
            } else {
                COAST_ERROR("coastd.super", "job {} (pid {}) has ignored {} SIGKILLs — it is "
                            "wedged in uninterruptible I/O (hung mount, dying disk). No further "
                            "signals will be sent; it is reaped when the kernel can reap it.",
                            id, c.pid, c.kills_sent);
                c.term_deadline.reset();   // stop escalating; reaping below still applies
            }
        }

        int st = 0;
        const pid_t w = ::waitpid(c.pid, &st, WNOHANG);
        if (w < 0 && errno != EINTR) {
            // waitpid FAILED — which this loop used to ignore, testing only for
            // `w == c.pid` and otherwise retrying forever. The dangerous errno is
            // ECHILD: "there is no such child to wait for". It is reachable without
            // anything crashing — if SIGCHLD's disposition is SIG_IGN and we inherit
            // that across exec (a shell or container init that sets it), the kernel
            // auto-reaps children and there is never a status to collect. The entry
            // then NEVER leaves children_, no child_exited is ever emitted, the
            // concurrency slot is never released, and the queue stalls for the
            // daemon's lifetime WITHOUT A SINGLE LOG LINE. A silent permanent stall
            // is the worst available outcome; synthesising an exit is strictly
            // better, because the job then flows through the normal crash-loop and
            // resume machinery, which is designed for "child gone, outcome unknown".
            //
            // Read the outcome from the run dir exactly as the normal path does: the
            // child may well have completed and checkpointed before vanishing, so
            // its real progress is on disk regardless of who reaped it.
            COAST_ERROR("coastd.super", "job {} (pid {}): waitpid failed: {} — treating the "
                        "child as gone (reaped by someone else?) so the slot is released "
                        "instead of stalling the queue", id, c.pid, std::strerror(errno));
            // Status 0 is not a claim the child SUCCEEDED: reconcile decides that
            // from steps_done/checkpoint, and a child that did nothing still trips
            // the crash-loop guard via `tailed == 0`.
            it = finish_child(it, 0);
            continue;
        }
        if (w == c.pid) {
            it = finish_child(it, st);
            continue;
        }
        ++it;   // still running
    }
}

// The indivisible post-reap step. Both operations that can realistically fail here
// — reading the outcome off the filesystem, and enqueueing the event — are wrapped,
// so neither can skip the erase; that skip is what left a reaped child in the
// registry (see poll_once). What is NOT covered is an allocation failure inside the
// logging calls themselves: that escapes the noexcept and terminates. Deliberate.
// Under genuine OOM the daemon is going down regardless, and dying at a std::format
// is at least loud and immediate, where a silently corrupted registry is neither.
// Returns the iterator past the erased entry so the caller's loop stays valid.
auto supervisor::finish_child(child_iter it, int wstatus) noexcept -> child_iter {
    const job_id& id = it->first;
    child&        c  = it->second;

    child_outcome oc;
    std::uint64_t tailed = 0;
    try {
        drain_log(c);                     // final bytes before we close
        oc = read_outcome(c.snapshot);
        // Last LIVE step from the log tail (0 if none seen). Lets the core's
        // crash-loop guard distinguish "ran real steps but was killed before a
        // checkpoint" from "died having done nothing", independent of checkpoint
        // interval.
        tailed = c.saw_progress ? c.last.step : 0;
    } catch (const std::exception& e) {
        // The child is already reaped; there is no way back. Report the exit with a
        // DEGRADED outcome (steps_done 0, no checkpoint) rather than not at all: the
        // core treats that as a no-progress launch, which costs one crash-loop strike
        // but keeps the job moving. Dropping the event instead would hold the
        // concurrency slot for the daemon's lifetime.
        COAST_ERROR("coastd.super", "job {}: could not read the outcome of the reaped child "
                    "(pid {}): {} — reporting the exit with no progress; a resume will "
                    "re-derive the checkpoint from disk", id, c.pid, e.what());
    } catch (...) {
        COAST_ERROR("coastd.super", "job {}: outcome read threw a non-exception; reporting "
                    "the exit with no progress", id);
    }

    // Close BEFORE the erase and blank the field: a stale fd number left in a
    // surviving entry is a wrong-fd close on the next pass (drain_log re-opens only
    // when log_fd < 0). Belt and braces — the entry is erased two lines down — but
    // the invariant "log_fd is either -1 or ours" now holds unconditionally.
    if (c.log_fd >= 0) {
        ::close(c.log_fd);
        c.log_fd = -1;
    }

    // push() takes a mutex and moves into a deque; a bad_alloc here is the one
    // remaining way to lose the event. Losing it stalls one job (loudly); letting it
    // escape a noexcept function terminates the daemon and orphans every child.
    try {
        out_.push(child_exited{id, wstatus, oc.steps_done, oc.checkpoint, oc.fatal, tailed});
        COAST_INFO("coastd.super", "job {} exited (wstatus={}) steps={} ckpt={}",
                   id, wstatus, oc.steps_done, oc.checkpoint.empty() ? "-" : oc.checkpoint);
    } catch (...) {
        COAST_ERROR("coastd.super", "job {}: could not enqueue its exit event — the job is "
                    "stalled until the daemon restarts (which re-derives it from disk)", id);
    }
    return children_.erase(it);
}

void supervisor::poll_loop(std::stop_token st) {
    // poll_once() can still throw. Its two known filesystem throw sources are gone —
    // read_outcome -> newest_checkpoint now uses the error_code overloads throughout
    // (child_cmd.cpp), and the reap step is noexcept — but drain_log's tailer still
    // allocates, and this guard is the backstop for whatever is added next. Escaping
    // a jthread's entry function calls std::terminate: the daemon would die outright
    // and orphan every running child, converting one unreadable directory into total
    // loss of supervision.
    //
    // Note the division of labour: this guard keeps the THREAD alive, and it is the
    // only thing it can do — it cannot repair state, which is why the reap step owns
    // its own invariant rather than relying on being caught here.
    //
    // core::run and worker_pool::worker_loop are both guarded this way already, with
    // comments making the same argument; this loop was the one that was not. Log and
    // continue: the next pass re-reads the same state, so a transient condition
    // self-heals and a persistent one is a recurring error for the operator rather
    // than a crash-loop.
    const auto guarded_poll = [this]() noexcept {
        try {
            poll_once();
        } catch (const std::exception& e) {
            COAST_ERROR("coastd.super", "supervisor poll failed (continuing): {}", e.what());
        } catch (...) {
            COAST_ERROR("coastd.super", "supervisor poll threw a non-exception (continuing)");
        }
    };

    while (!st.stop_requested()) {
        guarded_poll();
        poll_passes_.fetch_add(1, std::memory_order_acq_rel);
        std::this_thread::sleep_for(poll_period_);
    }
    guarded_poll();  // a last pass to reap anything that just finished
}

} // namespace coastd
