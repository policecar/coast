#ifndef COAST_COASTD_CHILD_PROC_H
#define COAST_COASTD_CHILD_PROC_H

// -----------------------------------------------------------------------------
// coastd::child_proc — the OS process primitive under the supervisor.
//
// Spawns a one-shot `coast` child with stdout+stderr redirected to a run-dir
// logfile (that file is both the human log and the machine progress feed the
// supervisor tails). Also: signal delivery for cancel, and the recycle-safe
// process-identity token used by the pid duplicate-guard.
// -----------------------------------------------------------------------------

#include <sys/types.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace coastd {

struct spawn_opts {
    std::string                exe;      // absolute path to the coast binary
    std::vector<std::string>   args;     // argv after exe (exe is prepended)
    std::filesystem::path      logfile;  // stdout+stderr -> here (created/truncated)
    std::filesystem::path      cwd;      // child working directory (empty = inherit)
};

// Spawn the process. Returns its pid, or nullopt on failure (logged). The child
// starts detached from coastd's controlling terminal only insofar as its stdio
// is redirected; process-group/cgroup membership is inherited (R1 relies on the
// systemd cgroup reap — coastd does not double-fork away from it).
[[nodiscard]] std::optional<pid_t> spawn_process(const spawn_opts& opts);

// Best-effort signal delivery (SIGTERM graceful / SIGKILL hard). Silent if the
// process is already gone.
void send_signal(pid_t pid, int sig);

// Same, but delivered to the child's whole PROCESS GROUP (kill(-pid, sig)).
//
// INVARIANT: a `coast` job child does not fork. It is a single process, spawned as
// its own group leader (POSIX_SPAWN_SETPGROUP above), so the group is exactly that
// one process and this is equivalent to send_signal — today.
//
// It is spelled as a group kill anyway because the invariant is coast's to keep,
// not the daemon's to verify: if coast ever grows a helper subprocess, signalling
// only the leader would leave that helper running against the run dir after the
// job was cancelled — a silent regression with no failing test, since the leader
// still dies and the job still looks cancelled. The group form contains that by
// construction. Only ever call this with a pid this daemon spawned and has since
// confirmed (the pgid equals the child pid); a recycled pid would otherwise aim
// the signal at an unrelated group.
void send_signal_group(pid_t pid, int sig);

// Has `pid` exited but not yet been reaped by its parent? A zombie answers
// kill(pid, 0) and keeps its /proc entry, so the ordinary liveness tests call it
// alive — but it can no longer execute or write anything, which is the only sense
// of "alive" this daemon's duplicate-run guard cares about. Linux reads the state
// from /proc/<pid>/stat, macOS from sysctl(KERN_PROC_PID); always false elsewhere.
[[nodiscard]] bool is_zombie(pid_t pid);

// ---- the platform seam -------------------------------------------------------
// Everything platform-specific in coastd's duplicate-run protection is the two
// functions below plus is_zombie above; read_start_token and is_zombie have the
// platform bodies. To PORT coastd to another OS, implement both there and add the
// platform to start_token_supported() — nothing else changes, and the daemon's
// startup gate (main.cpp) opens automatically because it asks
// start_token_supported(). Implemented for Linux (/proc) and macOS (sysctl
// KERN_PROC_PID; docs/content/coastd/decisions.md ADR-17).

// True iff this platform can produce a start token at all. Compile-time; drives
// the daemon's refusal to start where the pid-guard would be inoperative.
[[nodiscard]] constexpr bool start_token_supported() noexcept {
#if defined(__linux__) || defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

// A recycle-safe identity for `pid`: Linux /proc/<pid>/stat field 22 (starttime,
// clock ticks since boot); macOS kp_proc.p_starttime (microseconds since the
// epoch). Either is an opaque decimal string, only ever compared for equality.
// Two processes with the same pid but different starttime are different
// processes — this is what lets the boot-time pid-guard tell a still-running
// child from a recycled pid. nullopt if unavailable: the
// process is gone, OR the platform has no token source at all (in which case
// start_token_supported() is false and the daemon refuses to run).
[[nodiscard]] std::optional<std::string> read_start_token(pid_t pid);

// True iff a process with this pid is currently live AND its start-token matches
// `token` (i.e. it is the SAME process, not a recycled pid). A dead pid or a
// mismatched token both return false. Used by the recovery pid-guard.
[[nodiscard]] bool is_same_process(pid_t pid, const std::string& token);

} // namespace coastd

#endif // COAST_COASTD_CHILD_PROC_H
