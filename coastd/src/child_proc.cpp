#include "daemon/child_proc.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/proc.h>     // SZOMB
#include <sys/sysctl.h>   // sysctl, kinfo_proc
#endif

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "diag/diag.h"

// macOS's <unistd.h> does not declare `environ` for a program; the explicit
// declaration is portable across Linux and macOS.
extern char** environ;  // NOLINT(readability-redundant-declaration): portability, not redundant on macOS

namespace coastd {

std::optional<pid_t> spawn_process(const spawn_opts& opts) {
    // argv: exe + args + nullptr. c_str() views into `store` stay valid until spawn returns.
    std::vector<std::string> store;
    store.reserve(opts.args.size() + 1);
    store.push_back(opts.exe);
    store.insert(store.end(), opts.args.begin(), opts.args.end());
    std::vector<char*> argv;
    argv.reserve(store.size() + 1);
    for (auto& s : store) argv.push_back(s.data());
    argv.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    // Every file action's return code is CHECKED. These fail for real — ENOMEM on a
    // loaded box is enough — and posix_spawn then proceeds with the actions that DID
    // register. A failed addopen means the child inherits coastd's stdout instead of
    // its logfile: the job's output vanishes into the daemon log, the supervisor
    // tails an empty file, sees no progress, and reports a healthy run as a
    // crash-loop. Refuse to spawn rather than launch a mis-wired child.
    int fa_rc = 0;
    // Enter the run dir first so any stray relative write the child makes lands
    // there, not in coastd's CWD. All designed paths are absolute, so this is a
    // belt-and-braces default rather than a correctness dependency.
    if (!opts.cwd.empty())
        fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addchdir_np(&fa, opts.cwd.c_str());
    // stdin <- /dev/null. A job child has no console and must never acquire one: with
    // coastd's stdin inherited AND the child in a background process group, anything
    // that tries to READ the terminal takes SIGTTIN and STOPS — not exits — so the
    // supervisor sees a process that is neither making progress nor dying.
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null",
                                                             O_RDONLY, 0);
    // stdout -> logfile (truncate); stderr -> same fd.
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO,
                                                             opts.logfile.c_str(),
                                                             O_CREAT | O_TRUNC | O_WRONLY, 0644);
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO);
    if (fa_rc != 0) {
        posix_spawn_file_actions_destroy(&fa);
        COAST_ERROR("coastd.child", "could not prepare the child's file actions for {}: {} "
                    "— refusing to spawn a mis-wired child", opts.exe, std::strerror(fa_rc));
        return std::nullopt;
    }

    // Put the job child in its OWN process group. Without this it inherits
    // coastd's, so a terminal signal sent to the foreground group -- Ctrl-C on an
    // interactively-run daemon -- is delivered to every running job at once. That
    // is not the graceful path: run_headless catches SIGTERM only, so SIGINT keeps
    // its default disposition and each job dies instantly with NO --save-on-term
    // checkpoint, while coastd treats the same SIGINT as an orderly stop and
    // proceeds to SIGTERM corpses believing the final-checkpoint contract was
    // honoured. Isolating the group makes cancel/shutdown the only routes to a job
    // child, which is what the supervisor's escalation assumes. (run_capture does
    // this already, for the related reason of reaching a whole build tree.)
    //
    // POSIX_SPAWN_SETSIGDEF for SIGPIPE: an IGNORED disposition survives execve, and
    // main.cpp sets SIGPIPE to SIG_IGN for the daemon's own sockets. Without this
    // reset every child inherits it, so a coast child writing down a broken pipe
    // gets EPIPE where it expects to die — a failure mode that does not reproduce
    // when the same binary is run by hand. Reset the whole set we touch rather than
    // just SIGPIPE, so a future ignored signal in the daemon cannot leak the same way.
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    sigset_t defaults;
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    const int at_rc = posix_spawnattr_setpgroup(&at, 0)
                    | posix_spawnattr_setsigdefault(&at, &defaults)
                    | posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF);
    if (at_rc != 0) {
        posix_spawnattr_destroy(&at);
        posix_spawn_file_actions_destroy(&fa);
        COAST_ERROR("coastd.child", "could not prepare spawn attributes for {} — refusing "
                    "to spawn a child in the wrong process group or signal state", opts.exe);
        return std::nullopt;
    }

    pid_t pid = 0;
    const int rc = posix_spawn(&pid, opts.exe.c_str(), &fa, &at, argv.data(), environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) {
        COAST_ERROR("coastd.child", "spawn {} failed: {}", opts.exe, std::strerror(rc));
        return std::nullopt;
    }
    return pid;
}

void send_signal(pid_t pid, int sig) {
    // Guard the sentinel/negative values, as is_same_process already does. Every
    // present call site passes a live child's pid, but `pid` reaches the store from
    // DISK via job_json's tolerant defaulting, and the failure mode of a missing
    // precondition here is kill(-1, SIGKILL) — every process the daemon's uid owns.
    // A pgid kill, when we want one, is spelled explicitly at its call site.
    if (pid <= 0) {
        COAST_ERROR("coastd.child", "refusing to signal non-positive pid {}", pid);
        return;
    }
    if (::kill(pid, sig) != 0 && errno != ESRCH)
        COAST_WARN("coastd.child", "kill({}, {}) failed: {}", pid, sig, std::strerror(errno));
}

void send_signal_group(pid_t pid, int sig) {
    // The same sentinel guard, and it matters more here: kill(-1, SIGKILL) is
    // "every process this uid owns", and kill(-0, …) is "coastd's OWN group" —
    // which, run interactively, includes the shell that started it. Both are
    // catastrophic and both are one bad `pid` away.
    if (pid <= 0) {
        COAST_ERROR("coastd.child", "refusing to group-signal non-positive pid {}", pid);
        return;
    }
    if (::kill(-pid, sig) != 0 && errno != ESRCH)
        COAST_WARN("coastd.child", "kill(-{}, {}) failed: {}", pid, sig, std::strerror(errno));
}

#if defined(__APPLE__)
namespace {
// The kernel's process record for `pid`, or nullopt if there is none. sysctl
// rather than proc_pidinfo(PROC_PIDTBSDINFO): proc_pidinfo answers EPERM for a
// process owned by another user and ESRCH for a zombie, so it would read either as
// "gone" — the costly direction for the duplicate-run guard (see is_same_process).
// sysctl reports both, like /proc on Linux. A pid with no process is NOT an error
// here: the call succeeds and returns zero bytes, so the length is the test.
std::optional<kinfo_proc> process_record(pid_t pid) {
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
    kinfo_proc kp{};
    std::size_t len = sizeof kp;
    if (::sysctl(mib, 4, &kp, &len, nullptr, 0) != 0 || len != sizeof kp) return std::nullopt;
    return kp;
}
} // namespace
#endif

std::optional<std::string> read_start_token(pid_t pid) {
#if defined(__APPLE__)
    // Start time as microseconds since the epoch, as a decimal string. The kernel
    // records it once, at fork, so it is fixed for the life of the process (a zombie
    // keeps it, as on Linux).
    const auto kp = process_record(pid);
    if (!kp) return std::nullopt;
    const timeval& t = kp->kp_proc.p_starttime;
    return std::to_string(static_cast<long long>(t.tv_sec) * 1'000'000 + t.tv_usec);
#elif !defined(__linux__)
    // No token source on this platform. The daemon refuses to START here (see
    // start_token_supported() in child_proc.h and the gate in main.cpp), so this
    // arm is reached only by an in-process test — never by a running daemon that
    // could then relaunch a job alongside a surviving child.
    (void)pid;
    return std::nullopt;
#else
    // /proc/<pid>/stat: "pid (comm) state ppid ... " where comm may contain
    // spaces and parens. Split AFTER the last ')' so comm cannot corrupt the
    // field count; starttime is field 22 (1-indexed). Fields after ')' start at
    // field 3 (state), so field 22 is index 19 in the post-')' token list.
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    if (!in) return std::nullopt;
    std::string line;
    std::getline(in, line);
    const auto rp = line.rfind(')');
    if (rp == std::string::npos) return std::nullopt;

    std::istringstream rest(line.substr(rp + 1));
    std::string tok;
    for (int i = 0; i <= 19; ++i)
        if (!(rest >> tok)) return std::nullopt;
    return tok;  // starttime, as a decimal string
#endif
}

#if defined(__linux__)
// The process state character from /proc/<pid>/stat, or 0 if unreadable. Parsed
// the same way as read_start_token: split AFTER the last ')' so a comm containing
// spaces or parens cannot shift the field count. State is the first field there.
char process_state(pid_t pid) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    if (!in) return 0;
    std::string line;
    std::getline(in, line);
    const auto rp = line.rfind(')');
    if (rp == std::string::npos) return 0;
    std::istringstream rest(line.substr(rp + 1));
    std::string st;
    if (!(rest >> st) || st.empty()) return 0;
    return st[0];
}
#endif

bool is_zombie(pid_t pid) {
#if defined(__linux__)
    return process_state(pid) == 'Z';
#elif defined(__APPLE__)
    const auto kp = process_record(pid);
    return kp && kp->kp_proc.p_stat == SZOMB;
#else
    (void)pid;
    return false;
#endif
}

bool is_same_process(pid_t pid, const std::string& token) {
    if (pid <= 0) return false;
    // ONLY ESRCH means "gone". The previous `!= 0` test folded EPERM in with it,
    // and EPERM means the exact opposite: the process EXISTS, we merely may not
    // signal it. Reachable whenever the orphan runs under different credentials —
    // a uid-mapped container, or coastd restarted as another user after the child
    // was spawned. Answering "dead" there makes recover_at_boot relaunch the job
    // alongside a child that is still running and still writing, i.e. two `coast`
    // processes on one run dir: precisely the corruption this guard exists to
    // prevent, produced by the guard itself.
    //
    // Any other errno (only EINVAL is specified, for a bad signal — impossible with
    // 0) is treated as "still there" for the same reason the whole function is
    // conservative: a false "alive" costs a job that waits for an operator, a false
    // "dead" costs a corrupted run dir.
    if (::kill(pid, 0) != 0 && errno == ESRCH) return false;
    // A ZOMBIE is dead. It still answers kill(pid, 0) and still has a
    // /proc/<pid>/stat carrying the original start-time, so every other test here
    // says "alive" — but it has exited, holds no descriptors, and can never write
    // to the run dir again. Since that is the ONLY question this guard exists to
    // answer (is another process still mutating this job's state?), calling a
    // zombie alive is wrong in the costly direction: the boot guard waits out its
    // whole confirmation window and then refuses to run a job whose child is
    // already gone. Reachable whenever the orphan's parent has not reaped it yet —
    // a non-reaping subreaper, or a test that parents its own stand-in.
    if (is_zombie(pid)) return false;
    auto tok = read_start_token(pid);
    return tok && *tok == token;                // same pid AND same start-time
}

} // namespace coastd
