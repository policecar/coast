#include "daemon/proc_util.h"

#include <arpa/inet.h>     // ntohs
#include <fcntl.h>
#include <netinet/in.h>    // sockaddr_in
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "diag/diag.h"

extern char** environ;  // NOLINT(readability-redundant-declaration): portability, not redundant on macOS

namespace coastd {

void set_cloexec(int fd) noexcept {
    if (fd < 0) return;
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags >= 0) ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
}

int cloexec_pipe(int fds[2]) noexcept {
#if defined(__linux__)
    return ::pipe2(fds, O_CLOEXEC);
#else
    const int rc = ::pipe(fds);
    if (rc == 0) { set_cloexec(fds[0]); set_cloexec(fds[1]); }
    return rc;
#endif
}

int cloexec_accept(int listen_fd) noexcept {
#if defined(__linux__)
    return ::accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
#else
    const int fd = ::accept(listen_fd, nullptr, nullptr);
    set_cloexec(fd);
    return fd;
#endif
}

int cloexec_socket(int domain, int type, int protocol) noexcept {
#if defined(__linux__)
    return ::socket(domain, type | SOCK_CLOEXEC, protocol);
#else
    const int fd = ::socket(domain, type, protocol);
    set_cloexec(fd);
    return fd;
#endif
}

std::uint16_t local_port(int fd) noexcept {
    sockaddr_in a{};
    socklen_t   len = sizeof a;
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len) != 0 || a.sin_family != AF_INET)
        return 0;
    return ntohs(a.sin_port);
}

exec_result run_capture(const std::string& exe,
                        const std::vector<std::string>& args,
                        const std::filesystem::path& cwd,
                        std::chrono::seconds timeout) {
    exec_result r;

    int pipefd[2];
    if (cloexec_pipe(pipefd) != 0) {
        COAST_ERROR("coastd.proc", "pipe() failed: {}", std::strerror(errno));
        return r;
    }
    // Both ends are CLOEXEC from creation (cloexec_pipe), so no OTHER child
    // spawned concurrently — a coast job, a git commit — can inherit them: an
    // inherited write end would keep this pipe from ever reaching EOF, hanging the
    // read loop below until that unrelated child exits. The build child itself
    // dup2's the write end onto its stdout/stderr (dup2 clears CLOEXEC on the new
    // fd) and addclose's both raw ends, so it is unaffected.

    std::vector<std::string> store;
    store.reserve(args.size() + 1);
    store.push_back(exe);
    store.insert(store.end(), args.begin(), args.end());
    std::vector<char*> argv;
    argv.reserve(store.size() + 1);
    for (auto& s : store) argv.push_back(s.data());
    argv.push_back(nullptr);

    // Return codes CHECKED, not discarded: these fail under memory pressure, and
    // posix_spawn would then run with only the actions that registered — a build
    // whose output goes to the daemon's stdout instead of the capture pipe, read as
    // a silent build. Refuse instead.
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    int fa_rc = 0;
    if (!cwd.empty())
        fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addchdir_np(&fa, cwd.c_str());
    // stdin <- /dev/null. The child is in a BACKGROUND process group, so any read of
    // the terminal raises SIGTTIN and STOPS it: the capture pipe never reaches EOF
    // and this drain blocks for the full timeout (an hour, for a build) holding the
    // single build slot. `git` asking for credentials or an unknown host key is the
    // realistic trigger; see also GIT_TERMINAL_PROMPT below.
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null",
                                                             O_RDONLY, 0);
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_adddup2(&fa, pipefd[1], STDOUT_FILENO);
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_adddup2(&fa, pipefd[1], STDERR_FILENO);
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addclose(&fa, pipefd[0]);  // child does not read
    fa_rc = fa_rc ? fa_rc : posix_spawn_file_actions_addclose(&fa, pipefd[1]);  // dup'd already
    if (fa_rc != 0) {
        posix_spawn_file_actions_destroy(&fa);
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        COAST_ERROR("coastd.proc", "could not prepare file actions for {}: {}",
                    exe, std::strerror(fa_rc));
        return r;
    }

    // Put the child in its OWN process group (pgid == child pid) so a timeout can
    // signal the WHOLE build tree, not just the direct child. `cmake --build` forks
    // make + compilers/linkers as its children; killing only the cmake pid would
    // orphan those grandchildren to keep burning CPU/RAM after we gave up. With the
    // child a group leader, kill(-pid, ...) reaches the entire group.
    // SETSIGDEF for SIGPIPE: an ignored disposition survives execve, and the daemon
    // sets SIGPIPE to SIG_IGN for its own sockets. A build step containing a shell
    // pipeline (`... | head`) would then see EPIPE instead of dying on the signal —
    // a build failure that does not reproduce outside the daemon.
    posix_spawnattr_t at;
    posix_spawnattr_init(&at);
    sigset_t defaults;
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGPIPE);
    posix_spawnattr_setpgroup(&at, 0);
    posix_spawnattr_setsigdefault(&at, &defaults);
    posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF);

    // Never let git block on an interactive prompt. With stdin on /dev/null a
    // credential prompt fails instead of hanging, but git also probes askpass
    // helpers and ssh has its own prompt path; these two make the refusal explicit
    // and immediate rather than depending on the terminal state. Set on the child's
    // environment only — the daemon's own env is untouched.
    //
    // GIT_CONFIG_GLOBAL/SYSTEM pin the user- and system-level config files to
    // /dev/null (an empty config), for the same containment reason run_git injects
    // its `-c` options: a build/run child executes repo code as the daemon's uid and
    // can therefore write the daemon user's ~/.gitconfig. Every LATER daemon git
    // call would honour it — including core.sshCommand, whose value git executes —
    // turning one submitted job into persistent code execution across all others.
    // Unlike the `-c` options this belongs on the ENV, because it must also cover
    // the git processes cmake spawns for us (FetchContent) during a build.
    //
    // Two consequences worth stating rather than discovering: (1) a global
    // credential helper no longer applies, so `git fetch` against an https remote
    // needing auth fails — non-fatally, the builder warns and resolves against the
    // local clone (ssh remotes are unaffected: keys come from ~/.ssh, not git
    // config); (2) global url.*.insteadOf rewrites stop applying to FetchContent.
    // The audit repo is unaffected — job_store already provisions a LOCAL
    // user.name/user.email when the probe finds none, which is exactly this case.
    // GIT_CONFIG_PARAMETERS carries the same key overrides run_git passes as `-c`,
    // and it is here rather than only there because `-c` reaches only the git we
    // exec DIRECTLY. The build step execs `cmake` (below, via this same function),
    // and cmake/coast_build_info.cmake runs `git rev-parse` / `git status` itself at
    // configure time — measured: with the flags only on run_git's argv, a poisoned
    // core.fsmonitor in repo_root/.git/config executed on every cmake configure,
    // i.e. the key the seam closes was reopened on the very next build step.
    // FetchContent's git children are covered by the same reasoning.
    //
    // The format is git's own: space-separated 'key'='value' pairs. It is picky —
    // a malformed value makes git abort with "bogus format in GIT_CONFIG_PARAMETERS"
    // on EVERY call — so it is a fixed literal here, never built from input, and
    // unit.coastd.proc_util asserts a real git accepts it.
    // safe.directory is here for a different reason than the other two, and it is
    // REQUIRED, not optional. Git refuses to operate on a repository owned by another
    // uid ("detected dubious ownership") unless safe.directory permits it — and that
    // key is honoured ONLY in protected scope: system, global, or command line. By
    // pinning GIT_CONFIG_GLOBAL/SYSTEM to /dev/null we removed the only two places an
    // operator could have set it, and it cannot be rescued from repo-LOCAL config
    // because git deliberately ignores it there. Measured: without this entry, a
    // repo_root owned by a foreign uid makes `git rev-parse` exit 128, so ref
    // resolution fails, so core.cpp files EVERY submitted job to failed/ immediately
    // and permanently. That is precisely the "read-only clone maintained by a
    // separate privileged uid" deployment INSTALL §9.1 recommends as the STRONGER
    // security posture — this hardening would have broken it outright.
    //
    // `*` rather than the specific path because run_capture is generic (it also runs
    // cmake, whose FetchContent clones repos we never name). Permissive by intent:
    // the ownership check exists to stop you running code from a repo you do not
    // control, and this daemon builds and runs exactly such a repo BY DESIGN. It
    // protects nothing here, while its failure mode is a total outage.
    static constexpr const char* git_config_params =
        "GIT_CONFIG_PARAMETERS="
        "'core.hooksPath'='/dev/null' 'core.fsmonitor'='' 'safe.directory'='*'";

    std::vector<std::string> env_store;
    std::vector<char*>       envp;
    for (char** e = environ; e && *e; ++e) {
        const std::string_view v(*e);
        if (v.starts_with("GIT_TERMINAL_PROMPT=") || v.starts_with("GIT_ASKPASS=") ||
            v.starts_with("SSH_ASKPASS=") || v.starts_with("GIT_CONFIG_GLOBAL=") ||
            v.starts_with("GIT_CONFIG_SYSTEM=") || v.starts_with("GIT_CONFIG_PARAMETERS="))
            continue;                                  // replaced below
        env_store.emplace_back(*e);
    }
    env_store.emplace_back("GIT_TERMINAL_PROMPT=0");
    env_store.emplace_back("GIT_ASKPASS=");            // empty: no helper to consult
    env_store.emplace_back("SSH_ASKPASS=");
    env_store.emplace_back("GIT_CONFIG_GLOBAL=/dev/null");
    env_store.emplace_back("GIT_CONFIG_SYSTEM=/dev/null");
    env_store.emplace_back(git_config_params);
    envp.reserve(env_store.size() + 1);
    for (auto& s : env_store) envp.push_back(s.data());
    envp.push_back(nullptr);

    pid_t pid = 0;
    const int rc = posix_spawnp(&pid, exe.c_str(), &fa, &at, argv.data(), envp.data());
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    ::close(pipefd[1]);  // parent keeps only the read end
    if (rc != 0) {
        ::close(pipefd[0]);
        COAST_ERROR("coastd.proc", "spawn {} failed: {}", exe, std::strerror(rc));
        return r;
    }
    r.spawned = true;

    // Drain the child's output with an absolute wall-clock deadline. poll() bounds
    // each wait by the time remaining until the deadline, so we keep reading
    // streamed output yet can never block past `timeout` even if the child stalls
    // producing nothing (a hung FetchContent clone) OR floods output slowly. A
    // zero timeout means "no deadline" — poll waits indefinitely for the next
    // chunk, matching the old blocking behaviour.
    const bool bounded  = timeout > std::chrono::seconds::zero();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    bool killed   = false;   // deadline hit
    bool abnormal = false;   // poll/read error: child state unknown, drain aborted
    int  drain_errno = 0;    // errno AT the failure, before close() overwrites it
    char buf[8192];
    for (;;) {
        int wait_ms = -1;  // -1 = block indefinitely (unbounded case)
        if (bounded) {
            const auto rem = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
            if (rem <= 0) { killed = true; break; }
            wait_ms = rem > 3600'000 ? 3600'000 : static_cast<int>(rem);  // clamp to int-safe
        }
        pollfd pfd{pipefd[0], POLLIN, 0};
        const int pr = ::poll(&pfd, 1, wait_ms);
        if (pr < 0) {
            if (errno == EINTR) continue;
            drain_errno = errno;
            abnormal = true; break;                // unexpected poll error: stop reading
        }
        if (pr == 0) continue;                     // timed out this slice; deadline re-checked at loop top
        const ssize_t n = ::read(pipefd[0], buf, sizeof buf);
        if (n > 0) {
            r.output.append(buf, static_cast<std::size_t>(n));
            // Trim with slack so the O(n) erase is amortized over many reads rather
            // than run once per 8 KiB chunk past the cap.
            if (r.output.size() > max_captured_output + (1u << 20)) {
                r.output.erase(0, r.output.size() - max_captured_output);
                if (!r.truncated) {
                    r.truncated = true;
                    COAST_WARN("coastd.proc", "{} produced more than {} MiB of output; "
                               "keeping only the tail", exe, max_captured_output >> 20);
                }
            }
            continue;
        }
        if (n == 0) break;                         // EOF: child closed its write end
        if (errno == EINTR) continue;
        drain_errno = errno;
        abnormal = true; break;                    // read error
    }
    ::close(pipefd[0]);

    // On timeout OR an abnormal drain end the child's state is unknown, so kill its
    // whole PROCESS GROUP (grandchildren included, see the SETPGROUP note above).
    if (killed || abnormal) {
        if (killed) {
            r.timed_out = true;
            COAST_WARN("coastd.proc", "{} exceeded {}s timeout; killing process group",
                       exe, timeout.count());
        } else {
            // Capture errno BEFORE anything else can clobber it — ::close(pipefd[0])
            // above already ran, so reading errno here described the close, not the
            // drain failure that actually ended the loop.
            COAST_WARN("coastd.proc", "{} drain aborted ({}); killing process group",
                       exe, std::strerror(drain_errno));
        }
        ::kill(-pid, SIGKILL);
    }

    // BOUNDED reap — never the old blocking waitpid(pid, ..., 0). A child wedged in
    // uninterruptible (D-state) I/O (hung NFS, dying disk) cannot be reaped even
    // after SIGKILL, and a blocking wait would then park this worker thread forever
    // — defeating the very timeout above. Poll with WNOHANG up to a short grace,
    // then give up (the kernel reaps the zombie once the D-state clears).
    int status = 0;
    bool reaped = false;
    // Has the group already been signalled? On the killed/abnormal paths above, yes.
    // On the ORDINARY path — the pipe reached EOF — no signal has been sent, and
    // that is the case that used to leak a LIVE process: EOF means the child closed
    // (or dup2'd away) its output, NOT that it exited. A child that lingers past the
    // grace was previously logged as "abandoning the zombie" and left running, while
    // this function returned failure — so the builder could go on to `git worktree
    // remove --force` and remove_all the very tree the straggler was still writing
    // into. Nothing ever waits on it again either, so it becomes a permanent zombie
    // once it does exit. Escalate instead: the drain is over and the result is
    // already decided, so there is nothing left for the child to contribute.
    bool signalled = killed || abnormal;
    auto reap_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        const pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid) { reaped = true; break; }
        if (w < 0) { if (errno == EINTR) continue; break; }   // ECHILD/other: nothing to reap
        if (std::chrono::steady_clock::now() >= reap_deadline) {
            if (!signalled) {
                COAST_WARN("coastd.proc", "{} (pid {}) closed its output but had not exited "
                           "within 5s; killing the process group so it cannot keep writing "
                           "into a tree we are about to remove", exe, pid);
                ::kill(-pid, SIGKILL);
                signalled     = true;
                reap_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                continue;                       // one more grace, now that it is dying
            }
            COAST_WARN("coastd.proc", "{} (pid {}) not reaped within 5s of SIGKILL "
                       "(stuck in uninterruptible I/O?); abandoning the zombie", exe, pid);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    r.exit_code = (reaped && WIFEXITED(status)) ? WEXITSTATUS(status) : -1;
    return r;
}

exec_result run_git(const std::vector<std::string>& args,
                    const std::filesystem::path& cwd,
                    std::chrono::seconds timeout) {
    // Prepended, so they are main-command options: git requires `-c` BEFORE the
    // subcommand. Order relative to the caller's own `-C <repo>` does not matter;
    // both are parsed as main-command options.
    // Same three keys the env carries (run_capture), deliberately duplicated rather
    // than dropped: the env covers grandchildren, these cover this git even if the
    // environment is ever stripped, and keeping the two lists identical means the
    // hardening does not depend on which path a reader happens to look at.
    std::vector<std::string> full{"-c", "core.hooksPath=/dev/null",
                                 "-c", "core.fsmonitor=",
                                 "-c", "safe.directory=*"};
    full.insert(full.end(), args.begin(), args.end());
    return run_capture("git", full, cwd, timeout);
}

} // namespace coastd
