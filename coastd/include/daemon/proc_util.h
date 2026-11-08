#ifndef COAST_COASTD_PROC_UTIL_H
#define COAST_COASTD_PROC_UTIL_H

// -----------------------------------------------------------------------------
// coastd::run_capture — run a subprocess and capture its combined output.
//
// Used by the builder to drive `git` and `cmake` and surface a failure's tail
// back to the core as a worker_result message. Blocking (runs on a worker
// thread). Combined stdout+stderr so an error message is never split from its
// context.
// -----------------------------------------------------------------------------

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace coastd {

// Cap on the output run_capture keeps in memory. A build child can emit output
// for as long as build_timeout allows (an hour by default), and a template-error
// storm from a submitted SHA reaches hundreds of MB — enough to OOM the daemon or
// its cgroup, taking every RUNNING job with it. When the cap is hit the OLDEST
// bytes are dropped: consumers want the tail (the error), and builder::tail()
// already reports only the last 40 lines. line_tailer caps itself for the same
// reason, at 1 MiB.
inline constexpr std::size_t max_captured_output = 4u * 1024u * 1024u;

struct exec_result {
    bool        spawned   = false;   // false if the process could not start
    bool        timed_out = false;   // true if killed for exceeding the timeout
    int         exit_code = -1;      // process exit code; -1 if signaled / failed
    std::string output;              // combined stdout+stderr, TAIL-capped (see max_captured_output)
    bool        truncated = false;   // true if early output was dropped to stay under the cap

    [[nodiscard]] bool ok() const { return spawned && !timed_out && exit_code == 0; }
};

// Run `exe` with `args` (exe is prepended to argv) in `cwd` (empty = inherit),
// capturing combined stdout+stderr. Never throws.
//
// `timeout` is an absolute wall-clock ceiling on the whole invocation (measured
// from spawn). On expiry the child is SIGKILL'd, reaped, and the result comes
// back with timed_out=true / ok()==false — this is what keeps a hung git/cmake
// (an unreachable FetchContent remote, a stale lock) from parking the worker
// thread forever. Output already read before the kill is preserved. A zero
// timeout means unbounded (kept only for callers that genuinely cannot hang).
[[nodiscard]] exec_result run_capture(const std::string& exe,
                                      const std::vector<std::string>& args,
                                      const std::filesystem::path& cwd = {},
                                      std::chrono::seconds timeout = std::chrono::seconds::zero());

// Run `git` with `args` through run_capture, with the daemon's git hardening
// prepended. USE THIS FOR EVERY git INVOCATION — never run_capture("git", …).
//
// Why a seam instead of the flags at each call site: a submitted job's build and
// run steps execute arbitrary repo code as the DAEMON's uid. That is the accepted
// model (submitting a SHA means running it), but without hardening it is not
// confined to the submitting job. The child can write repo_root/.git/hooks/* and
// repo_root/.git/config, and the daemon then runs `git -C repo_root worktree add`
// — which fires post-checkout — for every future job on a new SHA, re-executing
// that code across OTHER users' jobs until .git is cleaned. A one-shot per submit
// becomes a persistent cross-job implant. The hardening below is only load-bearing
// if it is on EVERY call, so it lives at the one seam rather than in ten argv
// literals a future call site can forget.
//
// What the injected options close:
//   core.hooksPath=/dev/null  — no hook of any name resolves (/dev/null is not a
//                               directory), so post-checkout et al. never run, and
//                               a hooksPath poisoned in local config is overridden:
//                               `-c` outranks every config file.
//   core.fsmonitor=           — disables a poisoned fsmonitor, the other local
//                               config key whose value git executes as a command.
// `-c` also propagates to child git processes via GIT_CONFIG_PARAMETERS, so
// `submodule update` inherits both.
//
// What it does NOT close, recorded rather than implied — and the honest scope is
// wider than the two keys an earlier version of this comment named. Repo-LOCAL git
// config is an execution surface that no combination of flags closes. Demonstrated
// working implants that survive this hardening, both needing no remote and no
// network: a `filter.<x>.smudge` command in .git/config paired with
// `* filter=<x>` in .git/info/attributes (fires on worktree add, for ANY future
// SHA, since info/attributes is repo-scoped); and `submodule.<n>.update = !cmd`,
// which git ignores from .gitmodules but honours from .git/config (fires on
// submodule update). core.sshCommand and url.*.insteadOf are two further examples,
// not the list. Note core.attributesFile=/dev/null does NOT help — that governs the
// USER-level attributes file, not .git/info/attributes (measured).
//
// Closing this needs the build to run under a uid that cannot write repo_root/.git;
// that is a deployment change, not a flag. See INSTALL.md §9.1, which states the
// same scope so an operator is not misled into thinking the flags are sufficient.
[[nodiscard]] exec_result run_git(const std::vector<std::string>& args,
                                  const std::filesystem::path& cwd = {},
                                  std::chrono::seconds timeout = std::chrono::seconds::zero());

// Set FD_CLOEXEC on an already-open fd so it does NOT survive the exec of a
// spawned child (a coast job, git, cmake). Without this, every daemon socket,
// pipe, and log fd is inherited by every posix_spawn child process-wide — a
// leaked listen socket keeps the command port bound after the daemon exits, a
// leaked run_capture pipe never EOFs (its read loop hangs until the unrelated
// child dies), and a leaked client fd holds a TCP connection half-open. No-op on
// a negative fd. Prefer O_CLOEXEC at open() time where the fd comes from open(),
// and the atomic creators below where the fd comes from socket()/accept()/pipe():
// set-after-create is a real race, not a formality — the daemon spawns children
// from several threads at once (worker pool builds, the core thread's job spawn,
// the committer's git), so a concurrently-forked child can inherit the fd in the
// window between the two calls. Symptoms: a run_capture pipe that never EOFs
// (its read loop then blocks for the full build/git timeout, failing a job for a
// ref that resolved in milliseconds) and a client socket held half-open for a
// whole run. set_cloexec remains for fds that arrive from elsewhere.
void set_cloexec(int fd) noexcept;

// Atomic-CLOEXEC creators. On Linux these use pipe2/accept4/SOCK_CLOEXEC, which
// close the window entirely; elsewhere they fall back to create + set_cloexec so
// this source stays portable (the daemon refuses to START off Linux, but it must
// still compile there — see main.cpp's start-token gate). Return values match the
// underlying call: pipe2-style 0/-1, accept/socket-style fd or -1.
int cloexec_pipe(int fds[2]) noexcept;
int cloexec_accept(int listen_fd) noexcept;
int cloexec_socket(int domain, int type, int protocol) noexcept;

// The port a bound AF_INET socket actually holds (getsockname), or 0 on failure.
// What makes a requested port of 0 usable: the kernel picks a free port at bind
// time, and this reads the choice back.
std::uint16_t local_port(int fd) noexcept;

} // namespace coastd

#endif // COAST_COASTD_PROC_UTIL_H
