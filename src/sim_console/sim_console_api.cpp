// Implementation of the sim-control console façade (design/sim_console_api.md).
//
// The façade is deliberately thin: run_config edits a JSON config; run() writes
// it into a run directory; sim_run::step() spawns a headless child `coast`
// process to do the actual work. All simulation, threading, and persistence
// lives in the already-tested `coast` binary — we only assemble its command line.

#include "sim_console/sim_console_api.h"

#include "imgui.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#if defined(__APPLE__)
#include <crt_externs.h>
#include <mach-o/dyld.h>
#define environ (*_NSGetEnviron())
#endif
// environ is declared by <unistd.h> on glibc; on Apple it is provided via the
// _NSGetEnviron() shim above (a shared library may not reference environ directly).

namespace coast::simc {

namespace {

std::string self_exe(); // defined below

// Path to the `coast` binary to spawn. Normally the currently-running
// executable — the console lives inside `coast`, so re-spawning ourselves runs
// the same build headless. Overridable via COAST_SIMC_BINARY to target a
// specific build (e.g. a Release binary from a Debug GUI) and to make the façade
// testable from a non-coast host process. The value may be an absolute or
// relative path, or a bare command name — spawn_child uses posix_spawnp, so a
// slash-less name is resolved via PATH.
std::string coast_binary()
{
    if (const char* env = std::getenv("COAST_SIMC_BINARY"); env && *env) {
        return env;
    }
    return self_exe();
}

// Absolute path to the currently-running executable.
std::string self_exe()
{
#if defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);          // query length
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) {
        throw std::runtime_error("self_exe: _NSGetExecutablePath failed");
    }
    return std::filesystem::canonical(buf.c_str()).string();
#else
    std::error_code ec;
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) {
        throw std::runtime_error("self_exe: cannot read /proc/self/exe: " +
                                 ec.message());
    }
    return p.string();
#endif
}

// Spawn args[0] with args as argv (PATH-searched if it has no slash — see the
// posix_spawnp note below), redirecting the child's stdout AND stderr to
// log_path (truncated), inheriting the current environment and CWD, and return
// its pid WITHOUT waiting. Throws on spawn failure. The child inherits only fds
// 0/1/2 (see the closefrom note below); if log_path is empty, 1/2 still carry the
// console's fd-capture redirect so its output flows into the capture pipe.
::pid_t spawn_child(const std::vector<std::string>& args, const std::string& log_path)
{
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    // Every posix_spawn_* setup call returns 0 or an errno. Check each so a setup
    // failure (bad flag, ENOMEM) surfaces with its own cause here instead of as a
    // mystery posix_spawnp error — or worse, a child spawned with a half-built
    // file-action list. `check` throws on nonzero; RAII guards below destroy the
    // file-actions / attr objects on any throw or return.
    // system_category().message() (not std::strerror) because spawn_child runs on
    // both the console thread (step/step_async) and the GUI thread (poll_runs'
    // queue drain); strerror's shared static buffer would race on concurrent
    // failures.
    auto check = [](int rc, const char* what) {
        if (rc != 0) {
            throw std::runtime_error(std::string("spawn_child: ") + what +
                                     " failed: " + std::system_category().message(rc));
        }
    };

    posix_spawn_file_actions_t fa;
    check(posix_spawn_file_actions_init(&fa), "posix_spawn_file_actions_init");
    struct fa_guard {
        posix_spawn_file_actions_t* p;
        ~fa_guard() { posix_spawn_file_actions_destroy(p); }
    } fa_g{&fa};

    if (!log_path.empty()) {
        // stdout -> the log file; stderr -> same fd, so both streams interleave.
        check(posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, log_path.c_str(),
                                               O_WRONLY | O_CREAT | O_TRUNC, 0644),
              "addopen(stdout -> log)");
        check(posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO),
              "adddup2(stdout -> stderr)");
    }

    // Close every inherited fd >= 3 in the child before exec. A fork/exec copies
    // the whole parent fd table, and only close-on-exec fds drop automatically —
    // so without this the child would hold the console's open HDF5 file handles,
    // the fd-capture pipe's saved dups, and any other descriptor it never uses.
    // This is the comprehensive fix for that inheritance (superseding piecemeal
    // FD_CLOEXEC). The two supported platforms express it differently but achieve
    // the same outcome: the child inherits only fds 0/1/2.
    //
    // SCOPE: only Linux/glibc >= 2.34 and macOS are handled. On an older glibc or a
    // non-glibc libc (e.g. musl) NEITHER branch compiles in and fds >= 3 stay
    // inherited (still not a hang — the capture pipe is separately close-on-exec in
    // session.cpp). We deliberately do not support those toolchains rather than add
    // a fork+manual-close fallback; a reasonably recent glibc or macOS is required.
    //
    // Linux (glibc 2.34+): a closefrom file action runs close_range(3, ~0U, 0) in
    // the child, keeping 0/1/2 (1/2 are the log redirect above). Ordered LAST so
    // the >=3 sweep can't clobber the log setup.
    [[maybe_unused]] posix_spawnattr_t attr; // used only on macOS (see below)
    posix_spawnattr_t*                 attrp = nullptr;
#if defined(__linux__)
    // Nested (not `&&`-chained with `__GLIBC_PREREQ(2, 34)`): on a non-glibc libc
    // (or macOS misidentifying as __linux__, which doesn't happen, but in general)
    // `__GLIBC_PREREQ` is not a function-like macro at all, and both GCC and Clang
    // treat `IDENT(args)` for an undefined, non-function-like IDENT as a hard
    // preprocessor error in a `#if`/`#elif` line -- regardless of `defined(X) &&`
    // short-circuiting before it. The nested `#if` below only evaluates
    // `__GLIBC_PREREQ(2, 34)` where the outer `#if defined(__linux__)` already
    // guarantees `__GLIBC_PREREQ` is defined (from <features.h> on glibc). See
    // the (verified) macOS build break this masked pre-nesting.
#  if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 34)
    check(posix_spawn_file_actions_addclosefrom_np(&fa, 3),
          "addclosefrom_np");
#  endif
#elif defined(__APPLE__)
    // macOS has no close_range/closefrom spawn action. Instead the Apple spawnattr
    // flag POSIX_SPAWN_CLOEXEC_DEFAULT makes the child treat EVERY inherited fd as
    // close-on-exec by default, so only fds named by a file action survive — the
    // macOS-equivalent comprehensive close (available since macOS 10.7).
    //
    // Consequence: 0/1/2 are NOT special under this flag; they too close unless a
    // file action names them. Preserve them with dup2(fd,fd), the Apple idiom that
    // marks an fd inheritable under CLOEXEC_DEFAULT. When log_path is set the
    // addopen/adddup2 above already name 1/2 (redirected to the log), so only
    // stdin needs preserving; when it is empty, keep 1/2 as-is so the child's
    // output still flows into the console's fd-capture pipe.
    check(posix_spawnattr_init(&attr), "posix_spawnattr_init");
    struct attr_guard {
        posix_spawnattr_t* p;
        ~attr_guard() { posix_spawnattr_destroy(p); }
    } attr_g{&attr};
    check(posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT),
          "posix_spawnattr_setflags(CLOEXEC_DEFAULT)");
    attrp = &attr;
    check(posix_spawn_file_actions_adddup2(&fa, STDIN_FILENO, STDIN_FILENO),
          "adddup2(keep stdin)");
    if (log_path.empty()) {
        check(posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDOUT_FILENO),
              "adddup2(keep stdout)");
        check(posix_spawn_file_actions_adddup2(&fa, STDERR_FILENO, STDERR_FILENO),
              "adddup2(keep stderr)");
    }
#endif

    // posix_spawnp (not posix_spawn): it is fork + execvp, so a slash-less
    // COAST_SIMC_BINARY (a bare command name like "coast") is PATH-searched —
    // matching the natural "which command to run" expectation. posix_spawn is
    // execv, which does NOT search PATH and would resolve "coast" as ./coast in
    // CWD (surprising ENOENT, or silently the wrong binary). The absolute
    // self_exe() default is a path so it resolves identically under either. Both
    // Linux and macOS provide posix_spawnp.
    ::pid_t   pid = -1;
    const int rc  = ::posix_spawnp(&pid, argv[0], &fa, attrp, argv.data(), environ);
    // fa (and, on macOS, attr) are destroyed by their RAII guards on return.
    if (rc != 0) {
        throw std::runtime_error(std::string("posix_spawnp('") + args[0] +
                                 "') failed: " + std::system_category().message(rc));
    }
    return pid;
}

// Convert a waitpid status to an exit code (128 + signal if signalled).
int exit_code_of(int status)
{
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

// spawn_child + block until exit; returns the exit code.
int spawn_and_wait(const std::vector<std::string>& args, const std::string& log_path)
{
    const ::pid_t pid = spawn_child(args, log_path);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::runtime_error(std::string("waitpid failed: ") +
                                     std::system_category().message(errno));
        }
    }
    return exit_code_of(status);
}

// Local timestamp, "YYYYmmdd_HHMMSS", for per-step log filenames.
std::string timestamp()
{
    const std::time_t t = std::time(nullptr);
    std::tm            tm{};
    ::localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", &tm);
    return buf;
}

// Async run registry + process pool — defined below (after sim_run);
// forward-declared here so sim_run::step_async can submit into the pool.
struct submit_result { bool spawned; ::pid_t pid; }; // spawned now vs. queued
submit_result submit_run(std::string label, std::string run_dir,
                         std::string log_path, std::vector<std::string> pending_args,
                         std::function<void(bool)> on_done);
void set_runs_visible(bool v);
// Cancel the in-flight run whose log matches `log` (a handle's last_log). Used by
// sim_run::cancel(); defined with the registry below.
void cancel_run_by_log(const std::string& log);

} // namespace

// run_config and config() now live in coast_lib (coast::cfg), re-exported into
// coast::simc by sim_console_api.h — see design/run_config_extraction.md. run()
// below uses only their public surface, so it needs no access to the pimpl.

// ----------------------------------------------------------------------- run()
std::shared_ptr<sim_run> run(const run_config& cfg, const std::filesystem::path& dir_arg)
{
    // Precedence: explicit arg > config's run_dir() > runs/<base-config-stem>.
    std::string dir = !dir_arg.empty()      ? dir_arg.string()
                      : !cfg.run_dir_hint().empty() ? cfg.run_dir_hint()
                                                    : std::string{};
    if (dir.empty()) {
        std::string stem = std::filesystem::path(cfg.base_path()).stem().string();
        dir = "runs/" + (stem.empty() ? "run" : stem);
    }

    // Claim a unique run dir. Two runs resolving to the same path (e.g.
    // simc::run(c) called twice with no explicit dir) must not share
    // config.json / state files / logs, or their children would stomp each other's
    // state. create_directory() is atomic — it returns true only for the caller
    // that actually creates the dir — so the first resolution gets the exact
    // path and every later collision gets the next free _NNN suffix.
    {
        // Create the parent chain first, surfacing its real cause: without this
        // check a permission denial or a parent-is-a-regular-file only shows up
        // later as a generic "cannot create run dir … No such file or directory"
        // from create_directory() below, hiding what actually went wrong. An
        // already-existing parent is not an error (create_directories leaves pec
        // clear), so this only fires on a genuine failure.
        if (const auto parent = std::filesystem::path(dir).parent_path();
            !parent.empty()) {
            std::error_code pec;
            std::filesystem::create_directories(parent, pec);
            if (pec) {
                throw std::runtime_error("run: cannot create parent directory '" +
                                         parent.string() + "' for run dir '" + dir +
                                         "': " + pec.message());
            }
        }
        const std::string base_dir = dir;
        bool              claimed  = false;
        for (int n = 0; n < 1000; ++n) {
            std::string cand = base_dir;
            if (n > 0) {
                char suf[8];
                std::snprintf(suf, sizeof suf, "_%03d", n);
                cand += suf;
            }
            std::error_code cec;
            if (std::filesystem::create_directory(cand, cec)) {
                dir     = cand;
                claimed = true;
                break;
            }
            if (cec) { // a real error (permission, missing parent), not "exists"
                throw std::runtime_error("run: cannot create run dir '" + cand +
                                         "': " + cec.message());
            }
        }
        if (!claimed) {
            throw std::runtime_error("run: cannot claim a unique run dir near '" +
                                     base_dir + "' (1000 suffixes exhausted)");
        }
    }

    // Ensure outputs point into the run dir even when run_dir() was not called.
    // Write ABSOLUTE paths for self-containment: a relative snapshot_dir /
    // observe.output_dir leaf would be placed under the app config's config_base
    // (<config_dir>/<config>/…), not resolved against CWD — so absolute paths both
    // bypass config_base and record exactly where outputs went independent of the
    // caller's CWD, keeping the two consumers unambiguous wherever the child runs.
    std::error_code   aec;
    const std::string abs_dir = std::filesystem::absolute(dir, aec).string();
    const std::string out_dir = aec ? dir : abs_dir; // fall back to dir on error
    run_config staged = cfg;                 // copy; edit through the public API
    staged.set("/observe/output_dir", out_dir + "/observe");
    staged.set("/snapshot_dir",       out_dir + "/snapshots");

    const std::string cfg_path = dir + "/config.json";
    std::ofstream out(cfg_path);
    if (!out) {
        throw std::runtime_error("run: cannot write config to " + cfg_path);
    }
    out << staged.dump() << '\n';
    out.close();
    // Check the write too, not just the open: a disk-full / IO error mid-dump()
    // or on close would otherwise leave a truncated config.json that only fails
    // later, opaquely, when the child cannot parse it.
    if (!out) {
        throw std::runtime_error("run: failed writing config to " + cfg_path +
                                 " (disk full or I/O error?)");
    }

    return std::make_shared<sim_run>(sim_run::ctor_key{}, dir, cfg_path,
                                     cfg.seed_value());
}

// --------------------------------------------------------------------- sim_run

namespace {

// Everything step() and step_async() need to launch one chunk, built from the
// handle's current state (holds impl.mtx).
struct step_plan {
    std::string              log;        // fresh timestamped log path
    std::string              state_out;  // <run_dir>/state_<NNN>.h5 (--save-state)
    std::vector<std::string> args;       // full child argv
};

} // namespace

struct sim_run::impl {
    mutable std::mutex mtx;
    std::string        run_dir;
    std::string        config_path;
    std::string        latest_state;      // "" until a step saves one
    std::string        last_log;          // "" until the first step
    int                seed  = 0;
    int                chunk = 0;          // per-handle step counter (log + state names)
    std::uint64_t      total = 0;          // cumulative steps run via this handle
    // Set while a step is executing on this handle — blocking step() OR async
    // step_async(). Guards against a second concurrent step on the same handle
    // (which would fork the trajectory and double-count `total`). Named for
    // "a step is in flight", not "async", since step() now sets it too.
    // The race it closes is UNLIKELY — the console runs cells on a single thread
    // and step() blocks, so ordinary use can't overlap two steps; it is only
    // reachable by deliberately driving one handle from several threads (e.g.
    // wrapping step() in std::async). The guard makes that correct regardless.
    bool               step_in_flight = false;

    // Build the child argv + a fresh timestamped log for `n` steps, creating
    // <run_dir>/logs/. Caller holds mtx (reads latest_state, bumps chunk).
    step_plan plan(std::uint64_t n)
    {
        const std::string logs_dir = run_dir + "/logs";
        std::error_code   ec;
        std::filesystem::create_directories(logs_dir, ec);
        if (ec) {
            throw std::runtime_error("step: cannot create log dir '" + logs_dir +
                                     "': " + ec.message());
        }
        char idx[16];
        std::snprintf(idx, sizeof idx, "%03d", ++chunk);

        step_plan sp;
        sp.log = logs_dir + "/step_" + timestamp() + "_" + idx + ".log";
        // Per-chunk state file (state_001.h5, state_002.h5, …), sharing the
        // log's index so a state and the log that produced it correlate. Each
        // completed file is immutable — the next chunk --load-states the newest
        // and --save-states a fresh one — which (a) keeps a full rolling history
        // for later introspection, (b) makes a torn write survivable (earlier
        // files stay intact → resumable), and (c) removes the save_state() race
        // (it now copies a stable prior file, never one a child is mid-writing).
        sp.state_out = run_dir + "/state_" + idx + ".h5";
        sp.args = {coast_binary(), "-n", std::to_string(n),
                   "--seed-offset", std::to_string(seed)};
        if (!latest_state.empty()) {
            sp.args.push_back("--load-state");
            sp.args.push_back(latest_state);
        }
        sp.args.push_back("--save-state");
        sp.args.push_back(sp.state_out);
        // Positional config, last. Guard a run dir that starts with '-': coast's
        // CLI parser reads a positional beginning with '-' as an unknown option,
        // not the config path (so a run named e.g. "-baseline" would fail every
        // step with a confusing "Unknown option"). A relative "-foo/config.json"
        // gets a "./" prefix — resolves to the same file, but no longer looks
        // like a flag; absolute paths already start with '/'. Only the positional
        // needs this: --load-state/--save-state VALUES are taken literally.
        sp.args.push_back(config_path.starts_with('-') ? "./" + config_path
                                                       : config_path);
        return sp;
    }
};

sim_run::sim_run(ctor_key, std::string run_dir, std::string config_path, int seed)
    : p_(std::make_unique<impl>())
{
    p_->run_dir     = std::move(run_dir);
    p_->config_path = std::move(config_path);
    p_->seed        = seed;
}

sim_run::~sim_run() = default;

std::uint64_t sim_run::step(std::uint64_t n)
{
    if (n == 0) { std::scoped_lock lk(p_->mtx); return p_->total; }

    // DECISION: a blocking step() spawns immediately and does NOT consult the
    // async process pool (max_concurrent_runs). The pool governs BACKGROUND fan-out
    // — step_async runs the user launches and walks away from. step() is a
    // FOREGROUND call that holds the console thread until it returns, so at most one
    // exists at a time; queuing it behind a saturated async pool would stall the
    // interactive console (potentially for the length of a long sweep) for no real
    // benefit. The worst case is therefore cap+1 concurrent children, not cap+N.
    step_plan sp;
    {
        std::scoped_lock lk(p_->mtx);
        if (p_->step_in_flight) {
            throw std::runtime_error("step: a step is already running on "
                                     "this handle");
        }
        sp = p_->plan(n);
        p_->step_in_flight = true; // held for the whole blocking run
    }

    int rc = 0;
    try {
        rc = spawn_and_wait(sp.args, sp.log);
    } catch (...) { // spawn failure — clear the guard before propagating
        std::scoped_lock lk(p_->mtx);
        p_->step_in_flight = false;
        throw;
    }

    std::scoped_lock lk(p_->mtx);
    p_->step_in_flight = false;
    p_->last_log = sp.log; // available even on failure
    if (rc != 0) {
        throw std::runtime_error("sim run failed (child exit " +
                                 std::to_string(rc) + "); see " + sp.log);
    }
    p_->latest_state = sp.state_out;
    p_->total += n;
    std::printf("step: ran %llu steps (total %llu) -> %s\n",
                static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(p_->total), sp.log.c_str());
    return p_->total;
}

void sim_run::step_async(std::uint64_t n, bool show_status)
{
    if (n == 0) return;

    step_plan   sp;
    std::string run_dir_copy;
    {
        std::scoped_lock lk(p_->mtx);
        if (p_->step_in_flight) {
            throw std::runtime_error("step_async: a step is already running on "
                                     "this handle");
        }
        sp = p_->plan(n);
        p_->step_in_flight = true;  // reserve before spawning
        p_->last_log        = sp.log;
        run_dir_copy        = p_->run_dir;
    }

    // On completion, chain the saved state (or clear the in-flight flag on
    // failure) under this handle's lock. Captures a shared_ptr to keep the
    // handle alive until the child finishes. step_in_flight stays true from here
    // through completion — including any time the run spends QUEUED — so the
    // handle correctly reports running() and rejects a second step meanwhile.
    auto              self      = shared_from_this();
    const std::string state_out = sp.state_out;
    std::function<void(bool)> on_done = [self, state_out, n](bool ok) {
        std::scoped_lock lk(self->p_->mtx);
        self->p_->step_in_flight = false;
        if (ok) {
            self->p_->latest_state = state_out;
            self->p_->total += n;
        }
    };

    // Hand the run to the pool: it spawns now if a slot is free, else queues it
    // (poll_runs spawns it when a running child exits). An immediate-spawn failure
    // throws here — clear the guard before propagating, as the old direct path did.
    submit_result res{};
    try {
        res = submit_run(std::filesystem::path(run_dir_copy).filename().string(),
                         run_dir_copy, sp.log, sp.args, std::move(on_done));
    } catch (...) {
        std::scoped_lock lk(p_->mtx);
        p_->step_in_flight = false;
        throw;
    }
    if (show_status) set_runs_visible(true);

    if (res.spawned) {
        std::printf("step_async: launched %llu steps -> %s (pid %d)\n",
                    static_cast<unsigned long long>(n), sp.log.c_str(),
                    static_cast<int>(res.pid));
    } else {
        std::printf("step_async: queued %llu steps -> %s (pool full; will start "
                    "when a slot frees)\n",
                    static_cast<unsigned long long>(n), sp.log.c_str());
    }
}

std::uint64_t sim_run::total_steps() const { std::scoped_lock lk(p_->mtx); return p_->total; }
std::string   sim_run::run_dir()     const { std::scoped_lock lk(p_->mtx); return p_->run_dir; }
std::string   sim_run::observe_dir() const { std::scoped_lock lk(p_->mtx); return p_->run_dir + "/observe"; }
std::string   sim_run::config_path() const { std::scoped_lock lk(p_->mtx); return p_->config_path; }
std::string   sim_run::state_path()  const { std::scoped_lock lk(p_->mtx); return p_->latest_state; }
std::string   sim_run::last_log()    const { std::scoped_lock lk(p_->mtx); return p_->last_log; }
bool          sim_run::running()     const { std::scoped_lock lk(p_->mtx); return p_->step_in_flight; }

void sim_run::cancel()
{
    // Identify this handle's in-flight run by its log (last_log is the current
    // chunk's). If nothing is in flight, there is nothing to cancel. The actual
    // cancel + its off-lock on_done(false) live in the registry, which owns the
    // pid and the row state; on_done clears step_in_flight back here.
    std::string log;
    {
        std::scoped_lock lk(p_->mtx);
        if (!p_->step_in_flight) return;
        log = p_->last_log;
    }
    cancel_run_by_log(log);
}

void sim_run::save_state(const std::filesystem::path& path)
{
    std::string src;
    { std::scoped_lock lk(p_->mtx); src = p_->latest_state; }
    if (src.empty()) {
        throw std::runtime_error("save_state: no state yet — call step() first");
    }
    std::error_code ec;
    std::filesystem::copy_file(
        src, path, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error("save_state: copy to '" + path.string() +
                                 "' failed: " + ec.message());
    }
}

void sim_run::load_state(const std::filesystem::path& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        throw std::runtime_error("load_state: file not found: " + path.string());
    }
    std::scoped_lock lk(p_->mtx);
    p_->latest_state = path.string();
}

// ============================================================ async run registry
//
// step_async() records a row here; poll_runs() (called from the GUI frame or a
// script) reaps finished children and fires their completion callback;
// render_runs_window() draws the table. Lock discipline: the registry mutex is
// NEVER held while a completion callback runs — the callback locks the sim_run,
// and step_async locks the sim_run *then* the registry, so nesting only ever
// goes sim_run -> registry, never the reverse.

namespace {

// queued:    accepted by step_async but not yet spawned — waiting for a pool slot.
// cancelled: stopped by the user (running child signalled, or queued run dropped).
enum class run_state : std::uint8_t { queued, running, done, failed, cancelled };

struct run_row {
    std::string   label;         // shown in the table (run-dir stem)
    std::string   run_dir;
    std::string   log_path;
    ::pid_t       pid         = -1;
    std::uint64_t start_step   = 0; // "from step A ..." — progress base
    std::uint64_t target_step  = 0; // "... to B"        — progress end
    std::uint64_t cur_step     = 0; // last "Steps: N" / "Total steps: N"
    run_state     state        = run_state::running;
    int           exit_code    = 0;
    std::time_t   t_start       = 0;
    std::time_t   t_end         = 0;
    std::time_t   t_parsed      = 0; // throttle log parsing to ~1/s
    std::function<void(bool)> on_done;
    // For a queued row: the child argv to spawn once a slot frees (empty once the
    // row is running). Captured at submit time; safe to spawn later because the
    // owning handle's step_in_flight guard keeps its latest_state stable meanwhile.
    std::vector<std::string> pending_args;
    // Set when the user cancels a RUNNING row: the child is signalled here and the
    // reap in poll_runs finalizes the row as `cancelled` (not `failed`) on exit.
    bool cancel_requested = false;
};

// Default concurrent-run cap, overridable via COAST_SIMC_MAX_CONCURRENT (>= 1) or
// set_max_concurrent_runs(). Each coast child is itself TBB-parallel across all
// cores, so this is primarily a resource-exhaustion guard for large sweeps, not a
// throughput dial. 4 balances overlap against CPU/RSS/PID oversubscription.
constexpr int k_default_max_concurrent = 4;

int env_max_concurrent()
{
    if (const char* e = std::getenv("COAST_SIMC_MAX_CONCURRENT")) {
        char*           end = nullptr;
        const long      v   = std::strtol(e, &end, 10);
        if (end != e && v >= 1 && v <= 100000) return static_cast<int>(v);
    }
    return k_default_max_concurrent;
}

struct registry {
    std::mutex           mtx;
    std::vector<run_row> rows;
    std::atomic<bool>    visible{false};
    int                  max_concurrent = env_max_concurrent(); // guarded by mtx
};

registry& reg()
{
    static registry r;
    return r;
}

// Count rows in a given state. Caller holds reg().mtx.
int count_state(const registry& R, run_state s)
{
    int c = 0;
    for (const auto& row : R.rows) {
        if (row.state == s) ++c;
    }
    return c;
}

// Parse the child log for progress: start/target from "from step A to B", and
// current from the last "Steps: N" or "Total steps: N". Best-effort.
//
// NOTE: these substrings are string-coupled to run_headless's human-readable
// prints with nothing tying the two ends together — a reword there silently
// breaks the progress bar. Left as-is deliberately: a proper structured logging
// facility is planned in the near future, and progress parsing should be built
// on top of it so the emitter and the parser share one schema and cannot drift.
// Until then this stays best-effort and non-fatal (a parse miss just leaves the
// bar where it was).
void parse_log_progress(run_row& r)
{
    std::ifstream in(r.log_path);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        if (const auto p = line.find("from step "); p != std::string::npos) {
            unsigned long long a = 0, b = 0;
            if (std::sscanf(line.c_str() + p, "from step %llu to %llu", &a, &b) == 2) {
                r.start_step  = a;
                r.target_step = b;
            }
        } else if (const auto q = line.find("Steps: "); q != std::string::npos) {
            unsigned long long s = 0;
            if (std::sscanf(line.c_str() + q, "Steps: %llu", &s) == 1) r.cur_step = s;
        } else if (const auto t = line.find("Total steps: "); t != std::string::npos) {
            unsigned long long s = 0;
            if (std::sscanf(line.c_str() + t, "Total steps: %llu", &s) == 1) r.cur_step = s;
        }
    }
}

// Accept an async run into the pool. If fewer than max_concurrent children are
// running, spawn immediately (in the caller's thread — the common, single-run
// case, identical to the old direct path); otherwise queue it and let poll_runs()
// spawn it when a running child exits. Throws ONLY on an immediate-spawn failure,
// so step_async can clear its handle guard; a queued run that later fails to
// spawn is finalized as failed by poll_runs' drain via on_done(false).
submit_result submit_run(std::string label, std::string run_dir,
                         std::string log_path, std::vector<std::string> pending_args,
                         std::function<void(bool)> on_done)
{
    auto&             R   = reg();
    const std::time_t now = std::time(nullptr);
    std::scoped_lock  lk(R.mtx);
    run_row row;
    row.label    = label.empty() ? run_dir : std::move(label);
    row.run_dir  = std::move(run_dir);
    row.log_path = std::move(log_path);
    row.on_done  = std::move(on_done);

    submit_result res{};
    if (count_state(R, run_state::running) < R.max_concurrent) {
        // Slot free — spawn now. A throw unwinds before push_back, so nothing is
        // registered and the exception reaches step_async unchanged.
        row.pid     = spawn_child(pending_args, row.log_path);
        row.state   = run_state::running;
        row.t_start = now;
        res = {true, row.pid};
    } else {
        row.state        = run_state::queued;
        row.pending_args = std::move(pending_args);
        res = {false, -1};
    }
    R.rows.push_back(std::move(row));
    // DECISION (for future reviewers): finished (done/failed) rows are NOT
    // auto-pruned — they persist until clear_finished_runs() or the table's
    // "Clear finished" button. This is deliberate: the table doubles as a
    // session history of what was run, which is the point for a parameter sweep.
    // Growth is bounded by how many runs the user launches and each row is small
    // (on_done is moved out on completion, so no sim_run is retained). Revisit
    // only if a genuinely long-lived session shows the vector as a memory issue.
    return res;
}

void set_runs_visible(bool v) { reg().visible.store(v); }

using completion_list = std::vector<std::pair<std::function<void(bool)>, bool>>;

// Cancel one row in place (caller holds reg().mtx). RUNNING: flag it and SIGTERM
// the child — poll_runs reaps it and finalizes it as `cancelled` (not `failed`).
// QUEUED: finalize now (no child exists yet), deferring its on_done(false) into
// `defer` so the callback fires AFTER the registry lock is released (it locks the
// sim_run). Any other state (done/failed/cancelled) is a no-op.
void cancel_row_locked(run_row& row, std::time_t now, completion_list& defer)
{
    if (row.state == run_state::running) {
        row.cancel_requested = true;
        if (row.pid > 0) ::kill(row.pid, SIGTERM);
    } else if (row.state == run_state::queued) {
        row.state     = run_state::cancelled;
        row.t_end     = now;
        row.exit_code = -1;
        row.pending_args.clear();
        if (row.on_done) defer.emplace_back(std::move(row.on_done), false);
    }
}

void cancel_run_by_log(const std::string& log)
{
    completion_list defer;
    {
        auto&             R   = reg();
        const std::time_t now = std::time(nullptr);
        std::scoped_lock  lk(R.mtx);
        for (auto& row : R.rows) {
            if (row.log_path == log && (row.state == run_state::running ||
                                        row.state == run_state::queued)) {
                cancel_row_locked(row, now, defer);
            }
        }
    }
    for (auto& [cb, ok] : defer) cb(ok);
}

// ---- live log viewers (GUI-thread-only state; no registry lock needed) ------

struct log_viewer {
    std::string path;
    std::string label;
    bool        open      = true;
    std::string cached;         // last-read tail of the file
    std::time_t last_read = 0;  // throttle re-reads to ~1/s
};

// Only ever touched on the GUI thread inside render_runs_window().
std::vector<log_viewer>& viewers()
{
    static std::vector<log_viewer> v;
    return v;
}

void open_log_viewer(const std::string& path, const std::string& label)
{
    for (auto& v : viewers()) {
        if (v.path == path) { v.open = true; return; } // already open — reuse
    }
    viewers().push_back({path, label, true, {}, 0});
}

// Read the last max_bytes of a file (dropping a partial leading line). For the
// live log window — bounded so a huge log never blows up the string.
std::string read_tail(const std::string& path, std::size_t max_bytes)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return "(log not available yet: " + path + ")";
    const std::streamoff size = in.tellg();
    std::streamoff       start = 0;
    bool                 truncated = false;
    if (size > static_cast<std::streamoff>(max_bytes)) {
        start     = size - static_cast<std::streamoff>(max_bytes);
        truncated = true;
    }
    in.seekg(start);
    std::string s((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    if (truncated) {
        if (const auto nl = s.find('\n'); nl != std::string::npos) {
            s = "…\n" + s.substr(nl + 1);
        }
    }
    return s;
}

void render_log_viewers()
{
    auto&             vs  = viewers();
    const std::time_t now = std::time(nullptr);
    for (auto& v : vs) {
        if (!v.open) continue;
        if (now != v.last_read) { // throttle reads to ~1/s
            v.cached    = read_tail(v.path, std::size_t{64} * 1024);
            v.last_read = now;
        }
        ImGui::SetNextWindowSize(ImVec2(720, 420), ImGuiCond_FirstUseEver);
        // Stable ID by path (### part), human title by label.
        const std::string title = "Log: " + v.label + "###simc_log_" + v.path;
        if (ImGui::Begin(title.c_str(), &v.open)) {
            ImGui::TextDisabled("%s", v.path.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton("Copy path")) ImGui::SetClipboardText(v.path.c_str());
            ImGui::Separator();
            ImGui::BeginChild("body", ImVec2(0, 0), ImGuiChildFlags_Borders,
                              ImGuiWindowFlags_HorizontalScrollbar);
            ImGui::TextUnformatted(v.cached.c_str());
            // Follow the tail while the user is scrolled to the bottom.
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
                ImGui::SetScrollHereY(1.0f);
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }
    std::erase_if(vs, [](const log_viewer& v) { return !v.open; });
}

} // namespace

void poll_runs()
{
    // Collect completion callbacks under the lock, fire them AFTER releasing it
    // (they lock the sim_run, and step_async locks sim_run->registry).
    std::vector<std::pair<std::function<void(bool)>, bool>> completions;
    {
        auto&             R   = reg();
        const std::time_t now = std::time(nullptr);
        std::scoped_lock  lk(R.mtx);
        for (auto& row : R.rows) {
            if (row.state != run_state::running) continue;
            int           status = 0;
            const ::pid_t w      = ::waitpid(row.pid, &status, WNOHANG);
            if (w == 0) {
                if (now != row.t_parsed) { // throttle: at most once per second
                    parse_log_progress(row);
                    row.t_parsed = now;
                }
                continue;
            }
            // The child exited — finalize this row. In practice only w == row.pid
            // reaches here: w == 0 returned above, and waitpid(WNOHANG) on a
            // specific, not-yet-reaped child pid should not return -1 — EINTR
            // requires WNOHANG unset, and ECHILD can't occur (a finalized row is
            // never re-waited and nothing else in-process reaps our children).
            // That last part is a WHOLE-PROCESS invariant, though: if any linked
            // library ever sets SIGCHLD to SIG_IGN, children auto-reap and this
            // becomes -1/ECHILD. Rather than std::unreachable() (UB if that
            // happens), report it and finalize the row as failed so the handle
            // doesn't wedge with async_in_flight stuck true.
            if (w != row.pid) {
                std::fprintf(stderr,
                             "simc: waitpid(%d) returned %d (errno=%d) — treating "
                             "run '%s' as failed; a SIGCHLD=SIG_IGN somewhere would "
                             "explain it\n",
                             static_cast<int>(row.pid), static_cast<int>(w), errno,
                             row.label.c_str());
            }
            parse_log_progress(row);
            row.t_end     = now;
            row.exit_code = (w == row.pid) ? exit_code_of(status) : -1;
            // A cancelled run's child was SIGTERM'd, so it exits nonzero — but show
            // it as `cancelled`, not `failed` (and it still chains nothing, since
            // cancelled != done below).
            row.state = row.cancel_requested ? run_state::cancelled
                        : (row.exit_code == 0) ? run_state::done
                                               : run_state::failed;
            if (row.state == run_state::done && row.cur_step < row.target_step) {
                row.cur_step = row.target_step; // full bar on success
            }
            if (row.on_done) {
                completions.emplace_back(std::move(row.on_done),
                                         row.state == run_state::done);
            }
        }

        // Drain the queue: promote queued runs to running (oldest first) while the
        // reaping above may have freed pool slots. Spawning here keeps the pool at
        // its cap without the caller ever blocking. A spawn that fails at drain
        // time has no caller to throw to, so finalize that row as failed and fire
        // its completion — the same outcome as a child that exits nonzero.
        int running = count_state(R, run_state::running);
        for (auto& row : R.rows) {
            if (running >= R.max_concurrent) break;
            if (row.state != run_state::queued) continue;
            try {
                row.pid     = spawn_child(row.pending_args, row.log_path);
                row.state   = run_state::running;
                row.t_start = now; // clock starts when it actually runs, not when queued
                row.pending_args.clear();
                row.pending_args.shrink_to_fit();
                ++running;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "simc: queued run '%s' failed to spawn: %s\n",
                             row.label.c_str(), e.what());
                row.state     = run_state::failed;
                row.exit_code = -1;
                row.t_end     = now;
                row.pending_args.clear();
                if (row.on_done) {
                    completions.emplace_back(std::move(row.on_done), false);
                }
            }
        }
    }
    for (auto& [cb, ok] : completions) cb(ok);
}

int active_run_count()
{
    // Running AND queued: a queued run is still "in flight" from the caller's view
    // (its handle reports running(), and wait_all_runs() must not return until the
    // queue has drained and every child has exited).
    auto&            R = reg();
    std::scoped_lock lk(R.mtx);
    int c = 0;
    for (const auto& row : R.rows) {
        if (row.state == run_state::running || row.state == run_state::queued) ++c;
    }
    return c;
}

void wait_all_runs()
{
    for (;;) {
        poll_runs();
        if (active_run_count() == 0) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// True iff a row has finished — the predicate for "clear finished". Note it is
// NOT `!= running`: a queued row is unfinished too and must be kept.
bool is_finished(const run_row& r)
{
    return r.state == run_state::done || r.state == run_state::failed ||
           r.state == run_state::cancelled;
}

void clear_finished_runs()
{
    auto&            R = reg();
    std::scoped_lock lk(R.mtx);
    std::erase_if(R.rows, is_finished);
}

void cancel_all_runs()
{
    completion_list defer;
    {
        auto&             R   = reg();
        const std::time_t now = std::time(nullptr);
        std::scoped_lock  lk(R.mtx);
        for (auto& row : R.rows) cancel_row_locked(row, now, defer);
    }
    for (auto& [cb, ok] : defer) cb(ok);
}

void show_runs_window() { set_runs_visible(true); }
void hide_runs_window() { set_runs_visible(false); }

void set_max_concurrent_runs(int n)
{
    auto&            R = reg();
    std::scoped_lock lk(R.mtx);
    R.max_concurrent = n < 1 ? 1 : n; // at least 1; the next poll_runs() drains up to it
}

int max_concurrent_runs()
{
    auto&            R = reg();
    std::scoped_lock lk(R.mtx);
    return R.max_concurrent;
}

int queued_run_count()
{
    auto&            R = reg();
    std::scoped_lock lk(R.mtx);
    return count_state(R, run_state::queued);
}

void render_runs_window()
{
    poll_runs(); // advance state (and fire completions) every frame

    // Cancel-button clicks below defer their on_done(false) callbacks here; they
    // fire AFTER the registry lock is released (a callback locks the sim_run).
    completion_list defer;
    {
        auto&             R   = reg();
        const std::time_t now = std::time(nullptr);
        std::scoped_lock  lk(R.mtx);
        if (R.visible.load() && !R.rows.empty()) {
            bool open = true;
            ImGui::SetNextWindowSize(ImVec2(640, 260), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Simulation Runs", &open)) {
                int running = 0, queued = 0;
                for (const auto& row : R.rows) {
                    if (row.state == run_state::running) ++running;
                    else if (row.state == run_state::queued) ++queued;
                }
                if (ImGui::SmallButton("Clear finished")) {
                    std::erase_if(R.rows, is_finished);
                }
                ImGui::SameLine();
                const bool any_active = (running + queued) > 0;
                ImGui::BeginDisabled(!any_active);
                if (ImGui::SmallButton("Cancel all")) {
                    for (auto& row : R.rows) cancel_row_locked(row, now, defer);
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                ImGui::TextDisabled(
                    "%d running, %d queued (cap %d), %zu total  (double-click a row "
                    "to view its log)",
                    running, queued, R.max_concurrent, R.rows.size());

                const ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                                              ImGuiTableFlags_RowBg |
                                              ImGuiTableFlags_SizingStretchProp;
                if (ImGui::BeginTable("simc_runs", 6, flags)) {
                    ImGui::TableSetupColumn("Run");
                    ImGui::TableSetupColumn("Progress");
                    ImGui::TableSetupColumn("Steps");
                    ImGui::TableSetupColumn("Status");
                    ImGui::TableSetupColumn("Elapsed");
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed);
                    ImGui::TableHeadersRow();

                    for (auto& row : R.rows) {
                        ImGui::TableNextRow();
                        ImGui::PushID(row.log_path.c_str());

                        // Column 0: label, row-spanning selectable. Double-click
                        // opens a live viewer on the run's log. AllowOverlap so the
                        // per-row Cancel button in the last column still receives
                        // its click — without it the span-all-columns selectable,
                        // submitted first, swallows clicks over the whole row.
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Selectable(row.label.c_str(), false,
                                          ImGuiSelectableFlags_SpanAllColumns |
                                              ImGuiSelectableFlags_AllowDoubleClick |
                                              ImGuiSelectableFlags_AllowOverlap);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("%s\ndouble-click to view log",
                                              row.log_path.c_str());
                        }
                        if (ImGui::IsItemHovered() &&
                            ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                            open_log_viewer(row.log_path, row.label);
                        }

                        // Column 1: progress bar, tinted by state.
                        float frac = 0.0f;
                        if (row.state == run_state::done) {
                            frac = 1.0f;
                        } else if (row.target_step > row.start_step &&
                                   row.cur_step >= row.start_step) {
                            // Guard the unsigned subtraction: on a resumed chunk
                            // start_step parses from "from step N to 2N" (N>0)
                            // while cur_step stays 0 until the first "Steps:" line
                            // (~every 10k steps), so cur_step - start_step would
                            // underflow and pin the bar to full. Keep it at 0
                            // until real progress is parsed.
                            frac = static_cast<float>(
                                static_cast<double>(row.cur_step - row.start_step) /
                                static_cast<double>(row.target_step - row.start_step));
                        }
                        frac = std::clamp(frac, 0.0f, 1.0f);
                        ImGui::TableSetColumnIndex(1);
                        bool tinted = true;
                        if (row.state == run_state::failed) {
                            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                                  ImVec4(0.85f, 0.22f, 0.20f, 1.0f)); // red
                        } else if (row.state == run_state::done) {
                            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                                  ImVec4(0.30f, 0.66f, 0.35f, 1.0f)); // green
                        } else if (row.state == run_state::cancelled) {
                            ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                                                  ImVec4(0.70f, 0.55f, 0.20f, 1.0f)); // amber
                        } else {
                            tinted = false;
                        }
                        ImGui::ProgressBar(frac, ImVec2(-1.0f, 0.0f));
                        if (tinted) ImGui::PopStyleColor();

                        ImGui::TableSetColumnIndex(2);
                        if (row.target_step > 0) {
                            ImGui::Text("%llu / %llu",
                                        static_cast<unsigned long long>(row.cur_step),
                                        static_cast<unsigned long long>(row.target_step));
                        } else {
                            ImGui::TextUnformatted("starting…");
                        }

                        ImGui::TableSetColumnIndex(3);
                        switch (row.state) {
                            case run_state::queued:
                                ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.70f, 1.0f),
                                                   "queued");
                                break;
                            case run_state::running:
                                ImGui::TextUnformatted("running");
                                break;
                            case run_state::done:
                                ImGui::TextColored(ImVec4(0.45f, 0.80f, 0.50f, 1.0f), "done");
                                break;
                            case run_state::failed:
                                ImGui::TextColored(ImVec4(0.92f, 0.35f, 0.32f, 1.0f),
                                                   "failed (%d)", row.exit_code);
                                break;
                            case run_state::cancelled:
                                ImGui::TextColored(ImVec4(0.80f, 0.66f, 0.30f, 1.0f),
                                                   "cancelled");
                                break;
                        }

                        ImGui::TableSetColumnIndex(4);
                        const std::time_t elapsed =
                            (row.state == run_state::running ? now : row.t_end) - row.t_start;
                        ImGui::Text("%llds", static_cast<long long>(elapsed));

                        // Column 5: a Cancel button for an in-flight (running or
                        // queued) run. cancel_row_locked runs under the registry
                        // lock we already hold; its on_done(false) is deferred and
                        // fired after the lock releases.
                        ImGui::TableSetColumnIndex(5);
                        if (row.state == run_state::running ||
                            row.state == run_state::queued) {
                            if (ImGui::SmallButton("Cancel")) {
                                cancel_row_locked(row, now, defer);
                            }
                        }

                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::End();
            R.visible.store(open);
        }
    } // registry lock released before rendering viewers (they read files)

    // Fire any cancel callbacks now that the registry lock is released.
    for (auto& [cb, ok] : defer) cb(ok);

    render_log_viewers();
}

// ----------------------------------------------------------------- filesystem
void mkdir(const std::filesystem::path& p)
{
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error("mkdir('" + p.string() + "'): " + ec.message());
    }
}

bool exists(const std::filesystem::path& p)
{
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

std::vector<std::string> ls(const std::filesystem::path& p)
{
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(p, ec)) {
        out.push_back(e.path().filename().string());
    }
    if (ec) {
        throw std::runtime_error("ls('" + p.string() + "'): " + ec.message());
    }
    return out;
}

std::string cwd()
{
    std::error_code ec;
    auto p = std::filesystem::current_path(ec);
    if (ec) throw std::runtime_error("cwd: " + ec.message());
    return p.string();
}

} // namespace coast::simc
