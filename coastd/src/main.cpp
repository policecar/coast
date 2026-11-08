// -----------------------------------------------------------------------------
// coastd — resident per-machine headless-job daemon (entry point).
//
// Resolves the state/runs/worktrees roots, boots the store, then runs the core
// loop in its OWN thread while the comms edges (TCP command server + UDP
// discovery responder) serve clients. Shutdown is ordered on purpose: a signal
// only sets a flag; the main thread then stops the comms edges FIRST — while the
// core is still alive to drain any in-flight client future — and only then stops
// the core and joins. This is why a client blocked on a reply never wedges the
// shutdown.
// -----------------------------------------------------------------------------

#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "daemon/atomic_write.h"
#include "daemon/child_proc.h"   // start_token_supported — the platform gate below
#include "daemon/comms_server.h"
#include "daemon/config_load.h"
#include "daemon/core.h"
#include "daemon/discovery_server.h"
#include "daemon/types.h"
#include "daemon/version.h"
#include "diag/diag.h"

namespace {

// The shutdown latch. Set by the signal handler AND by the core thread's
// last-resort backstop, and polled by the main thread — so it is genuinely shared
// across threads, not merely between a handler and its own thread.
// `volatile sig_atomic_t` carries no inter-thread ordering and is a data race in
// that second role; std::atomic<int> is what the same latch in run_headless.cpp
// already uses. It stays lock-free and async-signal-safe for the handler:
// std::atomic<int> is always lock-free on every platform coastd supports, and the
// store below compiles to a plain store.
std::atomic<int> g_stop{0};

void handle_stop(int) { g_stop.store(1, std::memory_order_relaxed); }

std::filesystem::path default_data_root() {
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg)
        return std::filesystem::path(xdg) / "coastd";
    if (const char* home = std::getenv("HOME"); home && *home)
        return std::filesystem::path(home) / ".local" / "share" / "coastd";
    return std::filesystem::current_path() / ".coastd";
}

void print_usage(std::ostream& out, const char* prog) {
    out << "usage: " << prog << " [--data-root <dir>] [--repo <dir>] [--config <file>]\n"
           "             [--tcp-port <n>] [--udp-port <n>] [--term-grace <seconds>]\n"
           "             [--slots <n>] [--worker-pool-size <n>] [--checkpoint-interval <n>]\n"
           "             [--worktree-cooloff <seconds>] [--scan-period <ms>]\n"
           "             [--build-timeout <seconds>] [--git-timeout <seconds>]\n"
           "             [--max-launch-failures <n>] [--build-jobs <n>] [--cmake-arg <arg>]...\n"
           "             [--no-fetch] [--git-remote <name>] [--fetch-timeout <seconds>]\n"
           "             [--fetch-min-interval <seconds>]\n"
           "             [--install-service] [-v|-q] [--help|--version]\n"
        << "  coastd runs COAST simulations headless and supervises their lifecycle.\n\n"
           "  Operational knobs may be set in a JSON config file (--config, else\n"
           "  <data-root>/coastd.json if present); any CLI flag overrides the file.\n"
           "  --config <file>         load daemon config JSON (keys mirror the flags below).\n"
           "  --tcp-port <n>          command port (default 51900); --udp-port <n> is the\n"
           "                          discovery port (default 51901). 0 lets the OS pick a\n"
           "                          free port (CLI only); either way the ports in use are\n"
           "                          written to <data-root>/coastd.ports once listening.\n"
           "  --slots <n>             max concurrent running jobs (default 1).\n"
           "  --checkpoint-interval <n>  coast --checkpoint N (default 500000).\n"
           "  --build-jobs <n>        cmake --build -j N (default 0 = half the cores).\n"
           "  --cmake-arg <arg>       extra -D flag for each worktree's cmake configure\n"
           "                          (repeatable; e.g. --cmake-arg -DCOAST_GPU_BACKEND=webgpu).\n"
           "  --worktree-cooloff <s>  idle-worktree GC grace (default 21600 = 6h).\n"
           "  --no-fetch              do NOT `git fetch` before resolving a ref, and\n"
           "                          resolve local-first. Default is to fetch (throttled,\n"
           "                          best-effort) and prefer <remote>/<branch> for plain\n"
           "                          branch names, so 'main' means the REMOTE's main and a\n"
           "                          stale clone cannot silently run old code.\n"
           "  --git-remote <name>     remote to prefer refs from (default: auto — 'origin'\n"
           "                          if present, else the sole remote).\n"
           "  --fetch-timeout <s>     bound on `git fetch` (default 300). An overrun is not\n"
           "                          fatal: resolution falls back to the local clone.\n"
           "  --fetch-min-interval <s>  skip a fetch newer than this (default 30; 0 = always\n"
           "                          fetch). Coalesces a burst of submissions into one fetch.\n"
           "  --term-grace <seconds>  grace a cancelled/stopped job gets to write its\n"
           "                          final checkpoint before SIGKILL (default 10).\n\n"
           "  Config-file keys for the above: fetch_enabled (bool), git_remote,\n"
           "  fetch_timeout_seconds, fetch_min_interval_seconds.\n\n"
           "  Config-file-only knobs (no CLI flag): build_concurrency, max_build_failures,\n"
           "  audit_git_timeout_seconds, relaunch_backoff_base_seconds,\n"
           "  relaunch_backoff_max_seconds, and the\n"
           "  client-edge bounds comms_max_connections, comms_max_request_bytes,\n"
           "  comms_max_config_bytes, comms_first_request_timeout_seconds,\n"
           "  comms_line_assembly_timeout_seconds, comms_send_timeout_seconds.\n\n"
           "  Directory roots as config keys: state_root, runs_root, worktrees_root,\n"
           "  repo_root — each defaults to <data-root>/<name> (repo_root to the CWD) and\n"
           "  may be moved independently. A literal '<data-root>' in the value expands to\n"
           "  the resolved data root. --repo still overrides a repo_root key. data_root\n"
           "  itself is CLI/-env only, since the config file is found under it.\n";
}


// Resolve this running binary's absolute path (for a self-contained ExecStart).
std::filesystem::path self_exe_path(const char* argv0) {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0) { buf[n] = '\0'; return std::filesystem::path(buf); }
#endif
    std::error_code ec;
    return std::filesystem::absolute(argv0, ec);
}

// Quote one ExecStart argument for systemd's unit-file parser.
//
// Unquoted paths break on the first space, and a path containing a newline would
// end the ExecStart= line and let the remainder be read as further unit
// DIRECTIVES. The inputs are operator-supplied (--data-root/--repo, or argv[0]),
// so this is robustness rather than a privilege boundary — but a directory named
// "My Runs" is an ordinary mistake to make, and it currently produces a unit that
// fails to start with no clue why. systemd's double-quoted form takes C-style
// escapes, so backslash, double-quote and newline are all expressible.
std::string sd_quote(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            default:   out += c;      break;
        }
    }
    out += '"';
    return out;
}

// Write a systemd USER unit pointing at THIS binary and the given roots, then
// print the enable steps. Linux only (macOS uses the launchd plist template).
int install_service(const std::filesystem::path& exe,
                    const std::filesystem::path& data_root,
                    const std::filesystem::path& repo_root,
                    long timeout_stop_sec) {
#if defined(__linux__)
    const char* home = std::getenv("HOME");
    if (!home || !*home) { std::cerr << "coastd: HOME is not set — cannot locate the user unit dir\n"; return 1; }
    std::error_code ec;
    const std::filesystem::path unit_dir = std::filesystem::path(home) / ".config" / "systemd" / "user";
    std::filesystem::create_directories(unit_dir, ec);
    const std::filesystem::path unit = unit_dir / "coastd.service";

    std::ofstream out(unit, std::ios::trunc);
    if (!out) { std::cerr << "coastd: cannot write " << unit.string() << "\n"; return 1; }
    out << "[Unit]\n"
        << "Description=coastd — resident COAST headless-job daemon\n"
        << "After=network.target\n\n"
        << "[Service]\n"
        << "Type=simple\n"
        << "ExecStart=" << sd_quote(std::filesystem::absolute(exe, ec).string())
        << " --data-root " << sd_quote(std::filesystem::absolute(data_root, ec).string())
        << " --repo "      << sd_quote(std::filesystem::absolute(repo_root, ec).string()) << "\n"
        << "Restart=on-failure\n"
        << "RestartSec=2\n"
        // Stock cgroup reap IS the R1 recovery model — do not override.
        << "KillMode=control-group\n"
        // Must exceed the LONGEST graceful-shutdown wait, else systemd SIGKILLs the
        // whole cgroup mid-work. That wait is dominated by an in-flight builder step
        // (worker_pool::shutdown lets it finish) which is bounded by build_timeout,
        // NOT by term_grace — so it is budgeted as build_timeout + term_grace + a
        // margin from the RESOLVED config (default 3600 + 10 + 30 = 3640s).
        << "TimeoutStopSec=" << timeout_stop_sec << "\n\n"
        << "[Install]\n"
        << "WantedBy=default.target\n";
    if (!out) { std::cerr << "coastd: failed writing " << unit.string() << "\n"; return 1; }

    std::cout << "Wrote " << unit.string() << "\nEnable it with:\n"
              << "  systemctl --user daemon-reload\n"
              << "  systemctl --user enable --now coastd\n"
              << "  sudo loginctl enable-linger \"$USER\"   # start at boot without a login\n";
    return 0;
#else
    (void)exe; (void)data_root; (void)repo_root;
    std::cerr << "coastd: --install-service is Linux/systemd only. On macOS use the launchd\n"
                 "        template coastd/service/com.coast.coastd.plist (see the README).\n";
    return 1;
#endif
}

// Parse an unsigned 16-bit port, exiting with a clear message on garbage. 0 is
// valid here and means "let the OS pick": the daemon binds first and reports the
// port it got (coastd.ports, below), so a caller that needs a collision-free port
// — a test, two daemons on one box — never has to guess one. CLI only: the config
// file keeps 1..65535, because a service that comes up on a different port every
// boot is unreachable through every fixed host list that names it.
std::uint16_t parse_port(std::string_view flag, const char* value) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || v > 65535) {
        std::cerr << "coastd: " << flag << " needs a port in 0..65535 (0 = OS-assigned)\n";
        std::exit(2);
    }
    return static_cast<std::uint16_t>(v);
}

// Parse a positive whole-second duration, exiting with a clear message on garbage.
std::chrono::seconds parse_seconds(std::string_view flag, const char* value) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || v == 0 || v > 86400) {
        std::cerr << "coastd: " << flag << " needs a whole number of seconds in 1..86400\n";
        std::exit(2);
    }
    return std::chrono::seconds(static_cast<long>(v));
}

// Parse an integer in [0, max]. Separate from parse_uint because a few knobs
// (fetch coalescing window) give 0 a real meaning — "no window" — rather than
// treating it as the garbage value parse_uint rejects.
unsigned long parse_uint_incl_zero(std::string_view flag, const char* value, unsigned long max) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || v > max) {
        std::cerr << "coastd: " << flag << " needs a whole number in 0.." << max << "\n";
        std::exit(2);
    }
    return v;
}

// Parse a positive integer in [1, max], exiting with a clear message on garbage.
unsigned long parse_uint(std::string_view flag, const char* value, unsigned long max) {
    char* end = nullptr;
    const unsigned long v = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || v == 0 || v > max) {
        std::cerr << "coastd: " << flag << " needs a whole number in 1.." << max << "\n";
        std::exit(2);
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path data_root = default_data_root();
    std::optional<std::filesystem::path> config_path;
    // CLI overrides — applied AFTER the config file, so a flag always wins. repo
    // gets the same treatment as every scalar knob (an optional applied post-load)
    // precisely so a config-file "repo_root" key does not silently beat --repo.
    std::optional<std::filesystem::path> repo_cli;
    std::optional<std::uint16_t> tcp_port, udp_port;
    std::optional<std::chrono::seconds> term_grace, worktree_cooloff, build_timeout, git_timeout;
    std::optional<std::chrono::milliseconds> scan_period;
    std::optional<int> slots, worker_pool_size, max_launch_failures, build_jobs;
    std::optional<std::uint64_t> checkpoint_interval;
    std::optional<std::vector<std::string>> cmake_args;
    std::optional<bool> fetch_enabled;
    std::optional<std::string> git_remote;
    std::optional<std::chrono::seconds> fetch_timeout, fetch_min_interval;
    bool want_install_service = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto next = [&](std::string_view flag) -> const char* {
            if (i + 1 >= argc) { std::cerr << "coastd: " << flag << " needs a value\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h")         { print_usage(std::cout, argv[0]); return 0; }
        else if (a == "--version")              { std::cout << "coastd " << coastd::version << "\n"; return 0; }
        else if (a == "--data-root")            { data_root = next(a); }
        else if (a == "--repo")                 { repo_cli = next(a); }
        else if (a == "--config")               { config_path = next(a); }
        else if (a == "--tcp-port")             { tcp_port = parse_port(a, next(a)); }
        else if (a == "--udp-port")             { udp_port = parse_port(a, next(a)); }
        else if (a == "--term-grace")           { term_grace = parse_seconds(a, next(a)); }
        else if (a == "--slots")                { slots = static_cast<int>(parse_uint(a, next(a), 4096)); }
        else if (a == "--worker-pool-size")     { worker_pool_size = static_cast<int>(parse_uint(a, next(a), 4096)); }
        else if (a == "--checkpoint-interval")  { checkpoint_interval = static_cast<std::uint64_t>(parse_uint(a, next(a), 1'000'000'000'000UL)); }
        else if (a == "--worktree-cooloff")     { worktree_cooloff = parse_seconds(a, next(a)); }
        else if (a == "--scan-period")          { scan_period = std::chrono::milliseconds(parse_uint(a, next(a), 3'600'000)); }
        else if (a == "--build-timeout")        { build_timeout = parse_seconds(a, next(a)); }
        else if (a == "--git-timeout")          { git_timeout = parse_seconds(a, next(a)); }
        else if (a == "--max-launch-failures")  { max_launch_failures = static_cast<int>(parse_uint(a, next(a), 1000)); }
        else if (a == "--build-jobs")           { build_jobs = static_cast<int>(parse_uint(a, next(a), 4096)); }
        else if (a == "--cmake-arg")            { if (!cmake_args) cmake_args.emplace(); cmake_args->emplace_back(next(a)); }
        else if (a == "--no-fetch")             { fetch_enabled = false; }
        else if (a == "--fetch")                { fetch_enabled = true; }
        else if (a == "--git-remote")           { git_remote = next(a); }
        else if (a == "--fetch-timeout")        { fetch_timeout = parse_seconds(a, next(a)); }
        // 0 = fetch on every resolve; parse_seconds rejects 0, so parse it as a
        // plain count and build the duration here.
        else if (a == "--fetch-min-interval")   { fetch_min_interval = std::chrono::seconds(
                                                      static_cast<long>(parse_uint_incl_zero(a, next(a), 86400))); }
        else if (a == "--install-service")      { want_install_service = true; }
        else if (a == "-v" || a == "--verbose") { coast::diag::set_level(coast::diag::level::debug); }
        else if (a == "-q" || a == "--quiet")   { coast::diag::set_level(coast::diag::level::warn); }
        else { std::cerr << "coastd: unknown argument '" << a << "'\n"; print_usage(std::cerr, argv[0]); return 2; }
    }

    coastd::daemon_config cfg;
    // Directory-root DEFAULTS. Each is overridable by a coastd.json key of the same
    // name (load_config_file, below); --repo then overrides repo_root in turn, so
    // the precedence is default < file < CLI, matching every scalar knob. data_root
    // itself is deliberately NOT a config key: the file is discovered at
    // <data-root>/coastd.json, so it must be resolved before the file is read — and
    // that is also why a "<data-root>" token in a config path value still resolves.
    cfg.state_root     = data_root / "state";
    cfg.runs_root      = data_root / "runs";
    cfg.worktrees_root = data_root / "worktrees";
    cfg.repo_root      = std::filesystem::current_path();
    // Absolute: the daemon writes it here at boot, and each child (which chdirs to
    // its run dir) receives it as --app-config, so a relative path would resolve
    // against the wrong directory in the child.
    {
        std::error_code aec;
        cfg.app_config_path = std::filesystem::absolute(data_root / "coast_app_config.json", aec);
        if (aec) cfg.app_config_path = data_root / "coast_app_config.json";
    }

    // Config file (defaults < file < CLI): explicit --config must load; an
    // auto-discovered <data-root>/coastd.json is best-effort (warn + continue).
    {
        std::filesystem::path cpath;
        if (config_path) {
            cpath = *config_path;
        } else {
            std::error_code ec;
            if (auto def = data_root / "coastd.json"; std::filesystem::is_regular_file(def, ec))
                cpath = def;
        }
        if (!cpath.empty()) {
            std::string err;
            if (coastd::load_config_file(cpath, data_root, cfg, err)) {
                COAST_INFO("coastd", "loaded daemon config {}", cpath.string());
            } else if (config_path) {
                COAST_ERROR("coastd", "{}", err);   // an explicitly-named file MUST load
                return 1;
            } else {
                COAST_WARN("coastd", "{}", err);     // auto-discovered: ignore + continue
            }
        }
    }

    // CLI overrides win over the config file.
    if (repo_cli)            cfg.repo_root           = *repo_cli;
    if (tcp_port)            cfg.tcp_port            = *tcp_port;
    if (udp_port)            cfg.udp_port            = *udp_port;
    if (term_grace)          cfg.term_grace          = *term_grace;
    if (slots)               cfg.slots               = *slots;
    if (worker_pool_size)    cfg.worker_pool_size    = *worker_pool_size;
    if (max_launch_failures) cfg.max_launch_failures = *max_launch_failures;
    if (build_jobs)          cfg.build_jobs          = *build_jobs;
    if (checkpoint_interval) cfg.checkpoint_interval = *checkpoint_interval;
    if (worktree_cooloff)    cfg.worktree_cooloff    = *worktree_cooloff;
    if (scan_period)         cfg.scan_period         = *scan_period;
    if (build_timeout)       cfg.build_timeout       = *build_timeout;
    if (git_timeout)         cfg.git_op_timeout      = *git_timeout;
    if (cmake_args)          cfg.build_cmake_args    = *cmake_args;
    if (fetch_enabled)       cfg.fetch_enabled       = *fetch_enabled;
    if (git_remote)          cfg.git_remote          = *git_remote;
    if (fetch_timeout)       cfg.fetch_timeout       = *fetch_timeout;
    if (fetch_min_interval)  cfg.fetch_min_interval  = *fetch_min_interval;

    // A submit carries the config's content in its request line, so the line cap
    // must exceed the config cap (types.h). Correct-and-tell rather than fail: the
    // daemon stays up, but the operator learns, because the symptom otherwise looks
    // like a daemon bug rather than a misconfiguration.
    if (clamp_edge_bounds(cfg))
        COAST_WARN("coastd", "max_request_bytes was below the size a maximum config needs "
                             "on the wire; raised to {} bytes (max_config_bytes={})",
                   cfg.max_request_bytes, cfg.max_config_bytes);

    // Backoff sanity: base > max is not fatal (reconcile clamps the product to
    // max, degrading to a constant backoff at max) but it silently ignores base,
    // which is almost always a typo. Warn rather than clamp — the operator's
    // intent is honored, just flagged.
    if (cfg.relaunch_backoff_base > cfg.relaunch_backoff_max)
        COAST_WARN("coastd", "relaunch_backoff_base ({}s) exceeds relaunch_backoff_max ({}s); "
                             "backoff is effectively constant at max",
                   cfg.relaunch_backoff_base.count(), cfg.relaunch_backoff_max.count());

    // install_service bakes the EFFECTIVE repo (config or --repo, absolutized in
    // the unit) into ExecStart, so a service started without a CWD still resolves.
    if (want_install_service)
        return install_service(self_exe_path(argv[0]), data_root, cfg.repo_root,
                               cfg.build_timeout.count() + cfg.term_grace.count() + 30);

    // ---- platform gate -------------------------------------------------------
    // The daemon's duplicate-run protection (R1) rests on the boot pid-guard:
    // identify a child that outlived a crashed daemon by pid + process START TOKEN,
    // and SIGKILL it before relaunching its job. Without a token source the guard
    // cannot distinguish "my child is still running" from "that pid is gone", and
    // answers `gone` — so the daemon would relaunch a job whose child is still
    // stepping, putting two coast processes on one run dir. That is the corruption
    // R1 exists to make impossible, and it is fail-DANGEROUS: silent, and worst
    // exactly when things already went wrong.
    //
    // So refuse to run rather than run without the guarantee. There is deliberately
    // no override flag: a flag that re-enables a silent-corruption path is that path.
    // Lifting this is one function — see the platform seam in daemon/child_proc.h.
    // Only the DAEMON is gated: coastd_client, the console façade and the GUI panel
    // are pure TCP clients and stay fully portable (a macOS workstation driving a
    // Linux daemon is a supported, tested configuration).
    if constexpr (!coastd::start_token_supported()) {
        std::cerr << "coastd: this platform provides no process start-token source, so the\n"
                     "        boot pid-guard that prevents two coast children sharing one run\n"
                     "        dir cannot work. Refusing to start.\n"
                     "        The daemon is supported on Linux and macOS; clients (GUI panel,\n"
                     "        console, coastd_client) run anywhere and can drive a remote daemon.\n"
                     "        To port: implement read_start_token() + start_token_supported()\n"
                     "        for this platform (see coastd/include/daemon/child_proc.h).\n";
        return 1;
    }

    // Single-instance guard: exclusively flock a lockfile under data_root. Two
    // daemons on the same roots would both load the store and independently spawn
    // children into the SAME run dirs (duplicate-run corruption). Held (fd left
    // open) for the whole process lifetime; the OS releases it on exit.
    std::error_code lec;
    std::filesystem::create_directories(data_root, lec);
    const std::filesystem::path lockpath = data_root / "coastd.lock";
    const int lock_fd = ::open(lockpath.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (lock_fd < 0) {
        COAST_ERROR("coastd", "cannot open lock file {}: {}", lockpath.string(), std::strerror(errno));
        return 1;
    }
    if (::flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        COAST_ERROR("coastd", "another coastd is already running against {} "
                    "(lock {} is held); refusing to start", data_root.string(), lockpath.string());
        ::close(lock_fd);
        return 1;
    }

    // <data-root>/coastd.ports: the ports this daemon actually listens on, written
    // once both edges are up. A leftover from a predecessor that died without
    // removing it would name ports nobody serves, so it goes before anything else
    // can be read from this root. The pid inside lets a reader that raced this
    // removal tell the stale file from ours.
    const std::filesystem::path portspath = data_root / "coastd.ports";
    std::filesystem::remove(portspath, lec);

    COAST_INFO("coastd", "coastd {} starting — data={} repo={} slots={}", coastd::version,
               data_root.string(), cfg.repo_root.string(), cfg.slots);

    coastd::core core(cfg);
    try {
        core.boot();   // store I/O — a failure here must not std::terminate the process
    } catch (const std::exception& e) {
        COAST_ERROR("coastd", "boot failed: {} — aborting startup", e.what());
        return 1;
    }

    // A client disconnecting mid-write must never kill the daemon (SIGPIPE -> ignore).
    std::signal(SIGPIPE, SIG_IGN);

    coastd::comms_server comms(core, cfg, std::string(coastd::version));
    try {
        comms.start();               // fatal if the command port is unavailable
    } catch (const std::exception& e) {
        COAST_ERROR("coastd", "{}", e.what());
        return 1;
    }
    // Discovery replies advertise the command port, so it is built from the port
    // comms actually bound — which differs from cfg.tcp_port when that was 0.
    cfg.tcp_port = comms.port();
    coastd::discovery_server discovery(cfg, std::string(coastd::version));
    // Best-effort; logs + continues if the UDP port is taken. Guarded because
    // start() constructs a std::thread: a system_error here would escape main (which
    // has no outer try) and terminate a daemon that is otherwise fully up, with its
    // command port already serving. Discovery is the optional convenience layer over
    // the fixed host list — losing it must not cost the daemon.
    try {
        discovery.start();
    } catch (const std::exception& e) {
        COAST_WARN("coastd", "discovery service could not start: {} — the daemon runs "
                   "without UDP discovery; clients can still connect by name/address", e.what());
    }

    // Written only now, so its existence means "accepting connections". Fatal on
    // failure: with an OS-assigned port this file is the only way to find the
    // daemon, and a durable write that fails under data_root is serious anyway.
    try {
        const nlohmann::json ports = {{"pid", static_cast<long long>(::getpid())},
                                      {"tcp_port", comms.port()},
                                      {"udp_port", discovery.port()}};   // 0 = no responder
        coastd::atomic_write(portspath, ports.dump() + "\n");
    } catch (const std::exception& e) {
        COAST_ERROR("coastd", "cannot write {}: {}", portspath.string(), e.what());
        return 1;
    }

    std::signal(SIGINT, handle_stop);
    std::signal(SIGTERM, handle_stop);

    std::thread core_thread([&] {                    // the single-writer loop
        try {
            core.run();
        } catch (const std::exception& e) {
            // Last-resort backstop. run() already guards every loop iteration; this
            // catches a throw from supervisor start/shutdown or any path outside the
            // loop so it never escapes the thread entry (-> std::terminate). Log and
            // request an orderly daemon shutdown.
            COAST_ERROR("coastd", "core thread aborted: {}", e.what());
            // Close the event queue NOW (idempotent with the later core.stop()): the
            // main thread tears comms down first, and its join would otherwise hang
            // on a connection blocked in future::get() for a client_command this dead
            // core will never process. Closing drops the pending command, breaking
            // its promise so the connection unblocks.
            core.stop();
            g_stop.store(1, std::memory_order_relaxed);
        }
    });

    using namespace std::chrono_literals;
    while (g_stop.load(std::memory_order_relaxed) == 0) std::this_thread::sleep_for(200ms);

    COAST_INFO("coastd", "stopping");
    comms.stop();        // stop accepting + drain in-flight (core still alive)
    std::filesystem::remove(portspath, lec);   // no longer accepting
    discovery.stop();
    core.stop();         // ask the loop to return
    core_thread.join();

    COAST_INFO("coastd", "coastd stopped");
    return 0;
}
