#ifndef COAST_TESTS_COASTD_FAULT_RIG_H
#define COAST_TESTS_COASTD_FAULT_RIG_H

// -----------------------------------------------------------------------------
// The fault-matrix rig: one REAL coastd, a throwaway repo with a fake `coast`
// build, and a fake `coast` child — shared by test_fault_matrix.cpp (behaviour,
// pass/fail) and bench_coastd_latency.cpp (latency numbers, manual).
//
// States are held by CONSTRUCTION, not by time (CLAUDE.md "Testing"): the fake
// build blocks until build_release exists, the fake child runs until
// run_release exists. A test injects its fault while the job is provably in the
// state, then releases. Waits are on the awaited job's own heartbeat (its
// fingerprint, rig::wait_until), never on a deadline.
// -----------------------------------------------------------------------------

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "client/coastd_client.h"
#include "test_daemon.h"

#ifndef COASTD_BIN
#define COASTD_BIN "coastd"
#endif

extern char** environ;

namespace coastd_fault {

namespace fs = std::filesystem;
namespace cc = coastd::client;
using clk    = std::chrono::steady_clock;

inline double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

inline bool shell_ok(const std::string& cmd) { return std::system(cmd.c_str()) == 0; }

inline fs::path make_temp_dir() {
    static std::atomic<int> ctr{0};
    fs::path p = fs::temp_directory_path() /
                 ("coastd_fault_" + std::to_string(::getpid()) + "_" +
                  std::to_string(ctr.fetch_add(1)));
    fs::create_directories(p);
    return p;
}

// A fake `coast` that is realistic where the matrix needs realism: it honours the
// argv coastd builds, reports progress on stdout so the tailer sees it, runs
// until $COASTD_FAKE_RELEASE exists (so a fault provably lands mid-run), and
// finishes with a clean manifest + saved state. It does NOT checkpoint —
// deliberately: a resume after a kill then starts over, which is the harsher path
// for the accounting, and checkpoint-correct resume is already the subject of
// gate.coastd.real_coast.
inline constexpr const char* fake_coast = R"SH(#!/usr/bin/env bash
# A child that IGNORES SIGTERM models the realistic slow stop: a real coast
# handles --save-on-term by writing a final checkpoint first, so the daemon waits
# out term_grace. That wait is the window a client must be able to see.
[ -n "$COASTD_FAKE_IGNORE_TERM" ] && trap '' TERM
n=0; save=""; manifest=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n) n="$2"; shift 2;;
    --save-state) save="$2"; shift 2;;
    --emit-run-manifest) manifest="$2"; shift 2;;
    *) shift;;
  esac
done
echo "@coast/progress v1 phase=start step=0 start=0 target=$n sps=0.000 logged=0"
# Silent while held: a progress line per poll would keep the job's heartbeat
# (its fingerprint, incl. live_step) ticking while the job is stuck, and a test
# waiting for it to leave `running` would never end.
while [ ! -e "$COASTD_FAKE_RELEASE" ]; do sleep 0.2; done
echo "@coast/progress v1 phase=done step=$n start=0 target=$n sps=5.000 logged=0"
[ -n "$save" ] && echo fake-state > "$save"
[ -n "$manifest" ] && printf '{"n_steps_completed": %s, "exit_reason": "completed_n_steps"}\n' "$n" > "$manifest"
exit 0
)SH";

// A CMake project whose only target is `coast`: it copies the script above into
// <build>/coast, which is exactly what the builder checks for. `project(… NONE)`
// skips compiler detection. With `blocking`, configure waits until build_release
// exists — "building" is then a state a fault can be injected INTO for as long
// as the test needs, instead of a sleep that has to be raced.
inline std::string cmake_lists(bool blocking, const fs::path& build_release) {
    std::string s = "cmake_minimum_required(VERSION 3.20)\n"
                    "project(fake_coast NONE)\n";
    if (blocking)
        s += "execute_process(COMMAND sh -c \"while [ ! -e '" + build_release.string() +
             "' ]; do sleep 0.1; done\")\n";
    s += "add_custom_target(coast ALL\n"
         "  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/fake_coast.sh ${CMAKE_BINARY_DIR}/coast\n"
         "  COMMAND chmod +x ${CMAKE_BINARY_DIR}/coast)\n";
    return s;
}

inline bool is_settled(const coastd::proto::job_summary& j)  { return j.bucket != "active"; }
inline bool is_running(const coastd::proto::job_summary& j)  { return j.status == "running"; }
inline bool is_building(const coastd::proto::job_summary& j) { return j.status == "building"; }

struct rig_options {
    bool blocking_build     = false;   // build waits for build_release
    int  slots              = 1;
    bool child_ignores_term = false;
    int  term_grace_seconds = 3;
};

// One daemon + its world. Owns the temp root, the throwaway repo, and the coastd
// process; every case drives the daemon exclusively through the real wire client.
struct rig {
    fs::path    root, data_root, repo, build_release, run_release;
    pid_t       daemon = 0;
    // OS-assigned, so a restart (start() after kill_daemon_hard) gets a NEW port —
    // every client call goes through open_conn(), which reads this.
    std::uint16_t tcp_port = 0;
    std::string   sha;
    bool          ok = false;   // construction succeeded (checked by the caller)

    explicit rig(rig_options o = {}) {
        root          = make_temp_dir();
        data_root     = root / "data";
        repo          = root / "repo";
        build_release = root / "build_release";
        run_release   = root / "run_release";
        fs::create_directories(data_root);
        fs::create_directories(repo);

        { std::ofstream(repo / "CMakeLists.txt") << cmake_lists(o.blocking_build, build_release); }
        { std::ofstream(repo / "fake_coast.sh")  << fake_coast; }
        fs::permissions(repo / "fake_coast.sh",
                        fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
        if (!shell_ok("git -C '" + repo.string() + "' init -q") ||
            !shell_ok("git -C '" + repo.string() + "' add -A") ||
            !shell_ok("git -C '" + repo.string() +
                      "' -c user.email=t@t -c user.name=t commit -q -m init") ||
            !shell_ok("git -C '" + repo.string() + "' rev-parse HEAD > '" +
                      (root / "sha.txt").string() + "'"))
            return;
        { std::ifstream in(root / "sha.txt"); std::getline(in, sha); }
        if (sha.size() != 40) return;

        { std::ofstream(data_root / "good.json") << "{\"seed\":1}\n"; }
        { std::ofstream(data_root / "coastd.json")
              << "{\"slots\":" << o.slots << ",\"term_grace_seconds\":" << o.term_grace_seconds
              << ",\"fetch_enabled\":false}\n"; }

        // The fake child learns its release file from the environment it inherits
        // through the daemon.
        ::setenv("COASTD_FAKE_RELEASE", run_release.c_str(), 1);
        if (o.child_ignores_term) ::setenv("COASTD_FAKE_IGNORE_TERM", "1", 1);
        else                      ::unsetenv("COASTD_FAKE_IGNORE_TERM");
        ok = start();
    }

    ~rig() {
        release_build();
        release_run();
        stop();
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    void release_build() const { std::ofstream(build_release) << "go"; }
    void release_run() const { std::ofstream(run_release) << "go"; }

    // The daemon's log is APPENDED to across restarts and dumped by log_tail() when
    // a case fails. In a fault-injection harness this is not a nicety: the whole
    // point is what the daemon did with a fault, and a bare "the flag never
    // appeared" tells you nothing about which of the ten steps in between broke.
    fs::path log_path() const { return root / "daemon.log"; }

    bool start() {
        const char* argv[] = {COASTD_BIN,
                              "--data-root", data_root.c_str(),
                              "--repo",      repo.c_str(),
                              "--tcp-port",  "0",   // OS-assigned: see test_daemon.h
                              "--udp-port",  "0",
                              nullptr};
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, log_path().c_str(),
                                         O_CREAT | O_WRONLY | O_APPEND, 0644);
        posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO);
        const int rc = ::posix_spawn(&daemon, COASTD_BIN, &fa, nullptr,
                                     const_cast<char* const*>(argv), environ);
        posix_spawn_file_actions_destroy(&fa);
        if (rc != 0) return false;
        // Matches on the pid, so a port file the previous (SIGKILLed) daemon left
        // behind is never mistaken for this one's.
        const auto ports = coastd_test::wait_for_ports(data_root, daemon);
        if (!ports) return false;
        tcp_port = ports->tcp;
        return true;
    }

    std::string log_tail(int lines = 40) const {
        std::ifstream in(log_path());
        std::vector<std::string> all;
        for (std::string l; std::getline(in, l);) all.push_back(l);
        std::string out;
        for (std::size_t i = all.size() > std::size_t(lines) ? all.size() - lines : 0;
             i < all.size(); ++i)
            out += all[i] + "\n";
        return out;
    }

    // SIGKILL: the crash the recovery model is written for. No signal handler runs,
    // nothing is flushed, and the running child is left an orphan — on a test box
    // there is no cgroup to reap it, which is precisely the case the boot pid-guard
    // exists to clean up.
    void kill_daemon_hard() {
        if (daemon <= 0) return;
        ::kill(daemon, SIGKILL);
        int st = 0;
        ::waitpid(daemon, &st, 0);
        daemon = 0;
    }

    void stop() {
        if (daemon <= 0) return;
        ::kill(daemon, SIGTERM);
        int st = 0;
        ::waitpid(daemon, &st, 0);
        daemon = 0;
    }

    // Budgets are the stall limit: this is our own daemon (CLAUDE.md "Testing").
    std::optional<cc::connection> open_conn() const {
        cc::endpoint ep;
        ep.host = "localhost";
        ep.port = tcp_port;
        std::string err;
        return cc::connection::open(ep, coast_test::default_stall, err, coast_test::default_stall);
    }

    std::optional<std::string> submit(const std::string& label, std::uint64_t steps = 1000) const {
        auto c = open_conn();
        if (!c) return std::nullopt;
        std::string err;
        cc::submit_args a;
        a.ref = "HEAD";
        if (!cc::load_config_file(a, data_root / "good.json", err)) return std::nullopt;
        a.label        = label;
        a.steps_target = steps;
        return c->submit(a, err);
    }

    std::optional<coastd::proto::job_summary> job(const std::string& id) const {
        auto c = open_conn();
        if (!c) return std::nullopt;
        std::string err;
        auto l = c->list(err);
        if (!l) return std::nullopt;
        for (auto& j : *l) if (j.id == id) return j;
        return std::nullopt;
    }

    // How often the job's child has been launched: coastd writes one
    // <run_dir>/logs/run_* per launch. An observable the daemon already leaves.
    int launches(const std::string& id) const {
        std::error_code ec;
        int n = 0;
        for (const auto& de : fs::directory_iterator(data_root / "runs" / id / "logs", ec))
            if (de.path().filename().string().starts_with("run_")) ++n;
        return n;
    }

    // The heartbeat of ONE job: everything observable about it. It changes on
    // every state transition, step report, checkpoint and launch — and stands
    // still when the job is stuck, however busy the daemon itself is. (The
    // daemon's own pass counter would keep ticking on its timer and hide a stuck
    // job.)
    std::string fingerprint(const std::string& id) const {
        const auto j = job(id);
        if (!j) return "?";
        return j->bucket + '|' + j->status + '|' + std::to_string(j->steps_done) + '|' +
               std::to_string(j->live_step) + '|' + j->checkpoint + '|' +
               (j->cancel_requested ? "c" : "-") + '|' + j->failure_reason + '|' +
               std::to_string(launches(id));
    }

    // Wait until pred(job) holds — or the job has settled, or it has been
    // launched more than `max_launches` times; either of the last two ends the
    // wait too, so neither a job filed early nor a relaunch loop is waited on
    // forever. The job's own fingerprint is the heartbeat: a job that is stuck
    // ends the wait after the stall limit. Returns whether pred holds.
    template <class Pred>
    bool wait_until(const std::string& id, Pred pred, int max_launches = 1) const {
        std::optional<coastd::proto::job_summary> last;
        (void)coast_test::await_progress(
            [&] {
                last = job(id);
                return last && (pred(*last) || is_settled(*last) || launches(id) > max_launches);
            },
            [&] { return fingerprint(id); }, coast_test::default_stall,
            std::chrono::milliseconds{50});   // each poll is network round-trips
        return last && pred(*last) && launches(id) <= max_launches;
    }

    // The job is in the state `pred` names (building, running — the rig holds
    // those until their release, so they are always observed).
    template <class Pred>
    bool reaches(const std::string& id, Pred pred, int max_launches = 1) const {
        return wait_until(id, pred, max_launches);
    }

    // The job left active/, and into `bucket`.
    bool ends_in(const std::string& id, const std::string& bucket, int max_launches = 1) const {
        return wait_until(id, [&](const auto& j) { return j.bucket == bucket; }, max_launches);
    }
};


// A client that dies mid-conversation: it opens a real connection, sends a
// PARTIAL request line (so the daemon is mid-read), and is then SIGKILLed — the
// realistic client fault, not a clean close. The client touches `sent` after the
// partial line is written, and is killed only then: established, not slept for.
inline bool kill_a_client_mid_request(std::uint16_t tcp_port, const fs::path& sent) {
    pid_t pid = 0;
    const std::string script = "exec 3<>/dev/tcp/127.0.0.1/" + std::to_string(tcp_port) +
                               "; printf '{\"cmd\":\"li' >&3; touch '" + sent.string() +
                               "'; sleep 3600";
    const char* argv[] = {"/usr/bin/env", "bash", "-c", script.c_str(), nullptr};
    if (::posix_spawn(&pid, "/usr/bin/env", nullptr, nullptr,
                      const_cast<char* const*>(argv), environ) != 0)
        return false;
    const bool landed = coast_test::await_condition([&] { return fs::exists(sent); });
    ::kill(pid, SIGKILL);
    int st = 0;
    ::waitpid(pid, &st, 0);
    return landed;
}

} // namespace coastd_fault

#endif // COAST_TESTS_COASTD_FAULT_RIG_H
