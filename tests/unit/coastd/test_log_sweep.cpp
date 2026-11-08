#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "daemon/log_sweep.h"

using namespace coastd;
namespace fs = std::filesystem;

// Log housekeeping: compress what is old, delete what is ancient, and — the part
// that actually matters — NEVER touch a log that is still being written. The
// active-file protection is mtime-based rather than state-based, so it is worth
// asserting directly: a rotation that moves a file out from under the supervisor's
// open fd is the classic way this feature breaks a tailer.

namespace {
struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_logsweep_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

daemon_config cfg_at(const scratch& s) {
    daemon_config c;
    c.runs_root      = s.root / "runs";
    c.worktrees_root = s.root / "worktrees";
    c.state_root     = s.root / "state";
    fs::create_directories(c.runs_root);
    fs::create_directories(c.worktrees_root);
    return c;
}

void write_file(const fs::path& p, const std::string& body) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << body;
}

// Backdate a file so the sweep sees it as idle.
void age(const fs::path& p, std::chrono::seconds by) {
    std::error_code ec;
    const auto now = fs::last_write_time(p, ec);
    fs::last_write_time(p, now - by, ec);
}
} // namespace

TEST_CASE("sweep_logs compresses idle run logs and leaves fresh ones alone") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.log_compress_after = std::chrono::seconds(3600);
    cfg.log_delete_after   = std::chrono::seconds(0);       // deletion disabled

    const fs::path old_log   = cfg.runs_root / "job_a" / "logs" / "run_0_0_0.log";
    const fs::path fresh_log = cfg.runs_root / "job_b" / "logs" / "run_0_0_0.log";
    write_file(old_log,   "old output\n");
    write_file(fresh_log, "still being written\n");
    age(old_log, std::chrono::seconds(7200));               // 2h idle, past the 1h threshold

    const sweep_stats st = sweep_logs(cfg);

    CHECK(st.compressed == 1);
    CHECK_FALSE(fs::exists(old_log));                       // gzip removed the original
    CHECK(fs::exists(fs::path(old_log.string() + ".gz")));
    // The whole point: a log with a recent mtime is a log something may still be
    // writing to. It must be untouched, and untouched means BOTH not compressed and
    // still present at its original path (the supervisor tails it by that fd).
    CHECK(fs::exists(fresh_log));
    CHECK_FALSE(fs::exists(fs::path(fresh_log.string() + ".gz")));
}

TEST_CASE("sweep_logs deletes ancient archives, keeps recent ones") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.log_compress_after = std::chrono::seconds(0);       // compression disabled
    cfg.log_delete_after   = std::chrono::seconds(3600);

    const fs::path ancient = cfg.runs_root / "job_a" / "logs" / "run_0_0_0.log.gz";
    const fs::path recent  = cfg.runs_root / "job_a" / "logs" / "run_1_0_0.log.gz";
    write_file(ancient, "gz");
    write_file(recent,  "gz");
    age(ancient, std::chrono::seconds(7200));

    const sweep_stats st = sweep_logs(cfg);

    CHECK(st.deleted == 1);
    CHECK_FALSE(fs::exists(ancient));
    CHECK(fs::exists(recent));
}

TEST_CASE("sweep_logs covers per-SHA build logs too") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.log_compress_after = std::chrono::seconds(3600);
    cfg.log_delete_after   = std::chrono::seconds(0);

    const fs::path build_log = cfg.worktrees_root / "abc1234" / "coastd-build.log";
    write_file(build_log, "cmake output\n");
    age(build_log, std::chrono::seconds(7200));

    CHECK(sweep_logs(cfg).compressed == 1);
    CHECK(fs::exists(fs::path(build_log.string() + ".gz")));
}

TEST_CASE("sweep_logs does nothing when both stages are disabled") {
    scratch s;
    auto cfg = cfg_at(s);
    cfg.log_compress_after = std::chrono::seconds(0);
    cfg.log_delete_after   = std::chrono::seconds(0);

    const fs::path log = cfg.runs_root / "job_a" / "logs" / "run_0_0_0.log";
    write_file(log, "x");
    age(log, std::chrono::seconds(100 * 24 * 3600));        // ancient, but policy is off

    const sweep_stats st = sweep_logs(cfg);
    CHECK(st.compressed == 0);
    CHECK(st.deleted == 0);
    CHECK(fs::exists(log));
}

TEST_CASE("sweep_logs tolerates absent roots (first boot)") {
    scratch s;
    daemon_config cfg;
    cfg.runs_root      = s.root / "nope" / "runs";
    cfg.worktrees_root = s.root / "nope" / "worktrees";
    const sweep_stats st = sweep_logs(cfg);      // must not throw; noexcept contract
    CHECK(st.compressed == 0);
    CHECK(st.deleted == 0);
    CHECK(st.failed == 0);
}

TEST_CASE("append_capped restarts an append-only log past its ceiling") {
    scratch s;
    const fs::path log = s.root / "coastd-build.log";
    const std::string chunk(4096, 'x');

    // Under the cap: content accumulates.
    append_capped(log, "step one", chunk, 1u << 20);
    const auto after_first = fs::file_size(log);
    append_capped(log, "step two", chunk, 1u << 20);
    CHECK(fs::file_size(log) > after_first);

    // Past the cap: restarted, so the file shrinks rather than growing without
    // bound. A crash-looping build is what reaches this in the real system.
    append_capped(log, "step three", chunk, 1024);
    CHECK(fs::file_size(log) < after_first * 2);
    std::ifstream in(log);
    std::string first_line;
    std::getline(in, first_line);
    CHECK(first_line.find("log restarted") != std::string::npos);
}
