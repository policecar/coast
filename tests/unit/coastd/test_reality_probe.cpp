#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>

#include "daemon/event_queue.h"
#include "daemon/reality_probe.h"
#include "daemon/supervisor.h"

using namespace coastd;
namespace fs = std::filesystem;

// rescan's disk probe. The reachable states are: measurable + above the floor,
// measurable + below it, check disabled, and unmeasurable. The last two are the
// ones that decide whether a misconfigured or unreadable root stalls the daemon
// forever, so they are tested explicitly rather than reasoned about.

namespace {
struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_probe_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

daemon_config cfg_at(const scratch& s) {
    daemon_config c;
    c.state_root     = s.root / "state";
    c.runs_root      = s.root / "runs";
    c.worktrees_root = s.root / "worktrees";
    fs::create_directories(c.state_root);
    fs::create_directories(c.runs_root);
    fs::create_directories(c.worktrees_root);
    return c;
}
} // namespace

TEST_CASE("rescan: disk probe") {
    scratch s;
    event_queue q;

    SUBCASE("a normal filesystem is measured and reports no pressure") {
        auto cfg = cfg_at(s);
        cfg.min_free_disk_bytes = 1;          // one byte: any real fs clears this
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK(r.disk_free_bytes > 0);         // the probe actually ran
        CHECK_FALSE(r.disk_pressure);
    }

    SUBCASE("an unreachable floor trips pressure") {
        auto cfg = cfg_at(s);
        cfg.min_free_disk_bytes = 1ULL << 60;  // 1 EiB — no test box has this
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK(r.disk_pressure);
        CHECK(r.disk_free_bytes > 0);          // still reports the real number
    }

    SUBCASE("0 disables the check even on a would-be-full disk") {
        auto cfg = cfg_at(s);
        cfg.min_free_disk_bytes = 0;
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK_FALSE(r.disk_pressure);
        CHECK(r.disk_free_bytes > 0);          // measured, just not judged
    }

    SUBCASE("roots that do not exist yet -> no pressure, not a stall") {
        // First boot (or a bad path): statvfs cannot answer. Reporting pressure
        // here would block every build and launch for as long as the mistake
        // lasts, with the daemon otherwise healthy — a self-inflicted outage.
        daemon_config cfg;
        cfg.state_root     = s.root / "nope" / "state";
        cfg.runs_root      = s.root / "nope" / "runs";
        cfg.worktrees_root = s.root / "nope" / "worktrees";
        cfg.min_free_disk_bytes = 1ULL << 60;   // even with an absurd floor
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK_FALSE(r.disk_pressure);
        CHECK(r.disk_free_bytes == 0);          // 0 == "never measured"
    }
}

// rescan's worktrees scan. worktrees_unreadable means "the world could not be
// seen", and reconcile answers it by holding every build — so a FALSE POSITIVE is
// a total queue stall, which makes the absent-root case the one that must be
// nailed down rather than reasoned about.
//
// It is also the bug this test was written for: is_directory(p, ec) SETS ec for a
// path that does not exist, so sharing one error_code between the root probe and
// the directory walk made every first boot (worktrees root not yet created) report
// an unreadable root. Nothing ever built, so the root was never created, so it
// never recovered. The whole fault matrix hung on it.
TEST_CASE("rescan: the worktrees scan distinguishes absent from unreadable") {
    scratch s;
    event_queue q;

    SUBCASE("an ABSENT worktrees root is ordinary first-boot state, not unreadable") {
        auto cfg = cfg_at(s);
        fs::remove_all(cfg.worktrees_root);      // never created yet
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK_FALSE(r.worktrees_unreadable);     // must NOT hold builds
        CHECK(r.builds_ready.empty());
    }

    SUBCASE("an empty worktrees root is likewise readable") {
        auto cfg = cfg_at(s);                    // created, nothing in it
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK_FALSE(r.worktrees_unreadable);
        CHECK(r.worktrees_on_disk.empty());
    }

    SUBCASE("a populated root is scanned and its ready marker seen") {
        auto cfg = cfg_at(s);
        fs::create_directories(cfg.worktrees_root / "a1b2c3d4e5f6");
        { std::ofstream(cfg.worktrees_root / "a1b2c3d4e5f6" / ".coastd-ready") << "ok\n"; }
        fs::create_directories(cfg.worktrees_root / "9876543210ab");   // built dir, no marker
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK_FALSE(r.worktrees_unreadable);
        CHECK(r.worktrees_on_disk.size() == 2);
        CHECK(r.builds_ready.count("a1b2c3d4e5f6") == 1);
        CHECK(r.builds_ready.count("9876543210ab") == 0);   // present but not ready
    }

    SUBCASE("an UNREADABLE root is reported, and reports nothing partial") {
        if (::geteuid() == 0) {
            MESSAGE("running as root: chmod 000 does not deny access; skipping");
            return;
        }
        auto cfg = cfg_at(s);
        fs::create_directories(cfg.worktrees_root / "a1b2c3d4e5f6");
        fs::permissions(cfg.worktrees_root, fs::perms::none);
        supervisor sup(cfg, q);
        const reality r = rescan(sup, cfg);
        CHECK(r.worktrees_unreadable);
        CHECK(r.worktrees_on_disk.empty());      // no partial view survives
        CHECK(r.builds_ready.empty());
        fs::permissions(cfg.worktrees_root, fs::perms::owner_all);   // let cleanup work
    }
}

// The cool-off clock. reconcile GCs an unreferenced worktree once
// (now - worktree_idle_since) exceeds worktree_cooloff, and reality_probe derives
// idle_since from the worktree DIRECTORY's mtime. That only means "idle since" if
// something refreshes it on use — the builder writes .coastd-ready into the dir
// last, so straight after a build the mtime is the BUILD time and never moves
// again. The core now touches the dir on spawn and on a terminal move; these
// cases pin the contract that makes that work.
TEST_CASE("rescan: worktree_idle_since tracks the directory mtime and is refreshable") {
    scratch s;
    event_queue q;
    auto cfg = cfg_at(s);
    const auto wt = cfg.worktrees_root / "a1b2c3d4e5f6";
    fs::create_directories(wt);
    { std::ofstream(wt / ".coastd-ready") << "ok\n"; }   // as the builder does, LAST

    supervisor sup(cfg, q);

    // Backdate the tree well past any plausible cool-off: this is the state a
    // long-lived SHA is in by the time its last job ends.
    const auto old_t = fs::file_time_type::clock::now() - std::chrono::hours(48);
    fs::last_write_time(wt, old_t);

    const reality before = rescan(sup, cfg);
    auto it = before.worktree_idle_since.find("a1b2c3d4e5f6");
    REQUIRE(it != before.worktree_idle_since.end());
    const auto stale = it->second;
    CHECK(std::chrono::system_clock::now() - stale > std::chrono::hours(24));

    // What core::touch_worktree does. If this did not move idle_since, the cool-off
    // could not be restarted at all and the fix would be inert.
    fs::last_write_time(wt, fs::file_time_type::clock::now());

    const reality after = rescan(sup, cfg);
    auto it2 = after.worktree_idle_since.find("a1b2c3d4e5f6");
    REQUIRE(it2 != after.worktree_idle_since.end());
    CHECK(it2->second > stale);
    CHECK(std::chrono::system_clock::now() - it2->second < std::chrono::minutes(5));

    // And the refresh must not disturb what the tree IS: still on disk, still ready.
    CHECK(after.builds_ready.count("a1b2c3d4e5f6") == 1);
    CHECK(after.worktrees_on_disk.count("a1b2c3d4e5f6") == 1);
}
