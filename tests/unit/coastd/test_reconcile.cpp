#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <csignal>
#include <string>
#include <variant>
#include <vector>

#include "daemon/reconcile.h"

using namespace coastd;

namespace {

// A job that is fully ready to run: ref resolved, worktree assumed present by
// the reality the test supplies. Individual cases mutate what they exercise.
job ready_job(std::string id, std::string sha = "sha_a") {
    job j;
    j.id           = std::move(id);
    j.submitted_at = "2026-07-20T10:00:00Z";
    j.ref_symbolic = "feat/x";
    j.ref_sha      = std::move(sha);
    j.run_dir      = "/runs/" + j.id;
    j.steps_target = 1000;
    j.steps_done   = 0;
    return j;
}

daemon_config cfg1() {
    daemon_config c;
    c.slots            = 1;
    c.worktree_cooloff = std::chrono::seconds{6 * 3600};
    return c;
}

reality rt(sys_time now = sys_time{}) {
    reality r;
    r.now = now;
    return r;
}

// A fully-built, runnable worktree: directory on disk AND the .coastd-ready
// sentinel present. Readiness (spawn eligibility) keys off builds_ready; a bare
// worktrees_on_disk entry means "directory exists, build not finished".
void mark_ready(reality& r, const std::string& sha) {
    r.worktrees_on_disk.insert(sha);
    r.builds_ready.insert(sha);
}

template <class T>
int count_of(const std::vector<action>& as) {
    int n = 0;
    for (const auto& a : as) if (std::holds_alternative<T>(a)) ++n;
    return n;
}
template <class T>
const T* first_of(const std::vector<action>& as) {
    for (const auto& a : as) if (const auto* p = std::get_if<T>(&a)) return p;
    return nullptr;
}

} // namespace

TEST_CASE("reconcile: no jobs, no worktrees -> no actions") {
    std::vector<job> jobs;
    CHECK(reconcile(jobs, rt(), cfg1()).empty());
}

TEST_CASE("reconcile: unresolved ref -> resolve_ref (once, nothing else)") {
    auto j = ready_job("j1");
    j.ref_sha.clear();
    std::vector<job> jobs{j};
    auto as = reconcile(jobs, rt(), cfg1());
    REQUIRE(count_of<resolve_ref>(as) == 1);
    CHECK(first_of<resolve_ref>(as)->id == "j1");
    CHECK(as.size() == 1);  // no build/spawn until the SHA is known
}

TEST_CASE("reconcile: resolved ref, worktree absent -> build_worktree") {
    auto j = ready_job("j1", "sha_a");
    std::vector<job> jobs{j};
    auto as = reconcile(jobs, rt(), cfg1());  // reality has no worktrees on disk
    REQUIRE(count_of<build_worktree>(as) == 1);
    CHECK(first_of<build_worktree>(as)->sha == "sha_a");
    CHECK(count_of<spawn_child>(as) == 0);
}

// ---- regression: the build/spawn race (review 2026-07-21) ---------------------
// `git worktree add` creates the worktree DIRECTORY seconds into a minutes-long
// build, but the binary only lands when the build finishes. Keying readiness on
// directory presence spawned coast against a not-yet-existent binary, striking
// the job to failed/ mid-build (and, on a restart mid-build, never rebuilding).
// A worktree whose directory exists but whose .coastd-ready sentinel is absent
// must (re)build, NEVER spawn.
TEST_CASE("reconcile: worktree dir present but not ready -> build_worktree, not spawn") {
    auto j = ready_job("j1", "sha_a");
    std::vector<job> jobs{j};
    auto r = rt();
    r.worktrees_on_disk.insert("sha_a");   // directory exists (git worktree add ran)
    // builds_ready deliberately does NOT contain sha_a: the build is still running
    // (or was interrupted by a daemon restart before it finished).
    auto as = reconcile(jobs, r, cfg1());
    CHECK(count_of<build_worktree>(as) == 1);
    CHECK(count_of<spawn_child>(as) == 0);
}

TEST_CASE("reconcile: worktree present, dead child, slot free -> spawn_child") {
    auto j = ready_job("j1", "sha_a");
    std::vector<job> jobs{j};
    auto r = rt();
    mark_ready(r, "sha_a");
    auto as = reconcile(jobs, r, cfg1());
    REQUIRE(count_of<spawn_child>(as) == 1);
    CHECK(first_of<spawn_child>(as)->id == "j1");
}

TEST_CASE("reconcile: partial progress after crash -> spawn_child (resume)") {
    auto j = ready_job("j1", "sha_a");
    j.steps_done  = 500;                 // interrupted mid-run
    j.checkpoint  = "snapshots/x_500.h5";
    std::vector<job> jobs{j};
    auto r = rt();
    mark_ready(r, "sha_a");  // child NOT alive -> relaunch to resume
    auto as = reconcile(jobs, r, cfg1());
    CHECK(count_of<spawn_child>(as) == 1);
}

// Relaunch backoff. last_launch_ms is epoch ms and now persists across a daemon
// restart (it is written AND read back), so these three cases pin the whole
// branch: owed, paid, and the impossible-timestamp escape hatch.
TEST_CASE("reconcile: relaunch backoff") {
    constexpr std::uint64_t t0 = 1'753'000'000'000ULL;   // arbitrary epoch ms
    const auto at = [](std::uint64_t ms) { return sys_time{std::chrono::milliseconds{ms}}; };

    auto j = ready_job("j1", "sha_a");
    j.launch_failures = 1;                 // one strike -> base backoff (2 s) owed
    j.last_launch_ms  = t0;

    SUBCASE("inside the backoff window -> no spawn") {
        std::vector<job> jobs{j};
        auto r = rt(at(t0 + 1'000));       // 1 s after the launch, backoff is 2 s
        mark_ready(r, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("past the backoff window -> spawn") {
        std::vector<job> jobs{j};
        auto r = rt(at(t0 + 2'500));
        mark_ready(r, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 1);
    }
    SUBCASE("last_launch_ms in the FUTURE (clock skew) -> spawn, not parked") {
        // A restored job file from a host with a fast clock, or a wall clock
        // stepped backwards. Honouring it would park the job until real time
        // catches up — potentially hours. The launch_failures cap still bounds
        // the crash-loop, so spawning now is the safe reading.
        std::vector<job> jobs{j};
        auto r = rt(at(t0 - 3'600'000));   // "now" is an hour BEFORE the last launch
        mark_ready(r, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 1);
    }
}

// The core charges TWO no-progress counters on mutually exclusive branches, and the
// backoff is owed on either. This case pins the one that used to be unspaced:
// no_checkpoint_launches (ran real steps, killed before its first checkpoint) fired
// a relaunch every scan pass, burning all max_launch_failures strikes in ~5 s.
TEST_CASE("reconcile: backoff is owed on no_checkpoint_launches too") {
    constexpr std::uint64_t t0 = 1'753'000'000'000ULL;
    const auto at = [](std::uint64_t ms) { return sys_time{std::chrono::milliseconds{ms}}; };

    auto j = ready_job("j1", "sha_a");
    j.launch_failures       = 0;           // NOT a crash-loop: it ran real steps
    j.no_checkpoint_launches = 1;          // one strike -> base backoff (2 s) owed
    j.last_launch_ms        = t0;

    SUBCASE("inside the window -> no spawn") {
        std::vector<job> jobs{j};
        auto r = rt(at(t0 + 1'000));
        mark_ready(r, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("past the window -> spawn") {
        std::vector<job> jobs{j};
        auto r = rt(at(t0 + 2'500));
        mark_ready(r, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 1);
    }
    SUBCASE("the larger of the two counters sets the spacing") {
        // 3 no-checkpoint strikes -> base << 2 == 8 s, not the 2 s that
        // launch_failures==1 alone would buy.
        j.launch_failures        = 1;
        j.no_checkpoint_launches = 3;
        std::vector<job> jobs{j};
        auto r5 = rt(at(t0 + 5'000));
        mark_ready(r5, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r5, cfg1())) == 0);
        auto r9 = rt(at(t0 + 9'000));
        mark_ready(r9, "sha_a");
        CHECK(count_of<spawn_child>(reconcile(jobs, r9, cfg1())) == 1);
    }
}

TEST_CASE("reconcile: live child -> no spawn") {
    auto j = ready_job("j1", "sha_a");
    std::vector<job> jobs{j};
    auto r = rt();
    mark_ready(r, "sha_a");
    r.child_alive["j1"] = true;
    r.running_count     = 1;
    auto as = reconcile(jobs, r, cfg1());
    CHECK(count_of<spawn_child>(as) == 0);
    CHECK(as.empty());
}

TEST_CASE("reconcile: target reached, child dead -> move_bucket done") {
    auto j = ready_job("j1", "sha_a");
    j.steps_done = 1000;  // == target
    std::vector<job> jobs{j};
    auto r = rt();
    mark_ready(r, "sha_a");
    auto as = reconcile(jobs, r, cfg1());
    REQUIRE(count_of<move_bucket>(as) == 1);
    CHECK(first_of<move_bucket>(as)->to == bucket::done);
    CHECK(count_of<spawn_child>(as) == 0);
}

TEST_CASE("reconcile: cancel + live child -> kill_child SIGTERM") {
    auto j = ready_job("j1", "sha_a");
    j.cancel_requested = true;
    std::vector<job> jobs{j};
    auto r = rt();
    mark_ready(r, "sha_a");
    r.child_alive["j1"] = true;
    r.running_count     = 1;
    auto as = reconcile(jobs, r, cfg1());
    REQUIRE(count_of<kill_child>(as) == 1);
    CHECK(first_of<kill_child>(as)->signal == SIGTERM);
    CHECK(count_of<move_bucket>(as) == 0);
}

TEST_CASE("reconcile: cancel + dead child -> move_bucket cancelled") {
    auto j = ready_job("j1", "sha_a");
    j.cancel_requested = true;
    std::vector<job> jobs{j};
    auto as = reconcile(jobs, rt(), cfg1());
    REQUIRE(count_of<move_bucket>(as) == 1);
    CHECK(first_of<move_bucket>(as)->to == bucket::cancelled);
}

TEST_CASE("reconcile: recorded failure, dead child -> move_bucket failed") {
    auto j = ready_job("j1", "sha_a");
    j.failure_reason = "build failed: cmake error";
    std::vector<job> jobs{j};
    auto as = reconcile(jobs, rt(), cfg1());
    REQUIRE(count_of<move_bucket>(as) == 1);
    CHECK(first_of<move_bucket>(as)->to == bucket::failed);
}

TEST_CASE("reconcile: slot cap — 3 ready, 1 slot -> exactly 1 spawn, highest priority") {
    auto a = ready_job("a", "sha_a"); a.priority = 0;
    auto b = ready_job("b", "sha_a"); b.priority = 5;   // should win
    auto c = ready_job("c", "sha_a"); c.priority = 2;
    std::vector<job> jobs{a, b, c};
    auto r = rt();
    mark_ready(r, "sha_a");
    auto cfg = cfg1();  // slots = 1
    auto as = reconcile(jobs, r, cfg);
    REQUIRE(count_of<spawn_child>(as) == 1);
    CHECK(first_of<spawn_child>(as)->id == "b");
}

TEST_CASE("reconcile: FIFO tiebreak at equal priority") {
    auto a = ready_job("a", "sha_a"); a.submitted_at = "2026-07-20T10:00:02Z";
    auto b = ready_job("b", "sha_a"); b.submitted_at = "2026-07-20T10:00:01Z"; // earlier
    std::vector<job> jobs{a, b};
    auto r = rt();
    mark_ready(r, "sha_a");
    auto as = reconcile(jobs, r, cfg1());  // slots = 1
    REQUIRE(count_of<spawn_child>(as) == 1);
    CHECK(first_of<spawn_child>(as)->id == "b");
}

TEST_CASE("reconcile: all slots busy -> no spawn") {
    auto j = ready_job("j1", "sha_a");
    std::vector<job> jobs{j};
    auto r = rt();
    mark_ready(r, "sha_a");
    r.running_count = 1;  // slots == 1, already full (a different job runs)
    auto as = reconcile(jobs, r, cfg1());
    CHECK(count_of<spawn_child>(as) == 0);
}

TEST_CASE("reconcile: GC an idle, unreferenced worktree past cool-off") {
    std::vector<job> jobs;  // no job references sha_old
    auto now = sys_time{} + std::chrono::hours{100};
    auto r = rt(now);
    r.worktrees_on_disk.insert("sha_old");
    r.worktree_idle_since["sha_old"] = now - std::chrono::hours{7};  // > 6h cool-off
    auto as = reconcile(jobs, r, cfg1());
    REQUIRE(count_of<gc_worktree>(as) == 1);
    CHECK(first_of<gc_worktree>(as)->sha == "sha_old");
}

TEST_CASE("reconcile: do NOT GC within cool-off") {
    std::vector<job> jobs;
    auto now = sys_time{} + std::chrono::hours{100};
    auto r = rt(now);
    r.worktrees_on_disk.insert("sha_old");
    r.worktree_idle_since["sha_old"] = now - std::chrono::hours{5};  // < 6h
    auto as = reconcile(jobs, r, cfg1());
    CHECK(count_of<gc_worktree>(as) == 0);
}

TEST_CASE("reconcile: do NOT GC a referenced worktree even if idle") {
    auto j = ready_job("j1", "sha_a");   // references sha_a
    std::vector<job> jobs{j};
    auto now = sys_time{} + std::chrono::hours{100};
    auto r = rt(now);
    r.worktrees_on_disk.insert("sha_a");
    r.worktree_idle_since["sha_a"] = now - std::chrono::hours{99};  // very idle, but referenced
    auto as = reconcile(jobs, r, cfg1());
    CHECK(count_of<gc_worktree>(as) == 0);
}

// ---- disk pressure: gate STARTING work, never finishing or reclaiming it -----
// reality.disk_pressure is set by rescan when free space on any daemon root falls
// below cfg.min_free_disk_bytes. The policy has to be exact about which actions it
// suppresses: suppressing too much (GC, terminal moves) removes the daemon's only
// way out of the condition.
TEST_CASE("reconcile: disk pressure") {
    SUBCASE("no spawn while under pressure") {
        auto j = ready_job("j1", "sha_a");
        std::vector<job> jobs{j};
        auto r = rt();
        mark_ready(r, "sha_a");
        r.disk_pressure = true;
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("no build while under pressure") {
        auto j = ready_job("j1", "sha_a");   // ref resolved, worktree absent
        std::vector<job> jobs{j};
        auto r = rt();
        r.disk_pressure = true;
        CHECK(count_of<build_worktree>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("a RUNNING child is not killed") {
        auto j = ready_job("j1", "sha_a");
        std::vector<job> jobs{j};
        auto r = rt();
        mark_ready(r, "sha_a");
        r.child_alive["j1"] = true;
        r.running_count     = 1;
        r.disk_pressure     = true;
        CHECK(count_of<kill_child>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("GC still runs — it is the only action that frees space") {
        std::vector<job> jobs;
        auto now = sys_time{} + std::chrono::hours{100};
        auto r = rt(now);
        r.worktrees_on_disk.insert("sha_old");
        r.worktree_idle_since["sha_old"] = now - std::chrono::hours{7};
        r.disk_pressure = true;
        CHECK(count_of<gc_worktree>(reconcile(jobs, r, cfg1())) == 1);
    }
    SUBCASE("a finished job is still filed to done/") {
        auto j = ready_job("j1", "sha_a");
        j.steps_done = j.steps_target;       // complete, child gone
        std::vector<job> jobs{j};
        auto r = rt();
        mark_ready(r, "sha_a");
        r.disk_pressure = true;
        REQUIRE(count_of<move_bucket>(reconcile(jobs, r, cfg1())) == 1);
        CHECK(first_of<move_bucket>(reconcile(jobs, r, cfg1()))->to == bucket::done);
    }
    SUBCASE("a cancel is still honoured") {
        auto j = ready_job("j1", "sha_a");
        j.cancel_requested = true;
        std::vector<job> jobs{j};
        auto r = rt();
        mark_ready(r, "sha_a");
        r.disk_pressure = true;
        REQUIRE(count_of<move_bucket>(reconcile(jobs, r, cfg1())) == 1);
        CHECK(first_of<move_bucket>(reconcile(jobs, r, cfg1()))->to == bucket::cancelled);
    }
    SUBCASE("pressure lifted -> the held job spawns immediately") {
        auto j = ready_job("j1", "sha_a");
        std::vector<job> jobs{j};
        auto r = rt();
        mark_ready(r, "sha_a");
        r.disk_pressure = false;
        CHECK(count_of<spawn_child>(reconcile(jobs, r, cfg1())) == 1);
    }
}

// ---- completion is terminal, and independent of build readiness --------------
// Filing a finished job must not require a worktree to run it in. With the done
// check below the build gate, a completed job whose tree was GC'd emitted
// build_worktree — a full rebuild purely to be filed — and if that build failed
// max_build_failures times the core stamped failure_reason on it, filing a
// SUCCESSFULLY COMPLETED job to failed/. Under disk pressure it emitted nothing at
// all and sat in active/ forever.
TEST_CASE("reconcile: a completed job is filed to done/ even with no build ready") {
    SUBCASE("no worktree at all -> done, not build") {
        auto j = ready_job("j1", "sha_a");
        j.steps_done = j.steps_target;               // complete
        std::vector<job> jobs{j};
        auto r = rt();                               // builds_ready EMPTY
        const auto as = reconcile(jobs, r, cfg1());
        CHECK(count_of<build_worktree>(as) == 0);
        REQUIRE(count_of<move_bucket>(as) == 1);
        CHECK(first_of<move_bucket>(as)->to == bucket::done);
    }
    SUBCASE("under disk pressure it is still filed") {
        auto j = ready_job("j1", "sha_a");
        j.steps_done = j.steps_target;
        std::vector<job> jobs{j};
        auto r = rt();
        r.disk_pressure = true;
        REQUIRE(count_of<move_bucket>(reconcile(jobs, r, cfg1())) == 1);
    }
    SUBCASE("with an unreadable worktrees root it is still filed") {
        auto j = ready_job("j1", "sha_a");
        j.steps_done = j.steps_target;
        std::vector<job> jobs{j};
        auto r = rt();
        r.worktrees_unreadable = true;
        REQUIRE(count_of<move_bucket>(reconcile(jobs, r, cfg1())) == 1);
    }
    SUBCASE("a RUNNING child still owns the run dir — not filed yet") {
        auto j = ready_job("j1", "sha_a");
        j.steps_done = j.steps_target;
        std::vector<job> jobs{j};
        auto r = rt();
        r.child_alive["j1"] = true;
        r.running_count     = 1;
        CHECK(count_of<move_bucket>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("an UNBOUNDED job (steps_target 0) never completes") {
        auto j = ready_job("j1", "sha_a");
        j.steps_target = 0;
        j.steps_done   = 1'000'000;
        std::vector<job> jobs{j};
        auto r = rt();
        mark_ready(r, "sha_a");
        CHECK(count_of<move_bucket>(reconcile(jobs, r, cfg1())) == 0);
    }
}

// ---- an unreadable worktrees root is UNKNOWN, not empty ----------------------
// rescan sets reality.worktrees_unreadable when the root exists but cannot be
// scanned. Without the flag that state is indistinguishable from "nothing is built
// yet", and reconcile answers it by rebuilding — every pass, since the condition
// persists — until max_build_failures stamps a terminal failure_reason on every job
// on the SHA. The whole queue is then durably lost to one transient fd exhaustion.
// So the correct answer is the disk-pressure answer: hold, no strike.
TEST_CASE("reconcile: an unreadable worktrees root holds builds instead of forcing them") {
    SUBCASE("no build is emitted while the root is unreadable") {
        auto j = ready_job("j1", "sha_a");   // ref resolved, nothing visible on disk
        std::vector<job> jobs{j};
        auto r = rt();
        r.worktrees_unreadable = true;
        CHECK(count_of<build_worktree>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("the job is NOT failed or moved — it just waits") {
        auto j = ready_job("j1", "sha_a");
        std::vector<job> jobs{j};
        auto r = rt();
        r.worktrees_unreadable = true;
        const auto as = reconcile(jobs, r, cfg1());
        CHECK(count_of<move_bucket>(as) == 0);
        CHECK(count_of<kill_child>(as) == 0);
    }
    SUBCASE("a RUNNING child is left alone") {
        auto j = ready_job("j1", "sha_a");
        std::vector<job> jobs{j};
        auto r = rt();
        r.child_alive["j1"] = true;
        r.running_count     = 1;
        r.worktrees_unreadable = true;
        CHECK(count_of<kill_child>(reconcile(jobs, r, cfg1())) == 0);
    }
    SUBCASE("readable again -> the held job builds immediately") {
        auto j = ready_job("j1", "sha_a");
        std::vector<job> jobs{j};
        auto r = rt();
        r.worktrees_unreadable = false;
        CHECK(count_of<build_worktree>(reconcile(jobs, r, cfg1())) == 1);
    }
}

// ---- regression: failure is bucketed before cancel (review 2026-07-21) --------
// A job that recorded a terminal failure AND is then cancel_requested must be
// filed under failed/, not cancelled/ — otherwise the failure_reason is lost
// where no operator looks for it.
TEST_CASE("reconcile: recorded failure wins over a later cancel") {
    auto j = ready_job("j1", "sha_a");
    j.failure_reason   = "build failed";
    j.cancel_requested = true;
    std::vector<job> jobs{j};
    auto r = rt();                       // no live child
    mark_ready(r, "sha_a");
    auto as = reconcile(jobs, r, cfg1());
    const auto* mv = first_of<move_bucket>(as);
    REQUIRE(mv != nullptr);
    CHECK(mv->to == bucket::failed);     // NOT bucket::cancelled
    CHECK(count_of<move_bucket>(as) == 1);
}

// A failed job that still has a live child is SIGKILLed (hard), a cancel-only job
// gets SIGTERM (graceful) — the differentiation must follow the failure-first order.
TEST_CASE("reconcile: failed+running -> SIGKILL, not the cancel SIGTERM") {
    auto j = ready_job("j1", "sha_a");
    j.failure_reason   = "crash-loop";
    j.cancel_requested = true;
    std::vector<job> jobs{j};
    auto r = rt();
    r.child_alive["j1"]  = true;         // a live child
    r.running_count      = 1;
    auto as = reconcile(jobs, r, cfg1());
    const auto* k = first_of<kill_child>(as);
    REQUIRE(k != nullptr);
    CHECK(k->signal == SIGKILL);
}
