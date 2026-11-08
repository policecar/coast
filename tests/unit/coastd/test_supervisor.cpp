#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <variant>

#include "daemon/child_proc.h"   // is_zombie
#include "daemon/event_queue.h"
#include "daemon/supervisor.h"

#include "await_progress.h"

using namespace coastd;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

// supervisor::shutdown must SIGTERM every child (so its --save-on-term handler can
// write a final checkpoint), wait up to term_grace, then SIGKILL a straggler —
// NOT SIGKILL immediately. Real children that obey vs. ignore SIGTERM validate
// both arms of the escalation and that the wait is BOUNDED (the previous
// immediate-SIGKILL had no test at all).
//
// No wall-clock verdicts (CLAUDE.md "Testing"): which signal arrived is recorded
// by the child itself, "shutdown returns" is a hang check, and event counts are
// taken over supervisor poll passes, not over a time window. No case measures
// time.

// Arm the child's trap and wait for it, so a SIGTERM never races bash startup:
// children are spawned with POSIX_SPAWN_SETSIGDEF, so SIGTERM starts at its
// default (fatal) disposition until the trap line has run.
static std::string trap_script(const fs::path& armed, const fs::path& got_term, bool exit_on_term) {
    return "#!/usr/bin/env bash\n"
           "trap 'touch \"" + got_term.string() + "\"" + (exit_on_term ? "; exit 0" : "") + "' TERM\n"
           "touch \"" + armed.string() + "\"\n"
           "while true; do sleep 0.2; done\n";
}

// shutdown() in a helper thread; true once it has returned. A shutdown that
// never returns is a frozen one (await_condition's stall limit names it). The
// thread OWNS what the stuck call touches — the supervisor and the queue it
// pushes events into — so a failing case that unwinds while it is still blocked
// frees nothing under it; they are leaked instead. (A joining std::jthread would
// avoid the use-after-free too, but its join would block on the stuck call and
// turn the named failure into a hang.)
static std::shared_ptr<std::atomic<bool>> start_shutdown(std::shared_ptr<supervisor> sup,
                                                          std::shared_ptr<event_queue> q) {
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread([sup = std::move(sup), q = std::move(q), done] {
        sup->shutdown();
        done->store(true);
    }).detach();
    return done;
}

static bool shutdown_returns(std::shared_ptr<supervisor> sup, std::shared_ptr<event_queue> q) {
    const auto done = start_shutdown(std::move(sup), std::move(q));
    return coast_test::await_condition([&] { return done->load(); });
}

namespace {
struct scratch {
    fs::path root;
    scratch() {
        static std::atomic<int> ctr{0};
        root = fs::temp_directory_path() /
               ("coastd_super_" + std::to_string(::getpid()) + "_" +
                std::to_string(ctr.fetch_add(1)));
        fs::create_directories(root);
    }
    ~scratch() { std::error_code ec; fs::remove_all(root, ec); }
};

void write_script(const fs::path& p, const std::string& body) {
    { std::ofstream f(p); f << body; }
    fs::permissions(p, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
}

daemon_config cfg_at(const scratch& s, std::chrono::seconds grace) {
    daemon_config c;
    c.state_root     = s.root / "state";
    c.runs_root      = s.root / "runs";
    c.worktrees_root = s.root / "worktrees";
    c.repo_root      = ".";
    c.term_grace     = grace;
    return c;
}

job mk_job(const std::string& id, const scratch& s) {
    job j;
    j.id           = id;
    j.submitted_at = "2026-07-20T10:00:00Z";
    j.ref_symbolic = "HEAD";
    j.ref_sha      = "sha_test";
    j.run_dir      = (s.root / "runs" / id).string();
    j.steps_target = 100000;
    // plan_launch references logs/<...>.log but does not create the dir; the core's
    // spawn path prepares it. We spawn the supervisor directly, so create it here.
    std::error_code ec;
    fs::create_directories(fs::path(j.run_dir) / "logs", ec);
    return j;
}
} // namespace

TEST_CASE("supervisor::shutdown lets a well-behaved child exit on SIGTERM (prompt)") {
    scratch s;
    // An hour of grace: a shutdown that waited the grace out instead of
    // returning when the child exits would never return within the test — the
    // hang check below names it.
    const auto cfg      = cfg_at(s, 3600s);
    const auto bin      = s.root / "coast_obeys";
    const auto armed    = s.root / "coast_obeys.armed";
    const auto got_term = s.root / "coast_obeys.term";
    write_script(bin, trap_script(armed, got_term, /*exit_on_term=*/true));

    auto q   = std::make_shared<event_queue>();
    auto sup = std::make_shared<supervisor>(cfg, *q);
    sup->start();
    const job j = mk_job("job_obeys", s);
    std::ofstream(fs::path(j.run_dir) / "final_state.h5") << "stale";   // from an earlier launch
    auto info = sup->spawn(j, bin);
    REQUIRE(info.has_value());
    CHECK_FALSE(fs::exists(fs::path(j.run_dir) / "final_state.h5"));     // cleared before the launch
    REQUIRE(coast_test::await_condition([&] { return fs::exists(armed); }));

    REQUIRE(shutdown_returns(sup, q));   // the child's own exit ended the wait
    CHECK(fs::exists(got_term));      // it got SIGTERM (a SIGKILL never runs the trap)
    CHECK(::kill(info->pid, 0) != 0); // and is gone
}

// An unreadable <run_dir>/snapshots at the moment a child exits must still produce
// EXACTLY ONE child_exited, and must release the slot.
//
// This is the regression test for a deferred-erase bug: poll_once used to collect
// exited ids and erase them after the loop, with the throwing read_outcome in
// between. The throw was caught one level up, so the erase never ran and the reaped
// child stayed in the registry — and the next pass, finding waitpid returning ECHILD
// for an already-reaped pid, emitted a SECOND child_exited for the same launch.
// on_child_exited is not idempotent: the duplicate is an extra crash-loop strike,
// which can file a healthy job to failed/. Before the fix this test sees >= 2 events.
//
// chmod 000 is the injection, so it is meaningless as root (which bypasses the
// permission check entirely) — skip rather than pass vacuously.
TEST_CASE("supervisor emits exactly one child_exited when the outcome cannot be read") {
    if (::geteuid() == 0) {
        MESSAGE("running as root: chmod 000 does not deny access; skipping");
        return;
    }
    scratch s;
    const auto cfg = cfg_at(s, 5s);
    const auto bin = s.root / "coast_quick";
    write_script(bin, "#!/usr/bin/env bash\nexit 0\n");   // exits immediately

    const job j = mk_job("job_unreadable", s);
    const fs::path snaps = fs::path(j.run_dir) / "snapshots";
    fs::create_directories(snaps);
    // Deny the scan: the directory EXISTS (so this is not the ordinary
    // pre-first-checkpoint case) but cannot be read.
    fs::permissions(snaps, fs::perms::none);

    event_queue q;
    supervisor sup(cfg, q);
    sup.start();
    const auto info = sup.spawn(j, bin);
    REQUIRE(info.has_value());

    // Counted over poll PASSES, not a time window: wait for the reap, then for
    // further passes — the pass after the reap is where the duplicate used to be
    // emitted (waitpid -> ECHILD for the already-reaped pid).
    int exits = 0;
    const auto drain = [&] {
        event e;
        while (q.pop_until(std::chrono::steady_clock::now(), e)) {   // non-blocking
            if (std::holds_alternative<child_exited>(e)) {
                ++exits;
                CHECK(std::get<child_exited>(e).id == "job_unreadable");
            }
        }
    };
    const auto passes = [&] { return sup.poll_passes(); };
    // poll_passes ticks on the poll timer whether or not the child is reaped, so
    // it alone would wait forever on a supervisor that never reaps. Bound: once
    // the child has provably exited (a zombie, or already gone), the pass that
    // reaps it completes at most 2 passes later and pushes its event before the
    // counter ticks. So: `exited_at` is read AFTER seeing the exit (an upper
    // bound), `p` BEFORE draining (a pass counted in p has pushed its event, and
    // the drain sees it) — and p >= exited_at + 3 without the event is a failure
    // that no amount of descheduling can fake.
    std::optional<std::uint64_t> exited_at;
    REQUIRE(coast_test::await_progress(
        [&] {
            const auto p = sup.poll_passes();
            drain();
            if (exits >= 1) return true;
            if (!exited_at && (is_zombie(info->pid) || ::kill(info->pid, 0) != 0))
                exited_at = sup.poll_passes();
            return exited_at && p >= *exited_at + 3;
        },
        passes));
    REQUIRE(exits >= 1);   // reaped (not given up on)
    const auto after_reap = sup.poll_passes();
    REQUIRE(coast_test::await_progress(
        [&] { drain(); return sup.poll_passes() >= after_reap + 3; }, passes));
    drain();

    CHECK(exits == 1);                              // the whole point: not 0, and not 2+
    CHECK_FALSE(sup.is_supervising("job_unreadable"));   // slot released

    fs::permissions(snaps, fs::perms::owner_all);   // let scratch's remove_all work
}

// The escalation is checked in two cases, so that no verdict depends on how fast
// the child runs. The child's trap is armed before shutdown() signals it (the
// `armed` handshake): children are spawned with POSIX_SPAWN_SETSIGDEF, so SIGTERM
// starts at its default (fatal) disposition, and a SIGTERM that raced bash
// startup would kill the child outright (observed under load).
//
// It once was one case with a 1 s grace that also checked the trap's file. That
// check was a race: bash had to run the trap within the grace, before the
// SIGKILL, which a loaded machine need not do.

TEST_CASE("supervisor::shutdown SIGTERMs a child that ignores it and waits for it") {
    // An hour of grace, and a child that ignores SIGTERM and exits only when the
    // test releases it. The supervisor cannot reach its deadline, so the order is
    // fixed by construction: SIGTERM arrives (the trap's file), shutdown() is still
    // waiting (no SIGKILL instead), and once released the child exits on its own.
    scratch s;
    const auto cfg      = cfg_at(s, 3600s);
    const auto bin      = s.root / "coast_waits";
    const auto armed    = s.root / "coast_waits.armed";
    const auto got_term = s.root / "coast_waits.term";
    const auto release  = s.root / "coast_waits.release";
    const auto exited   = s.root / "coast_waits.exited";   // a SIGKILL never gets here
    write_script(bin, "#!/usr/bin/env bash\n"
                      "trap 'touch \"" + got_term.string() + "\"' TERM\n"
                      "touch \"" + armed.string() + "\"\n"
                      "while [ ! -e \"" + release.string() + "\" ]; do sleep 0.2; done\n"
                      "touch \"" + exited.string() + "\"\n");

    auto q   = std::make_shared<event_queue>();
    auto sup = std::make_shared<supervisor>(cfg, *q);
    sup->start();
    auto info = sup->spawn(mk_job("job_waits", s), bin);
    REQUIRE(info.has_value());
    const pid_t pid = info->pid;
    REQUIRE(coast_test::await_condition([&] { return fs::exists(armed); }));

    const auto done = start_shutdown(sup, q);
    REQUIRE(coast_test::await_condition([&] { return fs::exists(got_term); }));   // SIGTERM ...
    CHECK_FALSE(done->load());   // ... and shutdown() waits, the child is still running

    { std::ofstream(release) << "go"; }
    CHECK(coast_test::await_condition([&] { return done->load(); }));   // the child's exit ends it
    CHECK(fs::exists(exited));   // it exited on its own, not by SIGKILL
    CHECK(::kill(pid, 0) != 0);  // reaped
}

TEST_CASE("supervisor::shutdown SIGKILLs a child that outlives term_grace") {
    // A child that ignores SIGTERM and is never released: only the SIGKILL after
    // the grace can end it. Without the escalation shutdown() never returns, and
    // the hang check fails the case by name. The 1 s grace is the supervisor's
    // setting, not a verdict: load can delay the SIGKILL, never fail the case.
    scratch s;
    const auto cfg      = cfg_at(s, 1s);   // term_grace is whole seconds
    const auto bin      = s.root / "coast_ignores";
    const auto armed    = s.root / "coast_ignores.armed";
    const auto got_term = s.root / "coast_ignores.term";
    write_script(bin, trap_script(armed, got_term, /*exit_on_term=*/false));

    auto q   = std::make_shared<event_queue>();
    auto sup = std::make_shared<supervisor>(cfg, *q);
    sup->start();
    auto info = sup->spawn(mk_job("job_ignores", s), bin);
    REQUIRE(info.has_value());
    const pid_t pid = info->pid;
    REQUIRE(coast_test::await_condition([&] { return fs::exists(armed); }));

    REQUIRE(shutdown_returns(sup, q));   // only the SIGKILL can end this child
    CHECK(::kill(pid, 0) != 0);          // killed and reaped: the pid is gone
}
