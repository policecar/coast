// -----------------------------------------------------------------------------
// The M3 fault matrix, and the behaviour behind the latency criteria.
//
// MATRIX: {kill client, kill daemon} x {building, queued, running}. Every cell
// injects the fault against the REAL coastd binary driving REAL child processes
// and a REAL cmake build, then asserts the job reaches correct completion.
//
// The design's third fault column, "reboot", is deliberately OUT OF SCOPE here
// (decision 2026-07-26). Its daemon-visible content — every child gone, daemon
// restarts from the store — is exactly the kill-daemon column plus an orphan kill,
// which IS covered. What a reboot adds beyond that is systemd's cgroup reap, and
// that is systemd's behaviour rather than coastd's: testing it here would measure
// the test harness, not the daemon. It stays an operator-level check against the
// shipped unit file.
//
// No wall-clock verdicts (CLAUDE.md "Testing"): the fake build and the fake child
// block on release files (fault_rig.h), so a fault provably lands in the state it
// is meant for, and every wait is on the daemon's heartbeat. The latency NUMBERS
// the design asked for ("status stale <= seconds", "cancel latency: measure it")
// are measured by the manual benchmark bench_coastd_latency, not asserted here.
// -----------------------------------------------------------------------------
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include "daemon/child_proc.h"   // start_token_supported — the platform gate below
#include "fault_rig.h"

using namespace coastd_fault;

// ---------------------------------------------------------------------------
// PLATFORM GATE. Every case in this file drives a REAL coastd process, and the
// daemon refuses to start where start_token_supported() is false (main.cpp):
// without a recycle-safe start token its duplicate-run guard is inoperative, and
// shipping that silent-corruption path was declined (ADR-16). So off Linux these
// cannot pass, and they skip rather than hard-fail — macOS is a first-class dev
// platform here and a Mac dev running the documented `ctest` must not be handed
// red that means "wrong OS". Gated on the seam, not __linux__, so porting
// read_start_token() re-enables them automatically.
// ---------------------------------------------------------------------------
constexpr bool skip_off_linux = !coastd::start_token_supported();

namespace {
std::string submit_ok(const rig& r, const std::string& label) {
    const auto id = r.submit(label);
    REQUIRE_MESSAGE(id.has_value(), "submit failed\n" << r.log_tail());
    return *id;
}
} // namespace

// ---- kill client x {queued, building, running} -------------------------------
// A client is not part of the daemon's state. These cases prove that: the fault
// must leave both the daemon serving and the job's trajectory untouched.

TEST_CASE("fault: client killed mid-request while a job is BUILDING" * doctest::skip(skip_off_linux)) {
    rig r({.blocking_build = true});
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "cx-building");
    REQUIRE(r.reaches(id, is_building));   // held there until release_build()

    REQUIRE(kill_a_client_mid_request(r.tcp_port, r.root / "client_sent"));
    const auto j = r.job(id);
    REQUIRE(j.has_value());                   // the daemon still answers ...
    CHECK(j->status == "building");           // ... and the fault landed mid-build

    r.release_build();
    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done"), "job did not complete\n" << r.log_tail());
}

TEST_CASE("fault: client killed mid-request while a job is QUEUED" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    const std::string busy = submit_ok(r, "cx-busy");
    const std::string id   = submit_ok(r, "cx-queued");
    REQUIRE(r.reaches(busy, is_running));   // holds the one slot until release_run()
    // With the one slot taken by a child that cannot finish yet, the second job
    // is queued by construction.
    auto q = r.job(id);
    REQUIRE(q.has_value());
    CHECK(q->status != "running");

    REQUIRE(kill_a_client_mid_request(r.tcp_port, r.root / "client_sent"));

    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done"), "queued job never ran to completion\n" << r.log_tail());
}

TEST_CASE("fault: client killed mid-request while a job is RUNNING" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "cx-running");
    REQUIRE(r.reaches(id, is_running));     // runs until release_run()

    REQUIRE(kill_a_client_mid_request(r.tcp_port, r.root / "client_sent"));
    const auto j = r.job(id);
    REQUIRE(j.has_value());
    CHECK(j->status == "running");             // the fault landed mid-run

    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done"), "running job did not finish\n" << r.log_tail());
}

// ---- kill daemon x {queued, building, running} -------------------------------
// SIGKILL the daemon, restart it on the same roots, and require the job to reach
// correct completion with no operator action. This is the recovery model's whole
// claim, and "recovery == steady state" means the restarted daemon must re-derive
// what to do purely from the store plus the world.

TEST_CASE("fault: daemon SIGKILLed while a job is BUILDING" * doctest::skip(skip_off_linux)) {
    rig r({.blocking_build = true});
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "dx-building");
    REQUIRE(r.reaches(id, is_building));   // the build cannot finish before the kill

    r.kill_daemon_hard();
    // The interrupted build left a worktree WITHOUT the readiness sentinel. The
    // restarted daemon must treat that as "not runnable, rebuild" — not as a built
    // tree (which would spawn against a missing binary) and not as a hard failure.
    REQUIRE_MESSAGE(r.start(), "daemon did not restart\n" << r.log_tail());

    r.release_build();
    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done"),
                  "job did not complete after a daemon crash mid-build\n" << r.log_tail());
}

TEST_CASE("fault: daemon SIGKILLed while a job is QUEUED" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    const std::string busy = submit_ok(r, "dx-busy");
    const std::string id   = submit_ok(r, "dx-queued");
    REQUIRE(r.reaches(busy, is_running));   // id is queued behind it by construction

    r.kill_daemon_hard();
    REQUIRE_MESSAGE(r.start(), "daemon did not restart\n" << r.log_tail());

    // Both must finish: the queued one is pure store state, the running one had its
    // orphaned child killed by the boot pid-guard and is relaunched.
    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done"),   "queued job lost across the crash\n" << r.log_tail());
    // The busy job's orphan was killed at boot and relaunched once: 2 launches.
    CHECK_MESSAGE(r.ends_in(busy, "done", /*max_launches=*/2),
                  "running job lost across the crash\n" << r.log_tail());
}

TEST_CASE("fault: daemon SIGKILLed while a job is RUNNING" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "dx-running");
    REQUIRE(r.reaches(id, is_running));     // cannot finish before the kill

    // The child is now an orphan: the daemon that spawned it is gone and no cgroup
    // reap exists here. The restarted daemon must SIGKILL it via the pid+start-token
    // guard before relaunching — otherwise two children write one run dir.
    r.kill_daemon_hard();
    REQUIRE_MESSAGE(r.start(), "daemon did not restart\n" << r.log_tail());

    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done", /*max_launches=*/2),   // the relaunch after the guard
                  "job did not complete after a daemon crash mid-run\n" << r.log_tail());

    // Exactly one final state, written by whichever child finished — not two
    // children racing the same file.
    auto j = r.job(id);
    REQUIRE(j.has_value());
    CHECK(j->steps_done == j->steps_target);
}

// ---- the behaviour behind the latency criteria -------------------------------
// Discover §5 asked for "status stale <= seconds", "child-death detection <=
// seconds" and a measured cancel latency. The numbers are bench_coastd_latency's
// job; what is asserted here is the behaviour they are numbers OF.

TEST_CASE("status: a submitted job is visible to the next client at once" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    // submit returns after the core has stored the job (the RPC resolves on the
    // core thread), so the very next list — on a NEW connection — must show it.
    for (int i = 0; i < 3; ++i) {
        const std::string id = submit_ok(r, "vis" + std::to_string(i));
        CHECK(r.job(id).has_value());
    }
}

TEST_CASE("status: a killed child is detected and the job relaunched to completion" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "death");
    REQUIRE(r.reaches(id, is_running));     // runs until released

    // SIGKILL the child (the pid is server-side only, so kill by parentage).
    REQUIRE(shell_ok("pkill -9 -P " + std::to_string(r.daemon) + " -f coast || true"));
    // Noticed = it stopped reporting the dead child as running, or it has already
    // relaunched (a second launch log) before a poll caught the gap.
    CHECK_MESSAGE(r.wait_until(id,
                               [&](const auto& j) { return j.status != "running" || r.launches(id) >= 2; },
                               /*max_launches=*/2),
                  "the daemon never noticed the dead child\n" << r.log_tail());

    r.release_run();
    CHECK_MESSAGE(r.ends_in(id, "done", /*max_launches=*/2),
                  "the job was not relaunched to completion\n" << r.log_tail());
}

// Cancel has TWO observable stages: the request is recorded (and visible) at
// once, and the job settles once its child is gone. How long the second takes
// depends on the child: one that exits on SIGTERM goes at once, one that ignores
// it (a real coast writing its final checkpoint under --save-on-term) is
// SIGKILLed after term_grace.
TEST_CASE("cancel: an obedient child settles into cancelled/" * doctest::skip(skip_off_linux)) {
    rig r;
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "cq");
    REQUIRE(r.reaches(id, is_running));
    auto c = r.open_conn();
    REQUIRE(c.has_value());
    std::string err;
    CHECK(c->cancel(id, err));
    CHECK_MESSAGE(r.ends_in(id, "cancelled"),
                  r.log_tail());
}

TEST_CASE("cancel: a child that ignores SIGTERM shows the pending cancel, then settles" * doctest::skip(skip_off_linux)) {
    // An hour of grace and a child that ignores SIGTERM: the job stays "stopping"
    // until the test releases the child — the window the acceptance criterion was
    // written for (the stop is NOT instant, so the client must see it is under
    // way) is held open by construction.
    rig r({.child_ignores_term = true, .term_grace_seconds = 3600});
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "cs");
    REQUIRE(r.reaches(id, is_running));
    auto c = r.open_conn();
    REQUIRE(c.has_value());
    std::string err;
    CHECK(c->cancel(id, err));

    // core::do_cancel records the request inside the RPC: the next list shows it.
    const auto j = r.job(id);
    REQUIRE(j.has_value());
    CHECK_MESSAGE(j->cancel_requested, "a pending cancel is not visible on the wire\n"
                                           << r.log_tail());
    CHECK(j->bucket == "active");

    r.release_run();   // the child exits on its own now
    CHECK_MESSAGE(r.ends_in(id, "cancelled"),
                  r.log_tail());
}

TEST_CASE("cancel: a child that outlives term_grace is SIGKILLed into cancelled/" * doctest::skip(skip_off_linux)) {
    // The child ignores SIGTERM and is never released: only the supervisor's
    // escalation (SIGKILL once term_grace has passed, in the poll loop) can end it.
    // Without the escalation the job stays active, its fingerprint stops changing,
    // and the wait fails after the stall limit. A 1 s grace is far below that
    // limit, so load only delays the SIGKILL; it cannot fail the case.
    rig r({.child_ignores_term = true, .term_grace_seconds = 1});
    REQUIRE(r.ok);
    const std::string id = submit_ok(r, "ck");
    REQUIRE(r.reaches(id, is_running));
    auto c = r.open_conn();
    REQUIRE(c.has_value());
    std::string err;
    CHECK(c->cancel(id, err));
    CHECK_MESSAGE(r.ends_in(id, "cancelled"),
                  r.log_tail());
}
