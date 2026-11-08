// tests/unit/sim_core/test_thread_controller_hold.cpp
//
// Unit tests for sim_thread_controller::run_quiesced(fn) — the single quiescence
// seam that runs fn ON the sim thread at the post-swap boundary and returns its
// result. It is the sole mechanism for non-sim-thread access to sim state
// (state save/load/checkpoint, set_step_count); the earlier RAII park-the-thread
// hold it replaced has been retired.
//
// run_quiesced contract: the task runs on the sim thread at a boundary (not the
// caller's thread); the step count is frozen for its duration; the mode is
// untouched; stepping resumes after; it works whether the sim is free-running
// or parked (paused); the return value propagates and exceptions rethrow to the
// caller; and a requester is unblocked (not hung) if the controller tears down
// while its task is still queued (stop-escape).
//
// Uses an empty sim::env (no entities, no GPU context) — process()/swap_io()
// iterate nothing, so the controller's threading behavior is exercised in
// isolation, sub-second.
//
// Timing discipline: none. Waits are expressed in sim-loop iterations
// (tests/support/sim_sync.h) or on explicit observation points, never in
// wall-clock time, so a loaded machine cannot change a verdict.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"

#include "await_progress.h"
#include "sim_sync.h"

namespace {

TEST_CASE("controller destruction while free-running stays clean") {
    // Construct, run briefly, destroy — exercises the parked/notify paths in the
    // destructor (regression guard for the quiesced-task signaling sharing
    // wake_mutex_/wake_cv_ with shutdown).
    for (int i = 0; i < 3; ++i) {
        sim::env env;
        sim::sim_thread_controller ctrl(env);
        ctrl.set_mode(sim::sim_mode::unlimited);
        // Free-running for certain (it has stepped), not merely asked to.
        coast_test::sim_iterations(ctrl);
        REQUIRE(ctrl.get_step_count() > 0);
    }
}

// --- run_quiesced -----------------------------------------------------------

TEST_CASE("run_quiesced runs on the sim thread at a boundary, freezing the count") {
    sim::env env;
    // The thread that steps: a post-swap hook runs inside do_step, so it records
    // the stepping thread's id.
    std::atomic<std::thread::id> stepping_id{};
    env.set_post_swap_hook([&] { stepping_id.store(std::this_thread::get_id()); });

    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // at least one step
    REQUIRE(ctrl.get_step_count() > 0);

    std::thread::id task_id{};
    std::uint64_t   count_at_entry = 0;
    std::uint64_t   count_at_exit  = 0;

    ctrl.run_quiesced([&] {
        task_id        = std::this_thread::get_id();
        count_at_entry = ctrl.get_step_count();
        count_at_exit  = ctrl.get_step_count();
    });

    // The task ran on the very thread that steps. That thread cannot step while
    // it runs the task, so no step overlaps the task — by construction, not by
    // watching the count over some window.
    CHECK(task_id == stepping_id.load());
    CHECK(task_id != std::this_thread::get_id());
    CHECK(count_at_entry == count_at_exit);
    // The mode was not touched.
    CHECK(ctrl.get_mode() == sim::sim_mode::unlimited);

    // Stepping resumes after the task.
    const auto after = ctrl.get_step_count();
    coast_test::sim_iterations(ctrl);
    CHECK(ctrl.get_step_count() > after);
}

TEST_CASE("run_quiesced wakes a parked (paused) sim thread") {
    sim::env env;
    sim::sim_thread_controller ctrl(env);   // starts paused
    REQUIRE(ctrl.get_mode() == sim::sim_mode::paused);

    // The thread is parked in the paused wait; the enqueue must wake it.
    const int result = ctrl.run_quiesced([] { return 7; });
    CHECK(result == 7);

    // A quiesced task advances no sim time.
    CHECK(ctrl.get_step_count() == 0);
}

TEST_CASE("run_quiesced propagates return values and rethrows exceptions") {
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);

    CHECK(ctrl.run_quiesced([] { return std::string("ok"); }) == "ok");

    CHECK_THROWS_AS(
        ctrl.run_quiesced([]() -> int { throw std::runtime_error("boom"); }),
        std::runtime_error);

    // The controller is still healthy after a task throws.
    CHECK(ctrl.run_quiesced([] { return 1; }) == 1);
}

TEST_CASE("run_quiesced stop-escape: teardown unblocks a queued requester") {
    sim::env env;
    std::optional<sim::sim_thread_controller> ctrl;
    ctrl.emplace(env);   // paused

    std::atomic<bool> task1_started{false};
    std::atomic<bool> release1{false};

    // Requester 1 occupies the sim thread inside its task until released, so the
    // sim thread provably cannot drain anything queued behind it.
    std::thread t1([&] {
        try {
            ctrl->run_quiesced([&] {
                task1_started.store(true);
                // Held by the test, released below; not a wait on an outcome.
                while (!release1.load()) std::this_thread::yield();
            });
        } catch (...) { /* not expected */ }
    });
    REQUIRE(coast_test::await_condition([&] { return task1_started.load(); }));

    // Requester 2's task sits in the queue behind task1 and blocks in fut.get().
    std::atomic<bool> t2_done{false};
    std::atomic<bool> t2_threw{false};
    std::thread t2([&] {
        try {
            ctrl->run_quiesced([] { /* must NOT run: rejected at teardown */ });
        } catch (const std::exception&) {
            t2_threw.store(true);
        }
        t2_done.store(true);
    });
    // Established, not assumed: task2 is in the queue (task1 has left it, being
    // run), so t2 is blocked in fut.get() behind task1.
    REQUIRE(coast_test::await_condition([&] { return ctrl->queued_quiesced_tasks() == 1; }));
    REQUIRE_FALSE(t2_done.load());

    // Tear down from a helper thread. The destructor request_stop()s, then
    // rejects the queued task2 (unblocking t2) BEFORE it joins; the join then
    // waits for task1, which is still blocked. Since the sim thread is stuck in
    // task1, it never drains task2 — task2 takes the reject path deterministically.
    std::thread killer([&] { ctrl.reset(); });

    CHECK(coast_test::await_condition([&] { return t2_done.load(); }));   // unblocked by teardown
    CHECK(t2_threw.load());                              // via the stopping exception

    release1.store(true);   // let task1 finish so the join inside reset() completes
    killer.join();
    t1.join();
    t2.join();
}

} // namespace
