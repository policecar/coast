// tests/unit/sim_core/test_thread_controller_single_step.cpp
//
// Regression tests for #201: a single_step() request belongs to the paused state
// it was made in. One left pending while the sim runs must neither take an extra
// step past the step limit nor fire on a later, unrelated pause.
//
// Empty sim::env, as in test_thread_controller_hold.cpp: steps cost microseconds.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdint>

#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"

#include "await_progress.h"
#include "sim_sync.h"

// Waits are in sim-loop iterations (tests/support/sim_sync.h): "a stray step
// would land" means "within the next full iteration", which is where the
// dispatch that would take it runs — independent of how fast the machine is.

namespace {

TEST_CASE("a step requested while running does not overshoot the step limit") {
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // running for certain
    // Set the limit and the pending step together at a step boundary, so the
    // request is certainly made while running and before the limit is reached.
    // (Calling single_step() right after set_mode() races the limit: the thread
    // may already have paused, and a step taken while paused is legitimate.)
    std::uint64_t limit = 0;
    ctrl.run_quiesced([&] {
        limit = ctrl.get_step_count() + 1000;
        ctrl.set_step_limit(limit);
        ctrl.single_step();   // left pending: unlimited mode never consumes it
    });

    // Reaching the limit takes 1000 steps; the step count is the heartbeat. A
    // limit that does not hold would keep the heartbeat ticking forever, so
    // overshooting it ends the wait as well (and fails below).
    REQUIRE(coast_test::await_progress(
        [&] { return ctrl.get_mode() == sim::sim_mode::paused || ctrl.get_step_count() > limit; },
        [&] { return ctrl.get_step_count(); }));
    coast_test::sim_iterations(ctrl);   // where a stray step would land
    CHECK(ctrl.get_step_count() == limit);
}

TEST_CASE("a step requested while running does not fire on the next pause") {
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    ctrl.set_steps_per_second(2.0);
    ctrl.set_mode(sim::sim_mode::throttled);
    coast_test::sim_iterations(ctrl);   // it has stepped: running for certain
    REQUIRE(ctrl.get_step_count() >= 1);

    // The request and the pause, both made while running and with no step in
    // between — at a boundary, instead of hoping both land inside the ~500 ms
    // throttle sleep.
    const auto at_pause = ctrl.run_quiesced([&] {
        ctrl.single_step();
        ctrl.set_mode(sim::sim_mode::paused);
        return ctrl.get_step_count();
    });

    coast_test::sim_iterations(ctrl);   // the paused dispatch that would fire it
    CHECK(ctrl.get_step_count() == at_pause);
}

TEST_CASE("step requests at the step limit never advance past it") {
    // Hammer single_step() and set_mode() while parked at the limit. Each call
    // can land between the top-of-loop limit check and the step itself; before
    // the limit was checked in do_step() this failed in every run, overshooting
    // by 5 to over 1000 steps.
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    ctrl.set_step_limit(1);
    ctrl.set_mode(sim::sim_mode::unlimited);
    // Iteration 1 takes the one allowed step, iteration 2's limit check pauses.
    coast_test::sim_iterations(ctrl, 2);
    REQUIRE(ctrl.get_mode() == sim::sim_mode::paused);
    REQUIRE(ctrl.get_step_count() == 1);

    // A fixed number of requests, not "as many as fit in 200 ms": the amount of
    // pressure does not depend on the machine's speed.
    constexpr int k_calls = 200'000;
    for (int i = 0; i < k_calls; ++i) ctrl.single_step();
    for (int i = 0; i < k_calls; ++i) ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // where a stray step would land
    CHECK(ctrl.get_step_count() == 1);
}

TEST_CASE("a step requested while paused still advances exactly one step") {
    sim::env env;
    sim::sim_thread_controller ctrl(env);
    REQUIRE(ctrl.get_mode() == sim::sim_mode::paused);
    ctrl.single_step();
    coast_test::sim_iterations(ctrl);   // the paused dispatch consumes the request
    REQUIRE(ctrl.get_step_count() == 1);
    coast_test::sim_iterations(ctrl);   // and takes no second step
    CHECK(ctrl.get_step_count() == 1);
}

}  // namespace
