#ifndef COAST_TESTS_SUPPORT_SIM_SYNC_H
#define COAST_TESTS_SUPPORT_SIM_SYNC_H

// -----------------------------------------------------------------------------
// Logical-time synchronisation with the sim thread (CLAUDE.md, "Testing": no
// sleeps as synchronisation, no fixed observation windows).
//
// sim_thread_controller::thread_main runs, per loop iteration, in this order:
//   1. drain queued run_quiesced() tasks,
//   2. the step-limit and halt checks (which may force `paused`),
//   3. the mode dispatch: one do_step() (unlimited / throttled / a pending single
//      step in paused) or park.
// drain_quiesced() swaps the queue out before running it, so a task submitted
// after the previous one returned is drained at a LATER loop top. n+1 sequential
// run_quiesced() calls therefore bracket n complete iterations (steps 2 and 3
// included).
//
// sim_iterations(ctrl, n) returns once at least n complete iterations have run
// since the call began. Typical uses:
//   - "the run advances":       set_mode(unlimited); sim_iterations(ctrl, k);
//                               CHECK(count >= start + k)
//   - "nothing happens":        sim_iterations(ctrl, 1); CHECK(count == before)
//     — one full iteration is exactly where a stray step, or the forced pause of
//     a halted run, would have to happen.
//   - "a fault halts the run":  set_mode(unlimited); sim_iterations(ctrl, 1);
//                               REQUIRE(is_halted())
// It waits on the sim thread itself, so load only delays it. If the sim thread
// is wedged, run_quiesced() blocks; the ctest TIMEOUT is the backstop there.
// -----------------------------------------------------------------------------

#include "sim_core/sim_thread_controller.h"

namespace coast_test {

inline void sim_iterations(sim::sim_thread_controller& ctrl, int n = 1)
{
    for (int i = 0; i <= n; ++i) ctrl.run_quiesced([] {});
}

} // namespace coast_test

#endif // COAST_TESTS_SUPPORT_SIM_SYNC_H
