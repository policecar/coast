#ifndef COAST_TESTS_SUPPORT_AWAIT_PROGRESS_H
#define COAST_TESTS_SUPPORT_AWAIT_PROGRESS_H

// -----------------------------------------------------------------------------
// await_progress: wait for an outcome without making the verdict depend on how
// fast the machine is (CLAUDE.md, "Testing": hangs are detected by progress, not
// by a timeout).
//
// pred()      — the outcome the test waits for.
// heartbeat() — a progress counter of the process that has to bring pred() about
//               (steps, reconcile passes, served requests, ...). Any value
//               comparable with != ; it only has to CHANGE while work is done.
//
// Returns true as soon as pred() holds. Returns false only when heartbeat() has
// not changed for `stall` — the process is then frozen, not slow. A loaded box
// stretches the wait; it never turns a correct outcome red.
//
// To fail a WRONG outcome fast and deterministically, put the violation into the
// predicate and check it afterwards (e.g. wait for `halted || steps > 0`, then
// CHECK(steps == 0)) instead of waiting for a stall.
//
// Header-only and free of project headers, so the coastd tests (a separate
// subtree that does not link coast_test_support) can include it as well.
// -----------------------------------------------------------------------------

#include <chrono>
#include <thread>

namespace coast_test {

inline constexpr std::chrono::milliseconds default_stall{60'000};

// `poll` is only the polling interval (how soon a satisfied pred() is seen,
// and how hard pred()/heartbeat() are hit — e.g. a network round-trip each); it
// never decides the outcome.
template <class Pred, class Heartbeat>
[[nodiscard]] bool await_progress(Pred&& pred, Heartbeat&& heartbeat,
                                  std::chrono::milliseconds stall = default_stall,
                                  std::chrono::milliseconds poll  = std::chrono::milliseconds{1})
{
    using clock = std::chrono::steady_clock;
    auto last        = heartbeat();
    auto last_change = clock::now();
    while (!pred()) {
        const auto beat = heartbeat();
        const auto now  = clock::now();
        if (beat != last) {
            last        = beat;
            last_change = now;
        } else if (now - last_change > stall) {
            return pred();
        }
        std::this_thread::sleep_for(poll);
    }
    return true;
}

// await_condition: for waits on something that exposes NO progress counter —
// e.g. a helper thread the test started reaching its blocking call, or a
// teardown that must reject a queued task while everything else is blocked by
// design. Nothing can tick there, so the process counts as frozen once pred()
// has not come true for `stall`: the same rule as await_progress with a
// heartbeat that never advances. The limit is a hang detector (it names the
// failure instead of leaving it to the 1500 s ctest TIMEOUT); it is ~4 orders of
// magnitude above the time any of these waits take.
template <class Pred>
[[nodiscard]] bool await_condition(Pred&& pred,
                                   std::chrono::milliseconds stall = default_stall)
{
    return await_progress(std::forward<Pred>(pred), [] { return 0; }, stall);
}

} // namespace coast_test

#endif // COAST_TESTS_SUPPORT_AWAIT_PROGRESS_H
