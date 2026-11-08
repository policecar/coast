// plot_manager lifetime — the deinit() tombstone. Isolated in its OWN process
// because deinit() sets a PROCESS-WIDE tombstone: once it runs, instance() must
// never resurrect the singleton again, so sharing a process with the other
// console_plot cases (which call instance() freely) would poison them.
//
// This pins the invariant behind the run_gui shutdown ordering fix: after the
// console thread is joined and the plot subsystem is torn down, any stray access
// (a figure created/freed post-shutdown) fails loudly at the call site instead of
// silently allocating a second, GUI-host-less manager whose deferred GL frees then
// run at atexit with no GL context.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "gui_plot/plot_manager.h"

#include <stdexcept>

using coast::plot::plot_manager;

TEST_CASE("plot_manager: instance() is live before deinit()")
{
    // A plain access constructs the singleton and hands back a usable manager.
    CHECK(plot_manager::instance().plot_count() == 0);
}

TEST_CASE("plot_manager: deinit() tombstones — instance() refuses to resurrect")
{
    // Bring the singleton up, then tear it down.
    (void)plot_manager::instance().plot_count();
    plot_manager::deinit();

    // Post-deinit access must throw rather than silently allocate a second manager.
    CHECK_THROWS_AS((void)plot_manager::instance(), std::logic_error);

    // Still tombstoned on a repeat access (not a one-shot latch that then resurrects).
    CHECK_THROWS_AS((void)plot_manager::instance(), std::logic_error);
}

TEST_CASE("plot_manager: deinit() is idempotent after the tombstone is set")
{
    // A second deinit() (e.g. the std::atexit fallback after an explicit call) is a
    // no-op — delete nullptr — and leaves the tombstone in place.
    plot_manager::deinit();
    CHECK_THROWS_AS((void)plot_manager::instance(), std::logic_error);
}
