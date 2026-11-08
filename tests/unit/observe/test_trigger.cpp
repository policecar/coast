// Unit test 1.4 — observe programmatic trigger semantics (post-refactor model).
//
// Triggers are now driven through observe_accessor::trigger() (the only public
// entry; the old OBSERVE_TRIGGER macro / registry on_trigger() are gone). The
// model is a single per-group boolean:
//   START          -> prog_active = true
//   STOP           -> prog_active = false
//   SINGLE_CAPTURE -> prog_single_pending = true (consumed at end of pass)
// and `active_this_step` is recomputed live from (ranges, combine, prog_*).
//
// These cases use groups with NO configured ranges, so OR-combine reduces
// active_this_step to the trigger state alone. The range/combine interaction
// over a real run is covered end-to-end by the B-tier capture test.
//
// Singleton hygiene: distinct group id per case.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "observe/observe_registry.h"
#include "observe/observe_accessor.h"
#include "observe/observe_group.h"
#include "observe/observe_trigger.h"

using observe::trigger_mode_t;

namespace {
void check_fresh(const observe::observe_group_t& g) {
    CHECK_FALSE(g.prog_active.load());
    CHECK_FALSE(g.prog_single_pending.load());
    CHECK_FALSE(g.active_this_step.load());
}
} // namespace

TEST_CASE("observe trigger: a brand-new group starts fully disarmed") {
    auto& reg = observe::observe_registry::instance();
    auto& g = reg.get_or_create_group("trg_fresh");
    check_fresh(g);
}

TEST_CASE("observe trigger: cond=false is a no-op") {
    auto& reg = observe::observe_registry::instance();
    auto& g = reg.get_or_create_group("trg_cond");
    observe::observe_accessor acc(reg);
    acc.trigger("trg_cond", trigger_mode_t::START, /*cond=*/false);
    check_fresh(g);
}

TEST_CASE("observe trigger: START streams and activates the group") {
    auto& reg = observe::observe_registry::instance();
    reg.get_or_create_group("trg_start");
    observe::observe_accessor acc(reg);
    acc.trigger("trg_start", trigger_mode_t::START, true);

    auto* g = reg.group("trg_start");
    REQUIRE(g != nullptr);
    CHECK(g->prog_active.load());            // streaming activation
    CHECK(g->active_this_step.load());       // active this step (OR, no range)
    CHECK_FALSE(g->prog_single_pending.load());
}

TEST_CASE("observe trigger: SINGLE_CAPTURE activates via the one-shot pending flag") {
    auto& reg = observe::observe_registry::instance();
    reg.get_or_create_group("trg_single");
    observe::observe_accessor acc(reg);
    acc.trigger("trg_single", trigger_mode_t::SINGLE_CAPTURE, true);

    auto* g = reg.group("trg_single");
    REQUIRE(g != nullptr);
    CHECK(g->prog_single_pending.load());
    CHECK(g->active_this_step.load());
    CHECK_FALSE(g->prog_active.load());      // single-shot, not streaming
}

TEST_CASE("observe trigger: STOP clears streaming and deactivates (no range configured)") {
    auto& reg = observe::observe_registry::instance();
    reg.get_or_create_group("trg_stop");
    observe::observe_accessor acc(reg);
    acc.trigger("trg_stop", trigger_mode_t::START, true);
    acc.trigger("trg_stop", trigger_mode_t::STOP, true);

    auto* g = reg.group("trg_stop");
    REQUIRE(g != nullptr);
    CHECK_FALSE(g->prog_active.load());      // streaming turned off
    CHECK_FALSE(g->active_this_step.load()); // no range -> dormant once stopped
}

TEST_CASE("observe trigger: a repeated START keeps the group active (idempotent)") {
    auto& reg = observe::observe_registry::instance();
    reg.get_or_create_group("trg_repeat");
    observe::observe_accessor acc(reg);
    acc.trigger("trg_repeat", trigger_mode_t::START, true);
    acc.trigger("trg_repeat", trigger_mode_t::START, true);

    auto* g = reg.group("trg_repeat");
    REQUIRE(g != nullptr);
    CHECK(g->prog_active.load());
    CHECK(g->active_this_step.load());
}
