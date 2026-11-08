// tests/unit/sim_core/test_hook_removal.cpp
//
// Hook removal is synchronous with respect to hook invocation (#235): once an
// env::remove_*_hook() call made outside a hook has returned, the removed hook is
// not running and will never run again. Owners that capture `this` in a hook (the
// GUI visualizations) rely on this to free themselves right after removing it.
//
// The race the old deferred removal allowed: the sim thread applies pending hook
// changes at the top of process()/swap_io() and only then runs the hooks. A
// removal queued after that point took effect one step late, so the removed hook
// still ran — on its owner's freed memory. The deterministic cases below force
// that interleaving with a gate hook (or with the target hook itself) parked on
// the sim thread while the "GUI" thread removes the target and destroys its owner.
//
// The owner is modelled by a shared `destroyed` flag rather than real freed memory,
// so a violation is a CHECK failure, not undefined behaviour in the test itself.
// The stress case further down uses real heap owners and is meant for ASan/TSan.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"

#include "await_progress.h"
#include "sim_sync.h"

// No wall-clock time decides a verdict here (CLAUDE.md "Testing"): parked hooks
// wait on observable events — the owner's destruction, or env's
// hook_removal_waiting() — never on a timer; the controller cases wait in sim
// iterations.

namespace {

// A one-shot flag.
struct signal_t {
    std::atomic<bool> set{false};
    void raise() { set.store(true); }
    [[nodiscard]] bool is_set() const { return set.load(); }
    // Wait for it; the stall limit only names a frozen run (await_condition).
    [[nodiscard]] bool wait() const {
        return coast_test::await_condition([&] { return is_set(); });
    }
};

// A hook parked on the sim thread, waiting for the "GUI" thread. It returns when
// the owner has been destroyed — the remover did NOT wait for this hook phase
// (the #235 bug) — or when a removal is waiting for this phase: the fix, under
// which the remover blocks until the hook returns, and env's contract that a
// hook waiting on another thread must then stop waiting. Both outcomes are
// events, so the interleaving is forced in either direction and no timer
// decides which one happened.
void park(const signal_t& owner_destroyed, const sim::env& env) {
    CHECK(coast_test::await_condition(
        [&] { return owner_destroyed.is_set() || env.hook_removal_waiting(); }));
}

using add_fn    = std::size_t (sim::env::*)(std::function<void()>);
using remove_fn = void (sim::env::*)(std::size_t);

struct hook_kind {
    std::string name;
    add_fn      add;
    remove_fn   remove;
};

const hook_kind pre_process  {"pre_process",  &sim::env::set_pre_process_hook,  &sim::env::remove_pre_process_hook};
const hook_kind post_process {"post_process", &sim::env::set_post_process_hook, &sim::env::remove_post_process_hook};
const hook_kind pre_swap     {"pre_swap",     &sim::env::set_pre_swap_hook,     &sim::env::remove_pre_swap_hook};
const hook_kind post_swap    {"post_swap",    &sim::env::set_post_swap_hook,    &sim::env::remove_post_swap_hook};
const hook_kind barrier      {"post_swap_barrier", &sim::env::set_post_swap_barrier_hook,
                              &sim::env::remove_post_swap_barrier_hook};

// One step as the controller runs it.
void step(sim::env& env) {
    REQUIRE(env.process().has_value());
    REQUIRE(env.swap_io().has_value());
}

// The interleaving of issue #235: pending hook changes have been processed (the
// target is active) and the sim thread is past that point, but has not invoked the
// target yet. The "GUI" removes the target and destroys its owner there.
void check_removal_before_invocation(const hook_kind& gate_kind, const hook_kind& target) {
    CAPTURE(gate_kind.name);
    CAPTURE(target.name);

    sim::env env;
    signal_t gate_reached, owner_destroyed_sig;
    std::atomic<bool> destroyed{false};
    std::atomic<int>  calls_after_destroy{0};
    std::atomic<int>  target_calls{0};
    std::atomic<bool> armed{false};

    (env.*gate_kind.add)([&] {
        if (!armed.exchange(false)) return;
        gate_reached.raise();
        park(owner_destroyed_sig, env);
    });
    const std::size_t id = (env.*target.add)([&] {
        ++target_calls;
        if (destroyed.load()) ++calls_after_destroy;   // would touch freed `this`
    });

    step(env);                        // applies both hooks; target runs once
    REQUIRE(target_calls.load() == 1);

    armed = true;
    std::thread sim_thread([&] { step(env); });
    REQUIRE(gate_reached.wait());

    (env.*target.remove)(id);         // owner's destructor: remove the hook ...
    destroyed = true;                 // ... then free the captured state
    owner_destroyed_sig.raise();

    sim_thread.join();
    CHECK(calls_after_destroy.load() == 0);

    const int calls = target_calls.load();
    step(env);                        // and it stays removed
    CHECK(target_calls.load() == calls);
}

// The target is already running when its owner is destroyed: removal must wait
// for that invocation to return.
void check_removal_during_invocation(const hook_kind& target) {
    CAPTURE(target.name);

    sim::env env;
    signal_t running, owner_destroyed_sig;
    std::atomic<bool> destroyed{false};
    std::atomic<int>  touched_after_destroy{0};
    std::atomic<bool> armed{false};

    const std::size_t id = (env.*target.add)([&] {
        if (!armed.exchange(false)) return;
        running.raise();
        park(owner_destroyed_sig, env);
        if (destroyed.load()) ++touched_after_destroy;  // still using `this`
    });

    step(env);                        // applies the hook
    armed = true;
    std::thread sim_thread([&] { step(env); });
    REQUIRE(running.wait());

    (env.*target.remove)(id);
    destroyed = true;
    owner_destroyed_sig.raise();

    sim_thread.join();
    CHECK(touched_after_destroy.load() == 0);
}

// Run `fn` on another thread and report whether it returns, i.e. does not block.
// The removal paths below take no lock that anything else holds and wait for no
// hook phase (nothing is stepping), so "returns" is the property; a blocked
// call is a frozen one, named by the await_condition stall limit instead of
// hanging the binary. The thread OWNS `keepalive` (the env the call works on), so
// a failing case that unwinds while it is still blocked frees nothing under it;
// `fn` must capture everything else by value. (A joining std::jthread would avoid
// the use-after-free too, but its join would block on the stuck call and turn the
// named failure into a hang.)
bool returns(std::shared_ptr<void> keepalive, std::function<void()> fn) {
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread([keepalive = std::move(keepalive), fn = std::move(fn), done] {
        fn();
        done->store(true);
    }).detach();
    return coast_test::await_condition([&] { return done->load(); });
}

struct faulting_entity : sim::io_entity {
    sim::step_result process() override {
        return std::unexpected(core::step_error{core::error_kind::contract_violation,
                                                core::step_error::no_entity, "test"});
    }
    [[nodiscard]] sim::output_id_t get_outp_id() const override { return 1; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

} // namespace

TEST_CASE("a hook removed after pending changes were applied does not run (#235)") {
    // swap_io: pending changes, pre_swap, swap, post_swap, barrier.
    check_removal_before_invocation(pre_swap,  post_swap);
    check_removal_before_invocation(pre_swap,  barrier);
    check_removal_before_invocation(post_swap, barrier);
    // process: pending changes, pre_process, entities, post_process.
    check_removal_before_invocation(pre_process, post_process);
}

TEST_CASE("removing a hook waits for its running invocation (#235)") {
    check_removal_during_invocation(pre_process);
    check_removal_during_invocation(post_process);
    check_removal_during_invocation(pre_swap);
    check_removal_during_invocation(post_swap);
    check_removal_during_invocation(barrier);
}

TEST_CASE("a hook removed before it was ever applied never runs") {
    sim::env env;
    std::atomic<int> calls{0};
    const auto id = env.set_post_swap_hook([&] { ++calls; });
    env.remove_post_swap_hook(id);
    step(env);
    step(env);
    CHECK(calls.load() == 0);
}

TEST_CASE("a hook may remove itself from inside its invocation") {
    // Removal on the invoking thread cannot wait for itself; it takes effect at
    // the next step boundary, and the running invocation completes normally.
    sim::env env;
    std::atomic<int> calls{0};
    std::size_t id = 0;
    id = env.set_post_swap_hook([&] {
        ++calls;
        env.remove_post_swap_hook(id);
    });
    step(env);
    step(env);
    step(env);
    CHECK(calls.load() == 1);
}

TEST_CASE("removal returns promptly in every controller state") {
    auto      envp = std::make_shared<sim::env>();   // shared: see returns()
    sim::env& env  = *envp;
    std::atomic<int> calls{0};
    auto ctrl = std::make_unique<sim::sim_thread_controller>(env);

    auto add_and_remove = [&](const char* state) {
        CAPTURE(state);
        const auto id = env.set_post_swap_hook([&] { ++calls; });
        const auto bid = env.set_post_swap_barrier_hook([&] { ++calls; });
        CHECK(returns(envp, [&env, id, bid] {
            env.remove_post_swap_hook(id);
            env.remove_post_swap_barrier_hook(bid);
        }));
    };

    add_and_remove("paused, never stepped");   // parked; hooks still pending

    ctrl->set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(*ctrl, 11);
    REQUIRE(ctrl->get_step_count() > 10);
    add_and_remove("unlimited");

    ctrl->set_steps_per_second(50.0);
    ctrl->set_mode(sim::sim_mode::throttled);
    add_and_remove("throttled");

    ctrl->set_mode(sim::sim_mode::paused);
    coast_test::sim_iterations(*ctrl);   // it has processed the pause and parked
    add_and_remove("paused after stepping");

    ctrl.reset();
    add_and_remove("controller destroyed");

    // Halted: a faulting entity stops the run for good.
    auto      env2p = std::make_shared<sim::env>();   // shared: see returns()
    sim::env& env2  = *env2p;
    env2.emplace_back<faulting_entity>();
    sim::sim_thread_controller ctrl2(env2);
    ctrl2.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl2);   // the first step faults
    REQUIRE(ctrl2.is_halted());
    const auto id = env2.set_post_swap_hook([&] { ++calls; });
    CHECK(returns(env2p, [&env2, id] { env2.remove_post_swap_hook(id); }));
}

TEST_CASE("stress: owners freed right after removal while the sim runs (ASan/TSan)") {
    // The real pattern: the hook captures a raw pointer to a heap owner that is
    // deleted as soon as remove returns. Without synchronous removal ASan reports
    // a heap-use-after-free here; with it, nothing.
    struct owner_t {
        std::uint64_t counter = 0;
    };

    sim::env env;
    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);
    REQUIRE(ctrl.get_step_count() > 0);

    for (int i = 0; i < 2000; ++i) {
        auto* o = new owner_t;
        const auto id  = env.set_post_swap_hook([o] { ++o->counter; });
        const auto bid = env.set_post_swap_barrier_hook([o] { ++o->counter; });
        // Let the sim thread apply and run them for a little while.
        const auto until = ctrl.get_step_count() + 2;
        REQUIRE(coast_test::await_progress([&] { return ctrl.get_step_count() >= until; },
                                           [&] { return ctrl.get_step_count(); }));
        env.remove_post_swap_hook(id);
        env.remove_post_swap_barrier_hook(bid);
        delete o;
    }
    CHECK(ctrl.get_step_count() > 0);
}

TEST_CASE("removal does not wait out GUI-waiting barrier hooks (#235)") {
    // The visualizations' two-phase protocol: the post_swap hook marks an update
    // requested, the barrier hook waits for the GUI thread to service it, giving
    // up after a timeout. The GUI thread cannot service anything while it is
    // itself blocked removing a hook, so without hook_removal_waiting() a removal
    // that lands in the barrier phase waits out every requested visualization's
    // timeout in turn (K x timeout; destroying all K, as a config reload does,
    // compounds that per destroyed visualization).
    //
    // Asserted as HOW each waiting barrier hook stopped waiting — through
    // hook_removal_waiting() or through its timeout — not as how long the removal
    // took. The timeout stands for "the GUI never services it" and is the stall
    // limit: with the early exit no hook reaches it; without it the first one
    // that does is a frozen removal, and the count says so.
    constexpr int k_vis = 5;
    const auto vis_timeout = coast_test::default_stall;

    struct vis_t {
        std::atomic<bool> requested{false};
        std::size_t       signal_id = 0, wait_id = 0;
    };
    sim::env env;
    std::vector<std::unique_ptr<vis_t>> vis;
    signal_t in_barrier;
    std::atomic<bool> armed{false};
    // Until the test switches the GUI "off" right before the removals, updates
    // are not requested, so setup steps do not wait out any timeout. From then
    // on nothing services them: every wait ends by removal or by the stall.
    std::atomic<bool> gui_unresponsive{false};
    std::atomic<int>  exits_by_removal{0};
    std::atomic<int>  exits_by_timeout{0};

    for (int i = 0; i < k_vis; ++i) {
        auto v = std::make_unique<vis_t>();
        auto* p = v.get();
        p->signal_id = env.set_post_swap_hook([p, &gui_unresponsive] {
            if (gui_unresponsive.load()) p->requested = true;
        });
        p->wait_id   = env.set_post_swap_barrier_hook(
            [p, &env, &in_barrier, &armed, &gui_unresponsive, &exits_by_removal, &exits_by_timeout,
             vis_timeout] {
            const auto start = std::chrono::steady_clock::now();
            while (p->requested.load()) {     // nobody services it: the GUI is removing
                // Raised by a hook that IS waiting (its update was requested), so
                // the removals below provably land in a waiting barrier phase —
                // not by one whose post_swap ran before the GUI went unresponsive.
                if (armed.exchange(false)) in_barrier.raise();
                if (env.hook_removal_waiting()) {
                    ++exits_by_removal;
                    p->requested = false;      // cancel, as the visualizations do
                    break;
                }
                if (std::chrono::steady_clock::now() - start > vis_timeout) {
                    ++exits_by_timeout;
                    p->requested = false;
                    // Frozen removal detected: stop requesting, so the sim stops
                    // holding the hook lock in waiting phases, the remover gets
                    // through, and the CHECK below reports it by name instead of
                    // the binary hanging on a starved removal.
                    gui_unresponsive = false;
                    break;
                }
                std::this_thread::yield();
            }
        });
        vis.push_back(std::move(v));
    }

    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);
    REQUIRE(ctrl.get_step_count() > 0);

    armed = true;
    gui_unresponsive = true;
    REQUIRE(in_barrier.wait());   // a barrier phase is waiting for the "GUI" now
    for (auto& v : vis) {         // "config reload": destroy them all
        env.remove_post_swap_hook(v->signal_id);
        env.remove_post_swap_barrier_hook(v->wait_id);
        v.reset();
    }
    CHECK(exits_by_timeout.load() == 0);   // no removal waited a hook out
    CHECK(exits_by_removal.load() >= 1);   // the phase it landed in was released by it
}
