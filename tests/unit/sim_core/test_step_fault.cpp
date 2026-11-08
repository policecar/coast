// tests/unit/sim_core/test_step_fault.cpp
//
// End-to-end proof of the value-based hot-path error channel (Phase 5 of the
// error-handling redesign; see design/error_handling_plan.md). A minimal io_entity
// reports a fault from process(); we assert the fault propagates entity ->
// env::process() reduction -> controller halt, deterministically.
//
// This is the test that justifies the whole mechanism: it exercises the path that
// is otherwise only reachable on a real invariant violation (which build_env makes
// unreachable in valid configs).
//
// The same sticky halt stops a run whose runtime state load failed partway
// (controller::halt, called by logging_manager::load_state), and a run in which
// an entity or a step hook throws (#198).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "core/step_error.h"
#include "hdf5/executor.h"
#include "logging/logging_manager.h"
#include "logging/state_io.h"
#include "logging/state_logger.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_thread_controller.h"
#include "sim_sync.h"

namespace {

namespace fs = std::filesystem;

// An io_entity with state, whose load_state refuses the checkpoint while `refuse`
// is set, as a group does when a checkpoint does not fit it. With `non_std` it
// throws something that is not a std::exception.
struct loadable_entity : sim::io_entity {
    sim::output_id_t   oid;
    std::atomic<bool>& refuse;
    bool               non_std;

    loadable_entity(sim::output_id_t id, std::atomic<bool>& r, bool ns = false)
        : oid(id), refuse(r), non_std(ns) {}

    sim::step_result process() override { return {}; }

    void save_state(logging::state_writer& w) const override { w.write("x", std::uint32_t{1}); }
    void load_state(logging::state_reader&) override {
        if (!refuse) return;
        if (non_std) throw 42;
        throw std::runtime_error("checkpoint does not fit this group");
    }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// A bare io_entity that optionally faults from process(). It touches no io_buffers,
// so it exercises the channel in isolation. entity_id is left as no_entity by the
// entity (the macro contract) and stamped by env::process() from get_outp_id().
struct faulting_entity : sim::io_entity {
    sim::output_id_t oid;
    bool             fault;
    core::error_kind kind;

    explicit faulting_entity(sim::output_id_t id, bool f = true,
                             core::error_kind k = core::error_kind::contract_violation)
        : oid(id), fault(f), kind(k) {}

    sim::step_result process() override {
        if (fault)
            return std::unexpected(core::step_error{kind, core::step_error::no_entity, "test"});
        return {};
    }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// An io_entity whose process() throws, and counts its calls into an optional
// counter. With `non_std` it throws something that is not a std::exception.
struct throwing_entity : sim::io_entity {
    sim::output_id_t  oid;
    bool              non_std;
    std::atomic<int>* calls;

    explicit throwing_entity(sim::output_id_t id, bool ns = false,
                             std::atomic<int>* c = nullptr)
        : oid(id), non_std(ns), calls(c) {}

    sim::step_result process() override {
        if (calls) ++*calls;
        if (non_std) throw 42;
        throw std::out_of_range("index out of range");
    }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// Counts its process() calls; never faults.
struct counting_entity : sim::io_entity {
    sim::output_id_t  oid;
    std::atomic<int>& calls;

    counting_entity(sim::output_id_t id, std::atomic<int>& c) : oid(id), calls(c) {}

    sim::step_result process() override { ++calls; return {}; }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

// Counts its save_state calls. With `throws` set, save_state throws, as ngm3's
// does when its GPU counter readback fails; with `non_std` too, it throws
// something that is not a std::exception.
struct saving_entity : sim::io_entity {
    sim::output_id_t  oid;
    std::atomic<int>& saves;
    bool              throws;
    bool              non_std;

    saving_entity(sim::output_id_t id, std::atomic<int>& s, bool t = false, bool ns = false)
        : oid(id), saves(s), throws(t), non_std(ns) {}

    sim::step_result process() override { return {}; }

    void save_state(logging::state_writer& w) const override {
        ++saves;
        if (throws && non_std) throw 42;
        if (throws) throw std::runtime_error("counter readback failed");
        w.write("x", std::uint32_t{1});
    }

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return oid; }
    [[nodiscard]] std::size_t get_outp_size() const override { return 0; }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }
};

}  // namespace

TEST_CASE("env::process() reports a faulting entity as unexpected, stamped with its id") {
    sim::env env;
    env.emplace_back<faulting_entity>(sim::output_id_t{5});

    auto r = env.process();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == core::error_kind::contract_violation);
    CHECK(r.error().entity_id == 5);  // env stamped the dispatched entity's outp_id
}

TEST_CASE("env::process() succeeds when no entity faults") {
    sim::env env;
    env.emplace_back<faulting_entity>(sim::output_id_t{1}, /*fault=*/false);

    auto r = env.process();
    CHECK(r.has_value());
}

TEST_CASE("env::process() fault reduction is deterministic: smallest entity_id wins") {
    // Two faulting entities; whichever thread/order the partitioner picks, the
    // reported fault is the one with the smaller entity_id.
    for (int order = 0; order < 2; ++order) {
        sim::env env;
        if (order == 0) {
            env.emplace_back<faulting_entity>(sim::output_id_t{7});
            env.emplace_back<faulting_entity>(sim::output_id_t{3});
        } else {
            env.emplace_back<faulting_entity>(sim::output_id_t{3});
            env.emplace_back<faulting_entity>(sim::output_id_t{7});
        }
        auto r = env.process();
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().entity_id == 3);
    }
}

TEST_CASE("controller halts sticky on a hot-path fault") {
    sim::env env;
    env.emplace_back<faulting_entity>(sim::output_id_t{9});

    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);

    // One full loop iteration steps in unlimited mode, and that first step faults.
    coast_test::sim_iterations(ctrl);
    REQUIRE(ctrl.is_halted());

    auto f = ctrl.get_fault();
    REQUIRE(f.has_value());
    CHECK(f->error.kind == core::error_kind::contract_violation);
    CHECK(f->error.entity_id == 9);
    CHECK(f->step == 0);                  // the first step faults
    CHECK(ctrl.get_step_count() == 0);    // a faulted step is not counted

    // Sticky: a further full iteration — where a step would land — does not step.
    coast_test::sim_iterations(ctrl);
    CHECK(ctrl.get_step_count() == 0);
    CHECK(ctrl.get_mode() == sim::sim_mode::paused);
}

TEST_CASE("halt() stops a running sim, sticky, like a hot-path fault") {
    sim::env env;
    env.emplace_back<faulting_entity>(sim::output_id_t{1}, /*fault=*/false);

    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // at least one step
    REQUIRE(ctrl.get_step_count() > 0);

    // Inside a quiesced task, as logging_manager::load_state calls it: no step
    // runs between the task and the halt.
    const auto at = ctrl.run_quiesced([&] {
        ctrl.halt(core::step_error{core::error_kind::partial_load, core::step_error::no_entity, "test"});
        return ctrl.get_step_count();
    });

    CHECK(ctrl.is_halted());
    const auto f = ctrl.get_fault();
    REQUIRE(f.has_value());
    CHECK(f->error.kind == core::error_kind::partial_load);
    CHECK(f->step == at);

    // Sticky: an external resume does not advance the run. The next full iteration
    // runs the halt check (forcing paused) and the dispatch (where a step would land).
    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);
    CHECK(ctrl.get_step_count() == at);
    CHECK(ctrl.get_mode() == sim::sim_mode::paused);
}

TEST_CASE("a runtime state load that fails partway halts the run") {
    bool        non_std = false;
    std::string expected_message;
    SUBCASE("the entity throws a std::exception") {
        expected_message = "does not fit";
    }
    SUBCASE("the entity throws something else") {
        non_std          = true;
        expected_message = "unknown exception";
    }

    std::atomic<bool> refuse{false};
    sim::env env;
    env.emplace_back<loadable_entity>(sim::output_id_t{3}, refuse, non_std);

    sim::sim_thread_controller ctrl(env);
    hdf5::executor             h5;
    logging::logging_manager   lm(env, h5);

    const auto path = fs::temp_directory_path() /
        ("coast_load_halt_" + std::to_string(::getpid()) + ".h5");
    fs::remove(path);

    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // at least one step
    REQUIRE(ctrl.get_step_count() > 0);
    auto es = lm.extract_state(ctrl);
    logging::state_save_config cfg;
    cfg.output_path = path;
    REQUIRE(lm.serialize_state(std::move(es.tree), cfg).get().success);

    // Load while the sim runs, as from the GUI; the entity refuses the checkpoint.
    refuse = true;
    const auto res = lm.load_state(ctrl, path);
    fs::remove(path);

    CHECK_FALSE(res.success);
    CHECK(res.error_message.find(expected_message) != std::string::npos);
    CHECK(ctrl.is_halted());
    const auto f = ctrl.get_fault();
    REQUIRE(f.has_value());
    CHECK(f->error.kind == core::error_kind::partial_load);

    // No step runs on the partly loaded env.
    const auto at = ctrl.get_step_count();
    CHECK(f->step == at);
    coast_test::sim_iterations(ctrl);   // a full iteration in unlimited mode
    CHECK(ctrl.get_step_count() == at);
}

TEST_CASE("env::process() turns an entity's throw into a fault stamped with its id") {
    bool non_std = false;
    SUBCASE("std::exception") { non_std = false; }
    SUBCASE("not a std::exception") { non_std = true; }

    std::atomic<int> thrower_calls{0};
    std::atomic<int> other_calls{0};
    sim::env env;
    env.emplace_back<throwing_entity>(sim::output_id_t{4}, non_std, &thrower_calls);
    env.emplace_back<counting_entity>(sim::output_id_t{8}, other_calls);

    sim::step_result r;
    REQUIRE_NOTHROW(r = env.process());
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind == core::error_kind::exception);
    CHECK(r.error().entity_id == 4);
    CHECK(thrower_calls == 1);
    CHECK(other_calls == 1);   // the throw did not cancel the other entity
}

TEST_CASE("a thrown and a returned fault reduce together: smallest entity_id wins") {
    for (int order = 0; order < 2; ++order) {
        sim::env env;
        if (order == 0) {
            env.emplace_back<faulting_entity>(sim::output_id_t{7});
            env.emplace_back<throwing_entity>(sim::output_id_t{3});
        } else {
            env.emplace_back<throwing_entity>(sim::output_id_t{7});
            env.emplace_back<faulting_entity>(sim::output_id_t{3});
        }
        auto r = env.process();
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().entity_id == 3);
        CHECK(r.error().kind == (order == 0 ? core::error_kind::exception
                                            : core::error_kind::contract_violation));
    }
}

TEST_CASE("controller halts sticky when an entity throws") {
    sim::env env;
    env.emplace_back<throwing_entity>(sim::output_id_t{9});

    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);

    coast_test::sim_iterations(ctrl);   // the first step throws
    REQUIRE(ctrl.is_halted());
    const auto f = ctrl.get_fault();
    REQUIRE(f.has_value());
    CHECK(f->error.kind == core::error_kind::exception);
    CHECK(f->error.entity_id == 9);
    CHECK(f->step == 0);
    CHECK(ctrl.get_step_count() == 0);
}

TEST_CASE("controller halts sticky when a step hook throws") {
    std::atomic<int> calls{0};
    sim::env env;
    env.emplace_back<counting_entity>(sim::output_id_t{1}, calls);

    // Throws on the third swap, so steps 0 and 1 complete and step 2 faults.
    int swaps = 0;
    env.set_post_swap_hook([&] {
        if (++swaps == 3) throw std::runtime_error("hook failed");
    });

    sim::sim_thread_controller ctrl(env);
    ctrl.set_mode(sim::sim_mode::unlimited);

    coast_test::sim_iterations(ctrl, 3);   // steps 0, 1 and the faulting step 2
    REQUIRE(ctrl.is_halted());
    const auto f = ctrl.get_fault();
    REQUIRE(f.has_value());
    CHECK(f->error.kind == core::error_kind::exception);
    CHECK(f->error.entity_id == core::step_error::no_entity);
    CHECK(f->step == 2);
    CHECK(ctrl.get_step_count() == 2);   // the faulted step is not counted

    coast_test::sim_iterations(ctrl);
    CHECK(ctrl.get_step_count() == 2);
    CHECK(ctrl.get_mode() == sim::sim_mode::paused);
}

// try_extract_state backs the GUI's Save State and auto-save (#229, PR 227
// review): nothing above them catches, so it must not throw, and it must not
// read a halted env at all.
TEST_CASE("try_extract_state refuses a halted run without reading its state") {
    std::atomic<int> saves{0};
    sim::env env;
    env.emplace_back<saving_entity>(sim::output_id_t{1}, saves);

    sim::sim_thread_controller ctrl(env);
    hdf5::executor             h5;
    logging::logging_manager   lm(env, h5);

    ctrl.run_quiesced([&] {
        ctrl.halt(core::step_error{core::error_kind::exception, core::step_error::no_entity, "test"});
    });
    REQUIRE(ctrl.is_halted());

    const auto r = lm.try_extract_state(ctrl);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().find("halted") != std::string::npos);
    CHECK(saves == 0);
}

TEST_CASE("try_extract_state returns an extraction throw as an error") {
    bool        non_std = false;
    std::string expected_message;
    SUBCASE("a std::exception") { expected_message = "counter readback failed"; }
    SUBCASE("something else")   { non_std = true; expected_message = "unknown exception"; }

    std::atomic<int> saves{0};
    sim::env env;
    env.emplace_back<saving_entity>(sim::output_id_t{1}, saves, /*throws=*/true, non_std);

    sim::sim_thread_controller ctrl(env);
    hdf5::executor             h5;
    logging::logging_manager   lm(env, h5);

    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // at least one step
    REQUIRE(ctrl.get_step_count() > 0);

    std::expected<logging::logging_manager::extracted_state, std::string> r;
    CHECK_NOTHROW(r = lm.try_extract_state(ctrl));
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().find(expected_message) != std::string::npos);
    CHECK(saves == 1);

    // A failed save is not a fault: the run goes on.
    CHECK_FALSE(ctrl.is_halted());
    const auto at = ctrl.get_step_count();
    coast_test::sim_iterations(ctrl);
    CHECK(ctrl.get_step_count() > at);
}

TEST_CASE("try_extract_state extracts a running sim at a coherent step") {
    std::atomic<int> saves{0};
    sim::env env;
    env.emplace_back<saving_entity>(sim::output_id_t{1}, saves);

    sim::sim_thread_controller ctrl(env);
    hdf5::executor             h5;
    logging::logging_manager   lm(env, h5);

    ctrl.set_mode(sim::sim_mode::unlimited);
    coast_test::sim_iterations(ctrl);   // at least one step
    REQUIRE(ctrl.get_step_count() > 0);

    const auto r = lm.try_extract_state(ctrl);
    REQUIRE(r.has_value());
    CHECK(r->step > 0);
    CHECK(saves == 1);
}
