// tests/integration_ngm3/state_roundtrip_split_test.cpp
//
// Step 3 gate: the extract/serialize/parse/apply split reproduces full sim state
// (design §4–§5). This exercises the NEW split functions end-to-end —
// state_logger::extract_state -> serialize_state -> parse_state -> apply_state —
// as opposed to the monolithic save_state/load_state (covered by the B-tier and
// the existing readpage/roundtrip tests).
//
// Protocol (no stepping): run a reference env to step N, extract its state tree,
// serialize to HDF5. Then in a fresh env: parse the file, apply the tree, and
// re-extract. The io_buffers and entities subtrees must be bit-for-bit identical
// to the reference's — a "coast_compare-clean" reproduction expressed directly as
// mem_group equality (mem_value's type-sensitive operator==). Metadata is excluded
// from the comparison because serialize_state stamps a wall-clock timestamp.
//
// No stepping happens after the load, so this is RNG-immune and valid on CPU and
// GPU configs alike (the same source is registered for both). The CPU entities'
// stateful fast_rng_t gap (not persisted by save_state) only manifests when
// stepping after a load — the GPU step-once variant (state_resume_split_repro)
// covers that, counter-PRNG-reproducibly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>
#include <unistd.h>

#include "app/build_env.h"
#include "cli/options.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "logging/state_logger.h"
#include "logging/mem_state_io.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"
#include "sim_core/sim_thread_controller.h"

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t k_warmup_steps = 50;   // step N: enough for non-trivial state
constexpr seed_offset_t k_seed_offset  = {};

// True if any io_buffer in the extracted tree carries a non-zero "data" field —
// guards against a vacuous all-zero comparison that would pass even if apply did
// nothing.
bool any_buffer_nonzero(const logging::mem_group& tree) {
    auto it = tree.groups.find("io_buffers");
    if (it == tree.groups.end()) return false;
    for (const auto& [id, g] : it->second.groups) {
        (void)id;
        auto fit = g.fields.find("data");
        if (fit == g.fields.end()) continue;
        if (const auto* v = std::get_if<std::vector<float>>(&fit->second)) {
            if (std::any_of(v->begin(), v->end(), [](float x) { return x != 0.0f; })) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

TEST_CASE("state extract->serialize->parse->apply reproduces sim state (split path)") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;
    // Every env below is the same run, rebuilt: its checkpoints continue it, datasets
    // included (one run_id, as coastd fixes one per job; SPEC.md S.5.14).
    std::string run_id;

    const auto snap = fs::temp_directory_path() /
        ("coast_roundtrip_split_" + std::to_string(::getpid()) + ".h5");
    fs::remove(snap);

    // --- reference: run to step N, extract the tree, serialize to file -------
    logging::mem_group ref_tree;
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        run_id = sim->get_sim_env().run_id();
        coast::test::run_for(*sim, k_warmup_steps);

        auto* ctrl = sim->get_thread_controller();
        REQUIRE(ctrl);
        const std::uint64_t n = ctrl->get_step_count();
        REQUIRE(n == k_warmup_steps);

        ref_tree = logging::state_logger::extract_state(sim->get_sim_env(), n);

        logging::state_save_config cfg;
        cfg.output_path = snap;
        // serialize_state takes the tree by value; pass a copy so ref_tree stays
        // intact for the post-apply comparison.
        auto fut = logging::state_logger::serialize_state(ref_tree, cfg, h5);
        REQUIRE(fut.get().success);
        REQUIRE(fs::exists(snap));
    }   // drop the reference env (frees any GPU context) before building the next

    REQUIRE_MESSAGE(any_buffer_nonzero(ref_tree),
                    "reference io_buffers are all-zero; comparison would be vacuous");

    // --- fresh env: parse + apply, then re-extract and compare ---------------
    {
        auto sim = coast::app::build_env(opts, h5);
        REQUIRE(sim);
        sim->get_sim_env().set_run_id(run_id);

        auto pr = logging::state_logger::parse_state(snap, h5);
        REQUIRE(pr.result.success);
        for (const auto& w : pr.result.warnings) { INFO("parse warning: ", w); }
        CHECK(pr.result.warnings.empty());
        CHECK(pr.result.restored_step == k_warmup_steps);

        auto applied = logging::state_logger::apply_state(sim->get_sim_env(), pr.tree);
        REQUIRE(applied.success);
        for (const auto& w : applied.warnings) { INFO("apply warning: ", w); }
        CHECK(applied.warnings.empty());
        CHECK(applied.restored_step == k_warmup_steps);

        auto got_tree = logging::state_logger::extract_state(sim->get_sim_env(), k_warmup_steps);

        // Buffers + entities must reproduce bit-for-bit. (Metadata holds a
        // wall-clock timestamp, so it is deliberately excluded.)
        CHECK(got_tree.groups.at("io_buffers") == ref_tree.groups.at("io_buffers"));
        CHECK(got_tree.groups.at("entities")   == ref_tree.groups.at("entities"));
    }

    fs::remove(snap);
}
