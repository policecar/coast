// tests/integration_ngm3/synthetic_smoke_test.cpp
//
// A-tier-style smoke test for ngm3. Builds a tiny synthetic ngm3 sim,
// runs it for a fixed number of steps, then asserts sanity properties
// on the group state. Not a golden comparison — purely structural.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cmath>
#include <filesystem>

#include "app/build_env.h"
#include "cli/options.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "hdf5/executor.h"
#include "integration_helpers.h"
#include "sim_core/sim_env.h"
#include "sim_core/sim_instance.h"

namespace {

constexpr std::uint64_t       k_steps         = 20000; // multiple branch_intervals (branch_interval=5000)
constexpr sim::output_id_t    k_ngm3_id       = 1;
constexpr seed_offset_t       k_seed_offset   = {};

ngm3::neuron_group_t&
require_group(coast::sim_instance& sim, sim::output_id_t id)
{
    auto* ent = sim.get_sim_env().get_entity(id);
    REQUIRE(ent != nullptr);
    auto* ng  = dynamic_cast<ngm3::neuron_group_t*>(ent);
    REQUIRE(ng  != nullptr);
    return *ng;
}

template<typename T>
void check_finite(const std::span<T const> values, const char* label)
{
    for (T v : values) {
        const bool ok = std::isfinite(static_cast<double>(v));
        if (!ok) {
            INFO("non-finite value in ", label);
            CHECK(ok);
            return;
        }
    }
}

} // namespace

TEST_CASE("ngm3 synthetic smoke") {
    coast::cli::options opts;
    opts.config_path = COAST_TEST_CONFIG;
    opts.seed_offset = k_seed_offset;

    hdf5::executor h5;
    auto sim = coast::app::build_env(opts, h5);
    REQUIRE(sim);

    coast::test::run_for(*sim, k_steps);

    auto& g = require_group(*sim, k_ngm3_id);

    // --- structural sanity -----------------------------------------------
    CHECK(g.get_neuron_count()        == 8);
    CHECK(g.get_apical_count()        == 2);
    CHECK(g.get_proximal_count()      == 4);
    CHECK(g.get_branch_depth()        == 3);
    CHECK(g.get_rep_count()           == 8);
    CHECK(g.get_apical_input_size()    > 0);
    CHECK(g.get_proximal_input_size()  > 0);

    const std::size_t n     = g.get_neuron_count();
    const std::size_t a_cnt = g.get_apical_count();
    const std::size_t p_cnt = g.get_proximal_count();
    const std::size_t rep   = g.get_rep_count();

    // --- representations have valid segment_ids in [1, max_seg_id]
    // and finite permanences in [0, 1] -----------------------------------
    const uint8_t max_sid = static_cast<uint8_t>((1u << (g.get_branch_depth() + 1)) - 1);

    for (std::size_t ni = 0; ni < n; ++ni) {
        for (std::size_t di = 0; di < a_cnt; ++di) {
            for (std::size_t ri = 0; ri < rep; ++ri) {
                auto perm = g.get_apical_representation(ni, di, ri);
                auto sids = g.get_apical_segment_ids   (ni, di, ri);
                REQUIRE(perm.size() == g.get_apical_input_size());
                REQUIRE(sids.size() == g.get_apical_input_size());
                check_finite<float>(perm, "apical permanences");
                for (float p : perm) CHECK((p >= 0.0f && p <= 1.0f));
                for (uint8_t s : sids) {
                    CHECK(s >= 1);
                    CHECK(s <= max_sid);
                }
            }
        }
        for (std::size_t di = 0; di < p_cnt; ++di) {
            for (std::size_t ri = 0; ri < rep; ++ri) {
                auto perm = g.get_proximal_representation(ni, di, ri);
                auto sids = g.get_proximal_segment_ids   (ni, di, ri);
                REQUIRE(perm.size() == g.get_proximal_input_size());
                REQUIRE(sids.size() == g.get_proximal_input_size());
                check_finite<float>(perm, "proximal permanences");
                for (float p : perm) CHECK((p >= 0.0f && p <= 1.0f));
                for (uint8_t s : sids) {
                    CHECK(s >= 1);
                    CHECK(s <= max_sid);
                }
            }
        }
    }

    // --- at least one branch event occurred -------------------------------
    // Initial total: every dendrite of both banks starts with a single
    // distinct seg_id (=1) → total = n * (a_cnt + p_cnt). After at least
    // one split, total must exceed that baseline.
    const std::size_t initial_total = n * (a_cnt + p_cnt);
    const std::size_t cur_total     = g.get_total_representation_count();
    INFO("initial total reps = ", initial_total, "  cur total = ", cur_total);
    CHECK(cur_total > initial_total);

    // --- output buffer of the ngm3 group is finite and in [0, 1] ---------
    auto& buffers = sim->get_sim_env().get_io_buffers();
    auto it = buffers.find(k_ngm3_id);
    REQUIRE(it != buffers.end());
    auto out = it->second->cur_read_buffer();
    REQUIRE(out.size() == n);
    for (float v : out) {
        CHECK(std::isfinite(v));
        CHECK((v >= 0.0f && v <= 1.0f));
    }

    // --- mismatch sanity -------------------------------------------------
    CHECK(std::isfinite(g.get_max_mismatch()));
    CHECK(std::isfinite(g.get_avg_mismatch()));
    CHECK(g.get_max_mismatch() >= 0.0f);
    CHECK(g.get_avg_mismatch() >= 0.0f);
}
