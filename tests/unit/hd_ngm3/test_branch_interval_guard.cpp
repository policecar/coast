// tests/unit/hd_ngm3/test_branch_interval_guard.cpp
//
// Regression test for the branch_interval == 0 modulo-by-zero guard.
//
// neuron_group_t::process() gates branching on `(++input_cnt) % branch_interval`.
// branch_interval is a JSON-deserialized field, so a malformed config can set
// it to 0, which is undefined behaviour (modulo by zero). The constructor must
// reject that up front rather than letting it reach the per-step hot loop.
//
// Constructing a neuron_group_t needs only a cfg_store — no IO wiring, GPU
// context, or sim env — so this stays a pure sub-second unit test.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <stdexcept>

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"

namespace {

constexpr sim::output_id_t k_id           = 1;
constexpr seed_offset_t    k_seed_offset  = {};

} // namespace

TEST_CASE("ngm3 neuron_group_t rejects branch_interval == 0 at construction") {
    ngm3::cfg_store cs;                                  // "default" group, valid defaults
    cs.neuron_groups["default"].branch_interval = 0;     // malformed config

    CHECK_THROWS_AS(
        ngm3::neuron_group_t(cs, "default", k_id, k_seed_offset),
        std::runtime_error);
}

TEST_CASE("ngm3 neuron_group_t accepts a valid branch_interval") {
    ngm3::cfg_store cs;                                  // default branch_interval = 5000
    CHECK_NOTHROW(
        ngm3::neuron_group_t(cs, "default", k_id, k_seed_offset));

    cs.neuron_groups["default"].branch_interval = 1;     // smallest legal value
    CHECK_NOTHROW(
        ngm3::neuron_group_t(cs, "default", k_id, k_seed_offset));
}
