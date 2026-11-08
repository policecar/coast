// tests/unit/hd_ngm25/test_process_guard.cpp
//
// Review should-fix #6: the in-step post-repack consistency guard, exercised
// THROUGH process() itself.
//
// hd_ngm25_neuron_group.cpp's process() runs the full consistency checker after a
// branch/repack and halts via the value channel (COAST_REQUIRE ->
// contract_violation step_error) on a violation — the safety net for a repack bug
// that would otherwise corrupt silently. Prior tests only drove check_arena_
// consistency() directly (test_arena_consistency, test_branch) or via load_state;
// nothing proved process() actually invokes it. A correct repack never produces a
// corrupt arena, and the load path runs the same check, so the only way to reach
// this guard through process() is to inject a fault mid-life. We do that through a
// narrow test-only friend seam (neuron_group_test_access), corrupting a field that:
//   - the checker rejects (dend_neuron must follow the fixed directory order, inv2),
//   - repack_with_splits preserves verbatim (dend_type/dend_neuron are copied), and
//   - the compute path tolerates for the pre-branch steps (a DIFFERENT but in-range
//     neuron id — no out-of-bounds, so nothing crashes before the branch fires).
// The first branch after the corruption then trips the guard.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <memory>
#include <vector>

#include "core/step_error.h"
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"

namespace ngm25 {
// The seam declared `friend` by neuron_group_t: hands a unit test the private
// arena so it can inject a repack-preserved fault.
struct neuron_group_test_access {
    static synapse_arena_t& arena(neuron_group_t& g) { return g.arena_; }
};
} // namespace ngm25

using namespace ngm25;

namespace {

// Branch-forcing config (mirrors test_persistence::small_group): a short interval
// and low gates make qualifying branch events appear within a few dozen steps.
neuron_group_params_t branchy_group() {
    neuron_group_params_t p;
    p.neuron_cnt              = 8;
    p.apical_cnt              = 0;
    p.proximal_cnt            = 2;
    p.max_branch_level        = 3;
    p.branch_interval         = 15;
    p.accumulated_theta_thres = 0.05f;
    p.min_mismatch_percentage = 0.002f;
    p.random_seed             = seed_t{ 7 };
    p.inputs                  = { { "in", bank_t::proximal } };
    return p;
}

// A group wired to a fixed-content input buffer and a private output buffer.
struct rig {
    cfg_store                            cs;
    std::unique_ptr<neuron_group_t>      grp;
    std::unique_ptr<sim::io_buffer_cpu>  in_buf;
    std::unique_ptr<sim::io_buffer_cpu>  out_buf;
    static constexpr std::size_t      W     = 16;
    static constexpr sim::output_id_t in_id = 0;

    rig() {
        cs.neuron_groups["g"] = branchy_group();
        grp = std::make_unique<neuron_group_t>(cs, "g", /*output_id=*/1, seed_offset_t{ 0 });
        grp->init_io({{"in", sim::output_channel_t{in_id, W}}});

        in_buf  = std::make_unique<sim::io_buffer_cpu>(W);
        out_buf = std::make_unique<sim::io_buffer_cpu>(grp->get_outp_size());
        auto w = in_buf->cur_write_buffer();
        for (std::size_t i = 0; i < W; ++i)
            w[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;
        in_buf->swap_buffer();                       // fixed read page
        grp->set_input_buffer(in_id, *in_buf);
        grp->set_output_buffer(*out_buf);
    }
};

// The step at which this fixture first grows is found at runtime, not hardcoded:
// the run is chaotic, so it depends on floating-point codegen (179 with AppleClang
// on arm64; later with GCC, where a fixed 240-step budget saw no growth at all).
// k_step_cap only bounds the search.
constexpr int k_step_cap = 5000;

// The first step after which a healthy fixture has grown, or -1 within the cap.
int first_growth_step()
{
    rig r;
    const std::size_t syn0 = r.grp->get_synapse_count();
    for (int s = 0; s < k_step_cap; ++s) {
        REQUIRE(r.grp->process().has_value());
        if (r.grp->get_synapse_count() > syn0) return s;
    }
    return -1;
}

} // namespace

// Positive control: a healthy group branches within the window and every step
// succeeds. This proves the negative test below is not vacuous — a branch (hence a
// guard invocation) really does occur — and that a CORRECT repack passes the guard.
TEST_CASE("ngm25 process guard: a healthy group branches and never trips the guard") {
    rig r;
    const std::size_t syn0 = r.grp->get_synapse_count();
    int grew_at = -1;
    for (int s = 0; s < k_step_cap && grew_at < 0; ++s) {
        REQUIRE(r.grp->process().has_value());       // no false halt
        if (r.grp->get_synapse_count() > syn0) grew_at = s;
    }
    MESSAGE("first growth at step " << grew_at);
    REQUIRE(grew_at >= 0);   // a branch actually happened -> the guard ran and passed
    // A few more branch intervals past the first growth, still never halting.
    for (int s = 0; s < 60; ++s) REQUIRE(r.grp->process().has_value());
}

// Negative: corrupt dend_neuron[0] to a different in-range neuron id before any
// branch. The compute path tolerates it (valid index), repack preserves it, and
// the first branch's consistency check rejects it (inv2) -> process() returns a
// contract_violation step_error rather than corrupting on silently.
TEST_CASE("ngm25 process guard: a corrupted repack halts process() via the value channel") {
    // A healthy fixture grows within the cap, so a missing halt below means the
    // guard failed, not that no branch (hence no guard invocation) happened.
    const int grows_at = first_growth_step();
    REQUIRE(grows_at >= 0);

    rig r;

    auto& arena = neuron_group_test_access::arena(*r.grp);
    REQUIRE(arena.dend_neuron.size() >= 1);
    const std::uint16_t orig = arena.dend_neuron[0];         // dendrite 0 -> neuron 0
    arena.dend_neuron[0] = static_cast<std::uint16_t>((orig + 1) % 8);  // valid, wrong
    REQUIRE(arena.dend_neuron[0] != orig);

    bool halted = false;
    for (int s = 0; s <= grows_at && !halted; ++s) {
        const sim::step_result res = r.grp->process();
        if (!res.has_value()) {
            CHECK(res.error().kind == core::error_kind::contract_violation);
            halted = true;
        }
    }
    // The first branch after the corruption must have driven the guard to halt.
    CHECK(halted);
}
