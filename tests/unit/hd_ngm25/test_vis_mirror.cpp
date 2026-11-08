// tests/unit/hd_ngm25/test_vis_mirror.cpp
//
// ngm2.5 increment I8: the vis/introspection surface (pitfall P5). Validates
// that refresh_vis_mirror materializes the sparse CSR arena into the dense
// ngm3-shaped [neuron × dendrite × rep × input_size] mirror correctly:
//   - every rep slot is filled (seg != 0): each leaf is covered by exactly one
//     synapse per input (antichain-cut property);
//   - a slot's seg id genuinely covers its leaf row (seg_interval contains r);
//   - seg ids are in range and permanences in [0,1];
//   - a root-only arena gives every rep row identical (all seg == 1);
//   - after branching, deeper seg ids appear and the invariants still hold.
// The accessors are the exact duck-typed contract the ngm3 flat/tree renderers
// consume, so this locks the reuse path.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <bit>
#include <memory>
#include <vector>

#include "hd_ngm25/hd_ngm25_arena.h"       // detail::seg_interval / seg_depth
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"

using namespace ngm25;

namespace {

struct rig {
    cfg_store                            cs;
    std::unique_ptr<neuron_group_t>      grp;
    std::unique_ptr<sim::io_buffer_cpu>  in_buf;
    std::unique_ptr<sim::io_buffer_cpu>  out_buf;
    static constexpr std::size_t W = 16;

    rig()
    {
        neuron_group_params_t p;
        p.neuron_cnt   = 8;
        p.apical_cnt   = 0;
        p.proximal_cnt = 2;
        p.max_branch_level = 3;
        p.branch_interval          = 15;      // force branching
        p.accumulated_theta_thres  = 0.05f;
        p.random_seed  = seed_t{ 3 };
        p.inputs       = { { "in", bank_t::proximal } };
        cs.neuron_groups["g"] = p;

        grp = std::make_unique<neuron_group_t>(cs, "g", 1, seed_offset_t{ 0 });
        grp->init_io({{"in", sim::output_channel_t{0, W}}});
        in_buf  = std::make_unique<sim::io_buffer_cpu>(W);
        out_buf = std::make_unique<sim::io_buffer_cpu>(grp->get_outp_size());
        auto w = in_buf->cur_write_buffer();
        for (std::size_t i = 0; i < W; ++i)
            w[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;
        in_buf->swap_buffer();
        grp->set_input_buffer(0, *in_buf);
        grp->set_output_buffer(*out_buf);
    }
    void step() { REQUIRE(grp->process().has_value()); }
};

// Check the proximal mirror against the invariants for the current arena state.
void check_proximal_mirror(const neuron_group_t &g, bool expect_root_only)
{
    const std::size_t ncnt = g.get_neuron_count();
    const std::size_t dcnt = g.get_proximal_count();
    const std::size_t rep  = g.get_rep_count();
    const uint8_t     msid = g.get_max_segment_id();
    const uint8_t     depth = g.get_branch_depth();

    for (std::size_t n = 0; n < ncnt; ++n)
        for (std::size_t d = 0; d < dcnt; ++d) {
            const auto r0_perm = g.get_proximal_representation(n, d, 0);
            const auto r0_seg  = g.get_proximal_segment_ids(n, d, 0);
            REQUIRE(r0_perm.size() == r0_seg.size());
            REQUIRE(!r0_perm.empty());
            const std::size_t isize = r0_perm.size();

            for (std::size_t r = 0; r < rep; ++r) {
                const auto perm = g.get_proximal_representation(n, d, r);
                const auto seg  = g.get_proximal_segment_ids(n, d, r);
                REQUIRE(perm.size() == isize);
                REQUIRE(seg.size()  == isize);
                for (std::size_t i = 0; i < isize; ++i) {
                    // completeness: every slot filled by a real synapse.
                    CHECK(seg[i] >= 1);
                    CHECK(seg[i] <= msid);
                    CHECK(perm[i] >= 0.0f);
                    CHECK(perm[i] <= 1.0f);
                    // the slot's seg genuinely covers leaf row r.
                    const auto span = detail::seg_interval(seg[i], depth);
                    CHECK(static_cast<int>(r) >= span.lo);
                    CHECK(static_cast<int>(r) <  span.hi);
                    if (expect_root_only) CHECK(seg[i] == 1);
                }
            }
        }
}

} // namespace

TEST_CASE("I8 vis mirror: root-only arena gives uniform all-root rep rows") {
    rig r;
    r.grp->refresh_vis_mirror();
    CHECK(r.grp->get_rep_count() == 8u);          // 2^3
    CHECK(r.grp->get_max_segment_id() == 15u);
    check_proximal_mirror(*r.grp, /*expect_root_only=*/true);
}

TEST_CASE("I8 vis mirror: after branching, deeper segs appear and invariants hold") {
    rig r;
    // Run until the group has grown, then 60 more steps to cross further branch
    // events. Found at runtime, not hardcoded: when this chaotic fixture first
    // grows depends on floating-point codegen (step 179 with AppleClang on arm64).
    const std::size_t root_only = r.grp->get_neuron_count() * r.grp->get_proximal_count() * rig::W;
    int s = 0;
    for (; s < 5000 && r.grp->get_synapse_count() <= root_only; ++s) r.step();
    REQUIRE(r.grp->get_synapse_count() > root_only);  // grew
    for (int extra = 0; extra < 60; ++extra) r.step();
    r.grp->refresh_vis_mirror();
    check_proximal_mirror(*r.grp, /*expect_root_only=*/false);

    // At least one rep slot must now carry a non-root seg (branching happened).
    bool saw_deep = false;
    for (std::size_t n = 0; n < r.grp->get_neuron_count() && !saw_deep; ++n)
        for (std::size_t d = 0; d < r.grp->get_proximal_count() && !saw_deep; ++d)
            for (std::size_t rep = 0; rep < r.grp->get_rep_count() && !saw_deep; ++rep) {
                const auto seg = r.grp->get_proximal_segment_ids(n, d, rep);
                for (uint8_t s : seg) if (s > 1) { saw_deep = true; break; }
            }
    CHECK(saw_deep);
    // (Dropped three tautological aggregate-stat asserts here — max>=mean, avg>=0,
    // count>0 all hold by construction and validated nothing; the mirror content
    // and saw_deep checks above are the real coverage.)
}
