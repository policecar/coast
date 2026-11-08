// tests/unit/hd_ngm25/test_branch.cpp
//
// ngm2.5 increment I5: branch integration (map R8). Gates:
//   - decide predicate: hand-built synthetic states, the qualifying set computed
//     independently, verify branch_decide_dendrite flags exactly those; covers
//     the adapt_history / mismatch / seg-cap / percentage-gate conditions and
//     the variance-as-std threshold.
//   - vs REAL ngm2: inject a synthetic high-mismatch state into an ngm2 dendrite
//     (save -> mutate the tree -> load), run adapt_branches, and confirm ngm2.5
//     branches the SAME synapses into the same seg structure (schedule/growth
//     agreement — structure, not the wiggled permanences).
//   - cadence: maybe_branch_group fires only for neurons that crossed a
//     branch_interval boundary this step (winner's +2 crossing included).
//   - property: force many branch events; the consistency checker stays green
//     after every re-pack and branch_epoch advances.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_branch.h"

#include "hd_ngm2/hd_ngm2_dendrite.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"
#include "logging/mem_state_io.h"

using namespace ngm25;

namespace {

// A single proximal dendrite, W root synapses, max_depth `level`.
group_shape_t proximal_shape(std::size_t W, uint8_t level)
{
    group_shape_t s;
    s.neuron_cnt = 1; s.apical_cnt = 0; s.proximal_cnt = 1;
    s.max_depth  = level;
    s.max_seg_id = static_cast<uint8_t>((1u << (level + 1)) - 1);
    s.input_size[static_cast<int>(bank_t::proximal)] = W;
    s.input_lo  [static_cast<int>(bank_t::proximal)] = 0;
    s.input_hi  [static_cast<int>(bank_t::proximal)] = W;
    return s;
}

synapse_arena_t root_arena(const std::vector<float> &mismatch,
                           const std::vector<float> &adapt_hist)
{
    const std::size_t W = mismatch.size();
    synapse_arena_t a;
    a.perm.assign(W, 0.5f);
    a.seg.assign(W, 1);
    a.mismatch   = mismatch;
    a.adapt_hist = adapt_hist;
    a.input_idx.resize(W);
    for (uint32_t i = 0; i < W; ++i) a.input_idx[i] = static_cast<uint16_t>(i);
    a.dend_offset = {0, static_cast<uint32_t>(W)};
    a.dend_type   = {bank_t::proximal};
    a.dend_neuron = {0};
    return a;
}

// Independent oracle: the exact ngm2 predicate on a single root dendrite.
std::vector<uint8_t> oracle(const synapse_arena_t &a, const group_shape_t &shape,
                            const branch_params_t &bp)
{
    const std::size_t W = a.syn_count();
    const float mm_avg = std::reduce(a.mismatch.begin(), a.mismatch.end(), 0.0f)
                       / static_cast<float>(W);
    float var = 0.0f;
    for (float mm : a.mismatch) var += (mm_avg - mm) * (mm_avg - mm);
    var /= static_cast<float>(W);
    const float input_size = static_cast<float>(shape.input_size[1]);
    const float mm_thres = mm_avg + var * bp.min_mismatch_deviation + 1.0f / input_size;

    std::vector<uint8_t> q(W, 0);
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < W; ++i)
        if (a.adapt_hist[i] >= bp.accumulated_theta_thres && a.mismatch[i] >= mm_thres
            && 2u * a.seg[i] + 1u <= shape.max_seg_id) { q[i] = 1; ++cnt; }
    if (static_cast<float>(cnt) < input_size * bp.min_mismatch_percentage)
        std::fill(q.begin(), q.end(), 0);   // gate not met -> nothing branches
    return q;
}

// ngm2 dendrite harness (owns cfg/buffer/dendrite; see test_adapt.cpp).
struct ngm2_dend {
    ngm2::cfg_store                     cs;
    std::unique_ptr<sim::io_buffer_cpu> buf;
    std::unique_ptr<ngm2::dendrite_t>   d;
    sim::output_id_t                    id = 7;
};

ngm2_dend make_ngm2_dend(std::size_t W, uint8_t level, int seed)
{
    ngm2_dend h;
    ngm2::dendrite_params_t p;
    p.type = ngm2::dendrite_type_t::proximal;
    p.max_branch_level = level;
    h.cs.dendrites["t"] = p;
    h.d = std::make_unique<ngm2::dendrite_t>(h.cs, "t", seed);
    h.d->init_input({{"chan", ngm2::channel_source_t{ sim::output_channel_t{h.id, W}, "chan" }}}, "test dendrite");
    h.buf = std::make_unique<sim::io_buffer_cpu>(W);
    auto w = h.buf->cur_write_buffer(); std::fill(w.begin(), w.end(), 0.5f); h.buf->swap_buffer();
    h.d->set_input_buffer(h.id, *h.buf);
    return h;
}

std::vector<uint8_t> sorted_segs(const synapse_arena_t &a)
{
    std::vector<uint8_t> v = a.seg;
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

// ---------------------------------------------------------------------------
// Decide predicate — independent oracle.
// ---------------------------------------------------------------------------
TEST_CASE("R8 decide: qualifiers flagged exactly, clear branch") {
    const group_shape_t shape = proximal_shape(16, 3);
    branch_params_t bp;                                  // defaults

    std::vector<float> mm(16, 0.1f), ah(16, 0.0f);
    for (int i : {0, 1, 2}) { mm[i] = 0.9f; ah[i] = 3.0f; }   // three qualifiers
    synapse_arena_t a = root_arena(mm, ah);

    std::vector<uint8_t> flags(16, 0);
    const std::size_t n = detail::branch_decide_dendrite(a, 0, shape, bp, flags);
    CHECK(n == 3);
    CHECK(flags == oracle(a, shape, bp));
    CHECK(std::count(flags.begin(), flags.end(), uint8_t{1}) == 3);
}

TEST_CASE("R8 decide: adapt_history below threshold => no branch") {
    const group_shape_t shape = proximal_shape(16, 3);
    branch_params_t bp;
    std::vector<float> mm(16, 0.1f), ah(16, 0.0f);
    for (int i : {0, 1, 2}) { mm[i] = 0.9f; ah[i] = 1.0f; }   // adapt_history < 2.0
    synapse_arena_t a = root_arena(mm, ah);

    std::vector<uint8_t> flags(16, 0);
    CHECK(detail::branch_decide_dendrite(a, 0, shape, bp, flags) == 0);
    CHECK(std::count(flags.begin(), flags.end(), uint8_t{1}) == 0);
}

TEST_CASE("R8 decide: seg cap excludes leaf synapses at max depth") {
    // max_depth 1: seg ids 2,3 are leaves (2*2+1=5 > max_seg_id 3). A fully split
    // input (segs {2,3}) can never branch further, even at high mismatch.
    group_shape_t shape = proximal_shape(1, 1);
    branch_params_t bp;
    synapse_arena_t a;
    a.perm = {0.5f, 0.5f}; a.seg = {2, 3};
    a.mismatch = {0.9f, 0.9f}; a.adapt_hist = {3.0f, 3.0f};
    a.input_idx = {0, 0};
    a.dend_offset = {0, 2}; a.dend_type = {bank_t::proximal}; a.dend_neuron = {0};
    std::string err; REQUIRE(check_arena_consistency(a, shape, &err));

    std::vector<uint8_t> flags(2, 0);
    CHECK(detail::branch_decide_dendrite(a, 0, shape, bp, flags) == 0);   // capped
}

TEST_CASE("R8 decide: percentage gate blocks a too-small qualifying set") {
    const group_shape_t shape = proximal_shape(16, 3);
    branch_params_t bp; bp.min_mismatch_percentage = 0.5f;   // need >= 8 qualifiers
    std::vector<float> mm(16, 0.1f), ah(16, 0.0f);
    for (int i : {0, 1, 2}) { mm[i] = 0.9f; ah[i] = 3.0f; }   // only 3 -> below gate
    synapse_arena_t a = root_arena(mm, ah);

    std::vector<uint8_t> flags(16, 0);
    CHECK(detail::branch_decide_dendrite(a, 0, shape, bp, flags) == 0);
    CHECK(std::count(flags.begin(), flags.end(), uint8_t{1}) == 0);
    CHECK(flags == oracle(a, shape, bp));
}

// ---------------------------------------------------------------------------
// vs REAL ngm2: same decide => same branch structure.
// ---------------------------------------------------------------------------
TEST_CASE("R8 branch structure matches real ngm2 on an injected state") {
    const std::size_t W = 16; const uint8_t level = 3;
    ngm2_dend h = make_ngm2_dend(W, level, 3);

    // Inject a synthetic high-mismatch state via save -> mutate -> load.
    logging::mem_state_writer wtr;
    h.d->save_state(wtr);
    logging::mem_group tree = wtr.take_root();
    std::vector<float> mm(W, 0.1f), ah(W, 0.0f);
    for (int i : {0, 1, 2}) { mm[i] = 0.9f; ah[i] = 3.0f; }
    tree.fields["mismatch"]      = mm;
    tree.fields["adapt_history"] = ah;
    { logging::mem_state_reader rdr(tree, logging::k_format_version); h.d->load_state(rdr, "ngm2 dendrite"); }

    const std::size_t n_pre = h.d->get_synapses().size();
    h.d->adapt_branches();
    const auto &syn = h.d->get_synapses();
    const std::size_t n_post = syn.size();

    std::vector<uint8_t> ngm2_segs(syn.segment_idx.begin(), syn.segment_idx.end());
    std::sort(ngm2_segs.begin(), ngm2_segs.end());

    // ngm2.5 on the same injected state.
    const group_shape_t shape = proximal_shape(W, level);
    std::vector<float> perm(h.d->get_synapses().permanence.begin(),   // post-branch perms
                            h.d->get_synapses().permanence.end());
    (void)perm;
    synapse_arena_t a = root_arena(mm, ah);            // W root synapses, same mm/ah
    branch_params_t bp;
    std::vector<uint8_t> flags(W, 0);
    detail::branch_decide_dendrite(a, 0, shape, bp, flags);
    repack_with_splits(a, flags, shape, /*seed=*/123, /*epoch=*/0);

    CHECK(n_pre == W);
    CHECK(a.syn_count() == n_post);                    // same growth
    CHECK(sorted_segs(a) == ngm2_segs);                // same seg structure

    std::string err = "untouched";
    CHECK(check_arena_consistency(a, shape, &err));    // and consistent

    // Explicit expected structure: inputs 0,1,2 -> {2,3}, the rest -> {1}.
    for (uint32_t s = 0; s < a.syn_count(); ++s) {
        const uint16_t in = a.input_idx[s];
        if (in <= 2) CHECK((a.seg[s] == 2 || a.seg[s] == 3));
        else         CHECK(a.seg[s] == 1);
    }
}

// ---------------------------------------------------------------------------
// Cadence predicate: the crossing test must catch a winner's +2 jump-OVER, which
// a naive `count % interval == 0` landing test would silently skip.
// ---------------------------------------------------------------------------
TEST_CASE("R8 cadence: crossed_branch_interval detects a jump-over, not just a landing") {
    using detail::crossed_branch_interval;
    constexpr std::uint64_t I = 5000;

    // Non-winner (+1): fires exactly when the new count is a multiple — same as ngm2.
    CHECK(crossed_branch_interval(4999, 5000, I));           // lands on 5000
    CHECK_FALSE(crossed_branch_interval(5000, 5001, I));     // just past, no new multiple

    // Winner (+2): the load-bearing case. 4999 -> 5001 jumps OVER 5000 without ever
    // equalling it; the branch MUST still fire. `5001 % 5000 == 1`, so a modulo-on-
    // final-count test would wrongly return false here.
    CHECK(crossed_branch_interval(4999, 5001, I));           // jump-over -> still crosses
    CHECK(5001 % I != 0);                                    // ...and a %-test would miss it
    CHECK(crossed_branch_interval(4998, 5000, I));           // +2 landing on 5000
    CHECK_FALSE(crossed_branch_interval(5000, 5002, I));     // both past the boundary

    // Exhaustive small-interval check: for every start k, a +1 step crosses iff the
    // new count is a multiple, and a +2 step crosses iff EITHER intermediate or
    // final count is a multiple (the winner never skips a boundary).
    constexpr std::uint64_t J = 7;
    for (std::uint64_t k = 0; k < 100; ++k) {
        CHECK(crossed_branch_interval(k, k + 1, J) == ((k + 1) % J == 0));
        CHECK(crossed_branch_interval(k, k + 2, J)
              == (((k + 1) % J == 0) || ((k + 2) % J == 0)));
    }
}

// ---------------------------------------------------------------------------
// Cadence: maybe_branch_group fires per crossed neuron only.
// ---------------------------------------------------------------------------
TEST_CASE("R8 gate: only neurons crossing a branch_interval boundary branch") {
    // Two neurons, one proximal dendrite each, 4 root synapses per dendrite.
    group_shape_t shape;
    shape.neuron_cnt = 2; shape.apical_cnt = 0; shape.proximal_cnt = 1;
    shape.max_depth = 3; shape.max_seg_id = 15;
    shape.input_size[1] = 4; shape.input_lo[1] = 0; shape.input_hi[1] = 4;

    // Both dendrites carry a branch-worthy OUTLIER (one high-mismatch synapse
    // among low ones — the predicate branches outliers, not uniform-high sets:
    // equal mismatches give variance 0 and mm_thres = avg + 1/input_size clears
    // every synapse).
    synapse_arena_t a;
    a.perm.assign(8, 0.5f); a.seg.assign(8, 1);
    a.mismatch   = {0.9f, 0.1f, 0.1f, 0.1f,  0.9f, 0.1f, 0.1f, 0.1f};
    a.adapt_hist = {3.0f, 0.0f, 0.0f, 0.0f,  3.0f, 0.0f, 0.0f, 0.0f};
    a.input_idx = {0, 1, 2, 3, 0, 1, 2, 3};
    a.dend_offset = {0, 4, 8};
    a.dend_type   = {bank_t::proximal, bank_t::proximal};
    a.dend_neuron = {0, 1};
    std::string err; REQUIRE(check_arena_consistency(a, shape, &err));

    branch_params_t bp; bp.branch_interval = 5000;
    // Neuron 0 crosses (4999 -> 5001, a winner-style +2). Neuron 1 does not
    // (5000 -> 5001, boundary already passed last step).
    const std::vector<std::uint64_t> before = {4999, 5000};
    const std::vector<std::uint64_t> after  = {5001, 5001};

    uint32_t epoch = 7;
    const bool repacked = detail::maybe_branch_group(a, shape, bp, before, after, 42, epoch);
    CHECK(repacked);
    CHECK(epoch == 8);                                 // advanced once

    // Neuron 0's dendrite grew (4 -> 5: its one outlier split); neuron 1's is
    // untouched (it did not cross, so its outlier was never decided).
    CHECK(a.dendrite(0).end - a.dendrite(0).begin == 5);
    CHECK(a.dendrite(1).end - a.dendrite(1).begin == 4);
    for (uint32_t s = a.dendrite(1).begin; s < a.dendrite(1).end; ++s)
        CHECK(a.seg[s] == 1);                          // neuron 1 never branched
    CHECK(check_arena_consistency(a, shape, &err));

    // No crossing anywhere => no re-pack, epoch stable.
    uint32_t epoch2 = 8;
    const std::vector<std::uint64_t> b2 = {5001, 5001}, a2 = {5002, 5002};
    CHECK_FALSE(detail::maybe_branch_group(a, shape, bp, b2, a2, 42, epoch2));
    CHECK(epoch2 == 8);
}

// ---------------------------------------------------------------------------
// Property: force branching every round until the depth cap; checker stays
// green after every re-pack.
// ---------------------------------------------------------------------------
TEST_CASE("R8 repeated forced branching keeps the arena consistent") {
    group_shape_t shape;
    shape.neuron_cnt = 4; shape.apical_cnt = 1; shape.proximal_cnt = 2;
    shape.max_depth = 3; shape.max_seg_id = 15;
    shape.input_size[0] = 3; shape.input_lo[0] = 0; shape.input_hi[0] = 3;
    shape.input_size[1] = 5; shape.input_lo[1] = 3; shape.input_hi[1] = 8;

    synapse_arena_t a = build_arena(shape, 0.3f, 5);
    std::string err = "untouched";
    REQUIRE(check_arena_consistency(a, shape, &err));

    branch_params_t bp; bp.branch_interval = 1;        // every step crosses
    uint32_t epoch = 0;
    const std::vector<std::uint64_t> before(shape.neuron_cnt, 0);
    const std::vector<std::uint64_t> after(shape.neuron_cnt, 1);

    int events = 0;
    for (int round = 0; round < 8; ++round) {
        // Make ONE splittable synapse per dendrite a high-mismatch OUTLIER (the
        // rest low), so the predicate qualifies it — a uniform-high drive would
        // branch nothing (variance 0). This splits one synapse per dendrite per
        // round until every synapse reaches the depth cap.
        for (std::size_t s = 0; s < a.syn_count(); ++s) {
            a.mismatch[s]   = 0.05f;
            a.adapt_hist[s] = 0.0f;
        }
        for (std::size_t d = 0; d < a.dend_count(); ++d) {
            const auto rng = a.dendrite(d);
            for (uint32_t s = rng.begin; s < rng.end; ++s)
                if (2u * a.seg[s] + 1u <= shape.max_seg_id) {   // first splittable
                    a.mismatch[s]   = 0.95f;
                    a.adapt_hist[s] = 3.0f;
                    break;
                }
        }
        const std::size_t before_cnt = a.syn_count();
        const bool repacked = detail::maybe_branch_group(a, shape, bp, before, after, 99, epoch);

        err = "untouched";
        REQUIRE(check_arena_consistency(a, shape, &err));   // green after every event
        if (repacked) {
            ++events;
            CHECK(a.syn_count() > before_cnt);              // it grew
        } else {
            CHECK(a.syn_count() == before_cnt);             // all capped -> stable
        }
    }
    CHECK(events >= 3);                                     // depth 3 => several rounds
    CHECK(epoch == static_cast<uint32_t>(events));          // one epoch bump per event
}
