// tests/unit/hd_ngm25/test_arena_consistency.cpp
//
// ngm2.5 increment I1. Two things under test:
//   1. detail::seg_depth / detail::seg_interval — the heap-node -> integer
//      dyadic-interval math that invariant 4 (the antichain cut) rests on.
//   2. check_arena_consistency + build_arena — a freshly built arena passes
//      the checker, and corrupting each invariant makes the checker fail with
//      a message that names the invariant it caught.
//
// Pure module: no IO wiring, no sim env. Sub-second.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <string>

#include "hd_ngm25/hd_ngm25_arena.h"

using namespace ngm25;

namespace {

// A representative multi-neuron, two-type shape:
//   3 neurons x (1 apical + 2 proximal) = 9 dendrites.
//   apical reads positions [0,4), proximal reads [4,9) of the concatenated
//   input. max_depth 3 -> max_seg_id 15, rep axis has 2^3 = 8 leaf slots.
group_shape_t make_shape()
{
    group_shape_t s;
    s.neuron_cnt   = 3;
    s.apical_cnt   = 1;
    s.proximal_cnt = 2;
    s.max_depth    = 3;
    s.max_seg_id   = (1 << (s.max_depth + 1)) - 1;   // 15
    s.input_size[static_cast<int>(bank_t::apical)]   = 4;
    s.input_lo  [static_cast<int>(bank_t::apical)]   = 0;
    s.input_hi  [static_cast<int>(bank_t::apical)]   = 4;
    s.input_size[static_cast<int>(bank_t::proximal)] = 5;
    s.input_lo  [static_cast<int>(bank_t::proximal)] = 4;
    s.input_hi  [static_cast<int>(bank_t::proximal)] = 9;
    return s;
}

// Per neuron: apical(4) + proximal(5) + proximal(5) = 14 synapses.
constexpr std::size_t k_syn_per_neuron = 4 + 5 + 5;

} // namespace

// ---------------------------------------------------------------------------
// 1. Segment-tree geometry (invariant 4 support), tested directly.
// ---------------------------------------------------------------------------
TEST_CASE("detail::seg_depth is floor(log2(seg))") {
    CHECK(detail::seg_depth(1) == 0);
    CHECK(detail::seg_depth(2) == 1);
    CHECK(detail::seg_depth(3) == 1);
    CHECK(detail::seg_depth(4) == 2);
    CHECK(detail::seg_depth(7) == 2);
    CHECK(detail::seg_depth(8) == 3);
    CHECK(detail::seg_depth(15) == 3);
}

TEST_CASE("detail::seg_interval tiles the rep axis dyadically") {
    const uint8_t md = 3;                 // axis = [0,8)
    // root covers the whole axis
    CHECK(detail::seg_interval(1, md).lo == 0);
    CHECK(detail::seg_interval(1, md).hi == 8);
    // depth-1 children split it in half
    CHECK(detail::seg_interval(2, md).lo == 0);
    CHECK(detail::seg_interval(2, md).hi == 4);
    CHECK(detail::seg_interval(3, md).lo == 4);
    CHECK(detail::seg_interval(3, md).hi == 8);
    // depth-2 quarter and depth-3 leaf
    CHECK(detail::seg_interval(6, md).lo == 4);
    CHECK(detail::seg_interval(6, md).hi == 6);
    CHECK(detail::seg_interval(8,  md).lo == 0);
    CHECK(detail::seg_interval(8,  md).hi == 1);
    CHECK(detail::seg_interval(15, md).lo == 7);
    CHECK(detail::seg_interval(15, md).hi == 8);

    // A valid antichain cut {2, 6, 7} (left half + two right quarters) tiles
    // [0,8) with no gap or overlap; the checker relies on exactly this.
    auto a = detail::seg_interval(2, md);   // [0,4)
    auto b = detail::seg_interval(6, md);   // [4,6)
    auto c = detail::seg_interval(7, md);   // [6,8)
    CHECK(a.lo == 0); CHECK(a.hi == 4);
    CHECK(b.lo == 4); CHECK(b.hi == 6);
    CHECK(c.lo == 6); CHECK(c.hi == 8);
}

// ---------------------------------------------------------------------------
// 2. build_arena produces a checker-passing arena.
// ---------------------------------------------------------------------------
TEST_CASE("build_arena: fresh arena passes the consistency checker") {
    const group_shape_t shape = make_shape();
    const synapse_arena_t arena = build_arena(shape, /*perm_threshold=*/0.3f, /*seed=*/42);

    std::string err = "untouched";
    const bool ok = check_arena_consistency(arena, shape, &err);
    INFO("checker error: " << err);
    CHECK(ok);
    CHECK(err == "untouched");   // the checker must not clobber err on success
}

TEST_CASE("build_arena: structural sizes and initial contents match the shape") {
    const group_shape_t shape = make_shape();
    const synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    CHECK(arena.dend_count() == 9);
    CHECK(arena.syn_count()  == 3 * k_syn_per_neuron);   // 42

    // Directory follows the fixed order: neuron-major, apical-then-proximal.
    CHECK(arena.dend_type[0]   == bank_t::apical);
    CHECK(arena.dend_type[1]   == bank_t::proximal);
    CHECK(arena.dend_type[2]   == bank_t::proximal);
    CHECK(arena.dend_type[3]   == bank_t::apical);       // neuron 1 starts
    CHECK(arena.dend_neuron[0] == 0);
    CHECK(arena.dend_neuron[3] == 1);
    CHECK(arena.dend_neuron[8] == 2);

    // Every synapse starts at the root, permanences are valid, and input-major
    // ordering holds inside the first (apical) dendrite: inputs 0,1,2,3.
    for (std::size_t s = 0; s < arena.syn_count(); ++s) {
        CHECK(arena.seg[s] == 1);
        CHECK(arena.perm[s] >= 0.0f);
        CHECK(arena.perm[s] <= 1.0f);
        CHECK(arena.adapt_hist[s] == 0.0f);
        CHECK(arena.mismatch[s]   == 0.0f);
    }
    const auto d0 = arena.dendrite(0);
    CHECK(d0.end - d0.begin == 4);
    for (uint32_t i = d0.begin, pos = 0; i < d0.end; ++i, ++pos)
        CHECK(arena.input_idx[i] == pos);          // 0,1,2,3

    // Determinism: same seed -> identical permanences (D5 documented order).
    const synapse_arena_t again = build_arena(shape, 0.3f, 42);
    CHECK(again.perm == arena.perm);
    // Different seed -> a different draw (guards against a dead engine).
    const synapse_arena_t other = build_arena(shape, 0.3f, 43);
    CHECK(other.perm != arena.perm);
}

// ---------------------------------------------------------------------------
// 3. Each invariant, corrupted, is caught with a message that names it.
// ---------------------------------------------------------------------------
TEST_CASE("checker inv1: mismatched SoA vector length is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.perm.pop_back();          // now perm shorter than the other SoA arrays
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv1") != std::string::npos);
}

TEST_CASE("checker inv1: broken CSR back-offset is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.dend_offset.back() += 1;  // no longer == syn_count()
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv1") != std::string::npos);
}

TEST_CASE("checker inv2: wrong dendrite type in the directory is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.dend_type[0] = bank_t::proximal;   // slot 0 must be apical
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv2") != std::string::npos);
}

TEST_CASE("checker inv2: wrong owning neuron in the directory is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.dend_neuron[0] = 7;                // must be neuron 0
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv2") != std::string::npos);
}

TEST_CASE("checker inv3: input_idx outside the dendrite's type range is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.input_idx[0] = 99;                 // apical range is [0,4)
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv3") != std::string::npos);
}

TEST_CASE("checker inv3: a missing / skipped routed input is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    // Dendrite 0 (apical) has inputs 0,1,2,3 at slots 0..3. Rewrite slot 2's
    // input from 2 to 3, giving 0,1,3,3: input 2 is skipped and the ascending
    // walk trips on `expected 2, got 3` before reaching the duplicate.
    // (A skip within a fully-covered dendrite necessarily either skips a value
    // ascending — this branch — or duplicates one, which inv4 catches.)
    arena.input_idx[2] = 3;
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv3") != std::string::npos);
}

TEST_CASE("checker inv4: a seg set that no longer covers the root is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    // Dendrite 0, input 0 has the single root seg=1 covering [0,8). Move it to
    // seg=2 ([0,4)); the right half [4,8) is now uncovered -> not an antichain
    // cut of the root.
    arena.seg[0] = 2;
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv4") != std::string::npos);
}

TEST_CASE("checker inv5: permanence outside [0,1] is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.perm[0] = 2.0f;
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv5") != std::string::npos);
}

TEST_CASE("checker inv5: seg id outside [1, max_seg_id] is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.seg[0] = 99;                       // max_seg_id is 15
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv5") != std::string::npos);
}

TEST_CASE("checker inv5: a negative/non-finite accumulator is caught") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    arena.mismatch[0] = -1.0f;
    std::string err;
    CHECK_FALSE(check_arena_consistency(arena, shape, &err));
    CHECK(err.find("inv5") != std::string::npos);
}

// ---------------------------------------------------------------------------
// 4. Invariant 4 on a hand-built branched arena: overlap (ancestor / duplicate
//    seg) is caught, a valid two-leaf cut passes. build_arena only makes
//    root-only trees, so this exercises the post-split shape directly.
// ---------------------------------------------------------------------------
namespace {

// One neuron, one proximal dendrite, one routed input (position 0), max_depth 1
// (axis = [0,2), max_seg_id 3). `segs` are the seg ids of the input's synapses.
group_shape_t make_tiny_shape()
{
    group_shape_t s;
    s.neuron_cnt   = 1;
    s.apical_cnt   = 0;
    s.proximal_cnt = 1;
    s.max_depth    = 1;
    s.max_seg_id   = (1 << (s.max_depth + 1)) - 1;   // 3
    s.input_size[static_cast<int>(bank_t::proximal)] = 1;
    s.input_lo  [static_cast<int>(bank_t::proximal)] = 0;
    s.input_hi  [static_cast<int>(bank_t::proximal)] = 1;
    return s;
}

synapse_arena_t make_tiny_arena(const std::vector<uint8_t> &segs)
{
    synapse_arena_t a;
    for (uint8_t seg : segs) {
        a.perm.push_back(0.5f);
        a.input_idx.push_back(0);
        a.seg.push_back(seg);
        a.adapt_hist.push_back(0.0f);
        a.mismatch.push_back(0.0f);
    }
    a.dend_offset = {0, static_cast<uint32_t>(segs.size())};
    a.dend_type   = {bank_t::proximal};
    a.dend_neuron = {0};
    return a;
}

} // namespace

TEST_CASE("checker inv4: valid two-leaf cut {2,3} passes; overlaps fail") {
    const group_shape_t shape = make_tiny_shape();

    SUBCASE("valid antichain cut of a split root") {
        const synapse_arena_t a = make_tiny_arena({2, 3});   // [0,1) + [1,2)
        std::string err = "untouched";
        CHECK(check_arena_consistency(a, shape, &err));
        CHECK(err == "untouched");
    }
    SUBCASE("duplicate seg -> overlap") {
        const synapse_arena_t a = make_tiny_arena({2, 2});   // [0,1) twice
        std::string err;
        CHECK_FALSE(check_arena_consistency(a, shape, &err));
        CHECK(err.find("inv4") != std::string::npos);
    }
    SUBCASE("ancestor + descendant -> overlap") {
        const synapse_arena_t a = make_tiny_arena({1, 2});   // [0,2) and [0,1)
        std::string err;
        CHECK_FALSE(check_arena_consistency(a, shape, &err));
        CHECK(err.find("inv4") != std::string::npos);
    }
    SUBCASE("single leaf -> gap (root not covered)") {
        const synapse_arena_t a = make_tiny_arena({2});      // only [0,1)
        std::string err;
        CHECK_FALSE(check_arena_consistency(a, shape, &err));
        CHECK(err.find("inv4") != std::string::npos);
    }
}
