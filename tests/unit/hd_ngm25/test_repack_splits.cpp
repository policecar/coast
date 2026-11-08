// tests/unit/hd_ngm25/test_repack_splits.cpp
//
// ngm2.5 increment I2. Under test: repack_with_splits (map R8 split effect,
// design §7). Three things it must get right:
//   1. Structure — each flagged synapse becomes two children seg*2 / seg*2+1
//      in input-major order with clone adjacency; unflagged synapses pass
//      through untouched; the rebuilt arena satisfies check_arena_consistency.
//   2. Values — children reset adapt_hist/mismatch to 0 and take the parent's
//      permanence wiggled by an INDEPENDENT D4 draw keyed on the child's NEW
//      seg (key order syn_key(seed, branch_epoch, dend, new_seg, input_idx)).
//   3. Determinism — same (arena, flags, seed, epoch) → bit-identical output.
//
// A property test then runs hundreds of randomized split SEQUENCES, asserting
// the checker stays green and input-major/adjacency hold across deepening trees.
//
// Pure module: no IO wiring, no sim env. Sub-second.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_prng.h"

using namespace ngm25;

namespace {

// Same representative shape as the I1 tests: 3 neurons x (1 apical + 2 proximal)
// = 9 dendrites; apical reads [0,4), proximal reads [4,9); max_depth 3 so
// max_seg_id 15 and the rep axis has 8 leaf slots.
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

// Independent recomputation of the D4 child permanence, mirroring the .cpp's
// formula through the PUBLIC prng primitives. Duplicating the constants here is
// deliberate: it pins the WIRING (which key args, which child seg, the ±0.1
// range, the clamp) so a mis-keyed or mis-signed draw in the kernel is caught.
float expected_child_perm(float parent_perm, uint32_t seed, uint32_t epoch,
                          uint32_t dend, uint32_t new_seg, uint32_t input_idx)
{
    const float u = to_unit(pcg_hash(syn_key(seed, epoch, dend, new_seg, input_idx)));
    return std::clamp(parent_perm + (u * 0.2f - 0.1f), 0.0f, 1.0f);
}

// Explicit input-major + clone-adjacency check, independent of the consistency
// checker: within every dendrite the input_idx sequence is non-decreasing
// (ascending across inputs, equal-input clones therefore contiguous).
bool input_major_and_adjacent(const synapse_arena_t &arena)
{
    for (std::size_t d = 0; d < arena.dend_count(); ++d) {
        const auto rng = arena.dendrite(d);
        for (uint32_t s = rng.begin + 1; s < rng.end; ++s)
            if (arena.input_idx[s] < arena.input_idx[s - 1]) return false;
    }
    return true;
}

// All-zero flag vector sized to the arena.
std::vector<uint8_t> no_flags(const synapse_arena_t &arena)
{
    return std::vector<uint8_t>(arena.syn_count(), 0u);
}

} // namespace

// ---------------------------------------------------------------------------
// 1. No flags → identity (every array byte-for-byte unchanged), checker green.
// ---------------------------------------------------------------------------
TEST_CASE("repack: no flags leaves the arena unchanged") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);
    const synapse_arena_t before = arena;

    repack_with_splits(arena, no_flags(arena), shape, /*seed=*/7, /*epoch=*/1);

    CHECK(arena.perm       == before.perm);
    CHECK(arena.input_idx  == before.input_idx);
    CHECK(arena.seg        == before.seg);
    CHECK(arena.adapt_hist == before.adapt_hist);
    CHECK(arena.mismatch   == before.mismatch);
    CHECK(arena.dend_offset == before.dend_offset);

    std::string err = "untouched";
    CHECK(check_arena_consistency(arena, shape, &err));
    CHECK(err == "untouched");
}

// ---------------------------------------------------------------------------
// 2. A single split: structure, pass-through, reset, and exact D4 values.
// ---------------------------------------------------------------------------
TEST_CASE("repack: one flagged synapse splits into seg*2 / seg*2+1 with D4 values") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 42);

    // Flag the FIRST synapse of dendrite 0 (input 0, root seg=1). It sits at
    // arena slot 0; dendrite 0 is neuron 0's apical.
    const uint32_t flagged_slot = 0;
    const float    parent_perm  = arena.perm[flagged_slot];
    const uint16_t parent_input = arena.input_idx[flagged_slot];
    REQUIRE(arena.seg[flagged_slot] == 1);
    REQUIRE(parent_input == 0);

    std::vector<uint8_t> flags = no_flags(arena);
    flags[flagged_slot] = 1;

    const std::size_t old_count = arena.syn_count();
    const uint32_t seed = 7, epoch = 3;
    repack_with_splits(arena, flags, shape, seed, epoch);

    // One synapse became two → total grew by exactly one.
    CHECK(arena.syn_count() == old_count + 1);

    // Slots 0,1 are now the two children (seg 2 then 3), both on input 0.
    CHECK(arena.seg[0] == 2);
    CHECK(arena.seg[1] == 3);
    CHECK(arena.input_idx[0] == 0);
    CHECK(arena.input_idx[1] == 0);
    CHECK(arena.adapt_hist[0] == 0.0f);
    CHECK(arena.adapt_hist[1] == 0.0f);
    CHECK(arena.mismatch[0]   == 0.0f);
    CHECK(arena.mismatch[1]   == 0.0f);

    // Exact permanences: each child wiggled by its OWN new-seg-keyed draw
    // (dendrite index 0). This locks the key argument order and the ± range.
    CHECK(arena.perm[0] == expected_child_perm(parent_perm, seed, epoch, 0, 2, 0));
    CHECK(arena.perm[1] == expected_child_perm(parent_perm, seed, epoch, 0, 3, 0));
    // Independent draws: distinct keys, so (barring an astronomically unlikely
    // hash collision away from the clamp bound) the two children differ.
    CHECK(arena.perm[0] != arena.perm[1]);

    // The rest of dendrite 0 (its inputs 1,2,3) passed through unchanged and now
    // sit at slots 2,3,4.
    CHECK(arena.input_idx[2] == 1);
    CHECK(arena.input_idx[3] == 2);
    CHECK(arena.input_idx[4] == 3);
    CHECK(arena.seg[2] == 1);
    CHECK(arena.seg[3] == 1);
    CHECK(arena.seg[4] == 1);

    // Dendrite 0 grew from 4 to 5 synapses; every later dendrite offset shifted
    // by exactly one.
    CHECK(arena.dend_offset[0] == 0);
    CHECK(arena.dend_offset[1] == 5);
    CHECK(arena.dendrite(0).end - arena.dendrite(0).begin == 5);

    std::string err = "untouched";
    CHECK(check_arena_consistency(arena, shape, &err));
    CHECK(err == "untouched");
    CHECK(input_major_and_adjacent(arena));
}

// ---------------------------------------------------------------------------
// 3. Flags spread across dendrites: per-dendrite prefix-sum offsets, coverage.
// ---------------------------------------------------------------------------
TEST_CASE("repack: flags in different dendrites grow each independently") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 99);

    // Flag one synapse in dendrite 0 (apical, slot 0) and two in dendrite 1
    // (neuron 0 proximal, slots at its begin). All are root seg=1, splittable.
    std::vector<uint8_t> flags = no_flags(arena);
    const auto d0 = arena.dendrite(0);
    const auto d1 = arena.dendrite(1);
    flags[d0.begin]     = 1;                 // +1 in dendrite 0
    flags[d1.begin]     = 1;                 // +2 in dendrite 1
    flags[d1.begin + 1] = 1;

    const uint32_t d0_old = d0.end - d0.begin;
    const uint32_t d1_old = d1.end - d1.begin;

    repack_with_splits(arena, flags, shape, /*seed=*/11, /*epoch=*/2);

    CHECK(arena.dendrite(0).end - arena.dendrite(0).begin == d0_old + 1);
    CHECK(arena.dendrite(1).end - arena.dendrite(1).begin == d1_old + 2);
    // Dendrite 2 onward are untouched in size.
    CHECK(arena.dendrite(2).end - arena.dendrite(2).begin ==
          static_cast<uint32_t>(shape.input_size[static_cast<int>(bank_t::proximal)]));

    std::string err = "untouched";
    CHECK(check_arena_consistency(arena, shape, &err));
    CHECK(err == "untouched");
    CHECK(input_major_and_adjacent(arena));
}

// ---------------------------------------------------------------------------
// 4. Second-level split: a seg=2 synapse splits into 4/5 and stays a valid
//    antichain cut (checker green). Exercises depth > 1 explicitly.
// ---------------------------------------------------------------------------
TEST_CASE("repack: splitting a depth-1 seg yields 4/5 and remains consistent") {
    const group_shape_t shape = make_shape();
    synapse_arena_t arena = build_arena(shape, 0.3f, 5);

    // Round 1: split dendrite 0 / input 0 (slot 0) → children seg 2,3 at slots 0,1.
    {
        std::vector<uint8_t> flags = no_flags(arena);
        flags[0] = 1;
        repack_with_splits(arena, flags, shape, 13, 1);
        REQUIRE(arena.seg[0] == 2);
        REQUIRE(arena.seg[1] == 3);
    }
    // Round 2: split the seg=2 child (now at slot 0) → children 4,5.
    {
        std::vector<uint8_t> flags = no_flags(arena);
        flags[0] = 1;
        repack_with_splits(arena, flags, shape, 13, 2);
    }

    CHECK(arena.seg[0] == 4);
    CHECK(arena.seg[1] == 5);
    CHECK(arena.seg[2] == 3);                 // the untouched depth-1 sibling
    CHECK(arena.input_idx[0] == 0);
    CHECK(arena.input_idx[1] == 0);
    CHECK(arena.input_idx[2] == 0);

    std::string err = "untouched";
    CHECK(check_arena_consistency(arena, shape, &err));
    CHECK(err == "untouched");
}

// ---------------------------------------------------------------------------
// 5. Determinism: identical inputs → bit-identical output arena.
// ---------------------------------------------------------------------------
TEST_CASE("repack: same inputs produce bit-identical output") {
    const group_shape_t shape = make_shape();
    synapse_arena_t a = build_arena(shape, 0.3f, 77);
    synapse_arena_t b = build_arena(shape, 0.3f, 77);
    REQUIRE(a.perm == b.perm);               // same build to start

    // A non-trivial, reproducible flag pattern (every 3rd synapse, all root).
    std::vector<uint8_t> flags = no_flags(a);
    for (std::size_t s = 0; s < flags.size(); s += 3) flags[s] = 1;

    repack_with_splits(a, flags, shape, /*seed=*/123, /*epoch=*/9);
    repack_with_splits(b, flags, shape, /*seed=*/123, /*epoch=*/9);

    CHECK(a.perm        == b.perm);
    CHECK(a.input_idx   == b.input_idx);
    CHECK(a.seg         == b.seg);
    CHECK(a.adapt_hist  == b.adapt_hist);
    CHECK(a.mismatch    == b.mismatch);
    CHECK(a.dend_offset == b.dend_offset);
}

// ---------------------------------------------------------------------------
// 6. Property test: hundreds of randomized split SEQUENCES. Each trial builds a
//    fresh arena and runs several repack rounds, flagging a random splittable
//    subset each round (branch_epoch advancing). After every round the checker
//    must be green and input-major/adjacency must hold. This is where multi-
//    level, multi-dendrite deepening gets stressed.
// ---------------------------------------------------------------------------
TEST_CASE("repack: randomized split sequences keep the arena consistent") {
    const group_shape_t shape = make_shape();

    constexpr int k_trials = 300;
    for (int trial = 0; trial < k_trials; ++trial) {
        // Deterministic per-trial engine (no Date/Random dependence — the whole
        // test is reproducible). Seeds vary by trial index to explore patterns.
        std::mt19937 rng(static_cast<uint32_t>(trial) * 2654435761u + 1u);

        synapse_arena_t arena = build_arena(shape, 0.3f, static_cast<uint32_t>(trial));
        std::string err = "untouched";
        REQUIRE(check_arena_consistency(arena, shape, &err));

        const int rounds = 1 + static_cast<int>(rng() % 6u);   // 1..6 rounds
        std::bernoulli_distribution flag_it(0.35);

        for (uint32_t epoch = 1; epoch <= static_cast<uint32_t>(rounds); ++epoch) {
            std::vector<uint8_t> flags(arena.syn_count(), 0u);
            for (std::size_t s = 0; s < arena.syn_count(); ++s) {
                // Only flag SPLITTABLE synapses (precondition of repack); the
                // I5 decide predicate guarantees this for real callers.
                const uint32_t seg = arena.seg[s];
                if (2u * seg + 1u <= shape.max_seg_id && flag_it(rng))
                    flags[s] = 1;
            }

            repack_with_splits(arena, flags, shape,
                               /*seed=*/static_cast<uint32_t>(trial) + 1000u, epoch);

            err = "untouched";
            const bool ok = check_arena_consistency(arena, shape, &err);
            INFO("trial " << trial << " epoch " << epoch << " checker: " << err);
            REQUIRE(ok);
            REQUIRE(input_major_and_adjacent(arena));
        }
    }
}
