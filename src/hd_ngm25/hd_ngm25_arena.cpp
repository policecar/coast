//
// Created 09.07.26 — ngm2.5 increment I1: the flat-sparse synapse arena.
// Implements the consistency checker (design §9 — written and wired FIRST,
// everything downstream builds on the invariants it guarantees) and the
// initial arena construction with Poisson permanence init (RNG site D5).
//
// Contract and invariants: include/hd_ngm25/hd_ngm25_arena.h. Semantic
// reference for the Poisson init: ngm/cold_init.h, shared with ngm2 (map D5).
//

#include "hd_ngm25/hd_ngm25_arena.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hd_ngm25/hd_ngm25_prng.h"           // pcg_hash / syn_key / to_unit — D4 draws
#include "ngm/cold_init.h"

namespace ngm25 {

namespace {

// Small helper so every invariant reports the FIRST violation with enough
// context to localize it (a re-pack bug corrupts silently; a bare "false" is
// useless — design §9).
bool fail(std::string *err, std::string msg)
{
    if (err != nullptr) *err = std::move(msg);
    return false;
}

// The fixed dendrite directory order (invariant 2). Returns the (neuron, type)
// a dendrite index MUST have. Kept in one place so build_arena and the checker
// cannot disagree.
struct dend_identity_t { uint16_t neuron; bank_t type; };
dend_identity_t expected_dendrite(std::size_t d, const group_shape_t &shape)
{
    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;
    const std::size_t neuron     = d / per_neuron;
    const std::size_t local      = d % per_neuron;
    const bank_t      type       = (local < shape.apical_cnt) ? bank_t::apical
                                                              : bank_t::proximal;
    return { static_cast<uint16_t>(neuron), type };
}

// D4 permanence wiggle for a split child (map R8 split effect; ngm2 semantic
// reference hd_ngm2_dendrite.cpp:526,558-564). ngm2 draws
// uniform_real_distribution<float>(-0.1, 0.1) per child and adds it to the
// parent permanence, clamped to [0,1]. ngm2.5 replaces the engine draw with the
// D4 structural-key hash: to_unit(...) ∈ [0,1) is remapped to [-0.1, 0.1).
//
// Key argument order (confirmed against hd_ngm25_prng.h + map D4): the header
// contract tuple (dend, NEW seg, input_idx, branch_epoch) maps onto
// syn_key(seed, step, dend, seg, input_idx) with step := branch_epoch and
// seg := the child's NEW seg. The two children of a split therefore key on
// DISTINCT segs and draw two independent wiggles (P2: keyed by structural
// identity, never arena slot — the property that survives re-pack).
float split_child_perm(float parent_perm, uint32_t seed, uint32_t branch_epoch,
                       uint32_t dend, uint32_t new_seg, uint32_t input_idx)
{
    const float u      = to_unit(pcg_hash(syn_key(seed, branch_epoch, dend, new_seg, input_idx)));
    const float wiggle = u * 0.2f - 0.1f;                  // [-0.1, 0.1)
    return std::clamp(parent_perm + wiggle, 0.0f, 1.0f);
}

} // namespace

bool check_arena_consistency(const synapse_arena_t &arena,
                             const group_shape_t   &shape,
                             std::string           *err)
{
    const std::size_t n_syn  = arena.syn_count();
    const std::size_t n_dend = arena.dend_count();

    // --- Invariant 1: SoA vectors equal length; CSR offsets monotone,
    //     front == 0, back == syn_count(). ---
    if (arena.input_idx.size()  != n_syn ||
        arena.seg.size()        != n_syn ||
        arena.adapt_hist.size() != n_syn ||
        arena.mismatch.size()   != n_syn)
        return fail(err, "inv1: per-synapse SoA vectors have mismatched lengths");

    if (arena.dend_offset.empty())
        return fail(err, "inv1: dend_offset is empty (need at least the trailing sentinel)");
    if (arena.dend_offset.front() != 0)
        return fail(err, "inv1: dend_offset.front() != 0 (CSR must start at 0)");
    if (arena.dend_offset.back() != n_syn)
        return fail(err, "inv1: dend_offset.back()=" + std::to_string(arena.dend_offset.back())
                       + " != syn_count()=" + std::to_string(n_syn));
    for (std::size_t d = 0; d < n_dend; ++d)
        if (arena.dend_offset[d] > arena.dend_offset[d + 1])
            return fail(err, "inv1: dend_offset not monotone at dendrite " + std::to_string(d)
                           + " (" + std::to_string(arena.dend_offset[d]) + " > "
                           + std::to_string(arena.dend_offset[d + 1]) + ")");

    // --- Invariant 2: directory shape matches group_shape_t; per-dendrite
    //     neuron/type follow the fixed order. ---
    const std::size_t expect_dend = shape.neuron_cnt * (shape.apical_cnt + shape.proximal_cnt);
    if (n_dend != expect_dend)
        return fail(err, "inv2: dend_count=" + std::to_string(n_dend)
                       + " != neuron_cnt*(apical+proximal)=" + std::to_string(expect_dend));
    if (arena.dend_type.size() != n_dend || arena.dend_neuron.size() != n_dend)
        return fail(err, "inv2: dend_type/dend_neuron length != dend_count");
    for (std::size_t d = 0; d < n_dend; ++d) {
        const dend_identity_t want = expected_dendrite(d, shape);
        if (arena.dend_neuron[d] != want.neuron)
            return fail(err, "inv2: dendrite " + std::to_string(d) + " neuron="
                           + std::to_string(arena.dend_neuron[d]) + " expected "
                           + std::to_string(want.neuron));
        if (arena.dend_type[d] != want.type)
            return fail(err, "inv2: dendrite " + std::to_string(d) + " type mismatch (expected "
                           + (want.type == bank_t::apical ? "apical" : "proximal") + ")");
    }

    // --- Invariants 3, 4, 5, walked per dendrite. ---
    for (std::size_t d = 0; d < n_dend; ++d) {
        const auto      rng  = arena.dendrite(d);
        const bank_t    type = arena.dend_type[d];
        const std::size_t ti = static_cast<std::size_t>(type);
        const uint16_t  lo   = static_cast<uint16_t>(shape.input_lo[ti]);
        const uint16_t  hi   = static_cast<uint16_t>(shape.input_hi[ti]);

        // Walk synapses in slot order, grouping contiguous runs of equal
        // input_idx (invariant 3 guarantees they are contiguous). Each run is
        // one (dendrite, input); its seg set must tile the rep axis (inv 4).
        // Coverage: the runs must be exactly lo, lo+1, ..., hi-1 in order —
        // every routed position present exactly once as a complete tree
        // (splits only ever ADD synapses, so a missing input is corruption).
        const int  axis = 1 << shape.max_depth;            // 2^max_depth leaf slots
        uint16_t   expect_input = lo;                       // next input we must see
        uint32_t   s = rng.begin;
        while (s < rng.end) {
            const uint16_t cur = arena.input_idx[s];

            // inv 3: input-major / in-range / full coverage.
            if (cur < lo || cur >= hi)
                return fail(err, "inv3: dendrite " + std::to_string(d) + " slot "
                               + std::to_string(s) + " input_idx=" + std::to_string(cur)
                               + " outside type range [" + std::to_string(lo) + ","
                               + std::to_string(hi) + ")");
            if (cur != expect_input)
                return fail(err, "inv3: dendrite " + std::to_string(d)
                               + " input coverage/order broken: expected input "
                               + std::to_string(expect_input) + " got " + std::to_string(cur)
                               + " (inputs must be lo..hi-1 ascending, none missing)");

            // Consume the whole run of this input, collecting seg intervals.
            std::vector<detail::seg_span_t> spans;
            while (s < rng.end && arena.input_idx[s] == cur) {
                // inv 5 (checked here so inv 4's seg_interval precondition holds):
                // seg range, perm range, finite non-negative accumulators.
                const uint8_t seg = arena.seg[s];
                if (seg < 1 || seg > shape.max_seg_id)
                    return fail(err, "inv5: dendrite " + std::to_string(d) + " slot "
                                   + std::to_string(s) + " seg=" + std::to_string(seg)
                                   + " outside [1," + std::to_string(shape.max_seg_id) + "]");
                const float p = arena.perm[s];
                if (!(p >= 0.0f && p <= 1.0f))
                    return fail(err, "inv5: dendrite " + std::to_string(d) + " slot "
                                   + std::to_string(s) + " perm=" + std::to_string(p)
                                   + " outside [0,1]");
                const float ah = arena.adapt_hist[s];
                const float mm = arena.mismatch[s];
                if (!std::isfinite(ah) || ah < 0.0f)
                    return fail(err, "inv5: dendrite " + std::to_string(d) + " slot "
                                   + std::to_string(s) + " adapt_hist=" + std::to_string(ah)
                                   + " not finite/non-negative");
                if (!std::isfinite(mm) || mm < 0.0f)
                    return fail(err, "inv5: dendrite " + std::to_string(d) + " slot "
                                   + std::to_string(s) + " mismatch=" + std::to_string(mm)
                                   + " not finite/non-negative");

                spans.push_back(detail::seg_interval(seg, shape.max_depth));
                ++s;
            }

            // inv 4: the run's seg intervals must tile [0, axis) exactly — no
            // gap (some rep-axis region uncovered) and no overlap (some seg is
            // an ancestor of another, or a split produced only one child).
            std::sort(spans.begin(), spans.end(),
                      [](const detail::seg_span_t &a, const detail::seg_span_t &b)
                      { return a.lo < b.lo; });
            int cursor = 0;
            for (const auto &sp : spans) {
                if (sp.lo != cursor)
                    return fail(err, "inv4: dendrite " + std::to_string(d) + " input "
                                   + std::to_string(cur) + " seg set is not a valid antichain"
                                     " cut (gap or overlap at rep-axis " + std::to_string(cursor)
                                   + ", next span starts at " + std::to_string(sp.lo) + ")");
                cursor = sp.hi;
            }
            if (cursor != axis)
                return fail(err, "inv4: dendrite " + std::to_string(d) + " input "
                               + std::to_string(cur) + " seg set does not cover the root"
                                 " (covered " + std::to_string(cursor) + " of "
                               + std::to_string(axis) + " leaf slots)");

            ++expect_input;
        }

        // All routed positions must have appeared (coverage of the tail).
        if (expect_input != hi)
            return fail(err, "inv3: dendrite " + std::to_string(d) + " missing inputs "
                           + std::to_string(expect_input) + ".." + std::to_string(hi - 1)
                           + " (routed range not fully covered)");
    }

    return true;
}

synapse_arena_t build_arena(const group_shape_t &shape,
                            float                 perm_threshold,
                            uint32_t              seed)
{
    synapse_arena_t arena;

    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;
    const std::size_t n_dend     = shape.neuron_cnt * per_neuron;

    arena.dend_offset.reserve(n_dend + 1);
    arena.dend_type.reserve(n_dend);
    arena.dend_neuron.reserve(n_dend);

    // ONE per-group cold engine (D5). Drawn in arena-slot order below: the
    // outer dendrite loop advances the directory, the inner loop advances
    // input positions ascending — so draw order == arena slot order, the
    // documented D5 contract.
    std::mt19937             engine(seed);
    ngm::cold_permanence_t   cold_perm(perm_threshold);   // ngm/cold_init.h

    arena.dend_offset.push_back(0);
    for (std::size_t d = 0; d < n_dend; ++d) {
        const dend_identity_t id = expected_dendrite(d, shape);
        arena.dend_neuron.push_back(id.neuron);
        arena.dend_type.push_back(id.type);

        const std::size_t ti = static_cast<std::size_t>(id.type);
        const std::size_t lo = shape.input_lo[ti];
        const std::size_t hi = shape.input_hi[ti];
        for (std::size_t pos = lo; pos < hi; ++pos) {
            arena.perm.push_back(cold_perm(engine));
            arena.input_idx.push_back(static_cast<uint16_t>(pos));
            arena.seg.push_back(1);            // root
            arena.adapt_hist.push_back(0.0f);
            arena.mismatch.push_back(0.0f);
        }
        arena.dend_offset.push_back(static_cast<uint32_t>(arena.perm.size()));
    }

    return arena;
}

void repack_with_splits(synapse_arena_t          &arena,
                        const std::vector<uint8_t> &clone_flags,
                        // shape is read only by the debug precondition asserts
                        // below; under NDEBUG (Release) it is otherwise unused.
                        [[maybe_unused]] const group_shape_t &shape,
                        uint32_t                  rng_seed,
                        uint32_t                  branch_epoch)
{
    const std::size_t n_syn  = arena.syn_count();
    const std::size_t n_dend = arena.dend_count();
    assert(clone_flags.size() == n_syn &&
           "repack_with_splits: clone_flags must be sized to syn_count()");

    // --- Step 1: count flagged synapses. In the same pass validate the split
    //     precondition — a flagged synapse must be splittable (seg*2+1 within
    //     range). The R8 decide predicate (I5) guarantees this; asserting it
    //     here makes a caller bug surface at the source instead of as a silent
    //     inv5 (seg out of range) failure downstream (pitfalls checklist).
    //
    //     RELEASE ENFORCEMENT (this assert is a NDEBUG no-op): the invariant is
    //     NOT left unguarded in release — every current caller runs
    //     check_arena_consistency() on the repacked arena before it is used, which
    //     catches the resulting out-of-range seg as inv5. The CPU process() path
    //     halts via the value channel (COAST_REQUIRE, hd_ngm25_neuron_group.cpp),
    //     the GPU grow_if_branched path throws — both BEFORE the bad arena feeds
    //     any compute. The assert only adds source-precision in debug. A future
    //     caller of repack_with_splits MUST likewise run the checker afterward
    //     (there is no in-function release guard, since this returns void and
    //     COAST_REQUIRE needs a step_result-returning caller).
    std::size_t total_flagged = 0;
    for (std::size_t s = 0; s < n_syn; ++s)
        if (clone_flags[s] != 0) {
            assert(2u * static_cast<uint32_t>(arena.seg[s]) + 1u <= shape.max_seg_id &&
                   "repack_with_splits: flagged synapse not splittable (seg*2+1 > max_seg_id)");
            ++total_flagged;
        }

    // --- Step 2: prefix-sum the new per-dendrite CSR offsets. Each flagged
    //     synapse becomes two children (+1 slot); everything else copies 1:1,
    //     so a dendrite grows by exactly its flagged count.
    std::vector<uint32_t> new_dend_offset;
    new_dend_offset.reserve(n_dend + 1);
    new_dend_offset.push_back(0);
    for (std::size_t d = 0; d < n_dend; ++d) {
        const auto rng   = arena.dendrite(d);
        uint32_t   grown = rng.end - rng.begin;
        for (uint32_t s = rng.begin; s < rng.end; ++s)
            if (clone_flags[s] != 0) ++grown;
        new_dend_offset.push_back(new_dend_offset.back() + grown);
    }
    const std::size_t new_total = n_syn + total_flagged;

    // --- Step 3: allocate the rebuilt SoA. A single reserve to the final size
    //     IS the growth event — one allocation, amortized O(1) per synapse, no
    //     per-synapse reallocation (design §7 "capacity doubling"; std::vector
    //     realizes it with this one grow). The directory (dend_type/dend_neuron)
    //     is unchanged by a split — same dendrites, same fixed order.
    synapse_arena_t out;
    out.perm.reserve(new_total);
    out.input_idx.reserve(new_total);
    out.seg.reserve(new_total);
    out.adapt_hist.reserve(new_total);
    out.mismatch.reserve(new_total);
    out.dend_offset = std::move(new_dend_offset);
    out.dend_type   = arena.dend_type;
    out.dend_neuron = arena.dend_neuron;

    // --- Step 4: rebuild in dendrite-major, input-major slot order. Walking the
    //     old arena in order and emitting each flagged synapse's two children
    //     adjacently at the parent's slot preserves input-major order and clone
    //     adjacency (invariant 3): children carry the parent's input_idx, so the
    //     run for that input just grows by one — no re-sort needed.
    for (std::size_t d = 0; d < n_dend; ++d) {
        const auto rng = arena.dendrite(d);
        for (uint32_t s = rng.begin; s < rng.end; ++s) {
            if (clone_flags[s] == 0) {
                out.perm.push_back(arena.perm[s]);
                out.input_idx.push_back(arena.input_idx[s]);
                out.seg.push_back(arena.seg[s]);
                out.adapt_hist.push_back(arena.adapt_hist[s]);
                out.mismatch.push_back(arena.mismatch[s]);
                continue;
            }
            // Split: two children at seg*2 (lower slot) and seg*2+1, matching
            // ngm2's ordering (hd_ngm2_dendrite.cpp:545-546). adapt_hist and
            // mismatch reset to 0; each child's permanence is the parent's
            // wiggled by an INDEPENDENT D4 draw keyed on its OWN new seg.
            const uint16_t inp         = arena.input_idx[s];
            const float    parent_perm = arena.perm[s];
            const auto     parent_seg  = static_cast<uint32_t>(arena.seg[s]);
            const uint8_t  seg_children[2] = { static_cast<uint8_t>(parent_seg * 2u),
                                               static_cast<uint8_t>(parent_seg * 2u + 1u) };
            for (const uint8_t child_seg : seg_children) {
                out.perm.push_back(split_child_perm(parent_perm, rng_seed, branch_epoch,
                                                    static_cast<uint32_t>(d), child_seg, inp));
                out.input_idx.push_back(inp);
                out.seg.push_back(child_seg);
                out.adapt_hist.push_back(0.0f);
                out.mismatch.push_back(0.0f);
            }
        }
        assert(out.perm.size() == out.dend_offset[d + 1] &&
               "repack_with_splits: rebuilt dendrite size disagrees with prefix-sum offset");
    }

    arena = std::move(out);
}

} // namespace ngm25
