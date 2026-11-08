// tests/unit/hd_ngm25/test_adapt.cpp
//
// ngm2.5 increment I4: winner selection (R5) + adapt path (R6/R7). Gates:
//   (a) adapt magnitude agreement vs REAL ngm2 (permanence / adapt_history /
//       mismatch arrays) after 1 and 100 steps on the deterministic fixture —
//       both run on the SAME post-init permanences, uniform-M input; adapt has
//       no draws so the sweep is deterministic in both.
//   (b) explicit "no-primary-leaf" dendrite (type-max lives in a sibling): all
//       leaves take the secondary weight; still matches ngm2.
//   (c) adapt_all bitwise deterministic across 1/4/32 TBB threads, and every
//       neuron, the winner included, adapts exactly once (SPEC.md M.4.14).
//
// Off-grid threshold 0.305 (see test_response.cpp) keeps the perm>threshold
// decisions aligned between ngm2's float>double and ngm2.5's float>float.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <oneapi/tbb/global_control.h>

#include "hd_ngm25/hd_ngm25_adapt.h"
#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_prng.h"
#include "hd_ngm25/hd_ngm25_response.h"

#include "hd_ngm2/hd_ngm2_dendrite.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "sim_core/io_buffer.h"
#include "sim_core/io_entity.h"

using namespace ngm25;

// The learning window these tests gate adapt with (the former configured
// default; groups pass their tracked windows, ngm/window_tracking.h).
static const learning_window_t k_window = { { 0.66, 0.33 }, { 0.66, 0.66 } };

namespace {

constexpr float k_thres = 0.305f;

// ngm2 dendrite harness: owns cfg/buffer/dendrite so a whole step (get_response
// then adapt_synapses) can be driven repeatedly. The dendrite copies its params
// at construction and stores only the heap buffer pointer, so moving this struct
// out of the factory keeps every address valid.
struct ngm2_dend {
    ngm2::cfg_store                     cs;
    std::unique_ptr<sim::io_buffer_cpu> buf;
    std::unique_ptr<ngm2::dendrite_t>   d;
    ngm2::input_stats_map_t             stats;
    sim::output_id_t                    id = 7;
    uint8_t                             max_seg_id = 0;
    std::size_t                         width = 0;
};

ngm2_dend make_ngm2_dend(ngm2::dendrite_type_t type, std::size_t W, float value,
                         uint8_t level, int seed)
{
    ngm2_dend h;
    h.width      = W;
    h.max_seg_id = static_cast<uint8_t>((1u << (level + 1)) - 1);

    ngm2::dendrite_params_t p;
    p.type                 = type;
    p.permanence_threshold = k_thres;
    p.max_branch_level     = level;
    h.cs.dendrites["t"]    = p;

    h.d = std::make_unique<ngm2::dendrite_t>(h.cs, "t", seed);
    h.d->init_input({{"chan", ngm2::channel_source_t{ sim::output_channel_t{h.id, W}, "chan" }}}, "test dendrite");

    h.buf = std::make_unique<sim::io_buffer_cpu>(W);
    auto w = h.buf->cur_write_buffer();
    std::fill(w.begin(), w.end(), value);
    h.buf->swap_buffer();
    h.d->set_input_buffer(h.id, *h.buf);

    h.stats.emplace(h.id, ngm2::compute_input_stats(h.buf->cur_read_buffer()));
    return h;
}

// adapt_params_t matching ngm2's dendrite defaults (the fields R7 reads).
adapt_params_t dendrite_ap()
{
    adapt_params_t ap;
    ap.perm_threshold          = k_thres;
    ap.primary_learning_rate   = 0.01f;
    ap.secondary_learning_rate = 0.0001f;
    ap.mismatch_act_thres      = 0.8f;
    ap.mismatch_smoothing      = 0.001f;
    return ap;
}

group_shape_t single_shape(bank_t type, std::size_t W, uint8_t level)
{
    group_shape_t s;
    s.neuron_cnt   = 1;
    s.apical_cnt   = (type == bank_t::apical)   ? 1 : 0;
    s.proximal_cnt = (type == bank_t::proximal) ? 1 : 0;
    s.max_depth    = level;
    s.max_seg_id   = static_cast<uint8_t>((1u << (level + 1)) - 1);
    const auto ti  = static_cast<int>(type);
    s.input_size[ti] = W; s.input_lo[ti] = 0; s.input_hi[ti] = W;
    return s;
}

synapse_arena_t single_arena(const std::vector<float> &perm,
                             const std::vector<uint8_t> &seg, bank_t type)
{
    synapse_arena_t a;
    a.perm = perm;
    a.seg  = seg;
    a.adapt_hist.assign(perm.size(), 0.0f);
    a.mismatch.assign(perm.size(), 0.0f);
    a.input_idx.resize(perm.size());
    for (uint32_t i = 0; i < perm.size(); ++i) a.input_idx[i] = static_cast<uint16_t>(i);
    a.dend_offset = {0, static_cast<uint32_t>(perm.size())};
    a.dend_type   = {type};
    a.dend_neuron = {0};
    return a;
}

std::vector<uint8_t> segs_of(const ngm2::dendrite_t &d)
{
    const auto &syn = d.get_synapses();
    std::vector<uint8_t> v(syn.segment_idx.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(syn.segment_idx[i]);
    return v;
}

} // namespace

// ---------------------------------------------------------------------------
// (a) one-step R7 vs real ngm2 across sizes/values/depths.
// ---------------------------------------------------------------------------
TEST_CASE("R7 adapt_dendrite matches real ngm2 after one step") {
    struct casev { std::size_t W; float M; uint8_t level; int seed; float weight; };
    const casev cases[] = {
        {12, 0.6f, 3, 1, 0.5f}, {20, 0.7f, 2, 2, 0.8f},
        {9,  0.5f, 1, 3, 0.3f}, {28, 0.85f, 3, 4, 1.0f},
    };
    for (const auto &c : cases) {
        CAPTURE(c.W); CAPTURE(c.M); CAPTURE(c.level); CAPTURE(c.seed);
        ngm2_dend h = make_ngm2_dend(ngm2::dendrite_type_t::proximal, c.W, c.M, c.level, c.seed);

        const float a = h.d->get_response(h.stats);            // fills segment_activity
        const std::vector<float>   perm0 = {h.d->get_synapses().permanence.begin(),
                                            h.d->get_synapses().permanence.end()};
        const std::vector<uint8_t> seg0  = segs_of(*h.d);

        h.d->adapt_synapses(a, c.weight, h.stats);             // ngm2 adapt
        const auto &syn = h.d->get_synapses();

        // ngm2.5: same starting perms, recompute response, adapt.
        const group_shape_t shape = single_shape(bank_t::proximal, c.W, c.level);
        synapse_arena_t arena = single_arena(perm0, seg0, bank_t::proximal);
        channel_stats_t ch{bank_t::proximal, 0, static_cast<uint32_t>(c.W), h.stats.at(h.id)};
        step_ctx_t ctx;
        detail::build_step_ctx(shape, {&ch, 1}, ctx);

        std::vector<float> input(c.W, c.M);
        std::vector<float> seg_act(h.max_seg_id + 1u), seg_w(h.max_seg_id + 1u);
        detail::response_dendrite<stochastic_penalty>(
            arena, 0, input, ctx, bank_t::proximal, h.max_seg_id, k_thres,
            1.0f / (1.0f - k_thres), 123, 0, seg_act);
        detail::adapt_dendrite(arena, 0, input, ctx, bank_t::proximal, h.max_seg_id,
                               /*type_max=*/a, /*synapse_weight=*/c.weight,
                               dendrite_ap(), seg_act, seg_w);

        for (std::size_t i = 0; i < c.W; ++i) {
            CAPTURE(i);
            CHECK(arena.perm[i]       == doctest::Approx(syn.permanence[i]).epsilon(1e-5));
            CHECK(arena.adapt_hist[i] == doctest::Approx(syn.adapt_history[i]).epsilon(1e-5));
            CHECK(arena.mismatch[i]   == doctest::Approx(syn.mismatch[i]).epsilon(1e-5));
        }
    }
}

// ---------------------------------------------------------------------------
// (a) 100-step accumulation vs real ngm2 (adapt_history grows, mismatch IIR
//     evolves, permanences drift). Close, not bit-exact.
// ---------------------------------------------------------------------------
TEST_CASE("R7 adapt_dendrite tracks real ngm2 over 100 steps") {
    const std::size_t W = 16; const float M = 0.7f; const uint8_t level = 3;
    ngm2_dend h = make_ngm2_dend(ngm2::dendrite_type_t::proximal, W, M, level, 7);

    // ngm2.5 side starts from the same post-init perms.
    const float a0 = h.d->get_response(h.stats);
    (void)a0;
    const group_shape_t shape = single_shape(bank_t::proximal, W, level);
    synapse_arena_t arena = single_arena(
        {h.d->get_synapses().permanence.begin(), h.d->get_synapses().permanence.end()},
        segs_of(*h.d), bank_t::proximal);
    channel_stats_t ch{bank_t::proximal, 0, static_cast<uint32_t>(W), h.stats.at(h.id)};
    step_ctx_t ctx; detail::build_step_ctx(shape, {&ch, 1}, ctx);
    std::vector<float> input(W, M), seg_act(h.max_seg_id + 1u), seg_w(h.max_seg_id + 1u);
    const adapt_params_t ap = dendrite_ap();

    const float weight = 0.6f;
    for (int step = 0; step < 100; ++step) {
        const float a = h.d->get_response(h.stats);
        h.d->adapt_synapses(a, weight, h.stats);
        detail::response_dendrite<stochastic_penalty>(
            arena, 0, input, ctx, bank_t::proximal, h.max_seg_id, k_thres,
            1.0f / (1.0f - k_thres), 123, static_cast<uint32_t>(step), seg_act);
        detail::adapt_dendrite(arena, 0, input, ctx, bank_t::proximal, h.max_seg_id,
                               a, weight, ap, seg_act, seg_w);
    }

    const auto &syn = h.d->get_synapses();
    for (std::size_t i = 0; i < W; ++i) {
        CAPTURE(i);
        CHECK(arena.perm[i]       == doctest::Approx(syn.permanence[i]).epsilon(1e-4));
        CHECK(arena.adapt_hist[i] == doctest::Approx(syn.adapt_history[i]).epsilon(1e-4));
        CHECK(arena.mismatch[i]   == doctest::Approx(syn.mismatch[i]).epsilon(1e-3));
    }
    std::string err = "untouched";
    CHECK(check_arena_consistency(arena, shape, &err));
}

// ---------------------------------------------------------------------------
// (b) no-primary-leaf: type-max inflated above every leaf => all leaves take
//     the secondary weight; must still match ngm2 fed the same inflated max.
// ---------------------------------------------------------------------------
TEST_CASE("R7 no-primary-leaf dendrite matches ngm2 (all-secondary path)") {
    const std::size_t W = 15; const float M = 0.6f; const uint8_t level = 3;
    ngm2_dend h = make_ngm2_dend(ngm2::dendrite_type_t::proximal, W, M, level, 9);

    const float a = h.d->get_response(h.stats);
    const float inflated = a + 0.1f;      // type-max lives in a (hypothetical) sibling
    const std::vector<float>   perm0 = {h.d->get_synapses().permanence.begin(),
                                        h.d->get_synapses().permanence.end()};
    const std::vector<uint8_t> seg0  = segs_of(*h.d);

    h.d->adapt_synapses(inflated, 0.7f, h.stats);
    const auto &syn = h.d->get_synapses();

    const group_shape_t shape = single_shape(bank_t::proximal, W, level);
    synapse_arena_t arena = single_arena(perm0, seg0, bank_t::proximal);
    channel_stats_t ch{bank_t::proximal, 0, static_cast<uint32_t>(W), h.stats.at(h.id)};
    step_ctx_t ctx; detail::build_step_ctx(shape, {&ch, 1}, ctx);
    std::vector<float> input(W, M), seg_act(h.max_seg_id + 1u), seg_w(h.max_seg_id + 1u);
    detail::response_dendrite<stochastic_penalty>(
        arena, 0, input, ctx, bank_t::proximal, h.max_seg_id, k_thres,
        1.0f / (1.0f - k_thres), 123, 0, seg_act);
    detail::adapt_dendrite(arena, 0, input, ctx, bank_t::proximal, h.max_seg_id,
                           inflated, 0.7f, dendrite_ap(), seg_act, seg_w);

    for (std::size_t i = 0; i < W; ++i) {
        CAPTURE(i);
        CHECK(arena.perm[i]       == doctest::Approx(syn.permanence[i]).epsilon(1e-5));
        CHECK(arena.adapt_hist[i] == doctest::Approx(syn.adapt_history[i]).epsilon(1e-5));
    }
}

// ---------------------------------------------------------------------------
// (c) adapt_all: single adaptation + bitwise determinism across threads.
// ---------------------------------------------------------------------------
TEST_CASE("R5 adapt_all: every neuron adapts once and is thread-deterministic") {
    group_shape_t shape;
    shape.neuron_cnt = 6; shape.apical_cnt = 1; shape.proximal_cnt = 2;
    shape.max_depth = 3; shape.max_seg_id = 15;
    shape.input_size[0] = 4;  shape.input_lo[0] = 0;  shape.input_hi[0] = 4;
    shape.input_size[1] = 6;  shape.input_lo[1] = 4;  shape.input_hi[1] = 10;

    const synapse_arena_t arena0 = build_arena(shape, 0.3f, 2024);

    std::vector<float> input(10);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.15f + 0.8f * static_cast<float>((i * 5 + 2) % 9) / 9.0f;

    std::array<channel_stats_t, 2> ch{};
    ch[0] = {bank_t::apical,   0, 4,  ngm2::compute_input_stats(std::span<const float>(input).subspan(0, 4))};
    ch[1] = {bank_t::proximal, 4, 10, ngm2::compute_input_stats(std::span<const float>(input).subspan(4, 6))};
    step_ctx_t ctx; detail::build_step_ctx(shape, ch, ctx);

    const std::size_t nd = arena0.dend_count();
    const std::size_t sc = static_cast<std::size_t>(shape.max_seg_id) + 1;

    // R2/R3 to produce the adapt inputs.
    std::vector<float> dact(nd), nact(shape.neuron_cnt), seg_base(nd * sc);
    detail::response_all<stochastic_penalty>(arena0, shape, input, ctx, 0.3f,
                                             {0.0f, 0.0f}, 777, 5, seg_base, dact, nact);
    std::vector<float> mx_act = nact;                  // plausible running max
    std::vector<float> post_inhib = nact;              // no inhibitor in this unit test

    adapt_params_t ap;                                 // realistic defaults
    ap.perm_threshold = 0.3f;

    auto run_at = [&](int threads) {
        oneapi::tbb::global_control gc(
            oneapi::tbb::global_control::max_allowed_parallelism, threads);
        synapse_arena_t arena = arena0;
        std::vector<float>        seg_act = seg_base;  // folded in place by adapt
        std::vector<float>        seg_w(nd * sc);
        std::vector<std::uint64_t> icount(shape.neuron_cnt, 0);
        detail::adapt_all(arena, shape, input, ctx, dact, nact, mx_act, post_inhib,
                          ap, k_window, k_window,
                          seg_act, seg_w, icount, 777, 5);
        return std::tuple{arena.perm, arena.adapt_hist, arena.mismatch, icount};
    };

    const auto [p1, h1, m1, c1]    = run_at(1);
    const auto [p4, h4, m4, c4]    = run_at(4);
    const auto [p32, h32, m32, c32] = run_at(32);

    CHECK(p1 == p4);   CHECK(p1 == p32);
    CHECK(h1 == h4);   CHECK(h1 == h32);
    CHECK(m1 == m4);   CHECK(m1 == m32);
    CHECK(c1 == c4);   CHECK(c1 == c32);

    // Every neuron, the winner included, adapted exactly once (M.4.14).
    const std::size_t once = std::count(c1.begin(), c1.end(), std::uint64_t{1});
    CHECK(once == shape.neuron_cnt);

    // Adapt actually moved permanences (not a no-op).
    CHECK(p1 != arena0.perm);

    // Structure still consistent after adapt.
    synapse_arena_t after = arena0;
    { std::vector<float> sa = seg_base, sw(nd * sc); std::vector<std::uint64_t> ic(shape.neuron_cnt, 0);
      detail::adapt_all(after, shape, input, ctx, dact, nact, mx_act, post_inhib, ap,
                        k_window, k_window, sa, sw, ic, 777, 5); }
    std::string err = "untouched";
    CHECK(check_arena_consistency(after, shape, &err));
}

// ---------------------------------------------------------------------------
// (d) R5 winner SELECTION + both WEIGHT formulas value-validated against the
//     transcription-map spec (map R5.2/R5.3/R5.4). Gap closed: (c) checks only
//     the CADENCE (icount==1) and thread-determinism, never that the RIGHT
//     neuron wins or that adapt_all applies exactly
//         primary  (loop 3, winner only)        = sig(1 - out[win], wf)
//         secondary(loop 4, every other neuron) = sig(1 - out[i]/act_sum, wf)·common_lr
//     Strategy (deterministic, no ngm2 harness needed): reconstruct the D3
//     winner threshold from the SAME hash adapt_all uses (map D3), derive the
//     expected winner + the two weights from the map formulas written HERE
//     independently, then cross-check each neuron's arena delta from adapt_all
//     against a reference adapt_neuron driven by those hand-derived weights.
//     Bit-exact: a wrong winner, key, or weight formula in adapt_all mismatches.
//     (The full learning-quality ngm2-vs-ngm2.5 comparison of design §9 is a
//     separate, larger deliverable; this pins the R5 mechanics numerically.)
// ---------------------------------------------------------------------------
TEST_CASE("R5 adapt_all: winner selection + both weight formulas match the spec") {
    group_shape_t shape;
    shape.neuron_cnt = 6; shape.apical_cnt = 1; shape.proximal_cnt = 2;
    shape.max_depth = 3; shape.max_seg_id = 15;
    shape.input_size[0] = 4;  shape.input_lo[0] = 0;  shape.input_hi[0] = 4;
    shape.input_size[1] = 6;  shape.input_lo[1] = 4;  shape.input_hi[1] = 10;

    const synapse_arena_t arena0 = build_arena(shape, 0.3f, 4242);

    std::vector<float> input(10);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.2f + 0.7f * static_cast<float>((i * 3 + 1) % 7) / 7.0f;

    std::array<channel_stats_t, 2> ch{};
    ch[0] = {bank_t::apical,   0, 4,  ngm2::compute_input_stats(std::span<const float>(input).subspan(0, 4))};
    ch[1] = {bank_t::proximal, 4, 10, ngm2::compute_input_stats(std::span<const float>(input).subspan(4, 6))};
    step_ctx_t ctx; detail::build_step_ctx(shape, ch, ctx);

    const std::size_t nd = arena0.dend_count();
    const std::size_t sc = static_cast<std::size_t>(shape.max_seg_id) + 1;

    // Real R2/R3 to produce meaningful per-dendrite/neuron activities for adapt.
    std::vector<float> dact(nd), nact(shape.neuron_cnt), seg_base(nd * sc);
    detail::response_all<stochastic_penalty>(arena0, shape, input, ctx, 0.3f,
                                             {0.0f, 0.0f}, 777, 5, seg_base, dact, nact);

    // Post-inhibition outputs crafted so TWO neurons (2 and 3) sit above the
    // winner-threshold band [mx·win_thres, mx] — so which one wins genuinely
    // depends on the D3 draw, making the test sensitive to the win_act formula.
    const std::vector<float> post_inhib = {0.10f, 0.42f, 0.79f, 0.81f, 0.55f, 0.19f};
    const std::vector<float> mx_act     = nact;   // R6 running-max input (as in (c))

    adapt_params_t ap;                            // defaults are realistic
    ap.perm_threshold       = 0.3f;
    ap.stochastic_win_thres = 0.8f;               // exercise the real stochastic D3 draw
    const uint32_t seed = 777, step = 5;

    // Reconstruct adapt_all's EXACT D3 winner threshold (map D3: hash(seed, step)
    // via syn_key with the 0xFFFFFFFF sentinel dendrite; win_act ∈ U(mx·wt, mx)),
    // then the expected winner (SPEC.md M.4.4): uniform among the neurons with
    // out+eps >= win_act — the floor(u·k)-th of the k qualifying, ascending, u
    // the D6 draw (the D3 key with seg = 1).
    const float mx      = *std::max_element(post_inhib.begin(), post_inhib.end());
    const float lo      = mx * ap.stochastic_win_thres;
    const float win_act = lo + to_unit(pcg_hash(syn_key(seed, step, 0xFFFFFFFFu, 0, 0))) * (mx - lo);
    std::vector<std::size_t> qualifying;
    for (std::size_t i = 0; i < post_inhib.size(); ++i)
        if (post_inhib[i] + std::numeric_limits<float>::epsilon() >= win_act) qualifying.push_back(i);
    REQUIRE(!qualifying.empty());
    const float u = to_unit(pcg_hash(syn_key(seed, step, 0xFFFFFFFFu, 1, 0)));
    const std::size_t j = std::min(static_cast<std::size_t>(u * static_cast<float>(qualifying.size())),
                                   qualifying.size() - 1);
    const std::size_t expected_winner = qualifying[j];

    // The map's weight formulas, written HERE from the spec (not lifted from the
    // implementation): these are the ground truth adapt_all must reproduce.
    const ngm2::sigmoid_params_t wf(ap.weight_filter);
    const float act_sum   = std::accumulate(post_inhib.begin(), post_inhib.end(), 0.0f);
    const float primary_w = ngm2::sigmoid(1.0f - post_inhib[expected_winner], wf);
    auto secondary_w = [&](std::size_t i) {
        return ngm2::sigmoid(1.0f - post_inhib[i] / act_sum, wf) * ap.common_learning_rate;
    };

    // Run the real group adapt.
    synapse_arena_t arena_all = arena0;
    std::vector<float> seg_all = seg_base, seg_w_all(nd * sc);
    std::vector<std::uint64_t> icount(shape.neuron_cnt, 0);
    detail::adapt_all(arena_all, shape, input, ctx, dact, nact, mx_act, post_inhib,
                      ap, k_window, k_window,
                      seg_all, seg_w_all, icount, seed, step);

    // (1) CADENCE: every neuron, the winner included, adapted exactly once
    //     (M.4.14). Which neuron won is checked by (2): only the spec-predicted
    //     winner's delta matches the primary weight.
    for (std::size_t i = 0; i < shape.neuron_cnt; ++i) { CAPTURE(i); CHECK(icount[i] == 1u); }
    CHECK(arena_all.perm != arena0.perm);          // adapt actually moved perms

    // (2) WEIGHT FORMULAS: reproduce each neuron's arena delta with a reference
    //     adapt_neuron driven by the hand-derived weight. Same deterministic
    //     code path -> bit-exact. tmax reads dendrite_act (not perms), exactly as
    //     adapt_all precomputes.
    const std::size_t per_neuron = shape.apical_cnt + shape.proximal_cnt;
    for (std::size_t n = 0; n < shape.neuron_cnt; ++n) {
        CAPTURE(n);
        synapse_arena_t ref = arena0;
        std::vector<float> seg_ref = seg_base, seg_w_ref(nd * sc);
        std::uint64_t ic = 0;
        const detail::type_max_t tmax = detail::neuron_type_max(ref, n, shape, dact);
        const float w = (n == expected_winner) ? primary_w        // loop 3 only
                                               : secondary_w(n);  // loop 4 only
        detail::adapt_neuron(ref, n, shape, input, ctx, tmax, nact[n], mx_act[n],
                             w, k_window, ap, seg_ref, seg_w_ref, ic);
        const std::size_t b = arena0.dend_offset[n * per_neuron];
        const std::size_t e = arena0.dend_offset[(n + 1) * per_neuron];
        for (std::size_t s = b; s < e; ++s) {
            CAPTURE(s);
            CHECK(arena_all.perm[s]       == ref.perm[s]);
            CHECK(arena_all.adapt_hist[s] == ref.adapt_hist[s]);
            CHECK(arena_all.mismatch[s]   == ref.mismatch[s]);
        }
    }
}
