//
// Activity window tracking — shared by ngm2, ngm25 and ngm3 (CPU and GPU).
//
// The activity learning window (SPEC.md M.4.2) follows the activity it gates.
// A group keeps two windows: one for the step's winner and one for every other
// neuron. Each window's edges (its two sigmoid transition points) are running
// quantiles of that population's raw activity — the combined activity before
// inhibition, the value the window gates. Once per learning step, after
// adaptation, every edge q moves by
//
//     q = clamp(q + rate * (tau - share of the population's activities below q), 0, 1)
//
// (a Robbins-Monro quantile estimate), so it settles where a share tau of the
// activities lies below it. The winner's edges see one activity per step (the
// winner's), the other edges all non-winners' activities at once; a step without
// a winner moves only the other edges, over all neurons. A step adapts with the
// edges the previous learning step left; frozen and zero-output steps (M.4.15)
// move nothing.
//
// Everything is f32 so the GPU kernels (WGSL has no f64) run the same arithmetic;
// window_tracking_wgsl() hands them these constants and the update step.
//

#ifndef COAST_NGM_WINDOW_TRACKING_H
#define COAST_NGM_WINDOW_TRACKING_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "hd_ngm2/hd_ngm2_tools.h"
#include "logging/state_io.h"

namespace ngm::window_tracking {

// Values from the window-tracking experiments (MNIST, CIFAR-10, dSprites, 500k
// steps each): the quantiles reproduce the window fitted to MNIST winner activity
// at 500k steps, the steepness is that fit's, the start is the former default.
inline constexpr float rate           = 0.001f;  // edge step per learning step
inline constexpr float quantile_lower = 0.19f;   // share of activities below the lower edge
inline constexpr float quantile_upper = 0.94f;   // share of activities below the upper edge
inline constexpr float steep_lower    = 0.57f;
inline constexpr float steep_upper    = 0.78f;
inline constexpr float initial_lower  = 0.33f;
inline constexpr float initial_upper  = 0.66f;

static_assert(rate > 0.0f && rate <= 0.1f);
static_assert(0.0f < quantile_lower && quantile_lower < quantile_upper && quantile_upper < 1.0f);
static_assert(0.0f < steep_lower && steep_lower < 1.0f && 0.0f < steep_upper && steep_upper < 1.0f);
static_assert(0.0f <= initial_lower && initial_lower < initial_upper && initial_upper <= 1.0f);

// The four edges: learning state of a group.
struct edges_t {
    float winner_lower = initial_lower;
    float winner_upper = initial_upper;
    float other_lower  = initial_lower;
    float other_upper  = initial_upper;
};

using window_t = std::pair<ngm2::sigmoid_shape_t, ngm2::sigmoid_shape_t>;

inline window_t window(float lower, float upper)
{
    return { { steep_lower, lower }, { steep_upper, upper } };
}
inline window_t winner_window(const edges_t& e) { return window(e.winner_lower, e.winner_upper); }
inline window_t other_window (const edges_t& e) { return window(e.other_lower,  e.other_upper);  }

// One edge step. Mirrored by wt_step() in window_tracking_wgsl().
inline float step(float q, float tau, float share_below)
{
    return std::clamp(q + rate * (tau - share_below), 0.0f, 1.0f);
}

// Advance the edges by one learning step. `act` holds every neuron's raw activity
// this step, `winner` the winner's index (act.size() if there is none).
inline void update(edges_t& e, std::span<const float> act, std::size_t winner)
{
    if (winner < act.size()) {
        const float a = act[winner];
        e.winner_lower = step(e.winner_lower, quantile_lower, a < e.winner_lower ? 1.0f : 0.0f);
        e.winner_upper = step(e.winner_upper, quantile_upper, a < e.winner_upper ? 1.0f : 0.0f);
    }
    std::size_t below_lower = 0, below_upper = 0, count = 0;
    for (std::size_t i = 0; i < act.size(); ++i) {
        if (i == winner) continue;
        below_lower += act[i] < e.other_lower;
        below_upper += act[i] < e.other_upper;
        ++count;
    }
    if (count == 0) return;   // groups have >= 2 neurons (M.1.1); defensive only
    const float n = static_cast<float>(count);
    e.other_lower = step(e.other_lower, quantile_lower, static_cast<float>(below_lower) / n);
    e.other_upper = step(e.other_upper, quantile_upper, static_cast<float>(below_upper) / n);
}

// Checkpoint: a group subgroup "activity_window_tracking" with the four edges,
// the same in every implementation so checkpoints move between them as before.
inline void save(logging::state_writer& w, const edges_t& e)
{
    w.begin_group("activity_window_tracking");
    w.write("winner_lower", e.winner_lower);
    w.write("winner_upper", e.winner_upper);
    w.write("other_lower",  e.other_lower);
    w.write("other_upper",  e.other_upper);
    w.end_group();
}

// Read and check the edges of a checkpoint; assigns nothing, so a caller can
// read in its checking phase and assign in its commit phase. Format 2 always
// carries the group: a missing group, a missing field or an edge outside [0, 1]
// is a corrupt checkpoint (throws, naming `who`). A format-1 checkpoint predates
// tracking and has no edges: the trackers then start over at the initial edges,
// whatever they held before the load. (The edges may legitimately cross, so
// lower > upper is not refused.)
[[nodiscard]] inline edges_t read(logging::state_reader& r, std::string_view who)
{
    auto corrupt = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: corrupt checkpoint — {}", who, what));
    };
    if (!r.enter_group("activity_window_tracking")) {
        if (r.format_version() >= 2) throw corrupt("activity_window_tracking missing");
        return edges_t{};
    }
    auto edge = [&](std::string_view name) {
        if (!r.has_field(name))
            throw corrupt(std::format("activity_window_tracking.{} missing", name));
        const float v = r.read_float(name, 0.0f);
        if (!(std::isfinite(v) && v >= 0.0f && v <= 1.0f))
            throw corrupt(std::format("activity_window_tracking.{} = {} is outside [0, 1]", name, v));
        return v;
    };
    edges_t e;
    e.winner_lower = edge("winner_lower");
    e.winner_upper = edge("winner_upper");
    e.other_lower  = edge("other_lower");
    e.other_upper  = edge("other_upper");
    r.leave_group();
    return e;
}

// WGSL prelude for the GPU groups: the constants above and the edge step. The
// constants are printed with 9 significant digits, which round-trips an f32.
inline std::string window_tracking_wgsl()
{
    return std::format(
        "// activity window tracking (generated from ngm/window_tracking.h)\n"
        "const WT_RATE: f32 = {:.9g};\n"
        "const WT_Q_LOWER: f32 = {:.9g};\n"
        "const WT_Q_UPPER: f32 = {:.9g};\n"
        "const WT_STEEP_LOWER: f32 = {:.9g};\n"
        "const WT_STEEP_UPPER: f32 = {:.9g};\n"
        "fn wt_step(q: f32, tau: f32, share_below: f32) -> f32 {{\n"
        "    return clamp(q + WT_RATE * (tau - share_below), 0.0, 1.0);\n"
        "}}\n",
        rate, quantile_lower, quantile_upper, steep_lower, steep_upper);
}

} // namespace ngm::window_tracking

#endif // COAST_NGM_WINDOW_TRACKING_H
