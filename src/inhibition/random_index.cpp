//
// Created for the COAST swappable-inhibition feature.
//

#include "inhibition/random_index.h"

#include "diag/diag.h"

#include "core/rng/fast_rng.h"
#include "logging/state_io.h"

#include <algorithm>
#include <format>
#include <random>
#include <stdexcept>
#include <string>

namespace inhibition {

namespace {
// Config-contract check that fires in Release too (this project builds -O3 with
// NDEBUG, so assert() is a no-op and cannot guard real invariants). random_index_t
// is constructed during sim build (cold path), so a throw is caught by build_env
// and reported gracefully — like every other config error — instead of aborting
// the whole process.
[[noreturn]] void ri_fatal(const char* msg)
{
    throw std::runtime_error(std::string("random_index: ") + msg);
}
}  // namespace

random_index_t::random_index_t(const inhibition_params_t& p)
    : inhibitor_t(p)
{
    // Config contract: both knobs are normalized. An out-of-range config is a
    // mistake, so fail fast rather than silently clamp.
    if (!(params_.ri_reach >= 0.0 && params_.ri_reach <= 1.0))
        ri_fatal("ri_reach must be in [0,1]");
    if (!(params_.ri_strength >= 0.0 && params_.ri_strength <= 1.0))
        ri_fatal("ri_strength must be in [0,1]");

    if (params_.ri_seed_from_time) {
        // Canonical non-deterministic seed source. The resolved value replaces
        // ri_seed, so params() — and with it the save state — carries the seed M
        // is generated from. The group's configuration record leaves a drawn seed
        // out (inhibition::append_config): it is random state, not a setting.
        std::random_device rd;
        params_.ri_seed = seed_t{ static_cast<std::uint32_t>(rd()) };
    }
}

void random_index_t::ensure_init(std::size_t neuron_count)
{
    if (initialized) {
        // The owning group's neuron count is fixed for the inhibitor's lifetime.
        if (neuron_count != n)
            ri_fatal("neuron count changed after init");
        return;
    }

    n = neuron_count;
    matrix.assign(n * n, 0.0f);
    if (n > 0) {
        // Fill row-major with U[0,1), then force a zero diagonal (no self-
        // inhibition). uniform_real_distribution is the standard adaptor; we need
        // statistical uniformity, not cross-toolchain bit-reproducibility (M is
        // persisted directly, so save/load fidelity does not depend on the RNG).
        core::rng::fast_rng_t rng(std::to_underlying(params_.ri_seed));
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : matrix)
            v = dist(rng);
        for (std::size_t i = 0; i < n; ++i)
            matrix[i * n + i] = 0.0f;
    }

    weight.assign(n * n, 0.0f);
    inh.assign(n, 0.0f);
    recompute();
    initialized = true;
}

void random_index_t::recompute()
{
    switch (params_.ri_variant) {
        case random_index_variant_t::threshold_linear: {
            // T[i,j] = (M[i,j] < reach) ? strength : 0. The float M entry is
            // promoted to double for the comparison against the double `reach`.
            const float s = static_cast<float>(params_.ri_strength);
            for (std::size_t k = 0; k < matrix.size(); ++k)
                weight[k] = (static_cast<double>(matrix[k]) < params_.ri_reach) ? s : 0.0f;
            // M's diagonal is already 0, and 0 < reach would otherwise create a
            // self-connection; force the T diagonal to 0 to be explicit.
            for (std::size_t i = 0; i < n; ++i)
                weight[i * n + i] = 0.0f;
            break;
        }
        case random_index_variant_t::TYPE_COUNT:
            break;  // not a real variant
    }
}

void random_index_t::inhibit(std::span<const float> raw, std::span<float> out)
{
    if (raw.size() != out.size())
        ri_fatal("raw/out size mismatch");
    ensure_init(raw.size());
    if (n == 0)
        return;

    // inh = T · raw. Compute the whole vector from raw before writing any out, so
    // the transform is correct even when out aliases raw (the CPU in-place case).
    for (std::size_t i = 0; i < n; ++i) {
        const float* row = &weight[i * n];
        float acc = 0.0f;
        for (std::size_t j = 0; j < n; ++j)
            acc += row[j] * raw[j];
        inh[i] = acc;
    }
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::clamp(raw[i] - inh[i], 0.0f, 1.0f);
}

void random_index_t::save_state(logging::state_writer& w) const
{
    // Complete, self-describing checkpoint: all hyperparameters plus the matrix.
    w.write("initialized", initialized);
    w.write("variant",  static_cast<std::uint32_t>(params_.ri_variant));
    w.write("reach",    params_.ri_reach);
    w.write("strength", params_.ri_strength);
    w.write("seed",     std::uint64_t{ std::to_underlying(params_.ri_seed) });
    w.write("n",        static_cast<std::uint64_t>(n));
    if (initialized)
        w.write_2d("matrix", matrix, n, n);
}

void random_index_t::load_state(logging::state_reader& r, std::size_t neuron_count)
{
    // Read and check everything first, then assign (M.7.18): a throw below
    // leaves the inhibitor as it was.
    //
    // The random state comes from the checkpoint (M.7.8), so its seed is taken —
    // also before init: the first inhibit() then generates the M the saved run
    // would have, even when the live ri_seed differs (other seed offsets, or a
    // fresh ri_seed_from_time draw). params() names the seed of the M in use, and
    // a later save records it. Config-wins for the settings: variant/reach/
    // strength keep the live config's values (set by the constructor); their
    // persisted copies are not read back.
    // Stored as uint64. A checkpoint from before seeds were 32-bit (core/seed.h)
    // may hold a wider drawn seed; M was generated from its low 32 bits, which is
    // all the RNG took, so those are the seed of the M in use. A checkpoint
    // without a seed (predating this kind) keeps the live one.
    const seed_t seed{ static_cast<std::uint32_t>(
        r.read_uint64("seed", std::to_underlying(params_.ri_seed))) };

    // Pre-init checkpoint, or one predating this kind: stay lazy and let the
    // first inhibit() generate the matrix from that seed.
    bool                              staged_init = r.read_bool("initialized", false);
    logging::state_reader::array_2d_info info;
    std::vector<float>                staged_matrix;
    if (staged_init) {
        staged_matrix = r.read_float_2d("matrix", info);
        // Claimed initialized but the matrix is absent/malformed — treat as not
        // initialized and regenerate lazily rather than restore corrupt state.
        if (staged_matrix.empty() || info.rows != info.cols) {
            COAST_WARN("checkpoint", "random_index: the checkpoint claims a matrix but holds {}; "
                       "it is regenerated from seed {} at the first step",
                       staged_matrix.empty() ? std::string("none")
                                             : std::format("a {}x{} one", info.rows, info.cols),
                       std::to_underlying(seed));
            staged_init = false;
            staged_matrix.clear();
        }
    }
    if (staged_init) {
        // The explicit `n` field must agree with the matrix's 2D metadata, and the
        // matrix must fit the owning group (ensure_init would refuse it at the
        // first inhibit(), after the load had claimed success).
        if (r.read_uint64("n", info.rows) != info.rows)
            ri_fatal("stored n disagrees with matrix shape");
        if (info.rows != neuron_count)
            ri_fatal(("checkpoint matrix is " + std::to_string(info.rows) + "x" +
                      std::to_string(info.rows) + ", the group has " +
                      std::to_string(neuron_count) + " neurons").c_str());
    }

    const std::size_t  staged_n = staged_init ? info.rows : 0;
    std::vector<float> staged_weight(staged_n * staged_n, 0.0f);
    std::vector<float> staged_inh(staged_n, 0.0f);

    // Commit: moves only, nothing below throws.
    params_.ri_seed = seed;
    initialized     = staged_init;
    n               = staged_n;
    matrix          = std::move(staged_matrix);
    weight          = std::move(staged_weight);
    inh             = std::move(staged_inh);
    // T is rebuilt from the restored M against the live settings.
    if (initialized) recompute();
}

}
