//
// Created by jk on the ngm3 GPU port (S6).
//

#ifdef COAST_WITH_GPU

#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"

#include "core/gpu/shared_prng.h"
#include "core/require.h"
#include "diag/diag.h"
#include "hd_ngm2/hd_ngm2_tools.h"          // ngm2::response_noise_window, learning_noise_window
#include "hd_ngm3/hd_ngm3_cold_init.h"
#include "hd_ngm3/hd_ngm3_layout.h"
#include "logging/config_record.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <bit>
#include <cstdint>
#include <cstring>
#include <format>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "observe/observe_accessor.h"

namespace ngm3 {

namespace {

// Floats per channel in the stats record the stats kernel writes:
// {sum, avg, true-max, 1 - Hoyer, count}. The kernels get it as STATS_W.
constexpr std::uint32_t k_stats_width = 5;

// ===========================================================================
// DEVICE-SPECIFIC kernel source + uniform layouts. Selected by the GPU backend
// chosen at compile time (COAST_GPU_BACKEND). Today this is WebGPU/WGSL; the
// host-array `null` backend reuses the same std140-shaped structs because it is
// layout-agnostic and no-ops compute, so both share this arm.
//
// A future NATIVE backend (Metal / Vulkan / CUDA) MUST add its own arm here,
// providing the corresponding kernel source AND the matching uniform data
// structures (`params_data_t`, `adapt_params_data_t`, `bank_dims_data_t`).
//
// It must ALSO audit the rest of this TU before assuming "just swap the arm":
// the backend-agnostic methods (build_step_plan / uniform fill) reference these
// structs by NAME and assume their field layout, and the host paths bake in
// WGSL-driven choices (std140 alignment, u8->u32 seg widening because WGSL has
// no byte storage). If a new backend's properties differ, that shared code is
// relying on device-specific properties and a refactor is required — not just a
// second #ifdef arm.
// ===========================================================================
#if defined(COAST_WITH_WEBGPU) || defined(COAST_WITH_NULL_GPU)

// The S6 compute kernels (stats / response+combine / winner / adapt / branch),
// embedded at build time — same #embed mechanism as mnist_io_gpu, path relative
// to this .cpp. Local inhibition is NOT here: it is injected at build time by the
// swappable inhibition::gpu_inhibitor_t (its WGSL is a self-contained string).
constexpr unsigned char kNgm3GpuWgsl[] = {
#embed "hd_ngm3_gpu.wgsl"
};

// Constants the kernels share with the host, as WGSL prepended to the kernel
// source: the dendrite noise-gate windows (the host's
// ngm2::response_noise_window and ngm2::learning_noise_window, not copies of
// their values; 9 significant digits round-trip every float exactly) and the
// stats record width.
std::string host_constants_wgsl()
{
    const auto emit = [](std::string_view prefix, const ngm2::noise_window_t &w) {
        return std::format("const {0}_START: f32 = {1:.9g};\n"
                           "const {0}_WIDTH: f32 = {2:.9g};\n"
                           "const {0}_STEEPNESS: f32 = {3:.9g};\n"
                           "const {0}_TRANSITION: f32 = {4:.9g};\n",
                           prefix, w.start, w.width,
                           static_cast<float>(w.shape.steepness),
                           static_cast<float>(w.shape.transition_point));
    };
    return emit("NOISE_RESP", ngm2::response_noise_window)
         + emit("NOISE_LEARN", ngm2::learning_noise_window)
         + std::format("const STATS_W: u32 = {}u;\n", k_stats_width);
}

// 64-byte uniform mirrored into params_uniform_ each step. MUST match the WGSL
// `struct Params` field-for-field (16 scalars, std140-clean: all 4-byte scalars,
// struct size a multiple of 16).
struct params_data_t {
    std::uint32_t step;
    std::uint32_t seed;
    std::uint32_t neuron_cnt;
    std::uint32_t rep_cnt;
    std::uint32_t a_d_cnt;
    std::uint32_t p_d_cnt;
    std::uint32_t a_input_size;
    std::uint32_t p_input_size;
    float         perm_thres;
    float         inv_perm_range;
    float         _pad2;          // was local_inhib_strength (unread; the inhibitor has its own uniform)
    float         noise_lo;       // combine-noise low bound
    float         noise_range;    // combine-noise span (hi - lo)
    std::uint32_t frozen;         // unread by the kernels (a frozen plan omits the learning
                                  // passes, the mx_act update with them); layout only
    float         _pad3;
    float         _pad4;
};
static_assert(sizeof(params_data_t) == 64, "Params uniform must be 64 bytes");

// 96-byte AdaptParams uniform — must match the WGSL `struct AdaptParams`
// field-for-field (24 scalars, std140-clean). The trailing 4 scalars are the S8
// branch params; this same bank-agnostic uniform feeds winner / adapt / branch.
struct adapt_params_data_t {
    std::uint32_t step;
    std::uint32_t seed;
    std::uint32_t neuron_cnt;
    std::uint32_t rep_cnt;
    std::uint32_t max_seg_id;
    float perm_thres;
    float inv_perm_range;
    float primary_lr;
    float secondary_lr;
    float common_lr;
    float stochastic_win_thres;
    float mismatch_act_thres;
    float mismatch_smoothing;
    float wf_steep;
    float wf_trans;
    float _pad_w0;   // formerly activity_learning_window; the window is tracked now
    float _pad_w1;
    float _pad_w2;
    float _pad_w3;
    // ---- S8 branch scalars ----
    std::uint32_t branch_depth;
    float accumulated_theta_thres;
    float min_mismatch_deviation;
    float min_mismatch_percentage;
    std::uint32_t branch_interval;   // learning steps between branch checks; read by winner_main
};
static_assert(sizeof(adapt_params_data_t) == 96, "AdaptParams uniform must be 96 bytes");

// 16-byte BankDims uniform — per-bank {d_cnt, input_size} so one generic
// adapt_main entry point handles both banks via two bind groups.
struct bank_dims_data_t {
    std::uint32_t d_cnt;
    std::uint32_t input_size;
    std::uint32_t jitter_salt;   // per-bank salt for the branch jitter key (0 for proximal)
    std::uint32_t _p1;
};
static_assert(sizeof(bank_dims_data_t) == 16, "BankDims uniform must be 16 bytes");

#else
#  error "hd_ngm3 GPU entity: the selected COAST_GPU_BACKEND has no device-specific \
kernel source / uniform layout. Add an arm above (see the DEVICE-SPECIFIC note)."
#endif // device-specific kernel source + uniform layouts

// NOTE: bank_host_init_t / compute_bank_init below are BACKEND-AGNOSTIC host-side
// init (plain std::vector) consumed by the shared build_step_plan — they
// intentionally live OUTSIDE the device-specific guard.

// Host-side per-bank init arrays: the cold permanences from cold_init() (the
// draw contract shared with the CPU group, hd_ngm3_cold_init.h) plus their
// zero-init companions. Absent banks (d_cnt==0 || input_size==0) collapse to
// 1-element placeholders; the kernels guard on d_cnt and never read them.
struct bank_host_init_t {
    std::vector<float>         perm;
    std::vector<std::uint32_t> seg;
    std::vector<float>         adapt;
    std::vector<float>         mism;
    std::vector<float>         acts;
    std::vector<float>         inbuf;
    std::array<float, k_stats_width> stats;
};

bank_host_init_t compute_bank_init(std::vector<float>&& cold_perm, std::size_t neuron_cnt,
                                   std::size_t rep_cnt, bank_extent_t bank) {
    const bool        absent  = (bank.d_cnt == 0 || bank.input_size == 0);
    const std::size_t total   = absent ? 1 : neuron_cnt * bank.d_cnt * rep_cnt * bank.input_size;
    const std::size_t act_cnt = absent ? 1 : neuron_cnt * bank.d_cnt * rep_cnt;
    if (absent ? !cold_perm.empty() : cold_perm.size() != total)
        throw std::logic_error(std::format(
            "ngm3_gpu: cold permanences hold {} values, the bank needs {}",
            cold_perm.size(), absent ? 0 : total));

    bank_host_init_t h{};
    h.perm = absent ? std::vector<float>(1, 0.0f) : std::move(cold_perm);
    h.seg  .assign(total, 1u);
    h.adapt.assign(total, 0.0f);
    h.mism .assign(total, 0.0f);
    h.acts .assign(act_cnt, 0.0f);
    h.inbuf.assign(absent ? 1 : bank.input_size, 0.0f);
    h.stats = {};
    return h;
}

} // namespace

// ---------------------------------------------------------------------------
// ctor — params lookup + derived sizes; bank storage waits for init_io.
// ---------------------------------------------------------------------------

neuron_group_gpu_t::neuron_group_gpu_t(const cfg_store   &cs,
                                       const std::string &type,
                                       sim::output_id_t   output_id,
                                       seed_offset_t      seed_offset)
  : params(cs.group_or_default(type)),
    poisson_seed_(params.random_seed + seed_offset),
    id(output_id)
{
    validate_params(params, k_gpu_caps);   // hd_ngm3_host.h, shared with the CPU group

    rep_cnt    = std::size_t{1} << params.max_branch_level;
    max_seg_id = (1u << (params.max_branch_level + 1)) - 1u;

    // Construct this group's GPU inhibition (kind only; the device is bound later
    // in build_step_plan via emit()). Empty inhibition_cfg synthesizes a
    // max_softmax with beta = local_inhibition_strength.
    gpu_inhibitor_ = inhibition::make_gpu_inhibitor(cs.inhibitions, params.inhibition_cfg,
                                                    params.local_inhibition_strength);

    // Device buffers are allocated + initialized in build_step_plan() (called by
    // env at build time); the constructor only resolves params + derived sizes.
}

// ---------------------------------------------------------------------------
// init_io — resolve channel routing, then record the per-bank input sizes.
// ---------------------------------------------------------------------------

void neuron_group_gpu_t::init_io(
    const std::unordered_map<std::string, sim::output_channel_t> &input_map)
{
    // Routing and its guards are shared with the CPU group (hd_ngm3_host.h).
    auto routing    = route_inputs(params, input_map, k_gpu_caps, id);
    apical_inputs   = std::move(routing.apical);
    proximal_inputs = std::move(routing.proximal);
    inp_ids         = std::move(routing.inp_ids);
    // The per-bank input widths; the device buffers (cold state from
    // ngm3::cold_init) are allocated in build_step_plan().
    apical_input_size   = (params.apical_cnt == 0) ? 0 : apical_inputs.width();
    proximal_input_size = proximal_inputs.width();
    init_structure();
}

// ---------------------------------------------------------------------------
// io wiring
// ---------------------------------------------------------------------------

void neuron_group_gpu_t::set_output_buffer(sim::io_buffer& buf) {
    output_buf = &buf;
}

void neuron_group_gpu_t::set_input_buffer(sim::output_id_t id_, sim::io_buffer& buf) {
    apical_inputs  .bind(id_, buf);
    proximal_inputs.bind(id_, buf);
}

// ---------------------------------------------------------------------------
// process — host bookkeeping + per-step uniform write
// ---------------------------------------------------------------------------

sim::step_result neuron_group_gpu_t::process() {
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);

    params_data_t pd{};
    pd.step                 = step_;
    pd.seed                 = std::to_underlying(poisson_seed_);
    pd.neuron_cnt           = static_cast<std::uint32_t>(params.neuron_cnt);
    pd.rep_cnt              = static_cast<std::uint32_t>(rep_cnt);
    pd.a_d_cnt              = static_cast<std::uint32_t>(params.apical_cnt);
    pd.p_d_cnt              = static_cast<std::uint32_t>(params.proximal_cnt);
    pd.a_input_size         = static_cast<std::uint32_t>(apical_input_size);
    pd.p_input_size         = static_cast<std::uint32_t>(proximal_input_size);
    pd.perm_thres           = params.permanence_threshold;
    pd.inv_perm_range       = 1.0f / (1.0f - params.permanence_threshold);
    pd.noise_lo             = params.neuron_noise_range.first;
    pd.noise_range          = params.neuron_noise_range.second - params.neuron_noise_range.first;
    pd.frozen               = params.frozen ? 1u : 0u;

    // AdaptParams — bank-agnostic scalar set for winner_main / adapt_main. All
    // fields but `step` are static, but rewriting the whole 96 B each step is
    // negligible and keeps process() a single, obvious mirror.
    adapt_params_data_t ad{};
    ad.step                 = step_;
    ad.seed                 = std::to_underlying(poisson_seed_);
    ad.neuron_cnt           = static_cast<std::uint32_t>(params.neuron_cnt);
    ad.rep_cnt              = static_cast<std::uint32_t>(rep_cnt);
    ad.max_seg_id           = max_seg_id;
    ad.perm_thres           = params.permanence_threshold;
    ad.inv_perm_range       = 1.0f / (1.0f - params.permanence_threshold);
    ad.primary_lr           = params.primary_learning_rate;
    ad.secondary_lr         = params.secondary_learning_rate;
    ad.common_lr            = params.common_learning_rate;
    ad.stochastic_win_thres = params.stochastic_win_thres;
    ad.mismatch_act_thres   = params.mismatch_act_thres;
    ad.mismatch_smoothing   = params.mismatch_smoothing;
    ad.wf_steep             = static_cast<float>(params.weight_filter.steepness);
    ad.wf_trans             = static_cast<float>(params.weight_filter.transition_point);
    ad.branch_depth             = static_cast<std::uint32_t>(params.max_branch_level);
    ad.accumulated_theta_thres  = params.accumulated_theta_thres;
    ad.min_mismatch_deviation   = params.min_mismatch_deviation;
    ad.min_mismatch_percentage  = params.min_mismatch_percentage;
    // The branch gate is the learning-step count reaching a multiple of
    // branch_interval, as on the CPU ((++input_cnt) % branch_interval == 0). The
    // count lives on the device (learn_cnt_id_, advanced by winner_main): whether
    // a step learns depends on the group output summing to non-zero (SPEC.md
    // M.4.15), which only the device knows mid-step. Not from step_: that keys the
    // noise draws and advances on every step (M.7.21).
    ad.branch_interval          = static_cast<std::uint32_t>(params.branch_interval);

    // Refresh the two per-step uniforms through the agnostic data plane, ordered
    // ahead of this step's enqueue so the kernels see the values just written.
    dev_->upload(params_uniform_id_, std::as_bytes(std::span<const params_data_t>(&pd, 1)));
    dev_->upload(adapt_params_id_,   std::as_bytes(std::span<const adapt_params_data_t>(&ad, 1)));

    // step_ always advances: it keys the stateless noise draws (the GPU analogue
    // of the CPU RNG streams). The learning-step counter is device-side.
    ++step_;

    return {};
}

// ---------------------------------------------------------------------------
// build_step_plan — emit the static per-step work as a backend-agnostic
// step_plan (E1). Called once at build time by env. Registers + initializes this
// entity's private device buffers (cold permanences from ngm3::cold_init, the
// contract shared with the CPU group), compiles its 5 kernels (the inhibition kernel is contributed by the
// injected gpu_inhibitor_), reads its io_buffer page ids, and builds the plan.
// The op list is a strictly-ordered set of copies + compute passes (gather,
// stats, response/combine, inhibition, output-publish, winner, adapt, gated
// branch).
// ---------------------------------------------------------------------------

void neuron_group_gpu_t::build_step_plan(coast::gpu::gpu_device& dev) {
    dev_ = &dev;
    register_device_buffers(dev);
    compile_kernels(dev);

    // The op list, op-for-op identical to record().
    coast::gpu::step_plan sp;
    append_forward_ops(dev, sp);
    // A frozen (inference-only) group has no learning passes (winner/adapt/
    // branch): the plan is static, so their kernels are never dispatched. Mirrors the CPU
    // group's early return; the mx_act update is the winner kernel's, so it goes too. Note:
    // adapt_scalars_ stays zero, so the "max activity"/"winning thres" probes
    // report 0 while frozen.
    if (!params.frozen) append_learning_ops(sp);

    plan_id_ = dev.register_plan();
    dev.build_plan(plan_id_, sp);
}

void neuron_group_gpu_t::register_device_buffers(coast::gpu::gpu_device& dev) {
    using namespace coast::gpu;
    // ---- register + init private device buffers ----------------------------
    // Byte sizes come from the host init arrays, which already encode the
    // absent-bank 1-element placeholder collapse — so the backend's whole-buffer
    // bind size matches record()'s perm_bytes/in_bytes/etc. exactly.
    auto reg_init = [&](topology topo, std::span<const std::byte> data) -> buffer_id {
        const buffer_id id = dev.register_buffer(topo, data.size(), /*ping_pong=*/false);
        dev.upload(id, data);
        return id;
    };

    auto register_bank = [&](bank_ids_t& ids, const bank_host_init_t& h,
                             std::size_t d_cnt, std::size_t input_size,
                             std::uint32_t jitter_salt, const bank_inputs_t& routes) {
        ids.permanences   = reg_init(topology::gpu_resident, std::as_bytes(std::span(h.perm)));
        ids.segment_ids   = reg_init(topology::gpu_resident, std::as_bytes(std::span(h.seg)));
        ids.adapt_history = reg_init(topology::gpu_resident, std::as_bytes(std::span(h.adapt)));
        ids.mismatch      = reg_init(topology::gpu_resident, std::as_bytes(std::span(h.mism)));
        ids.rep_act       = reg_init(topology::gpu_resident, std::as_bytes(std::span(h.acts)));
        ids.input_buf     = reg_init(topology::gpu_resident, std::as_bytes(std::span(h.inbuf)));
        // Per-channel input statistics (SPEC.md M.3.2): one [sum, avg, max, noise, count]
        // record per routed channel, and a static channel index per input
        // position for the response and adapt kernels. An absent bank keeps a
        // single placeholder channel.
        const bool absent = (d_cnt == 0 || input_size == 0);
        ids.chan_cnt = absent || routes.sizes.empty()
                     ? 1u : static_cast<std::uint32_t>(routes.sizes.size());
        std::vector<std::uint32_t> chan_of(absent ? 1 : input_size, 0u);
        if (!absent)
            for (std::size_t c = 0; c < routes.sizes.size(); ++c)
                for (std::size_t q = 0; q < routes.sizes[c]; ++q)
                    chan_of[routes.offsets[c] + q] = static_cast<std::uint32_t>(c);
        const std::vector<float> stats0(std::size_t{k_stats_width} * ids.chan_cnt, 0.0f);
        ids.stats   = reg_init(topology::gpu_resident, std::as_bytes(std::span(stats0)));
        ids.chan_of = reg_init(topology::gpu_resident, std::as_bytes(std::span(chan_of)));
        const bank_dims_data_t dims{static_cast<std::uint32_t>(d_cnt),
                                    static_cast<std::uint32_t>(input_size), jitter_salt, 0u};
        ids.bank_dims = dev.register_buffer(topology::uniform, sizeof(dims), false);
        dev.upload(ids.bank_dims, std::as_bytes(std::span<const bank_dims_data_t>(&dims, 1)));
    };

    // The cold permanences come from the contract the CPU group uses too
    // (hd_ngm3_cold_init.h), so a CPU and a GPU group built from one config start
    // bit-identical. The per-neuron streams it also derives are the CPU's; the
    // device draws are hash-keyed and do not use them.
    {
        const bank_extent_t apical  { params.apical_cnt,   apical_input_size   };
        const bank_extent_t proximal{ params.proximal_cnt, proximal_input_size };
        cold_state_t cold = cold_init(poisson_seed_, params.neuron_cnt, rep_cnt,
                                      params.permanence_threshold, apical, proximal);
        const bank_host_init_t a_init = compute_bank_init(
            std::move(cold.apical_perm), params.neuron_cnt, rep_cnt, apical);
        // The apical salt makes its branch jitter independent of the proximal bank's
        // (same flat cell index, same step). Proximal keeps salt 0, so its draws are
        // unchanged.
        register_bank(apical_ids_, a_init, params.apical_cnt, apical_input_size, 0x632BE5ABu, apical_inputs);
        const bank_host_init_t p_init = compute_bank_init(
            std::move(cold.proximal_perm), params.neuron_cnt, rep_cnt, proximal);
        register_bank(proximal_ids_, p_init, params.proximal_cnt, proximal_input_size, 0u, proximal_inputs);
    }

    // Per-neuron scratch + group buffers (zero-init; apical_max defaults to 1.0
    // when apical is absent — mirrors the CPU init_storage's apmax_init).
    const std::vector<float> zeros_n(params.neuron_cnt, 0.0f);
    const std::vector<float> apmax_init(params.neuron_cnt,
                                        params.apical_cnt == 0 ? 1.0f : 0.0f);
    neuron_apical_max_id_   = reg_init(topology::gpu_resident, std::as_bytes(std::span(apmax_init)));
    neuron_proximal_max_id_ = reg_init(topology::gpu_resident, std::as_bytes(std::span(zeros_n)));
    neuron_act_id_          = reg_init(topology::gpu_resident, std::as_bytes(std::span(zeros_n)));
    neuron_mx_act_id_       = reg_init(topology::gpu_resident, std::as_bytes(std::span(zeros_n)));
    group_outp_id_          = reg_init(topology::gpu_resident, std::as_bytes(std::span(zeros_n)));
    // [winner_idx, act_sum, win_act, mx_act, window edges x4] (see winner_main).
    const std::array<std::uint32_t, 8> scal_init{};
    adapt_scalars_id_       = reg_init(topology::gpu_resident, std::as_bytes(std::span(scal_init)));
    // Activity window edges (learning state), from the host mirror.
    {
        const auto& e = window_edges_;
        const std::array<float, 4> track_init{ e.winner_lower, e.winner_upper, e.other_lower, e.other_upper };
        window_track_id_ = reg_init(topology::gpu_resident, std::as_bytes(std::span(track_init)));
    }
    // Learning-step counter [count lo, count hi, count % branch_interval, due].
    const std::array<std::uint32_t, 4> cnt_init{0u, 0u, 0u, 0u};
    learn_cnt_id_           = reg_init(topology::gpu_resident, std::as_bytes(std::span(cnt_init)));

    // Uniforms: registered only — process() writes them each step (matches
    // the CPU init_storage, where params/adapt uniforms are make_buffer-only and the first
    // process() before the first enqueue fills them).
    params_uniform_id_ = dev.register_buffer(topology::uniform, sizeof(params_data_t), false);
    adapt_params_id_   = dev.register_buffer(topology::uniform, sizeof(adapt_params_data_t), false);
}

void neuron_group_gpu_t::compile_kernels(coast::gpu::gpu_device& dev) {
    // ---- compile the 5 kernels (one module by source, 5 entry points) -------
    // DEVICE-SPECIFIC kernel generation — paired with the kernel-source/uniform
    // arm at the top of this TU. A native backend supplies its own source dialect
    // and entry-point names here. (Inhibition is injected separately by
    // gpu_inhibitor_->emit() at its slot below, not compiled here.)
#if defined(COAST_WITH_WEBGPU) || defined(COAST_WITH_NULL_GPU)
    const std::string src = std::string(coast::gpu::shared_prng_wgsl()) + host_constants_wgsl() +
        ngm::window_tracking::window_tracking_wgsl() +
        std::string(reinterpret_cast<const char*>(kNgm3GpuWgsl), sizeof(kNgm3GpuWgsl));
    k_stats_   = dev.register_kernel(src, "stats_main");
    k_respcmb_ = dev.register_kernel(src, "response_combine_main");
    k_winner_  = dev.register_kernel(src, "winner_main");
    k_adapt_   = dev.register_kernel(src, "adapt_main");
    k_branch_  = dev.register_kernel(src, "branch_main");
#else
#  error "hd_ngm3 GPU entity: no kernel generation for the selected COAST_GPU_BACKEND."
#endif
}

// (1)-(5): gather, stats, response+combine, inhibition, publish.
void neuron_group_gpu_t::append_forward_ops(coast::gpu::gpu_device& dev,
                                            coast::gpu::step_plan& sp) {
    using namespace coast::gpu;
    // io_buffer page id (ping-pong; registered by env at F).
    const buffer_id     out_page = output_buf->gpu_page_id();
    const std::uint32_t n_wg     = static_cast<std::uint32_t>(params.neuron_cnt);
    const std::uint64_t n_bytes  = static_cast<std::uint64_t>(params.neuron_cnt) * sizeof(float);

    // (1) gather copies: apical bank then proximal bank, each route in order. The
    //     source is an UPSTREAM io_buffer produced last step -> READ side (the
    //     opposite/older page = one-step latency); dst (input_buf) is non-ping_pong.
    auto emit_gather = [&](const bank_inputs_t& b, buffer_id dst) {
        for (std::size_t i = 0; i < b.bufs.size(); ++i) {
            if (b.bufs[i] == nullptr) continue;
            plan_op op{};
            op.kind            = op_kind::copy;
            op.copy.src        = b.bufs[i]->gpu_page_id();
            op.copy.dst        = dst;
            op.copy.src_offset = 0;
            op.copy.dst_offset = static_cast<std::uint64_t>(b.offsets[i]) * sizeof(float);
            op.copy.bytes      = static_cast<std::uint64_t>(b.sizes[i])   * sizeof(float);
            op.copy.src_side   = access::read;
            op.copy.dst_side   = access::write;
            sp.ops.push_back(std::move(op));
        }
    };
    emit_gather(apical_inputs,   apical_ids_.input_buf);
    emit_gather(proximal_inputs, proximal_ids_.input_buf);

    // (2) stats: one pass, proximal dispatch first then apical-if-present.
    //     bindings {0: input_buf, 1: stats}; one workgroup each.
    {
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = k_stats_;
        auto stats_disp = [&](const bank_ids_t& ids) {
            plan_dispatch d{};
            d.bindings = {{ids.input_buf, access::read, 0}, {ids.stats, access::write, 1},
                          {ids.chan_of, access::read, 2}};
            d.dispatch = {ids.chan_cnt, 1, 1};   // one workgroup per input channel
            return d;
        };
        op.dispatches.push_back(stats_disp(proximal_ids_));
        if (params.apical_cnt > 0) op.dispatches.push_back(stats_disp(apical_ids_));
        sp.ops.push_back(std::move(op));
    }

    // (3) response+combine: one dispatch, 13 bindings, one workgroup per neuron.
    {
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = k_respcmb_;
        plan_dispatch d{};
        d.bindings = {
            {apical_ids_.permanences,   access::read,  0},
            {apical_ids_.input_buf,     access::read,  1},
            {apical_ids_.stats,         access::read,  2},
            {apical_ids_.rep_act,       access::write, 3},
            {proximal_ids_.permanences, access::read,  4},
            {proximal_ids_.input_buf,   access::read,  5},
            {proximal_ids_.stats,       access::read,  6},
            {proximal_ids_.rep_act,     access::write, 7},
            {neuron_apical_max_id_,     access::write, 8},
            {neuron_proximal_max_id_,   access::write, 9},
            {neuron_act_id_,            access::write, 10},
            {params_uniform_id_,        access::read,  12},
            // segment ids: the response sums each segment on its own (per-segment
            // floor, SPEC.md M.7.31)
            {apical_ids_.segment_ids,   access::read,  13},
            {proximal_ids_.segment_ids, access::read,  14},
            // per-position channel index: per-channel 1/sum and draw range
            {apical_ids_.chan_of,       access::read,  15},
            {proximal_ids_.chan_of,     access::read,  16},
        };
        d.dispatch = {n_wg, 1, 1};
        op.dispatches.push_back(std::move(d));
        sp.ops.push_back(std::move(op));
    }

    // (4) local inhibition via the swappable GPU emitter (kernel injection): the
    //     emitter registers its own kernel + uniform and appends one compute op
    //     binding our activities (read) and the output WRITE page (read_write;
    //     the only ping-pong-bound compute pass -> backend builds both even/odd
    //     bind groups). Dispatch is a single workgroup, as before.
    gpu_inhibitor_->emit(dev, { neuron_act_id_, out_page,
                                static_cast<std::uint32_t>(params.neuron_cnt),
                                {1u, 1u, 1u} }, sp);

    // (5) publish: copy the post-inhibition output (out_page WRITE side = this
    //     step's fresh output) into the stable group_outp_ read source.
    {
        plan_op op{};
        op.kind            = op_kind::copy;
        op.copy.src        = out_page;
        op.copy.dst        = group_outp_id_;
        op.copy.src_offset = 0;
        op.copy.dst_offset = 0;
        op.copy.bytes      = n_bytes;
        op.copy.src_side   = access::write;
        op.copy.dst_side   = access::write;
        sp.ops.push_back(std::move(op));
    }
}

// (6)-(8): winner, adapt, gated branch.
void neuron_group_gpu_t::append_learning_ops(coast::gpu::step_plan& sp) {
    using namespace coast::gpu;
    const std::uint32_t n_wg = static_cast<std::uint32_t>(params.neuron_cnt);

    // (6) winner: one dispatch, 7 bindings, one workgroup. Also updates the
    //     running max, the learning-step counter and the activity window edges
    //     on learning steps (M.4.15, M.4.2a).
    {
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = k_winner_;
        plan_dispatch d{};
        d.bindings = {
            {group_outp_id_,    access::read,  0},
            {adapt_scalars_id_, access::write, 1},
            {adapt_params_id_,  access::read,  2},
            {neuron_act_id_,    access::read,  3},
            {neuron_mx_act_id_, access::write, 4},
            {learn_cnt_id_,     access::write, 5},
            {window_track_id_,  access::write, 6},
        };
        d.dispatch = {1, 1, 1};
        op.dispatches.push_back(std::move(d));
        sp.ops.push_back(std::move(op));
    }

    // (7) adapt: one pass, proximal then apical-if-present, 14 bindings each,
    //     one workgroup per neuron.
    {
        plan_op op{};
        op.kind   = op_kind::compute;
        op.kernel = k_adapt_;
        auto adapt_disp = [&](const bank_ids_t& ids, buffer_id bank_max) {
            plan_dispatch d{};
            d.bindings = {
                {ids.permanences,   access::write, 0},
                {ids.segment_ids,   access::read,  1},
                {ids.adapt_history, access::write, 2},
                {ids.mismatch,      access::write, 3},
                {ids.rep_act,       access::read,  4},
                {ids.input_buf,     access::read,  5},
                {bank_max,          access::read,  6},
                {ids.stats,         access::read,  7},
                {neuron_act_id_,    access::read,  8},
                {neuron_mx_act_id_, access::read,  9},
                {group_outp_id_,    access::read,  10},
                {adapt_scalars_id_, access::read,  11},
                {adapt_params_id_,  access::read,  12},
                {ids.bank_dims,     access::read,  13},
                {ids.chan_of,       access::read,  14},
            };
            d.dispatch = {n_wg, 1, 1};
            return d;
        };
        op.dispatches.push_back(adapt_disp(proximal_ids_, neuron_proximal_max_id_));
        if (params.apical_cnt > 0)
            op.dispatches.push_back(adapt_disp(apical_ids_, neuron_apical_max_id_));
        sp.ops.push_back(std::move(op));
    }

    // (8) branch: GATED pass (kernel early-outs unless winner_main marked a branch
    //     due in learn_cnt; C1), proximal then apical-if-present, 7 bindings each,
    //     one workgroup per neuron.
    //     segment_ids is rw here (read-only in adapt) -> distinct layout.
    {
        plan_op op{};
        op.kind          = op_kind::compute;
        op.kernel        = k_branch_;
        op.uniform_gated = true;
        auto branch_disp = [&](const bank_ids_t& ids) {
            plan_dispatch d{};
            d.bindings = {
                {ids.permanences,   access::write, 0},
                {ids.segment_ids,   access::write, 1},
                {ids.adapt_history, access::write, 2},
                {ids.mismatch,      access::write, 3},
                {adapt_params_id_,  access::read,  4},
                {ids.bank_dims,     access::read,  5},
                {learn_cnt_id_,     access::read,  6},
            };
            d.dispatch = {n_wg, 1, 1};
            return d;
        };
        op.dispatches.push_back(branch_disp(proximal_ids_));
        if (params.apical_cnt > 0) op.dispatches.push_back(branch_disp(apical_ids_));
        sp.ops.push_back(std::move(op));
    }
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------

sim::output_id_t neuron_group_gpu_t::get_outp_id() const { return id; }
std::size_t      neuron_group_gpu_t::get_outp_size() const { return params.neuron_cnt; }
std::span<const std::size_t> neuron_group_gpu_t::get_inp_ids() const {
    return { inp_ids.data(), inp_ids.size() };
}

std::string neuron_group_gpu_t::status_str() const {
    std::string s { "NGM3 Group (GPU)" };
    s += " | id: "        + std::to_string(get_outp_id());
    s += " | neurons: "   + std::to_string(params.neuron_cnt);
    s += " | apical/prox: " + std::to_string(params.apical_cnt)
       + "/"              + std::to_string(params.proximal_cnt);
    s += " | depth: "     + std::to_string(static_cast<int>(params.max_branch_level));
    s += " | step: "      + std::to_string(step_);
    return s;
}

// ---------------------------------------------------------------------------
// save/load_state — S7 persists the learned bank state (permanences /
// segment_ids / adapt_history / mismatch), schema-compatible with the CPU
// ngm3 group: same group/dataset names, segment_ids narrowed to u8 on save and
// widened back on load (values <= max_seg_id = 31).
// ---------------------------------------------------------------------------

// A failed readback throws: the vectors are zero-initialized, so returning one
// after a failure would hand save_state fabricated zeros as learned state (#234).
std::vector<float>
neuron_group_gpu_t::readback_dev_f32(coast::gpu::buffer_id id, std::size_t count,
                                     std::string_view what) const {
    std::vector<float> out(count);
    if (count > 0 && !dev_->readback(id, std::as_writable_bytes(std::span<float>(out))))
        throw std::runtime_error(std::format("ngm3_gpu: {} readback failed", what));
    return out;
}

std::vector<std::uint32_t>
neuron_group_gpu_t::readback_dev_u32(coast::gpu::buffer_id id, std::size_t count,
                                     std::string_view what) const {
    std::vector<std::uint32_t> out(count);
    if (count > 0 && !dev_->readback(id, std::as_writable_bytes(std::span<std::uint32_t>(out))))
        throw std::runtime_error(std::format("ngm3_gpu: {} readback failed", what));
    return out;
}

std::uint64_t neuron_group_gpu_t::learning_step_count() const {
    // Before build_step_plan the counter does not exist yet: no step has run.
    if (dev_ == nullptr || learn_cnt_id_ == coast::gpu::invalid_buffer) return 0;
    std::array<std::uint32_t, 4> c{0u, 0u, 0u, 0u};
    // A failure must not pass as 0: the count goes into checkpoints and keys the
    // branch cadence.
    if (!dev_->readback(learn_cnt_id_, std::as_writable_bytes(std::span<std::uint32_t>(c))))
        throw std::runtime_error("ngm3_gpu: learning-step counter readback failed");
    return (std::uint64_t{c[1]} << 32) | c[0];
}

std::vector<float>
neuron_group_gpu_t::readback_permanences(bank_t bank) const {
    const bool apical = (bank == bank_t::apical);
    const std::size_t d_cnt      = apical ? params.apical_cnt   : params.proximal_cnt;
    const std::size_t input_size = apical ? apical_input_size   : proximal_input_size;
    if (d_cnt == 0 || input_size == 0) return {};
    const std::size_t total = params.neuron_cnt * d_cnt * rep_cnt * input_size;
    return readback_dev_f32((apical ? apical_ids_ : proximal_ids_).permanences, total,
                            apical ? "apical permanences" : "proximal permanences");
}
std::vector<std::uint32_t>
neuron_group_gpu_t::readback_segment_ids(bank_t bank) const {
    const bool apical = (bank == bank_t::apical);
    const std::size_t d_cnt      = apical ? params.apical_cnt   : params.proximal_cnt;
    const std::size_t input_size = apical ? apical_input_size   : proximal_input_size;
    if (d_cnt == 0 || input_size == 0) return {};
    const std::size_t total = params.neuron_cnt * d_cnt * rep_cnt * input_size;
    return readback_dev_u32((apical ? apical_ids_ : proximal_ids_).segment_ids, total,
                            apical ? "apical segment_ids" : "proximal segment_ids");
}

// ---- D7-vis CPU mirror (S10) ----

namespace {
// Flat-SoA offset of representation (n, d, r), identical to the CPU group's
// rep_offset in hd_ngm3_neuron_group.cpp — so the same mirror data indexes the
// same way for both entities.
inline std::size_t vis_rep_offset(std::size_t n, std::size_t d_cnt,
                                  std::size_t d, std::size_t rep_cnt,
                                  std::size_t r, std::size_t input_size) {
    return n * d_cnt * rep_cnt * input_size
         + d         * rep_cnt * input_size
         + r                  * input_size;
}
} // namespace

void neuron_group_gpu_t::refresh_vis_mirror() const {
    // One bulk GPU->CPU copy per bank/data-view. The device must be idle — the
    // caller (visualizer update() on the GUI thread, inside the post-swap barrier)
    // guarantees this. readback_* return {} for an absent bank, leaving the mirror
    // empty (the accessors then return {} too).
    //
    // All four views are read before any mirror is replaced. A failed readback
    // keeps the last good mirrors (and their step) rather than showing zeros or a
    // mix of two refreshes, and does not throw: this runs on the GUI thread inside
    // the post-swap barrier, where nothing above would catch it (#234).

    // seg_ids: GPU stores u32 (WGSL has no byte storage); narrow to the u8 the CPU
    // type reports. Lossless — values are < MAX_SEG_CNT (32).
    auto narrow = [](const std::vector<std::uint32_t>& src) {
        std::vector<std::uint8_t> out(src.size());
        for (std::size_t i = 0; i < src.size(); ++i)
            out[i] = static_cast<std::uint8_t>(src[i]);
        return out;
    };
    try {
        auto ap = readback_permanences(bank_t::apical);
        auto pp = readback_permanences(bank_t::proximal);
        auto as = narrow(readback_segment_ids(bank_t::apical));
        auto ps = narrow(readback_segment_ids(bank_t::proximal));
        apical_perm_mirror_    = std::move(ap);
        proximal_perm_mirror_  = std::move(pp);
        apical_segid_mirror_   = std::move(as);
        proximal_segid_mirror_ = std::move(ps);
    } catch (const std::runtime_error& e) {
        COAST_WARN("ngm3_gpu", "group {}: vis mirror not refreshed, keeping the last one: {}",
                   id, e.what());
        return;
    }

    // The host-side sim step, not learning_step_count(): no readback by design,
    // so a GUI refresh issues no extra synchronous transfer and cannot throw on
    // one (the counter's readback throws, rightly, for save_state/load_state).
    vis_mirror_step_.store(step_, std::memory_order_release);
}

std::span<const float>
neuron_group_gpu_t::get_apical_representation(std::size_t n, std::size_t d, std::size_t r) const {
    if (params.apical_cnt == 0 || apical_perm_mirror_.empty()) return {};
    const std::size_t off = vis_rep_offset(n, params.apical_cnt, d, rep_cnt, r, apical_input_size);
    return { apical_perm_mirror_.data() + off, apical_input_size };
}

std::span<const float>
neuron_group_gpu_t::get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const {
    if (proximal_perm_mirror_.empty()) return {};
    const std::size_t off = vis_rep_offset(n, params.proximal_cnt, d, rep_cnt, r, proximal_input_size);
    return { proximal_perm_mirror_.data() + off, proximal_input_size };
}

std::span<const std::uint8_t>
neuron_group_gpu_t::get_apical_segment_ids(std::size_t n, std::size_t d, std::size_t r) const {
    if (params.apical_cnt == 0 || apical_segid_mirror_.empty()) return {};
    const std::size_t off = vis_rep_offset(n, params.apical_cnt, d, rep_cnt, r, apical_input_size);
    return { apical_segid_mirror_.data() + off, apical_input_size };
}

std::span<const std::uint8_t>
neuron_group_gpu_t::get_proximal_segment_ids(std::size_t n, std::size_t d, std::size_t r) const {
    if (proximal_segid_mirror_.empty()) return {};
    const std::size_t off = vis_rep_offset(n, params.proximal_cnt, d, rep_cnt, r, proximal_input_size);
    return { proximal_segid_mirror_.data() + off, proximal_input_size };
}

void neuron_group_gpu_t::read_probes() {
    // S9 per-step fan-out of group-level scalars. adapt_scalars_ is written by
    // winner_main every non-frozen step (the α-3 "kernel always writes"
    // property). A frozen plan never dispatches winner_main, so the buffer stays
    // init-zeroed and the readback is skipped -- the probes report 0. Layout:
    // {winner_idx (u32), act_sum, win_act, mx_act} with the last three bit-cast
    // f32 (see hd_ngm3_gpu.wgsl winner_main).
    //
    // Gate ONLY the readback on any_probe_active(); the PROBE() statements MUST
    // run every step so their registration statics fire (else the probes are
    // never discoverable). When inactive the values stay zero and probe()
    // discards them on its own active-count check.
    //
    // A failed readback publishes nothing this step rather than values of 0, like
    // the window edges below (probes are live diagnostics; nothing is persisted).
    float         mx_act = 0.0f, win_act = 0.0f, act_sum = 0.0f;
    std::uint32_t winner = 0;
    bool          scalars_ok = true;
    if (!params.frozen && any_probe_active()) {
        try {
            const auto s = readback_dev_u32(adapt_scalars_id_, 8, "probe scalars");   // device is idle here
            winner  = s[0];
            act_sum = std::bit_cast<float>(s[1]);
            win_act = std::bit_cast<float>(s[2]);
            mx_act  = std::bit_cast<float>(s[3]);
        } catch (const std::runtime_error&) {
            scalars_ok = false;
        }
    }
    if (scalars_ok) {
        PROBE("max activity",  mx_act);    // names match the ngm2 group probes so the
        PROBE("winning thres", win_act);   // same Observer visualizers apply
        PROBE("activity sum",  act_sum);
        PROBE("winner",        static_cast<double>(winner));
    }

    // Activity window edges as of the end of this step (device-resident). A
    // frozen group reads them too: its edges stay valid, they just stop moving.
    // A failed readback publishes nothing this step rather than edges of 0.
    ngm::window_tracking::edges_t we{ 0.0f, 0.0f, 0.0f, 0.0f };
    if (any_probe_active()) {
        try { we = window_edges(); }
        catch (const std::runtime_error&) { return; }
    }
    PROBE("window winner lower", we.winner_lower);   // same names as the ngm2 group's
    PROBE("window winner upper", we.winner_upper);
    PROBE("window other lower",  we.other_lower);
    PROBE("window other upper",  we.other_upper);
}

void neuron_group_gpu_t::perform_observations(observe::observe_accessor& acc) {
    // Observe the full post-inhibition output vector plus the input the group
    // received this step. All vectors are GPU-resident, so gate the GPU->CPU
    // readbacks on group_active("population") to avoid paying them when nothing
    // is observing. The pass runs in the post-process hook, after
    // rendezvous_step() and read_probes(), so the device is idle here. A failed
    // readback throws (and halts the run): these rows are recorded data, and a
    // row of zeros would be a fabricated observation (#234).
    //   population/out       — post-inhibition output (mirrors ngm2 / ngm3-CPU)
    //   population/in        — proximal feedforward = the presented sample
    //   population/in_apical — top-down/context input, only when configured
    if (acc.group_active("population")) {
        const auto out = readback_dev_f32(group_outp_id_, params.neuron_cnt, "group output");
        acc.observe(*this, "population", "out", std::span<const float>(out));
        const auto in_prox = readback_dev_f32(proximal_ids_.input_buf, proximal_input_size, "proximal input");
        acc.observe(*this, "population", "in", std::span<const float>(in_prox));
        if (apical_input_size > 0) {
            const auto in_apic = readback_dev_f32(apical_ids_.input_buf, apical_input_size, "apical input");
            acc.observe(*this, "population", "in_apical", std::span<const float>(in_apic));
        }
    }
}

void neuron_group_gpu_t::save_state(logging::state_writer &w) const {
    // Header (mirrors the CPU schema). Settings are not state (M.7.8): they are
    // in the configuration record written last.
    w.write("input_cnt", learning_step_count());
    w.write("step",      static_cast<std::uint64_t>(step_));
    w.write("effective_seed", std::to_underlying(poisson_seed_));

    write_checkpoint_shape(w, { params, apical_input_size, proximal_input_size,
                                apical_inputs, proximal_inputs });

    auto save_bank = [&](std::string_view bank_name, const bank_ids_t &ids,
                         std::size_t d_cnt, std::size_t input_size) {
        if (d_cnt == 0 || input_size == 0) return;   // absent bank -> no group
        const std::size_t total = params.neuron_cnt * d_cnt * rep_cnt * input_size;

        // Required state: a failed read throws, so the save fails before it is
        // published instead of recording zeros (#234).
        const std::string b(bank_name);
        const auto perm  = readback_dev_f32(ids.permanences,   total, b + " permanences");
        const auto seg32 = readback_dev_u32(ids.segment_ids,   total, b + " segment_ids");
        const auto adapt = readback_dev_f32(ids.adapt_history, total, b + " adapt_history");
        const auto mm    = readback_dev_f32(ids.mismatch,      total, b + " mismatch");

        std::vector<std::uint8_t> seg8(total);
        for (std::size_t i = 0; i < total; ++i)
            seg8[i] = static_cast<std::uint8_t>(seg32[i]);

        w.begin_group(bank_name);
        w.write("permanences"  , std::span<const float>       (perm));
        w.write("segment_ids"  , std::span<const std::uint8_t>(seg8));
        w.write("adapt_history", std::span<const float>       (adapt));
        w.write("mismatch"     , std::span<const float>       (mm));
        w.end_group();
    };
    save_bank("apical",   apical_ids_,   params.apical_cnt,   apical_input_size);
    save_bank("proximal", proximal_ids_, params.proximal_cnt, proximal_input_size);

    // Decaying per-neuron running max (device mirror of the CPU neuron_mx_act).
    const auto mx = readback_dev_f32(neuron_mx_act_id_, params.neuron_cnt, "neuron_mx_act");
    w.write("neuron_mx_act", std::span<const float>(mx));
    ngm::window_tracking::save(w, window_edges());

    logging::write_config_record(w, config_record(params, poisson_seed_, gpu_inhibitor_->params()));
}

void neuron_group_gpu_t::load_state(logging::state_reader &r) {
    // A checkpoint from a differently shaped group cannot be loaded into this one.
    const auto who = std::format("ngm3-gpu neuron group {}", id);
    check_checkpoint_shape(r, { params, apical_input_size, proximal_input_size,
                                apical_inputs, proximal_inputs }, who);

    // The configuration wins (M.7.8); only the differences are reported. This
    // runs before the payload checks below, so a corrupt checkpoint saved under
    // another configuration reports both.
    logging::check_config_record(r, config_record(params, poisson_seed_, gpu_inhibitor_->params()), who);

    // The payload is read and checked before anything is uploaded or assigned:
    // a corrupt checkpoint throws and leaves this group as it was. The split
    // kernel relies on the layout as much as the CPU split does (a misaligned
    // run is a cross-workgroup write), so the same check runs here.
    const auto apical   = stage_bank(r, "apical",   params, params.apical_cnt,   apical_input_size,   who);
    const auto proximal = stage_bank(r, "proximal", params, params.proximal_cnt, proximal_input_size, who);
    const auto mx       = read_neuron_mx_act(r, params.neuron_cnt, who);

    // Window edges, checked here and uploaded below with the rest (a format-1
    // checkpoint has none: the trackers start over, see read()).
    const auto edges = ngm::window_tracking::read(r, who);

    // The seed is random state: every per-step draw is keyed on (coordinates,
    // step, seed, salt), so the checkpoint's seed continues its streams (M.7.17)
    // whatever the current configuration says. A pre-field checkpoint keeps the
    // configured seed.
    poisson_seed_ = seed_t{ r.read_uint32("effective_seed", std::to_underlying(poisson_seed_)) };

    {   // Learning-step counter: device-resident; the branch phase follows from it.
        const std::uint64_t cnt = r.read_uint64("input_cnt", learning_step_count());
        const std::array<std::uint32_t, 4> c{
            static_cast<std::uint32_t>(cnt & 0xFFFFFFFFu),
            static_cast<std::uint32_t>(cnt >> 32),
            static_cast<std::uint32_t>(cnt % params.branch_interval),
            0u };
        dev_->upload(learn_cnt_id_, std::as_bytes(std::span<const std::uint32_t>(c)));
    }
    step_     = static_cast<std::uint32_t>(r.read_uint64("step", step_));

    auto upload_bank = [&](const staged_bank_t &b, const bank_ids_t &ids) {
        if (!b.present) return;
        std::vector<std::uint32_t> seg32(b.seg.begin(), b.seg.end());
        // Write into the live device buffers (non-ping_pong; side ignored).
        dev_->upload(ids.permanences,   std::as_bytes(std::span<const float>        (b.perm)));
        dev_->upload(ids.segment_ids,   std::as_bytes(std::span<const std::uint32_t>(seg32)));
        dev_->upload(ids.adapt_history, std::as_bytes(std::span<const float>        (b.adapt)));
        dev_->upload(ids.mismatch,      std::as_bytes(std::span<const float>        (b.mm)));
    };
    upload_bank(apical,   apical_ids_);
    upload_bank(proximal, proximal_ids_);

    if (mx)   // a pre-field checkpoint keeps the constructed running max
        dev_->upload(neuron_mx_act_id_, std::as_bytes(std::span<const float>(*mx)));

    window_edges_ = edges;
    if (dev_ != nullptr && window_track_id_ != coast::gpu::invalid_buffer) {
        const std::array<float, 4> t{ edges.winner_lower, edges.winner_upper, edges.other_lower, edges.other_upper };
        dev_->upload(window_track_id_, std::as_bytes(std::span<const float>(t)));
    }
}

ngm::window_tracking::edges_t neuron_group_gpu_t::window_edges() const
{
    // Device-resident once the plan is built (winner_main advances them); the host
    // mirror before that.
    if (dev_ == nullptr || window_track_id_ == coast::gpu::invalid_buffer) return window_edges_;
    // The edges go into checkpoints: a failed readback must not pass as edges
    // of 0, like learning_step_count().
    std::array<float, 4> t{};
    if (!dev_->readback(window_track_id_, std::as_writable_bytes(std::span<float>(t))))
        throw std::runtime_error("ngm3_gpu: activity window edges readback failed");
    return { t[0], t[1], t[2], t[3] };
}

// ---------------------------------------------------------------------------
// move ctor + dtor
// ---------------------------------------------------------------------------

neuron_group_gpu_t::neuron_group_gpu_t(neuron_group_gpu_t&& o) noexcept :
    sim::gpu_io_entity(std::move(o)),
    params(o.params),
    rep_cnt(o.rep_cnt), max_seg_id(o.max_seg_id),
    apical_input_size(o.apical_input_size), proximal_input_size(o.proximal_input_size),
    poisson_seed_(o.poisson_seed_), step_(o.step_),
    id(o.id), inp_ids(std::move(o.inp_ids)), output_buf(o.output_buf),
    apical_inputs(std::move(o.apical_inputs)), proximal_inputs(std::move(o.proximal_inputs)),
    dev_(o.dev_), apical_ids_(o.apical_ids_), proximal_ids_(o.proximal_ids_),
    neuron_apical_max_id_(o.neuron_apical_max_id_),
    neuron_proximal_max_id_(o.neuron_proximal_max_id_),
    neuron_act_id_(o.neuron_act_id_), neuron_mx_act_id_(o.neuron_mx_act_id_),
    group_outp_id_(o.group_outp_id_), adapt_scalars_id_(o.adapt_scalars_id_),
    learn_cnt_id_(o.learn_cnt_id_), window_track_id_(o.window_track_id_),
    window_edges_(o.window_edges_),
    params_uniform_id_(o.params_uniform_id_), adapt_params_id_(o.adapt_params_id_),
    k_stats_(o.k_stats_), k_respcmb_(o.k_respcmb_),
    k_winner_(o.k_winner_), k_adapt_(o.k_adapt_), k_branch_(o.k_branch_),
    plan_id_(o.plan_id_),
    gpu_inhibitor_(std::move(o.gpu_inhibitor_))
{
    // The device owns the GPU resources (referenced by the opaque ids copied
    // above); nothing to release on the source. The vis mirrors + atomic step are
    // left default-constructed (refreshed on demand by the consumer).
}

} // namespace ngm3

#endif // COAST_WITH_GPU
