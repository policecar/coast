//
// Created 2026-07-10 — ngm2.5 GPU port (GI1). CPU-arena-backed device entity:
// the host synapse_arena_t is the source of truth (reusing build_arena and, in
// GI4, repack_with_splits), mirrored into GPU storage buffers; per-step response
// (R2) + combine (R3) run as WGSL kernels via the gpu_device facade. Forward path
// only in GI1 (winner/adapt = GI3, growth = GI4). See the header for the v1 input
// limitation (host-visible upstream required).
//

#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#include "hd_ngm25/hd_ngm25_host.h"

#ifdef COAST_WITH_GPU

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <stdexcept>
#include <utility>

#include "core/gpu/shared_prng.h"
#include "core/require.h"
#include "diag/diag.h"
#include "hd_ngm2/hd_ngm2_tools.h"              // ngm2::compute_input_stats
#include "logging/config_record.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "observe/observe_accessor.h"
#include "sim_core/io_buffer.h"

namespace ngm25 {

// WGSL Params mirror (hd_ngm25_gpu.wgsl). std140: three 16-B scalar blocks then
// four vec2<f32> (8-B aligned). Total 80 B.
namespace {
struct gpu_params_t {
    std::uint32_t seed, step, neuron_cnt, dend_cnt;
    std::uint32_t apical_cnt, proximal_cnt, max_seg_id, frozen;
    float perm_thres, inv_perm_range, noise_lo, noise_range;
    float inp_sum[2];
    float inv_inp_sum[2];
    float atten[2];
    float pad[2];
};
static_assert(sizeof(gpu_params_t) == 80, "gpu_params_t must match the WGSL std140 Params");

// WGSL AdaptParams mirror. Two 16-B u32 blocks, two 16-B f32 blocks, six
// vec2<f32> pairs (8-B aligned): 32 + 32 + 48 = 112 B.
struct gpu_adapt_params_t {
    std::uint32_t seed, step, neuron_cnt, max_seg_id;
    std::uint32_t apical_cnt, proximal_cnt, dend_cnt, _pu;
    float primary_lr, secondary_lr, common_lr, perm_thres;
    float win_thres, mismatch_act_thres, mismatch_smoothing, _pf;
    float sig_default[2];
    float sig_wf[2];
    float wt_inv_step[2];   // tracked window: inv_step of the lower / upper edge's steepness
    float _pad_w[2];
    float inv_last_max_inp[2];
    float _pv[2];
};
static_assert(sizeof(gpu_adapt_params_t) == 112, "gpu_adapt_params_t must match WGSL AdaptParams");

// The kernel source, #embed'd (C23) relative to this TU — same mechanism as ngm3.
constexpr unsigned char kNgm25GpuWgsl[] = {
#embed "hd_ngm25_gpu.wgsl"
};

template <class T>
std::span<const std::byte> bytes_of(const std::vector<T>& v) {
    return std::as_bytes(std::span<const T>(v));
}
} // namespace

neuron_group_gpu_t::neuron_group_gpu_t(const cfg_store   &cs,
                                       const std::string &type,
                                       sim::output_id_t   output_id,
                                       seed_offset_t      seed_offset)
  : params(cs.group_or_default(type)),
    id_(output_id)
{
    effective_seed_ = params.random_seed + seed_offset;

    validate_params(params, k_gpu_caps);

    shape_         = base_shape(params);           // input ranges: init_io
    adapt_params_  = make_adapt_params(params);    // map R5/R6/R7
    branch_params_ = make_branch_params(params);   // map R8 (GI4)

    neuron_mx_act_.assign(params.neuron_cnt, 0.0f);
    input_count_.assign(params.neuron_cnt, 0);
    recompute_sync_due();

    gpu_inhibitor_ = inhibition::make_gpu_inhibitor(cs.inhibitions, params.inhibition_cfg,
                                                    params.local_inhibition_strength);
}

void neuron_group_gpu_t::init_io(
    const std::unordered_map<std::string, sim::output_channel_t> &input_map)
{
    auto routing = route_inputs(params, input_map, shape_, k_gpu_caps, id_);
    routes_  = std::move(routing.routes);
    inp_ids_ = std::move(routing.inp_ids);
    input_concat_.assign(routing.width, 0.0f);
    arena_ = build_arena(shape_, params.permanence_threshold, std::to_underlying(effective_seed_));
    init_structure();
}

void neuron_group_gpu_t::set_output_buffer(sim::io_buffer& buf) { output_buf_ = &buf; }

void neuron_group_gpu_t::set_input_buffer(sim::output_id_t id, sim::io_buffer& buf)
{
    for (auto &r : routes_)
        if (r.id == id) r.buf = &buf;
}

// -------------------------- build-time plan assembly -----------------------

void neuron_group_gpu_t::build_step_plan(coast::gpu::gpu_device& dev)
{
    namespace g = coast::gpu;
    dev_ = &dev;

    // Capacity-sized arena buffers (design §5): the dispatch grid is fixed, so a
    // branch that stays within capacity is a pure data re-upload; only a capacity
    // crossing (GI4 grow_if_branched) rebuilds the plan. capacity_syn_ may already
    // be doubled by a prior grow before a rebuild — never shrink it.
    capacity_syn_ = std::max(capacity_syn_, arena_.syn_count());

    // A rebuild (growth capacity crossing, or a load carrying a post-branch arena)
    // re-enters here with plan_id_ already valid. Only the five synapse buffers are
    // capacity-sized, so ONLY they change size across a branch: repack copies
    // dend_type/dend_neuron verbatim (dend_count invariant) and max_seg_id is a
    // fixed config bound (seg_cnt invariant); neuron_cnt and input_size are fixed.
    // So re-register just those five on a rebuild and reuse every other handle.
    // The orphaned old syn buffers leak (the backend exposes no release_buffer) —
    // bounded to ~log2(final/initial) capacity crossings, unavoidable here.
    const bool first_build = (plan_id_ == g::invalid_plan);
    register_device_buffers(dev, first_build);

    // Upload the (re-packed) host arena to device (widening u16/u8 -> u32). Every
    // build: the fixed-size buffers keep their handles but their CONTENTS change
    // on a branch.
    upload_arena_to_device();

    if (first_build) compile_kernels(dev);

    g::step_plan sp;
    append_forward_ops(dev, sp, first_build);
    // Learning passes (R5/R6/R7) — omitted entirely for a frozen (inference-only)
    // group, exactly as the CPU process() stops after inhibition when frozen.
    if (!params.frozen) append_learning_ops(sp);

    // Reuse the plan handle across rebuilds: register_plan() once, then build_plan()
    // on the SAME id every time. build_plan() drains in-flight work and releases the
    // prior build's bind groups (backend release_plan) — minting a fresh plan_id_ on
    // each rebuild would orphan that plan_rec, leaking its bind groups.
    if (first_build) plan_id_ = dev.register_plan();
    dev.build_plan(plan_id_, sp);
}

// The five capacity-sized synapse buffers on every build; the fixed-size ones
// on the first only.
void neuron_group_gpu_t::register_device_buffers(coast::gpu::gpu_device& dev, bool first_build)
{
    namespace g = coast::gpu;
    const std::size_t dend_cnt   = arena_.dend_count();
    const std::size_t neuron_cnt = params.neuron_cnt;
    const std::size_t input_size = input_concat_.size();
    auto reg = [&](std::size_t bytes) {
        return dev.register_buffer(g::topology::gpu_resident, bytes, /*ping_pong=*/false);
    };

    b_syn_perm_       = reg(capacity_syn_ * sizeof(float));
    b_syn_input_idx_  = reg(capacity_syn_ * sizeof(std::uint32_t));
    b_syn_seg_        = reg(capacity_syn_ * sizeof(std::uint32_t));
    b_syn_adapt_hist_ = reg(capacity_syn_ * sizeof(float));
    b_syn_mismatch_   = reg(capacity_syn_ * sizeof(float));

    if (first_build) {
        // Fixed-size buffers (invariant across growth): register once. Re-
        // registering them on every rebuild would leak ~14 buffers per crossing.
        b_dend_offset_    = reg((dend_cnt + 1) * sizeof(std::uint32_t));
        b_dend_type_      = reg(dend_cnt * sizeof(std::uint32_t));
        b_input_concat_   = reg(input_size * sizeof(float));
        b_half_max_       = reg(input_size * sizeof(float));
        b_inv_sum_        = reg(input_size * sizeof(float));
        b_dend_act_       = reg(dend_cnt * sizeof(float));
        const std::size_t seg_cnt = static_cast<std::size_t>(shape_.max_seg_id) + 1;
        b_seg_activity_   = reg(dend_cnt * seg_cnt * sizeof(float));
        b_neuron_act_     = reg(neuron_cnt * sizeof(float));
        b_neuron_mx_act_  = reg(neuron_cnt * sizeof(float));
        b_group_outp_     = reg(neuron_cnt * sizeof(float));
        b_ctx4_           = reg(input_size * 4 * sizeof(float));   // vec4 per position
        b_wscal_          = reg(8 * sizeof(std::uint32_t));   // [idx, act_sum, win_act, mx_act, edges x4]
        b_learn_cnt_      = reg(sizeof(std::uint32_t));
        // Activity window edges (learning state), device-resident from here on:
        // seeded once from the host copy, then advanced by winner_main. A growth
        // rebuild keeps this buffer and its contents.
        b_window_track_   = reg(4 * sizeof(float));
        upload_window_edges();
        b_params_         = dev.register_buffer(g::topology::uniform, sizeof(gpu_params_t), false);
        b_adapt_params_   = dev.register_buffer(g::topology::uniform, sizeof(gpu_adapt_params_t), false);
    }
}

void neuron_group_gpu_t::compile_kernels(coast::gpu::gpu_device& dev)
{
    namespace g = coast::gpu;
    // Kernels: shared_prng prepended, one pipeline per entry point. The WGSL is
    // static (no dependence on counts), and register_kernel mints a fresh
    // WGPUComputePipeline per call, so register once — re-registering on a
    // rebuild would leak one pipeline per kernel per capacity crossing.
    const std::string src = std::string(g::shared_prng_wgsl()) +
        ngm::window_tracking::window_tracking_wgsl() +
        std::string(reinterpret_cast<const char*>(kNgm25GpuWgsl), sizeof(kNgm25GpuWgsl));
    k_response_        = dev.register_kernel(src, "response_main");
    k_combine_         = dev.register_kernel(src, "combine_main");
    k_winner_          = dev.register_kernel(src, "winner_main");
    k_adapt_primary_   = dev.register_kernel(src, "adapt_primary");
    k_adapt_secondary_ = dev.register_kernel(src, "adapt_secondary");
}

// op1-op4: response, combine, inhibition, publish.
void neuron_group_gpu_t::append_forward_ops(coast::gpu::gpu_device& dev, coast::gpu::step_plan& sp,
                                            bool first_build)
{
    namespace g = coast::gpu;
    const std::size_t  dend_cnt   = arena_.dend_count();
    const std::size_t  neuron_cnt = params.neuron_cnt;
    const g::buffer_id out_page   = output_buf_->gpu_page_id();

    // op1 — response (R2), one workgroup per dendrite.
    {
        g::plan_op op{};
        op.kind = g::op_kind::compute;
        op.kernel = k_response_;
        g::plan_dispatch d{};
        d.bindings = {
            {b_syn_perm_,      g::access::read,  0},
            {b_syn_input_idx_, g::access::read,  1},
            {b_syn_seg_,       g::access::read,  2},
            {b_dend_offset_,   g::access::read,  3},
            {b_dend_type_,     g::access::read,  4},
            {b_input_concat_,  g::access::read,  5},
            {b_half_max_,      g::access::read,  6},
            {b_inv_sum_,       g::access::read,  7},
            {b_dend_act_,      g::access::write, 8},
            {b_params_,        g::access::read,  9},
            {b_seg_activity_,  g::access::write, 10},
        };
        d.dispatch = { static_cast<std::uint32_t>(dend_cnt), 1, 1 };
        op.dispatches = { std::move(d) };
        sp.ops.push_back(std::move(op));
    }
    // op2 — combine (R3), one thread per neuron.
    {
        g::plan_op op{};
        op.kind = g::op_kind::compute;
        op.kernel = k_combine_;
        g::plan_dispatch d{};
        d.bindings = {
            {b_dend_act_,      g::access::read,      0},
            {b_dend_type_,     g::access::read,      1},
            {b_neuron_act_,    g::access::write,     2},
            {b_params_,        g::access::read,      4},
        };
        d.dispatch = { static_cast<std::uint32_t>((neuron_cnt + 63) / 64), 1, 1 };
        op.dispatches = { std::move(d) };
        sp.ops.push_back(std::move(op));
    }
    // op3 — inhibition (R4): neuron_act -> out_page (injected emitter). emit()
    // registers the emitter's own kernel + uniform as a side effect, so run it
    // exactly once (first build) and cache the ops it appends; on a rebuild replay
    // the cached ops. Their bindings reference only stable handles (b_neuron_act_
    // and out_page are fixed-size, registered once; the emitter's uniform likewise),
    // so replaying is correct and avoids leaking a pipeline + uniform per crossing.
    if (first_build) {
        g::step_plan inh;
        gpu_inhibitor_->emit(dev,
            inhibition::gpu_host_handles{ b_neuron_act_, out_page,
                                          static_cast<std::uint32_t>(neuron_cnt), {1, 1, 1} },
            inh);
        inhibition_ops_ = std::move(inh.ops);
    }
    for (const g::plan_op& op : inhibition_ops_) sp.ops.push_back(op);
    // op4 — publish the post-inhibition output page into the stable group_outp
    // buffer (winner/adapt read this non-ping-pong source in GI3).
    {
        g::plan_op op{};
        op.kind = g::op_kind::copy;
        op.copy = g::plan_copy{ out_page, b_group_outp_, 0, 0,
                                neuron_cnt * sizeof(float),
                                g::access::write, g::access::write };
        sp.ops.push_back(std::move(op));
    }
}

// op5-op7: winner, adapt primary, adapt secondary (R5/R6/R7).
void neuron_group_gpu_t::append_learning_ops(coast::gpu::step_plan& sp)
{
    namespace g = coast::gpu;
    const std::size_t neuron_cnt = params.neuron_cnt;

    // op5 — winner (R5), single workgroup -> w_scal {idx, act_sum, win_act, mx_act,
    //       window edges x4}; advances the activity window edges (M.4.2a).
    {
        g::plan_op op{}; op.kind = g::op_kind::compute; op.kernel = k_winner_;
        g::plan_dispatch d{};
        d.bindings = {
            {b_group_outp_,    g::access::read,  0},
            {b_wscal_,         g::access::write, 1},
            {b_adapt_params_,  g::access::read,  2},
            {b_neuron_act_,    g::access::read,  3},
            {b_neuron_mx_act_, g::access::write, 4},
            {b_learn_cnt_,     g::access::write, 5},
            {b_window_track_,  g::access::write, 6},
        };
        d.dispatch = { 1, 1, 1 };
        op.dispatches = { std::move(d) };
        sp.ops.push_back(std::move(op));
    }
    // The adapt bind set is shared by primary + secondary (same globals).
    auto adapt_bindings = [&]() {
        return std::vector<g::plan_binding>{
            {b_syn_perm_,       g::access::write, 0},
            {b_syn_input_idx_,  g::access::read,  1},
            {b_syn_seg_,        g::access::read,  2},
            {b_syn_adapt_hist_, g::access::write, 3},
            {b_syn_mismatch_,   g::access::write, 4},
            {b_dend_offset_,    g::access::read,  5},
            {b_dend_type_,      g::access::read,  6},
            {b_dend_act_,       g::access::read,  7},
            {b_seg_activity_,   g::access::read,  8},
            {b_neuron_act_,     g::access::read,  9},
            {b_neuron_mx_act_,  g::access::read,  10},
            {b_group_outp_,     g::access::read,  11},
            {b_input_concat_,   g::access::read,  12},
            {b_ctx4_,           g::access::read,  13},
            {b_wscal_,          g::access::read,  14},
            {b_adapt_params_,   g::access::read,  15},
        };
    };
    // op6 — adapt primary (winner only), its own dispatch.
    {
        g::plan_op op{}; op.kind = g::op_kind::compute; op.kernel = k_adapt_primary_;
        g::plan_dispatch d{}; d.bindings = adapt_bindings(); d.dispatch = { 1, 1, 1 };
        op.dispatches = { std::move(d) };
        sp.ops.push_back(std::move(op));
    }
    // op7 — adapt secondary (every neuron except the winner), one wg/neuron.
    {
        g::plan_op op{}; op.kind = g::op_kind::compute; op.kernel = k_adapt_secondary_;
        g::plan_dispatch d{}; d.bindings = adapt_bindings();
        d.dispatch = { static_cast<std::uint32_t>(neuron_cnt), 1, 1 };
        op.dispatches = { std::move(d) };
        sp.ops.push_back(std::move(op));
    }
}

void neuron_group_gpu_t::upload_arena_to_device()
{
    if (dev_ == nullptr) return;
    // GI4 grows capacity_syn_ + rebuilds when this would be exceeded; until then a
    // breach is a bug, not a runtime step fault (this is not a step_result path).
    if (arena_.syn_count() > capacity_syn_)
        throw std::runtime_error("ngm25-gpu: arena exceeds device capacity (growth is GI4)");

    std::vector<std::uint32_t> iidx(arena_.input_idx.begin(), arena_.input_idx.end());
    std::vector<std::uint32_t> seg (arena_.seg.begin(),        arena_.seg.end());
    std::vector<std::uint32_t> dty (arena_.dend_type.size());
    for (std::size_t i = 0; i < dty.size(); ++i)
        dty[i] = static_cast<std::uint32_t>(arena_.dend_type[i]);

    dev_->upload(b_syn_perm_,       bytes_of(arena_.perm));
    dev_->upload(b_syn_input_idx_,  bytes_of(iidx));
    dev_->upload(b_syn_seg_,        bytes_of(seg));
    dev_->upload(b_syn_adapt_hist_, bytes_of(arena_.adapt_hist));
    dev_->upload(b_syn_mismatch_,   bytes_of(arena_.mismatch));
    dev_->upload(b_dend_offset_,    bytes_of(arena_.dend_offset));
    dev_->upload(b_dend_type_,      bytes_of(dty));
    dev_->upload(b_neuron_mx_act_,  bytes_of(neuron_mx_act_));
}

// ------------------------------- per-step ----------------------------------

void neuron_group_gpu_t::upload_step_uniforms()
{
    // R1 stats already in ctx_. Build the Params uniform + per-position arrays.
    gpu_params_t p{};
    p.seed = std::to_underlying(effective_seed_); p.step = step_;
    p.neuron_cnt   = static_cast<std::uint32_t>(params.neuron_cnt);
    p.dend_cnt     = static_cast<std::uint32_t>(arena_.dend_count());
    p.apical_cnt   = static_cast<std::uint32_t>(params.apical_cnt);
    p.proximal_cnt = static_cast<std::uint32_t>(params.proximal_cnt);
    p.max_seg_id   = shape_.max_seg_id;
    p.frozen       = params.frozen ? 1u : 0u;
    p.perm_thres     = params.permanence_threshold;
    p.inv_perm_range = 1.0f / (1.0f - params.permanence_threshold);
    p.noise_lo       = params.neuron_noise_range.first;
    p.noise_range    = params.neuron_noise_range.second - params.neuron_noise_range.first;
    for (int t = 0; t < 2; ++t) {
        p.inp_sum[t]     = ctx_.inp_sum[t];
        p.inv_inp_sum[t] = ctx_.inv_inp_sum[t];
        p.atten[t]       = ctx_.response_atten[t];
    }
    dev_->upload(b_params_, std::as_bytes(std::span<const gpu_params_t>(&p, 1)));
    dev_->upload(b_input_concat_, bytes_of(input_concat_));
    dev_->upload(b_half_max_,     bytes_of(ctx_.half_max_pos));
    dev_->upload(b_inv_sum_,      bytes_of(ctx_.inv_sum_pos));

    if (params.frozen) return;   // adapt buffers unused (not in the plan) when frozen

    // Packed adapt per-position ctx: vec4(high_thres, inv_high_range, inv_high_thres,
    // adapt_atten). These come from the same CPU build_step_ctx -> CPU/GPU parity.
    const std::size_t w = input_concat_.size();
    std::vector<std::array<float,4>> ctx4(w);
    for (std::size_t i = 0; i < w; ++i)
        ctx4[i] = { ctx_.high_thres_pos[i], ctx_.inv_high_range_pos[i],
                    ctx_.inv_high_thres_pos[i], ctx_.adapt_atten_pos[i] };
    dev_->upload(b_ctx4_, std::as_bytes(std::span<const std::array<float,4>>(ctx4)));

    // AdaptParams uniform. Sigmoid params precomputed from the shapes (ngm2 form).
    auto sp = [](const ngm2::sigmoid_shape_t& sh) {
        const ngm2::sigmoid_params_t q(sh);
        return std::array<float,2>{ q.inv_step_size, q.starting_point };
    };
    const auto sig_def  = sp(ngm2::sigmoid_shape_t{});                       // {0.5,0.5}
    const auto sig_wf   = sp(adapt_params_.weight_filter);
    // Tracked window: only the steepness is fixed; the edges come from the
    // device (winner_main), so the shader forms start = -edge * inv_step.
    const auto sig_wt   = sp(ngm::window_tracking::window(0.0f, 0.0f).first);
    const auto sig_wt2  = sp(ngm::window_tracking::window(0.0f, 0.0f).second);

    gpu_adapt_params_t a{};
    a.seed = std::to_underlying(effective_seed_); a.step = step_;
    a.neuron_cnt = static_cast<std::uint32_t>(params.neuron_cnt);
    a.max_seg_id = shape_.max_seg_id;
    a.apical_cnt = static_cast<std::uint32_t>(params.apical_cnt);
    a.proximal_cnt = static_cast<std::uint32_t>(params.proximal_cnt);
    a.dend_cnt = static_cast<std::uint32_t>(arena_.dend_count());
    a.primary_lr = adapt_params_.primary_learning_rate;
    a.secondary_lr = adapt_params_.secondary_learning_rate;
    a.common_lr = adapt_params_.common_learning_rate;
    a.perm_thres = adapt_params_.perm_threshold;
    a.win_thres = adapt_params_.stochastic_win_thres;
    a.mismatch_act_thres = adapt_params_.mismatch_act_thres;
    a.mismatch_smoothing = adapt_params_.mismatch_smoothing;
    a.sig_default[0] = sig_def[0];  a.sig_default[1] = sig_def[1];
    a.sig_wf[0]      = sig_wf[0];    a.sig_wf[1]      = sig_wf[1];
    a.wt_inv_step[0] = sig_wt[0];    a.wt_inv_step[1] = sig_wt2[0];
    a.inv_last_max_inp[0] = ctx_.inv_last_max_inp[0];
    a.inv_last_max_inp[1] = ctx_.inv_last_max_inp[1];
    dev_->upload(b_adapt_params_, std::as_bytes(std::span<const gpu_adapt_params_t>(&a, 1)));
}

sim::step_result neuron_group_gpu_t::process()
{
    COAST_REQUIRE(output_buf_ != nullptr, core::error_kind::contract_violation);
    COAST_REQUIRE(dev_ != nullptr, core::error_kind::contract_violation);

    // --- host gather (v1: upstream must be host-visible) ---
    for (const auto &r : routes_) {
        if (r.buf == nullptr) continue;
        const auto src = r.buf->cur_read_buffer();
        COAST_REQUIRE(src.size() == r.pos_hi - r.pos_lo, core::error_kind::contract_violation);
        std::copy(src.begin(), src.end(), input_concat_.begin() + r.pos_lo);
    }

    // --- R1 stats on the host (reuse CPU build_step_ctx -> CPU/GPU parity) ---
    std::vector<channel_stats_t> channels;
    channels.reserve(routes_.size());
    for (const auto &r : routes_) {
        const std::span<const float> slice(input_concat_.data() + r.pos_lo, r.pos_hi - r.pos_lo);
        channels.push_back({r.bank, r.pos_lo, r.pos_hi, ngm2::compute_input_stats(slice)});
    }
    detail::build_step_ctx(shape_, channels, ctx_);

    // --- upload uniforms + input for this step; the plan (enqueued by env) runs
    //     response -> combine -> inhibition -> publish on device ---
    upload_step_uniforms();
    ++step_;
    return {};
}

// ------------------------------- readback ----------------------------------

ngm::window_tracking::edges_t neuron_group_gpu_t::window_edges() const
{
    // Device-resident once the plan is built (winner_main advances them); the host
    // copy before that.
    if (dev_ == nullptr || b_window_track_ == coast::gpu::invalid_buffer) return window_edges_;
    // The edges go into checkpoints: a failed readback must not pass as edges of
    // 0, like read_learn_cnt().
    std::array<float, 4> t{};
    if (!dev_->readback(b_window_track_, std::as_writable_bytes(std::span<float>(t))))
        throw std::runtime_error("ngm25-gpu: activity window edges readback failed");
    return { t[0], t[1], t[2], t[3] };
}

void neuron_group_gpu_t::upload_window_edges()
{
    if (dev_ == nullptr || b_window_track_ == coast::gpu::invalid_buffer) return;
    const auto& e = window_edges_;
    const std::array<float, 4> t{ e.winner_lower, e.winner_upper, e.other_lower, e.other_upper };
    dev_->upload(b_window_track_, std::as_bytes(std::span<const float>(t)));
}

std::vector<float> neuron_group_gpu_t::readback_f32(coast::gpu::buffer_id id,
                                                    std::size_t count,
                                                    std::string_view what) const
{
    std::vector<float> out(count, 0.0f);
    if (dev_ == nullptr || count == 0) return out;
    // A failed readback leaves `out` zero-initialized; returning it would hand the
    // host arena (and from there a checkpoint or a growth re-upload) fabricated
    // zeros as learned state (#234). Throw instead.
    if (!dev_->readback(id, std::as_writable_bytes(std::span<float>(out))))
        throw std::runtime_error(std::format("ngm25-gpu: {} readback failed", what));
    return out;
}

std::uint32_t neuron_group_gpu_t::read_learn_cnt() const
{
    std::uint32_t c = 0;
    // A failure must not pass as a count: it keys the branch cadence and checkpoints.
    if (!dev_->readback(b_learn_cnt_, std::as_writable_bytes(std::span<std::uint32_t>(&c, 1))))
        throw std::runtime_error("ngm25 GPU: learning-step counter readback failed");
    return c;
}

void neuron_group_gpu_t::recompute_sync_due()
{
    const std::uint64_t interval = branch_params_.branch_interval;
    std::uint64_t due = interval;
    for (const std::uint64_t c : input_count_)
        due = std::min(due, interval - c % interval);
    sync_due_in_ = due;   // >= 1
}

// GI4 — host-side branch gate + growth. Runs post-rendezvous (device idle). The
// per-step adapt already ran on device; input_count is tracked here (per adapt
// call, matching CPU: every neuron is adapted exactly once per learning step, so
// +1 each — SPEC.md M.4.14 — and not at all on a zero-output step, M.4.15).
// When a neuron crosses
// a branch_interval boundary, pull the device-updated arena home, run the SAME CPU
// branch code (single source of truth for split semantics, §5), and re-mirror.
void neuron_group_gpu_t::grow_if_branched()
{
    if (dev_ == nullptr || params.frozen) return;

    // No counter can reach a boundary before sync_due_in_ more steps (see the
    // header); until then there is nothing to decide and no readback.
    if (++steps_unsynced_ < sync_due_in_) return;

    // Sync: every neuron adapted once per learning step since the last sync (loop 3
    // the winner, loop 4 every other neuron); a zero-output step (M.4.15) was not
    // counted by the winner kernel.
    //
    // Every device read of the sync comes before any state is committed: a failed
    // read throws with the counters, the host arena and the device as they were,
    // so nothing is branched on fabricated zeros and a later sync retries this one
    // (#234). (In a run the throw halts the sim; the ordering keeps the halted
    // state coherent rather than half-synced.)
    const std::uint32_t now     = read_learn_cnt();
    const std::uint64_t learned = static_cast<std::uint32_t>(now - learn_synced_);   // u32 wrap-safe
    std::vector<std::uint64_t> after = input_count_;
    for (auto& c : after) c += learned;

    // Cheap gate: skip the (costly) device readback + branch phase unless some
    // neuron crossed a branch_interval boundary. Same shared crossing predicate as
    // the CPU maybe_branch_group, so the two paths cannot disagree.
    const std::size_t interval = branch_params_.branch_interval;
    bool any_cross = false;
    for (std::size_t n = 0; n < after.size(); ++n)
        if (detail::crossed_branch_interval(input_count_[n], after[n], interval)) {
            any_cross = true;
            break;
        }
    if (any_cross)
        readback_arena_into_host();   // perm/adapt_hist/mismatch/neuron_mx_act -> host; throws

    // Commit the sync.
    const std::vector<std::uint64_t> before = std::exchange(input_count_, std::move(after));
    learn_synced_   = now;
    steps_unsynced_ = 0;
    recompute_sync_due();
    if (!any_cross) return;

    const bool grew = detail::maybe_branch_group(arena_, shape_, branch_params_,
                                                 before, input_count_,
                                                 std::to_underlying(effective_seed_), branch_epoch_);
    if (!grew) return;

    // Post-repack safety net (design §9), mirroring CPU process(). grow_if_branched
    // is a void, non-value-channel path (like upload_arena_to_device below), so a
    // broken invariant surfaces as a throw carrying the offending invariant rather
    // than through step_result. The host arena is complete here (device readback +
    // host repack done); the subsequent device upload does not change it.
    std::string why;
    if (!check_arena_consistency(arena_, shape_, &why))
        throw std::runtime_error("ngm25-gpu: arena inconsistent after repack: " + why);

    sync_arena_to_device();
}

void neuron_group_gpu_t::sync_arena_to_device()
{
    if (dev_ != nullptr && arena_.syn_count() > capacity_syn_) {
        // Capacity crossing (rare, ~log2 over a run): double the capacity and
        // rebuild the plan (build_step_plan says what it re-registers and leaks).
        while (capacity_syn_ < arena_.syn_count()) capacity_syn_ *= 2;
        build_step_plan(*dev_);
    } else {
        // Within capacity the dispatch grid is unchanged (§5): a data upload
        // (a no-op before build_step_plan has attached the device).
        upload_arena_to_device();
    }
}

void neuron_group_gpu_t::read_probes()
{
    grow_if_branched();

    // Activity window edges as of the end of this step (device-resident). The
    // PROBE statements run every step (their registration statics must fire);
    // only the readback is gated. A frozen group reads them too: its edges stay
    // valid, they just stop moving. A failed readback publishes nothing this
    // step rather than edges of 0.
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

std::vector<float> neuron_group_gpu_t::readback_neuron_act() const
{ return readback_f32(b_neuron_act_, params.neuron_cnt, "neuron activity"); }

std::vector<float> neuron_group_gpu_t::readback_group_outp() const
{ return readback_f32(b_group_outp_, params.neuron_cnt, "group output"); }

void neuron_group_gpu_t::readback_arena_into_host() const
{
    if (dev_ == nullptr) return;
    const std::size_t n = arena_.syn_count();
    if (n == 0) return;
    // The syn_* buffers are CAPACITY-sized (>= n after growth); the backend readback
    // requires dst.size() == buffer bytes, so read the full capacity and keep the
    // first n (the live synapses; dend_offset bounds them). perm / adapt_hist /
    // mismatch are device-updated (adapt, GI3); input_idx / seg / dend_* are
    // host-authoritative (host repack, §5) and not read back.
    //
    // Staged: all four arrays are read before any is committed, so a failed read
    // (which throws) leaves the host arena and neuron_mx_act_ exactly as they were
    // — never zeros, never a mix of two device states (#234).
    const std::size_t cap = std::max(capacity_syn_, n);
    auto perm = readback_f32(b_syn_perm_,       cap, "synapse perm");
    auto ah   = readback_f32(b_syn_adapt_hist_, cap, "synapse adapt_hist");
    auto mm   = readback_f32(b_syn_mismatch_,   cap, "synapse mismatch");
    auto mx   = readback_f32(b_neuron_mx_act_,  params.neuron_cnt, "neuron_mx_act");
    std::copy_n(perm.begin(), n, arena_.perm.begin());
    std::copy_n(ah.begin(),   n, arena_.adapt_hist.begin());
    std::copy_n(mm.begin(),   n, arena_.mismatch.begin());
    neuron_mx_act_ = std::move(mx);
}

// ------------------------------- vis mirror --------------------------------
// The shared dense mirror (hd_ngm25_host.h). The perm values are read back from
// the device first; seg/structure are host-authoritative.

void neuron_group_gpu_t::refresh_vis_mirror() const
{
    // Pull device perm (adapt-updated) into arena_. A failed readback keeps the
    // last good mirror rather than showing zeros, and does not throw: this runs on
    // the GUI thread inside the post-swap barrier, with nothing above to catch it
    // (#234). The staged readback left the arena untouched.
    try {
        readback_arena_into_host();
    } catch (const std::runtime_error& e) {
        COAST_WARN("ngm25_gpu", "group {}: vis mirror not refreshed, keeping the last one: {}",
                   id_, e.what());
        return;
    }
    mirror_.refresh(arena_, shape_);
}

std::span<const float>
neuron_group_gpu_t::get_apical_representation(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.perm(bank_t::apical, n, d, r); }

std::span<const float>
neuron_group_gpu_t::get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.perm(bank_t::proximal, n, d, r); }

std::span<const std::uint8_t>
neuron_group_gpu_t::get_apical_segment_ids(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.seg(bank_t::apical, n, d, r); }

std::span<const std::uint8_t>
neuron_group_gpu_t::get_proximal_segment_ids(std::size_t n, std::size_t d, std::size_t r) const
{ return mirror_.seg(bank_t::proximal, n, d, r); }

// ------------------------------- misc iface --------------------------------

sim::output_id_t neuron_group_gpu_t::get_outp_id()  const { return id_; }
std::size_t      neuron_group_gpu_t::get_outp_size() const { return params.neuron_cnt; }

std::span<const std::size_t> neuron_group_gpu_t::get_inp_ids() const
{ return { inp_ids_.data(), inp_ids_.size() }; }

std::string neuron_group_gpu_t::status_str() const
{
    return "NGM2.5-GPU Group " + std::to_string(id_)
         + " | neurons "  + std::to_string(params.neuron_cnt)
         + " | apical "   + std::to_string(params.apical_cnt)
         + " | proximal " + std::to_string(params.proximal_cnt)
         + " | depth "    + std::to_string(static_cast<int>(params.max_branch_level))
         + " | synapses " + std::to_string(arena_.syn_count())
         + " | step "     + std::to_string(step_);
}

void neuron_group_gpu_t::perform_observations(observe::observe_accessor& acc)
{
    // Population in/out channels, mirroring the CPU ngm25 entity. `in` is the
    // host-gathered input; `out` is the post-inhibition group output, read back
    // from the device (device idle: observe runs on the sim thread at a step
    // boundary). Gated so the readback only fires when the channel is active. A
    // failed readback throws (and halts the run): the row is recorded data, and a
    // row of zeros would be a fabricated observation (#234).
    if (acc.group_active("population")) {
        acc.observe(*this, "population", "in", std::span<const float>(input_concat_));
        const std::vector<float> outp = readback_group_outp();
        acc.observe(*this, "population", "out", std::span<const float>(outp));
    }
    entity_element::perform_observations(acc);
}

// ------------------------------- persistence -------------------------------
// Same schema as the CPU neuron_group_t (so a CPU checkpoint loads on GPU and
// vice versa — GI5 gate). Device-updated fields are refreshed from the device
// first; the host arena is otherwise authoritative.

void neuron_group_gpu_t::save_state(logging::state_writer &w) const
{
    // Device idle at a step boundary. Every device read below (the arena, the
    // counter, the window edges) throws on failure, so a failed read fails the save
    // before anything is published (#234).
    readback_arena_into_host();

    // The counters as of now: add the learning steps not yet synced.
    const std::uint64_t pending =
        (dev_ != nullptr) ? static_cast<std::uint32_t>(read_learn_cnt() - learn_synced_) : 0;
    std::vector<std::uint64_t> counts = input_count_;
    for (auto& c : counts) c += pending;

    const auto who = std::format("ngm25-gpu neuron group {}", id_);
    const auto cfg = config_record(params, effective_seed_, gpu_inhibitor_->params());
    write_checkpoint(w, { who, params, shape_, routes_, cfg },
                     step_, branch_epoch_, effective_seed_, arena_,
                     counts, neuron_mx_act_, window_edges());

    // Inhibition (delegated) — the same schema as the CPU group, so a CPU
    // checkpoint loads on GPU and vice versa. The current GPU inhibitor
    // (max_softmax) is stateless, so this writes an empty subgroup today.
    w.begin_group("inhibition");
    gpu_inhibitor_->save_state(w);
    w.end_group();
}

void neuron_group_gpu_t::load_state(logging::state_reader &r)
{
    // Read and check the whole checkpoint (the CPU group's schema), then the
    // inhibitor's part, before anything of this group is assigned: a misfit or
    // corrupt checkpoint throws and leaves the group as it was (M.7.18).
    const auto who = std::format("ngm25-gpu neuron group {}", id_);
    const auto cfg = config_record(params, effective_seed_, gpu_inhibitor_->params());
    checkpoint_state_t s = stage_checkpoint(r, { who, params, shape_, routes_, cfg },
                                            step_, branch_epoch_, effective_seed_);

    if (r.enter_group("inhibition")) { gpu_inhibitor_->load_state(r, params.neuron_cnt); r.leave_group(); }

    // Commit the host state (moves only).
    step_           = s.step;
    branch_epoch_   = s.branch_epoch;
    effective_seed_ = s.effective_seed;
    arena_          = std::move(s.arena);
    input_count_    = std::move(s.input_count);
    neuron_mx_act_  = std::move(s.neuron_mx_act);
    window_edges_   = s.window_edges;

    // Then mirror it to the device. The loaded counters are current: resync with
    // the device count from here.
    learn_synced_   = (dev_ != nullptr) ? read_learn_cnt() : 0;
    steps_unsynced_ = 0;
    recompute_sync_due();
    upload_window_edges();

    // A checkpoint taken after a branch can carry an arena larger than a fresh
    // group's capacity: grow and rebuild as on a live crossing. Before the device
    // is attached, build_step_plan later sizes capacity from the loaded arena.
    sync_arena_to_device();
}

} // namespace ngm25

#endif // COAST_WITH_GPU
