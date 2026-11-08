//
// Created by jk on the ngm3 GPU port (S6).
//

#ifndef HD_NGM3_GPU_NEURON_GROUP_H
#define HD_NGM3_GPU_NEURON_GROUP_H

#ifdef COAST_WITH_GPU

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/entity_element_macros.h"
#include "core/gpu/gpu_device.h"  // agnostic seam: opaque buffer/kernel/plan ids
#include "ngm/window_tracking.h"
#include "hd_ngm3/hd_ngm3_cfg.h"
#include "hd_ngm3/hd_ngm3_host.h"   // bank_inputs_t, routing, validation
#include "inhibition/gpu_inhibition.h"
#include "logging/config_record.h"
#include "sim_core/gpu_io_entity.h"

namespace ngm3 {

// GPU sibling of ngm3::neuron_group_t (flat, no shared base — they diverge in
// how the per-step compute is expressed: CPU TBB loops vs. WGSL kernels). The
// storage layout is a direct port of the CPU flat-SoA (hd_ngm3_neuron_group.h
// lines 200-227), uploaded into GPU storage buffers; per-step process() does
// host bookkeeping + a single uniform write and record() appends the kernel
// dispatches.
//
// S6 SCOPE — the read/output path only: per-bank input gather (CopyBufferToBuffer)
// + stats reduction + fused response+combine (workgroup-per-neuron) + local
// inhibition into the output write page. Adapt (S7) and branch (S8) are NOT here,
// so permanences stay frozen at their Poisson init; the S6 bar is "sane / in
// [0,1] / no NaN", NOT statistical parity with a *learning* CPU run.
//
// RNG: no per-neuron RNG state on the GPU. The CPU's per-synapse low_dis(nrng)
// draw is replaced by the stateless PCG-1D32 hash (shared_prng.wgsl) keyed on
// (synapse coordinate ^ step ^ seed ^ bank-salt). Deterministic per (seed, step)
// but NOT bit-equivalent to the CPU mt19937 stream — the intentional MH3
// divergence (IDD §13). seg_ids widen u8 -> u32 (WGSL has no byte storage).
//
// Compiles only under COAST_WITH_GPU.
class neuron_group_gpu_t : public sim::gpu_io_entity {

public:
    ELEMENT_TYPE(neuron_group_gpu_t, "ngm3_neuron_group_gpu")

    // sim_builder-compatible ctor: looks up the named config in the cfg_store and
    // constructs the group with output_id as its identity. The GPU device is
    // bound later via build_step_plan() (called by env at build time).
    neuron_group_gpu_t(const cfg_store   &cs,
                       const std::string &type,
                       sim::output_id_t   output_id,
                       seed_offset_t      seed_offset = {});

    // Movable (env stores entities in a vector); nulls the source handles so the
    // moved-from dtor releases nothing.
    neuron_group_gpu_t(neuron_group_gpu_t&& o) noexcept;
    neuron_group_gpu_t& operator=(neuron_group_gpu_t&&) = delete;
    neuron_group_gpu_t(const neuron_group_gpu_t&)            = delete;
    neuron_group_gpu_t& operator=(const neuron_group_gpu_t&) = delete;

    ~neuron_group_gpu_t() override = default;   // device owns the GPU resources

    // Resolve input channels against the env's channel map (per-bank concatenated
    // widths, inp_ids), then ALLOCATE + Poisson-init + upload the bank GPU
    // storage. Called once by sim_builder's init-func after construction.
    void init_io(const std::unordered_map<std::string, sim::output_channel_t> &input_map);

    // --- io_entity / gpu_io_entity interface ---
    void set_output_buffer(sim::io_buffer& buf) override;
    void set_input_buffer (sim::output_id_t id, sim::io_buffer& buf) override;

    sim::step_result process() override;                    // host bookkeeping + uniform refresh

    // E1 (gpu-abstraction-layer): emit this entity's static per-step work as a
    // backend-agnostic step_plan through the gpu_device facade. Called once at
    // build time by env. Registers this entity's private device buffers (+
    // initializes them from ngm3::cold_init, shared with the CPU group), compiles the 5
    // kernels, reads its io_buffer page ids, emits the op list, and builds the
    // plan on the device.
    void build_step_plan(coast::gpu::gpu_device& dev) override;

    // The plan id registered by build_step_plan (invalid until then). env (F3)
    // enqueues this per step.
    [[nodiscard]] coast::gpu::plan_id plan() const override { return plan_id_; }

    // Structural-change epoch for the dirty/rebuild path (design §8). The current
    // model never changes structure at runtime, so this is constant 0 (dormant);
    // proven end-to-end by the forced-bump test G7.
    [[nodiscard]] std::uint64_t graph_epoch() const override { return 0; }

    void read_probes() override;                // S9: fan group scalars into probes
    void perform_observations(observe::observe_accessor& acc) override;  // OBSERVE population/out

    [[nodiscard]] sim::output_id_t              get_outp_id  () const override;
    [[nodiscard]] std::size_t                   get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t>  get_inp_ids  () const override;
    [[nodiscard]] std::string                   status_str   () const override;

    void save_state(logging::state_writer &writer) const override;
    void load_state(logging::state_reader &reader)       override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override { return poisson_seed_; }

    [[nodiscard]] const neuron_group_params_t& get_params() const { return params; }

    // ---- D7-vis surface (S10) ----
    // Structural getters mirroring ngm3::neuron_group_t exactly, so the
    // (templated) flat/tree visualizers bind to either entity via the same
    // duck-typed accessor contract. All cheap scalars; no GPU traffic.
    [[nodiscard]] std::size_t get_neuron_count()   const { return params.neuron_cnt; }
    [[nodiscard]] std::size_t get_apical_count()   const { return params.apical_cnt; }
    [[nodiscard]] std::size_t get_proximal_count() const { return params.proximal_cnt; }
    [[nodiscard]] std::size_t get_rep_count()      const { return rep_cnt; }
    [[nodiscard]] std::uint8_t get_branch_depth()  const { return params.max_branch_level; }
    // seg_ids are < MAX_SEG_CNT (32) so the u32 GPU width narrows losslessly to
    // the u8 the CPU type reports.
    [[nodiscard]] std::uint8_t get_max_segment_id() const { return static_cast<std::uint8_t>(max_seg_id); }

    // CONSUMER-PULLED refresh (IDD §5.7): one bulk GPU->CPU copy per bank into
    // the CPU mirrors below, reusing readback_permanences()/readback_segment_ids().
    // The entity does NOT auto-refresh — the visualizer calls this from its update()
    // during the post-swap barrier window, where the sim thread is parked and the
    // device is idle (the precondition the readbacks document). const because it
    // only refreshes a cache (mirrors are mutable); observable entity state is
    // unchanged. Cadence is the consumer's (~1 per 10k steps) so the few-MB copy is
    // negligible. Absent banks (apical_cnt==0) leave their mirrors empty. A failed
    // readback does not throw (the caller is the GUI thread) and does not show
    // zeros: the mirrors and their step keep the last good refresh (#234).
    void refresh_vis_mirror() const;

    // Data-view accessors reading from the mirrors (signatures identical to the CPU
    // type; same flat-SoA offset). Return empty if the mirror has not been refreshed
    // or the bank is absent. Each span is one representation = input_size cells.
    [[nodiscard]] std::span<const float>        get_apical_representation  (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const float>        get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const std::uint8_t> get_apical_segment_ids     (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const std::uint8_t> get_proximal_segment_ids   (std::size_t n, std::size_t d, std::size_t r) const;

    // Sim step the mirrors currently reflect, stamped by refresh_vis_mirror() from
    // the host-side step_ counter — no readback by design. Not the learning-step
    // count: the two differ on frozen and zero-output steps.
    [[nodiscard]] std::uint64_t get_vis_mirror_step() const { return vis_mirror_step_.load(std::memory_order_acquire); }

    // GPU->CPU readback of one bank's permanences (flat SoA order,
    // [n*d*rep*input_size]). Returns empty if the bank is absent; throws if the
    // readback fails. Assumes the device is idle (call at a step boundary). For
    // tests / vis.
    [[nodiscard]] std::vector<float> readback_permanences(bank_t bank) const;

    // GPU->CPU readback of one bank's segment_ids (flat SoA order,
    // [n*d*rep*input_size]). Returns empty if the bank is absent; throws if the
    // readback fails. Assumes the device is idle (call at a step boundary). For S8
    // branch validation / vis.
    [[nodiscard]] std::vector<std::uint32_t> readback_segment_ids(bank_t bank) const;

    // GPU->CPU readback of this step's post-inhibition group output (the stable
    // group_outp_ copy published after inhibition); throws if the readback fails.
    // Assumes the device is idle (call at a step boundary). Mirrors ngm25's
    // readback_group_outp().
    [[nodiscard]] std::vector<float> readback_group_outp() const
    { return readback_dev_f32(group_outp_id_, params.neuron_cnt, "group output"); }

private:
    // build_step_plan's parts, in order: the device buffers (cold state from
    // ngm3::cold_init), the kernels, the forward ops (gather -> publish) and,
    // unless frozen, the learning ops (winner, adapt, branch).
    void register_device_buffers(coast::gpu::gpu_device& dev);
    void compile_kernels(coast::gpu::gpu_device& dev);
    void append_forward_ops(coast::gpu::gpu_device& dev, coast::gpu::step_plan& sp);
    void append_learning_ops(coast::gpu::step_plan& sp);

    // gpu_device-path readback of a registered (non-ping_pong) entity buffer, used
    // by save_state / readback_permanences. Assumes the device is idle (true at the
    // dispatcher's step boundary). Throws std::runtime_error naming `what` if the
    // readback fails: most callers read learned state (save_state), and a failed
    // read must never pass as zeros (#234). Diagnostic callers that must not throw
    // (vis refresh, probes) catch it and publish nothing new.
    [[nodiscard]] std::vector<float>         readback_dev_f32(coast::gpu::buffer_id id, std::size_t count,
                                                              std::string_view what) const;
    [[nodiscard]] std::vector<std::uint32_t> readback_dev_u32(coast::gpu::buffer_id id, std::size_t count,
                                                              std::string_view what) const;
    // Learning steps so far (the branch counter), read back from the device: the
    // count of non-frozen steps whose group output did not sum to 0 (M.4.15).
    // Assumes the device is idle (a step boundary).
    [[nodiscard]] std::uint64_t learning_step_count() const;
    // The activity learning window edges as of now (device readback once the plan
    // is built; ngm/window_tracking.h).
    [[nodiscard]] ngm::window_tracking::edges_t window_edges() const;

    // ----------------- members -----------------

    const neuron_group_params_t params;

    // derived
    std::size_t   rep_cnt             = 0;
    std::uint32_t max_seg_id          = 0;
    std::size_t   apical_input_size   = 0;
    std::size_t   proximal_input_size = 0;

    // The effective seed (random_seed + seed_offset): seeds the host Poisson init
    // (identically to the CPU group, so the initial permanences are bit-identical)
    // and keys every per-step draw (params/adapt uniforms). Random state, so it is
    // checkpointed and restored (M.7.8, M.7.17).
    seed_t poisson_seed_ = {};

    // host bookkeeping (the learning-step counter is device-resident:
    // learn_cnt_id_, read back by learning_step_count())
    std::uint32_t step_     = 0;

    // ---- D7-vis CPU mirrors (S10) ----
    // Cache of the GPU bank state, filled on demand by refresh_vis_mirror() and
    // read by the visualizers' accessors. mutable: a const refresh logically only
    // updates this cache. seg_id mirrors are u8 (narrowed from the u32 GPU width).
    // Layout per bank is the flat SoA [n*d*rep*input_size] — identical to the CPU
    // type's vectors, so rep_offset() indexes both the same way.
    mutable std::vector<float>         apical_perm_mirror_;
    mutable std::vector<float>         proximal_perm_mirror_;
    mutable std::vector<std::uint8_t>  apical_segid_mirror_;
    mutable std::vector<std::uint8_t>  proximal_segid_mirror_;
    mutable std::atomic<std::uint64_t> vis_mirror_step_{0};

    // io_entity identity + routing
    sim::output_id_t              id;
    std::vector<sim::output_id_t> inp_ids;
    sim::io_buffer*               output_buf = nullptr;

    // Per-bank channel routing (hd_ngm3_host.h, shared with the CPU group): each
    // routed channel's source io_buffer + its offset into the concatenated bank
    // buffer.
    bank_inputs_t apical_inputs;
    bank_inputs_t proximal_inputs;

    // ---- agnostic gpu_device handles (E1; populated by build_step_plan) -------
    // The device buffers/kernels/plan backing this entity's per-step work. The
    // device owns the GPU resources; these are opaque ids into its registry. All
    // invalid/null until build_step_plan() runs at build time.
    coast::gpu::gpu_device* dev_ = nullptr;

    struct bank_ids_t {
        coast::gpu::buffer_id permanences   = coast::gpu::invalid_buffer;
        coast::gpu::buffer_id segment_ids   = coast::gpu::invalid_buffer;
        coast::gpu::buffer_id adapt_history = coast::gpu::invalid_buffer;
        coast::gpu::buffer_id mismatch      = coast::gpu::invalid_buffer;
        coast::gpu::buffer_id rep_act       = coast::gpu::invalid_buffer;
        coast::gpu::buffer_id input_buf     = coast::gpu::invalid_buffer;
        coast::gpu::buffer_id stats         = coast::gpu::invalid_buffer;  // [chan_cnt][5]: sum, avg, max, noise, count
        coast::gpu::buffer_id chan_of       = coast::gpu::invalid_buffer;  // channel index per input position
        coast::gpu::buffer_id bank_dims     = coast::gpu::invalid_buffer;  // uniform
        std::uint32_t         chan_cnt      = 1;
    };
    bank_ids_t apical_ids_;
    bank_ids_t proximal_ids_;

    coast::gpu::buffer_id neuron_apical_max_id_   = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id neuron_proximal_max_id_ = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id neuron_act_id_          = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id neuron_mx_act_id_       = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id group_outp_id_          = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id adapt_scalars_id_       = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id learn_cnt_id_           = coast::gpu::invalid_buffer;  // u32[4], see winner_main
    coast::gpu::buffer_id window_track_id_        = coast::gpu::invalid_buffer;  // f32[4] window edges, see winner_main
    // Host copy of the window edges: the initial edges until the plan is built,
    // then only a load target (the device copy is authoritative).
    ngm::window_tracking::edges_t window_edges_;
    coast::gpu::buffer_id params_uniform_id_      = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id adapt_params_id_        = coast::gpu::invalid_buffer;

    coast::gpu::kernel_id k_stats_   = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_respcmb_ = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_winner_  = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_adapt_   = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_branch_  = coast::gpu::invalid_kernel;

    coast::gpu::plan_id   plan_id_   = coast::gpu::invalid_plan;

    // Swappable GPU inhibition (kernel-injection emitter). Owns its kernel +
    // uniform; emit()ted into the step_plan at the inhibition slot.
    std::unique_ptr<inhibition::gpu_inhibitor_t> gpu_inhibitor_;
};

} // namespace ngm3

#endif // COAST_WITH_GPU

#endif // HD_NGM3_GPU_NEURON_GROUP_H
