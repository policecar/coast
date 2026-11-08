//
// Created 2026-07-10 — ngm2.5 GPU port (GI1, design/ngm2_5_gpu.md).
//
// GPU sibling of ngm25::neuron_group_t. It is CPU-ARENA-BACKED: the host
// synapse_arena_t / group_shape_t / step_ctx_t are the source of truth (built and,
// later, re-packed by the SAME CPU code the CPU model uses — design §5, single
// source of truth for split semantics), and each step's arena + input + stats are
// uploaded to device mirrors. The per-step compute (response R2 + combine R3, and
// in later increments winner/adapt/branch) runs as WGSL kernels in a static
// step_plan through the coast::gpu::gpu_device facade.
//
// GI1 scope — the forward path only: host gather + host R1 stats (reusing the CPU
// build_step_ctx, so CPU<->GPU stats agree by construction) -> upload -> device
// response + combine + inhibition + publish. Winner/adapt (GI3) and growth (GI4)
// are not here; permanences stay at their Poisson init (like ngm3-GPU S6).
//
// v1 INPUT LIMITATION (documented default, design §6.1 conflict): inputs are
// gathered + stats computed on the HOST, so the upstream must be host-visible
// (CPU / cpu_to_gpu io_buffer). GPU->GPU in-plan gather copies (design §6.2) are
// deferred; CPU<->GPU validation drives both models from the same CPU input.
//
// Compiles only under COAST_WITH_GPU.
//

#ifndef HD_NGM25_GPU_NEURON_GROUP_H
#define HD_NGM25_GPU_NEURON_GROUP_H

#ifdef COAST_WITH_GPU

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/entity_element_macros.h"
#include "core/gpu/gpu_device.h"
#include "ngm/window_tracking.h"
#include "hd_ngm25/hd_ngm25_adapt.h"          // adapt_params_t
#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_branch.h"         // branch_params_t
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "hd_ngm25/hd_ngm25_host.h"
#include "hd_ngm25/hd_ngm25_response.h"       // group_shape_t, step_ctx_t, channel_stats_t
#include "inhibition/gpu_inhibition.h"
#include "logging/config_record.h"
#include "sim_core/gpu_io_entity.h"

namespace ngm25 {

class neuron_group_gpu_t : public sim::gpu_io_entity {

public:
    ELEMENT_TYPE(neuron_group_gpu_t, "ngm25_neuron_group_gpu")

    neuron_group_gpu_t(const cfg_store   &cs,
                       const std::string &type,
                       sim::output_id_t   output_id,
                       seed_offset_t      seed_offset = {});

    // Defaulted move is noexcept (all members move-noexcept; params is a plain
    // member, not const, precisely so this stays noexcept for the entity vector).
    // The device owns the GPU resources; the entity holds only opaque ids, so a
    // moved-from dtor releases nothing.
    neuron_group_gpu_t(neuron_group_gpu_t&&) noexcept = default;
    neuron_group_gpu_t& operator=(neuron_group_gpu_t&&) = delete;
    neuron_group_gpu_t(const neuron_group_gpu_t&)            = delete;
    neuron_group_gpu_t& operator=(const neuron_group_gpu_t&) = delete;
    ~neuron_group_gpu_t() override = default;

    // Resolve routing (apical channels first, then proximal — group_shape_t
    // contract), build the host arena (D5 Poisson init), size host scratch.
    void init_io(const std::unordered_map<std::string, sim::output_channel_t> &input_map);

    // --- io_entity / gpu_io_entity interface ---
    void set_output_buffer(sim::io_buffer& buf) override;
    void set_input_buffer (sim::output_id_t id, sim::io_buffer& buf) override;

    sim::step_result process() override;                 // host gather+stats, uniform+input upload
    void build_step_plan(coast::gpu::gpu_device& dev) override;
    [[nodiscard]] coast::gpu::plan_id plan() const override { return plan_id_; }
    [[nodiscard]] std::uint64_t graph_epoch() const override { return 0; }
    // Post-rendezvous (device-idle) hook: GI4 host-side branch gate + growth.
    void read_probes() override;
    void perform_observations(observe::observe_accessor& acc) override;

    [[nodiscard]] sim::output_id_t             get_outp_id  () const override;
    [[nodiscard]] std::size_t                  get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t> get_inp_ids  () const override;
    [[nodiscard]] std::string                  status_str   () const override;

    void save_state(logging::state_writer &writer) const override;
    void load_state(logging::state_reader &reader)       override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override
    { return effective_seed_; }

    [[nodiscard]] const neuron_group_params_t& get_params() const { return params; }

    [[nodiscard]] std::size_t get_neuron_count()  const { return params.neuron_cnt; }
    [[nodiscard]] std::size_t get_synapse_count() const { return arena_.syn_count(); }
    [[nodiscard]] std::uint32_t get_branch_epoch() const { return branch_epoch_; }

    // ---- vis / introspection surface (duck-typed to the ngm3 flat/tree
    // renderers, exactly like the CPU ngm25 entity). refresh_vis_mirror() does the
    // GPU->CPU readback + materializes the dense [n x d x rep x input_size] mirror;
    // the visualizer calls it from update() in the post-swap barrier window (sim
    // parked, device idle) -- the same contract ngm3-GPU uses. A failed readback
    // keeps the last good mirror and does not throw (#234). ----
    [[nodiscard]] std::size_t get_apical_count()   const { return params.apical_cnt; }
    [[nodiscard]] std::size_t get_proximal_count() const { return params.proximal_cnt; }
    [[nodiscard]] uint8_t     get_branch_depth()   const { return params.max_branch_level; }
    [[nodiscard]] std::size_t get_rep_count()      const
    { return std::size_t{1} << params.max_branch_level; }
    [[nodiscard]] uint8_t     get_max_segment_id() const { return shape_.max_seg_id; }
    void refresh_vis_mirror() const;

    [[nodiscard]] std::span<const float>   get_apical_representation  (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const float>   get_proximal_representation(std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const uint8_t> get_apical_segment_ids     (std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const uint8_t> get_proximal_segment_ids   (std::size_t n, std::size_t d, std::size_t r) const;
    // Consistency of the (host-authoritative structure + device-updated fields)
    // arena — GI4 checks this green after branch events. Refreshes device fields.
    [[nodiscard]] bool check_consistency(std::string* err) const
    { readback_arena_into_host(); return check_arena_consistency(arena_, shape_, err); }

    // GPU->CPU readback of the per-neuron post-combine activity and the published
    // post-inhibition output (device idle assumed — a step boundary). Throw if
    // the readback fails.
    [[nodiscard]] std::vector<float> readback_neuron_act()  const;
    [[nodiscard]] std::vector<float> readback_group_outp()  const;
    // Refresh the host arena's device-updated fields (perm/adapt_hist/mismatch) +
    // neuron_mx_act from the device. const: only mutates mirrors/caches. Staged:
    // throws on a failed readback with the arena untouched (#234).
    void readback_arena_into_host() const;
    // Device-idle snapshots of the learned arrays (for CPU<->GPU adapt validation).
    [[nodiscard]] std::vector<float> readback_perm()       const { readback_arena_into_host(); return arena_.perm; }
    [[nodiscard]] std::vector<float> readback_adapt_hist() const { readback_arena_into_host(); return arena_.adapt_hist; }
    [[nodiscard]] std::vector<float> readback_mismatch()   const { readback_arena_into_host(); return arena_.mismatch; }

private:
    // Throws std::runtime_error naming `what` on a failed readback (#234).
    [[nodiscard]] std::vector<float> readback_f32(coast::gpu::buffer_id id, std::size_t count,
                                                  std::string_view what) const;
    // The activity learning window edges as of now (device readback once the plan
    // is built; ngm/window_tracking.h), and the host copy -> device upload.
    [[nodiscard]] ngm::window_tracking::edges_t window_edges() const;
    void upload_window_edges();
    [[nodiscard]] std::uint32_t      read_learn_cnt() const;   // throws on a failed readback
    void recompute_sync_due();                                 // from input_count_
    void upload_step_uniforms();     // Params + input_concat + half_max + inv_sum
    void upload_arena_to_device();   // widen u16/u8 -> u32, push host arena to device
    void sync_arena_to_device();     // upload, or grow capacity + rebuild the plan if needed
    // build_step_plan's parts: device buffers, kernels (first build only), the
    // forward ops (response -> publish) and, unless frozen, the learning ops.
    void register_device_buffers(coast::gpu::gpu_device& dev, bool first_build);
    void compile_kernels(coast::gpu::gpu_device& dev);
    void append_forward_ops(coast::gpu::gpu_device& dev, coast::gpu::step_plan& sp,
                            bool first_build);
    void append_learning_ops(coast::gpu::step_plan& sp);
    void grow_if_branched();         // GI4: host branch gate + repack + re-upload/rebuild

    // ----- config / host model (source of truth) -----
    neuron_group_params_t params;   // non-const: keeps the defaulted move noexcept
    group_shape_t               shape_;
    mutable synapse_arena_t     arena_;   // host arena; device is a mirror
    step_ctx_t                  ctx_;     // rebuilt each step (R1)
    adapt_params_t              adapt_params_;   // derived from params (R5/R6/R7)
    branch_params_t             branch_params_;  // derived from params (R8, GI4)

    // per-neuron carried state (host mirror; neuron_mx_act is device-updated and
    // refreshed on readback / save).
    mutable std::vector<float>  neuron_mx_act_;
    std::vector<std::uint64_t>  input_count_;   // per-neuron branch counters, as of the
                                                // last sync with b_learn_cnt_ (see below)

    // Lazy branch-counter sync. Only the device knows whether a step learned (its
    // group output may sum to 0, SPEC.md M.4.15); the winner kernel counts learning
    // steps in b_learn_cnt_. Every counter advances by at most 1 per step, so no
    // neuron can reach its next branch boundary for sync_due_in_ steps; the host
    // reads the count back only then — on the first step a crossing is possible,
    // so branching still happens on exactly the CPU's steps.
    std::uint32_t learn_synced_   = 0;   // device count at the last sync
    std::uint64_t steps_unsynced_ = 0;   // non-frozen steps since the last sync
    std::uint64_t sync_due_in_    = 1;   // steps until a counter could reach a boundary

    // Dense vis mirror, filled on demand by refresh_vis_mirror() (mutable: a
    // const refresh only updates cache).
    mutable dense_mirror_t mirror_;

    sim::output_id_t id_;
    seed_t           effective_seed_ = {};
    std::uint32_t    step_           = 0;
    std::uint32_t    branch_epoch_   = 0;

    // io routing (route_inputs, hd_ngm25_host.h — shared with the CPU entity)
    std::vector<channel_route_t> routes_;
    std::vector<std::size_t>     inp_ids_;
    sim::io_buffer*              output_buf_ = nullptr;
    std::vector<float>           input_concat_;   // host gather target

    // ----- device handles (populated by build_step_plan) -----
    coast::gpu::gpu_device* dev_ = nullptr;
    std::size_t capacity_syn_ = 0;                // device arena capacity (GI4 grows)

    coast::gpu::buffer_id b_syn_perm_       = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_syn_input_idx_  = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_syn_seg_        = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_syn_adapt_hist_ = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_syn_mismatch_   = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_dend_offset_    = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_dend_type_      = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_input_concat_   = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_half_max_       = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_inv_sum_        = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_dend_act_       = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_seg_activity_   = coast::gpu::invalid_buffer;  // response->adapt
    coast::gpu::buffer_id b_neuron_act_     = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_neuron_mx_act_  = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_group_outp_     = coast::gpu::invalid_buffer;
    coast::gpu::buffer_id b_ctx4_           = coast::gpu::invalid_buffer;  // adapt per-pos ctx (vec4)
    coast::gpu::buffer_id b_wscal_          = coast::gpu::invalid_buffer;  // winner {idx,sum,win,mx}
    coast::gpu::buffer_id b_learn_cnt_      = coast::gpu::invalid_buffer;  // u32 learning steps (winner)
    coast::gpu::buffer_id b_window_track_   = coast::gpu::invalid_buffer;  // f32[4] window edges (winner)
    // Host copy of the window edges: the initial edges until the plan is built,
    // then only a load target (the device copy is authoritative).
    ngm::window_tracking::edges_t window_edges_;
    coast::gpu::buffer_id b_params_         = coast::gpu::invalid_buffer;  // uniform (Params)
    coast::gpu::buffer_id b_adapt_params_   = coast::gpu::invalid_buffer;  // uniform (AdaptParams)

    coast::gpu::kernel_id k_response_       = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_combine_        = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_winner_         = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_adapt_primary_  = coast::gpu::invalid_kernel;
    coast::gpu::kernel_id k_adapt_secondary_= coast::gpu::invalid_kernel;
    coast::gpu::plan_id   plan_id_          = coast::gpu::invalid_plan;

    // Inhibition ops captured on the first build_step_plan() and replayed on every
    // rebuild. gpu_inhibitor_->emit() registers the emitter's own kernel + uniform
    // as a side effect, so it must run exactly once; caching its ops lets a growth
    // rebuild reuse them (their bindings reference only stable handles) instead of
    // re-emitting and leaking a pipeline + uniform buffer each capacity crossing.
    std::vector<coast::gpu::plan_op> inhibition_ops_;

    std::unique_ptr<inhibition::gpu_inhibitor_t> gpu_inhibitor_;
};

} // namespace ngm25

#endif // COAST_WITH_GPU
#endif // HD_NGM25_GPU_NEURON_GROUP_H
