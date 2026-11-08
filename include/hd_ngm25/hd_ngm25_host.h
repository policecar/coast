//
// Host-side code shared by the CPU (neuron_group_t) and GPU (neuron_group_gpu_t)
// ngm2.5 groups. Both keep the host arena as the source of truth, so all of the
// following is one implementation, not two kept in step by hand:
//
//   * configuration contracts (validate_params; the GPU's tighter limits are
//     passed in as group_caps_t)
//   * input routing (route_inputs)
//   * the dense vis mirror (dense_mirror_t)
//   * the checkpoint schema (write_checkpoint / stage_checkpoint)
//
// stage_checkpoint reads AND validates a whole checkpoint into a staging
// value without touching the group, so load_state assigns nothing until
// everything has passed (SPEC.md M.7.18): a misfit or corrupt checkpoint
// leaves the group exactly as it was.
//

#ifndef HD_NGM25_HOST_H
#define HD_NGM25_HOST_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/seed.h"
#include "hd_ngm25/hd_ngm25_adapt.h"     // adapt_params_t
#include "hd_ngm25/hd_ngm25_arena.h"
#include "hd_ngm25/hd_ngm25_branch.h"    // branch_params_t
#include "hd_ngm25/hd_ngm25_cfg.h"
#include "logging/config_record.h"       // config_record, channel_layout
#include "ngm/window_tracking.h"
#include "sim_core/io_entity.h"          // output_id_t, output_channel_t, io_buffer

namespace logging { class state_reader; class state_writer; }

namespace ngm25 {

// ------------------------------- configuration -----------------------------

// What a backend can run beyond the model's own contracts.
struct group_caps_t {
    std::string_view model;                 // message prefix: "ngm25" / "ngm25-gpu"
    std::uint8_t     max_branch_level;
    std::string_view depth_note;            // appended to the depth message ("" or " (...)")
    bool             stochastic_penalty_only;
    std::size_t      max_dend_count;        // 0 = no limit
    bool             branch_interval_u32;   // branch_interval must be < 2^32
};

inline constexpr group_caps_t k_cpu_caps{
    "ngm25", 6, "", false, 0, false };

// GPU: depth <= 4 (the WGSL response reduction's WG*SEG_MAX(=32) shared buffer,
// as in ngm3-GPU); only the stochastic penalty has a kernel; the response
// kernel dispatches one workgroup per dendrite (WebGPU's guaranteed
// maxComputeWorkgroupsPerDimension, 65535); the device learning-step count is
// a wrapping u32 synced at least once per branch_interval.
inline constexpr group_caps_t k_gpu_caps{
    "ngm25-gpu", 4, " (GPU seg-scratch cap; use the CPU model for depth 5-6)",
    true, 65535, true };

// Throws std::runtime_error naming the first violated contract.
void validate_params(const neuron_group_params_t &p, const group_caps_t &caps);

// The parts of the group shape fixed by params (input ranges: route_inputs).
[[nodiscard]] group_shape_t   base_shape        (const neuron_group_params_t &p);
[[nodiscard]] adapt_params_t  make_adapt_params (const neuron_group_params_t &p);
[[nodiscard]] branch_params_t make_branch_params(const neuron_group_params_t &p);

// --------------------------------- routing ---------------------------------

// One route per declared input channel; apical routes occupy the low positions
// of the concatenated input buffer, proximal the high ones.
struct channel_route_t {
    sim::output_id_t id     = 0;
    bank_t           bank   = bank_t::proximal;
    std::uint32_t    pos_lo = 0;
    std::uint32_t    pos_hi = 0;
    sim::io_buffer  *buf    = nullptr;   // filled by set_input_buffer
    std::string      name;               // the configured channel name
};

struct input_routing_t {
    std::vector<channel_route_t> routes;
    std::vector<std::size_t>     inp_ids;   // flat list for get_inp_ids
    std::uint32_t                width = 0; // total routed input positions
};

// Routes apical channels first, then proximal (one contiguous position range
// per type — the group_shape_t / §13.1 contract; declared order kept within a
// type), and fills shape.input_lo/hi/size. Throws on a configuration that would
// build a silently dead or uncheckpointable group.
[[nodiscard]] input_routing_t route_inputs(
    const neuron_group_params_t                                  &p,
    const std::unordered_map<std::string, sim::output_channel_t> &input_map,
    group_shape_t                                                &shape,
    const group_caps_t                                           &caps,
    sim::output_id_t                                              group_id);

// A bank's channel record (M.7.5): its channels' names and widths in order.
[[nodiscard]] logging::channel_layout channel_layout(std::span<const channel_route_t> routes,
                                                     bank_t                           bank);

// ------------------------------- vis mirror --------------------------------

// The sparse arena materialized into ngm3's dense [neuron x dendrite x rep x
// input_size] layout per bank, so ngm2.5 reuses the ngm3 flat/tree renderers
// unchanged (pitfall P5). Filled on demand by refresh().
class dense_mirror_t {
public:
    void refresh(const synapse_arena_t &arena, const group_shape_t &shape);

    // One (neuron, dendrite, rep) row; empty if out of range or not refreshed.
    [[nodiscard]] std::span<const float>        perm(bank_t b, std::size_t n, std::size_t d, std::size_t r) const;
    [[nodiscard]] std::span<const std::uint8_t> seg (bank_t b, std::size_t n, std::size_t d, std::size_t r) const;

private:
    struct bank_mirror_t {
        std::vector<float>        perm;
        std::vector<std::uint8_t> seg;
        std::size_t               d_cnt = 0, rep_cnt = 0, input_size = 0;
        [[nodiscard]] std::size_t offset(std::size_t n, std::size_t d, std::size_t r) const
        { return ((n * d_cnt + d) * rep_cnt + r) * input_size; }
    };
    bank_mirror_t banks_[2];   // indexed by bank_t
};

// ------------------------------- checkpoint --------------------------------

// The group state a checkpoint carries besides inhibition and the config record.
struct checkpoint_state_t {
    std::uint32_t                 step           = 0;
    std::uint32_t                 branch_epoch   = 0;
    seed_t                        effective_seed = {};
    synapse_arena_t               arena;
    std::vector<std::uint64_t>    input_count;     // [neuron_cnt]
    std::vector<float>            neuron_mx_act;   // [neuron_cnt]
    ngm::window_tracking::edges_t window_edges;
};

// What a group is, for writing and checking a checkpoint against it.
struct group_identity_t {
    std::string_view                 who;      // "ngm25 neuron group 3"
    const neuron_group_params_t     &params;
    const group_shape_t             &shape;
    std::span<const channel_route_t> routes;
    const logging::config_record    &config;   // config_record(params, seed, inhibitor)
};

// Writes everything but the "inhibition" group (the caller's inhibitor writes
// that). The CPU and the GPU group share this schema, so a checkpoint of one
// loads into the other.
void write_checkpoint(logging::state_writer          &w,
                      const group_identity_t         &g,
                      std::uint32_t                   step,
                      std::uint32_t                   branch_epoch,
                      seed_t                          effective_seed,
                      const synapse_arena_t          &arena,
                      std::span<const std::uint64_t>  input_count,
                      std::span<const float>          neuron_mx_act,
                      const ngm::window_tracking::edges_t &window_edges);

// Reads the checkpoint and checks all of it — structure, channel and config
// records, payload presence, per-neuron lengths, check_arena_consistency —
// without touching the group. Throws std::runtime_error ("<who>: ...") on any
// misfit or corruption. step / branch_epoch / effective_seed are the values a
// checkpoint predating the field keeps (the group's current ones).
[[nodiscard]] checkpoint_state_t stage_checkpoint(logging::state_reader  &r,
                                                  const group_identity_t &g,
                                                  std::uint32_t           step,
                                                  std::uint32_t           branch_epoch,
                                                  seed_t                  effective_seed);

} // namespace ngm25

#endif // HD_NGM25_HOST_H
