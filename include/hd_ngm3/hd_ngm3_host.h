//
// Host-side code shared by the ngm3 CPU (neuron_group_t) and GPU
// (neuron_group_gpu_t) groups: the configuration contracts and the input
// routing. One implementation, so a config the CPU refuses the GPU refuses
// too, and both lay out their banks' inputs the same way.
//

#ifndef HD_NGM3_HOST_H
#define HD_NGM3_HOST_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "hd_ngm3/hd_ngm3_cfg.h"
#include "logging/config_record.h"   // channel_layout
#include "sim_core/io_entity.h"      // output_id_t, output_channel_t, io_buffer

namespace logging { class state_reader; class state_writer; }

namespace ngm3 {

// What a backend adds to the model's own contracts.
struct group_caps_t {
    std::string_view model;                 // message prefix: "ngm3" / "ngm3_gpu"
    bool             branch_interval_u32;   // the device keeps the branch phase in 32 bits
};

inline constexpr group_caps_t k_cpu_caps{ "ngm3",     false };
inline constexpr group_caps_t k_gpu_caps{ "ngm3_gpu", true  };

// Throws std::runtime_error naming the first violated contract. Configs are
// JSON-deserialized, so a malformed one is a Release-time event: these are
// throws, not asserts.
void validate_params(const neuron_group_params_t &p, const group_caps_t &caps);

// One bank's routed input channels, in declared order.
struct bank_inputs_t {
    std::vector<std::string>      names;     // the configured channel names
    std::vector<sim::output_id_t> ids;
    std::vector<std::size_t>      sizes;
    std::vector<std::size_t>      offsets;   // prefix sum into the bank's concat buffer
    std::vector<sim::io_buffer*>  bufs;      // one per channel, filled by set_input_buffer

    [[nodiscard]] std::size_t width() const
    { return offsets.empty() ? 0 : offsets.back() + sizes.back(); }

    // Points every channel fed by output `id` at `buf`.
    void bind(sim::output_id_t id, sim::io_buffer &buf);

    // The bank's channel record (M.7.5): names and widths in order.
    [[nodiscard]] logging::channel_layout layout() const;
};

struct input_routing_t {
    bank_inputs_t            apical;
    bank_inputs_t            proximal;
    std::vector<std::size_t> inp_ids;   // every routed channel's id, declared order
};

// Walks params.inputs in declared order (a deterministic layout) and routes
// each channel to its bank. Throws on an unnamed or unknown channel, a channel
// routed to an absent apical bank, an empty proximal bank, and apical
// dendrites without apical input.
[[nodiscard]] input_routing_t route_inputs(
    const neuron_group_params_t                                  &p,
    const std::unordered_map<std::string, sim::output_channel_t> &input_map,
    const group_caps_t                                           &caps,
    sim::output_id_t                                              group_id);

// ---- checkpoint parts the CPU and GPU groups share (one schema, M.7.17) ----

struct checkpoint_shape_t {
    const neuron_group_params_t &params;
    std::size_t                  apical_input_size, proximal_input_size;
    const bank_inputs_t         &apical, &proximal;
};
// The structural fields and the channel records (M.7.5).
void write_checkpoint_shape(logging::state_writer &w, const checkpoint_shape_t &s);
// Throws unless the checkpoint's shape and channel records fit (M.7.18).
void check_checkpoint_shape(logging::state_reader &r, const checkpoint_shape_t &s,
                            const std::string &who);

[[nodiscard]] std::runtime_error corrupt_checkpoint(const std::string &who, std::string_view what);

// One bank's payload, read and checked against the replica layout the compute
// relies on (check_bank_layout). An absent bank is not read: present == false.
struct staged_bank_t {
    std::vector<float>        perm, adapt, mm;
    std::vector<std::uint8_t> seg;
    bool                      present = false;
};
[[nodiscard]] staged_bank_t stage_bank(logging::state_reader &r, std::string_view bank_name,
                                       const neuron_group_params_t &p, std::size_t d_cnt,
                                       std::size_t input_size, const std::string &who);

// The running max, checked against neuron_cnt; nullopt for a checkpoint written
// before the format had it (the group keeps its own, state_checks.h).
[[nodiscard]] std::optional<std::vector<float>>
read_neuron_mx_act(logging::state_reader &r, std::size_t neuron_cnt, const std::string &who);

} // namespace ngm3

#endif // HD_NGM3_HOST_H
