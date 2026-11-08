//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_DENDRITE_H
#define HD_NGM2_DENDRITE_H

#include <cstdint>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <functional>

#include "core/rng/fast_rng.h"
#include "sim_core/io_buffer.h"
#include "hd_ngm2/hd_ngm2_cfg.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "sim_core/io_entity.h"
#include "core/entity_element.h"
#include "core/entity_element_macros.h"
#include "logging/config_record.h"

namespace logging {
class state_writer;
class state_reader;
}

namespace ngm2 {

/*
 * The channels of input_map in the order input_channels declares them; input_map
 * is unordered, so its iteration order must not be used (SPEC.md M.7.5). The
 * declaration must name exactly input_map's channels, each once; a single channel
 * needs no declaration. Shared by the neuron group and the dendrite. Throws
 * std::runtime_error starting with who; feeder names what feeds a channel at that
 * level ("group input", "channel mapping").
 */
template <class Channel>   // sim::output_channel_t or channel_source_t (instantiated in the .cpp)
[[nodiscard]] std::vector<Channel> declared_channels(
    const std::vector<std::string> &input_channels,
    const std::unordered_map<std::string, Channel> &input_map,
    std::string_view who, std::string_view feeder);

/*
 * Main model of a dendritic branch used by the neuron model.
 */
class dendrite_t : public core::entity_element {

public:

    // helper type wrappers to deal with SOA (struct of arrays)
    using syn_tuple_t     = std::tuple<float, float, float, uint16_t, uint8_t>;
    using syn_tuple_ref_t = std::tuple<float&,float&,float&,uint16_t&,uint8_t&>;

    using seg_id_t = uint16_t;

    // main data structure to model the synapses on the dendritic branch.
    // the data structure is layout as "struct of arrays" (SOA) to allow for optimal
    // cache friendliness and facilitate SIMD optimization by the compiler
    struct synapses_t {
        std::vector<float>    permanence;
        std::vector<float>    mismatch;
        std::vector<float>    adapt_history;
        std::vector<seg_id_t> segment_idx;
        std::vector<uint8_t>  input_inc;

        // helper functions to manage SOA layout
        void reserve(std::size_t size);
        void resize(std::size_t size);
        [[nodiscard]] std::size_t size() const;

        // operator overload to access all members synchronously (where needed)
        syn_tuple_ref_t operator[](std::size_t idx);
        syn_tuple_t     operator[](std::size_t idx) const;
    };

private:
    const dendrite_params_t params;

    std::size_t input_size;

    // derived params
    const seg_id_t    max_segment_idx;
    // The permanence threshold in float, as every other implementation holds it:
    // permanences are float, and a float permanence compared with the double
    // params value disagrees exactly where it matters (0.3f > 0.3 is true, and
    // ~7 % of the initial Pois(30)/100 permanences are 0.3f).
    const float       perm_thres_;
    const float       inv_perm_range_;  // precomputed 1.0f / (1.0f - perm_thres_)

    // state
    synapses_t         synapses;
    std::vector<float> segment_activity;
    std::vector<float> segment_weights;
    float              primary_learning_rate;
    float              secondary_learning_rate;
    float              mismatch_smoothing;
    float              accumulated_theta_thres;
    float              min_mismatch_deviation;
    float              min_mismatch_percentage;
    float              mismatch_act_thres;
    float              last_max_inp;

    // helper structures
    // The dendrite's input channels in their declared order (params.input_channels):
    // the source output id and the buffer this dendrite reads for it (null until
    // set_input_buffer). The synapse array is laid out channel by channel in this
    // order, so the order is part of the learned state's meaning and must not
    // depend on how the channels were wired (SPEC.md M.7.5). The dendrite reads
    // the current data via cur_read_buffer(); the matching per-buffer statistics
    // are supplied per step by the owning neuron group (input_stats_map_t).
    std::vector<std::pair<sim::output_id_t, sim::io_buffer*>> input_buf;
    // input_buf's channels by name and width: the dendrite's channel record.
    logging::channel_layout channels;

    fast_rng_t rgen;

    // Observed values captured at their (transient) compute point in
    // get_response(), read back by perform_observations() in the post-process
    // pass. max_activity and attenuation are locals there, so they must be
    // snapshotted here to survive to end-of-step.
    float observed_segment_activity_{0.0f};
    float observed_attenuation_{0.0f};

    // helper functions
    static constexpr seg_id_t calc_max_segment_idx(seg_id_t max_branch_level);


public:
    // Entity element interface
    ELEMENT_TYPE(dendrite_t, "dendrite")
    void perform_observations(observe::observe_accessor& acc) override;

    // stream_index: this dendrite's index in its neuron; parent_seed: the
    // neuron's stream seed. Both feed ngm2::stream_seed.
    dendrite_t(const cfg_store &cs, const std::string &type, int stream_index = 0, seed_t parent_seed = {});

    // input initialization. `input_map` is keyed by this dendrite's declared
    // channel names; the channel record (M.7.5) stores each entry's `source`.
    // `who` names this dendrite in an error.
    void init_input(const channel_source_map_t &input_map, std::string_view who);

    // param read access
    const dendrite_params_t& get_params() const { return params; }

    // the core processing functions
    void set_input_buffer(sim::output_id_t id, sim::io_buffer& buf);
    float get_response(const input_stats_map_t& stats);
    void  adapt_synapses(float max_activity, float weight, const input_stats_map_t& stats);
    void  adapt_branches();

    // runtime parameterization
    void set_primary_learning_rate(float rate)      { primary_learning_rate   = rate;    }
    void set_secondary_learning_rate(float rate)    { secondary_learning_rate = rate;    }
    void set_mismatch_smoothing(float weight)       { mismatch_smoothing      = weight;  }
    void set_accumulated_theta_thres(float thres)   { accumulated_theta_thres = thres;   }
    void set_min_mismatch_deviation(float factor)   { min_mismatch_deviation  = factor;  }
    void set_min_mismatch_percentage(float percent) { min_mismatch_percentage = percent; }

    [[nodiscard]] float get_primary_learning_rate()   const { return primary_learning_rate;   }
    [[nodiscard]] float get_secondary_learning_rate() const { return secondary_learning_rate; }
    [[nodiscard]] float get_mismatch_smoothing()      const { return mismatch_smoothing;      }
    [[nodiscard]] float get_accumulated_theta_thres() const { return accumulated_theta_thres; }
    [[nodiscard]] float get_min_mismatch_deviation()  const { return min_mismatch_deviation;  }
    [[nodiscard]] float get_min_mismatch_percentage() const { return min_mismatch_percentage; }

    // introspection support
    [[nodiscard]] std::vector<uint8_t> get_leaf_mask()                  const;
    [[nodiscard]] seg_id_t             get_representation_count()       const;
    [[nodiscard]] std::vector<float>   get_representation(seg_id_t idx) const;
    [[nodiscard]] std::size_t          get_representation_size()        const;
    [[nodiscard]] std::size_t          get_synapse_count()              const;
    [[nodiscard]] const synapses_t&    get_synapses()                   const;
    [[nodiscard]] seg_id_t             get_max_segment_idx()            const;
    [[nodiscard]] std::size_t          get_input_size()                 const;
    [[nodiscard]] std::size_t          get_channel_count() const { return input_buf.size(); }
    [[nodiscard]] const logging::channel_layout& get_channel_layout() const { return channels; }

    // State persistence. `who` names this dendrite in a load error (e.g.
    // "ngm2 neuron group 3: neuron 5: dendrite 2"); a corrupt checkpoint throws.
    void save_state(logging::state_writer& writer) const;
    void load_state(logging::state_reader& reader, std::string_view who);

    // Reads this dendrite's synapse arrays and checks them against its shape
    // (M.7.18): equal lengths, input_inc summing to the input width, segment ids
    // within the tree depth. Read-only, so the group runs it on every dendrite
    // before assigning anything; load_state keeps the result.
    synapses_t check_shape(logging::state_reader& reader, std::string_view who) const;

    // Appends this dendrite's settings to its group's configuration record (M.7.8).
    void append_config(logging::config_record& rec, const std::string& prefix) const;
};

}

#endif //HD_NGM2_DENDRITE_H
