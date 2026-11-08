//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_NEURON_GROUP_H
#define HD_NGM2_NEURON_GROUP_H

#include <vector>
#include <unordered_map>
#include <string>
#include <functional>
#include <span>
#include <memory>
#include <utility>

#include "core/rng/fast_rng.h"
#include "sim_core/io_entity.h"
#include "sim_core/io_buffer.h"
#include "hd_ngm2/hd_ngm2_neuron.h"
#include "hd_ngm2/hd_ngm2_cfg.h"
#include "hd_ngm2/hd_ngm2_tools.h"
#include "ngm/window_tracking.h"
#include "core/entity_element_macros.h"
#include "inhibition/inhibition.h"

using namespace sim;

namespace ngm2 {

/*
 * The main neuron group model is defined within this class. It interfaces with the simulation
 * environment by implementing the sim::io_entity interface
 */
class neuron_group_t : public sim::io_entity {

    const neuron_group_params_t params;

    // derived params
    output_id_t              id;
    // source id of each input channel, in params.input_channels order
    std::vector<std::size_t> inp_ids;

    // state
    std::vector<neuron_t> neurons;
    sim::io_buffer* output_buf {nullptr};
    // input channels in params.input_channels order: source id and buffer
    // (null until set_input_buffer)
    std::vector<std::pair<sim::output_id_t, sim::io_buffer*>> input_bufs;

    float           local_inhibition_strength;
    float           common_learning_rate;
    sigmoid_shape_t weight_filter;
    float           stochastic_win_thres;

    // Swappable inhibition for this group (one instance; per-group state). The
    // call site in process() delegates to inhibitor_->inhibit(). Future
    // composition (multiple inhibitions) is the group's concern, not the
    // interface's.
    std::unique_ptr<inhibition::inhibitor_t> inhibitor_;

    fast_rng_t rgen;

    // Effective RNG seed (params.random_seed + seed_offset) stored at construction
    // so the run manifest can surface it without re-deriving coast's composition rule.
    seed_t effective_seed_ = {};

    // buffer for input observe
    std::vector<float> inp_observe;

    // Activity window tracking (ngm/window_tracking.h): the edges of the
    // winner's and the other neurons' learning windows. Learning state.
    ngm::window_tracking::edges_t window_edges;
    std::vector<float>       activity_scratch;   // per-step raw activities for the update

public:
    // Entity element interface. Children: the neuron vector (path "neuron_t[i]",
    // unchanged) plus this group's single inhibition instance (path
    // "inhibition[0]"), exposing the inhibitor for probing/introspection.
    ELEMENT_TYPE(neuron_group_t, "neuron_group")
    ELEMENT_CHILDREN_1_AND_PTR(neurons, neuron_t, inhibitor_, inhibition::inhibitor_t, "inhibition")
    void perform_observations(observe::observe_accessor& acc) override;

    // main constructor the sets up the neuron group
    explicit neuron_group_t(const cfg_store &cs, const std::string &type, sim::output_id_t output_id, seed_offset_t seed_offset = {});

    // io initialization
    void init_io(const std::unordered_map<std::string,output_channel_t> &input_map);

    // read-only parameter access
    [[nodiscard]] const neuron_group_params_t& get_params() const { return params; }

    // core functionality / io_entity interface
    void set_output_buffer(sim::io_buffer& buf) override;
    void set_input_buffer(output_id_t id, sim::io_buffer& buf) override;

    // main function that models one processing step of the neuron group
    sim::step_result process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override;

    [[nodiscard]] std::string status_str() const override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override { return effective_seed_; }
    [[nodiscard]] std::optional<seed_t> inhibition_seed() const override
    { return inhibition::random_state_seed(inhibitor_->params()); }

    // runtime parameterization
    void set_common_learning_rate(const float rate)          { common_learning_rate      = rate;     }
    void set_weight_filter(const sigmoid_shape_t filter)     { weight_filter             = filter;   }

    [[nodiscard]] float get_common_learning_rate()      const { return common_learning_rate;      }
    [[nodiscard]] sigmoid_shape_t get_weight_filter()   const { return weight_filter;             }

    // introspection support - used by the visualizations
    [[nodiscard]] const neuron_t&      get_neuron(std::size_t idx)    const;
    [[nodiscard]] std::size_t          get_neuron_count()             const;
    [[nodiscard]] std::size_t          get_max_representation_count() const;
    [[nodiscard]] std::size_t          get_representation_count()     const;
    [[nodiscard]] std::size_t          get_synapse_count()            const;
    [[nodiscard]] float                get_max_mismatch()             const;
    [[nodiscard]] float                get_avg_mismatch()             const;
    [[nodiscard]] float                get_max_acc_theta()            const;
    [[nodiscard]] float                get_avg_acc_theta()            const;
    // Tracked activity learning windows: first = the winner's, second = every
    // other neuron's (ngm/window_tracking.h).
    [[nodiscard]] std::pair<learning_window_t,learning_window_t> get_tracked_windows() const;

    // State persistence (implements io_entity interface)
    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    // This group's configuration record (M.7.8), from its current settings: the
    // group's own, then those of each neuron cfg and dendrite cfg it uses, read
    // from the first neuron / dendrite built from that cfg, and for every later
    // instance the settings in which it differs from that first one.
    [[nodiscard]] logging::config_record current_config() const;
};

}

#endif //HD_NGM2_NEURON_GROUP_H
