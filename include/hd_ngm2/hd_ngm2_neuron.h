//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_NEURON_H
#define HD_NGM2_NEURON_H

#include <vector>
#include <array>
#include <functional>

#include "core/rng/fast_rng.h"
#include "hd_ngm2_dendrite.h"
#include "io_buffer.h"
#include "hd_ngm2_cfg.h"
#include "core/entity_element.h"
#include "core/entity_element_macros.h"


namespace ngm2 {


/*
 * The main neuron model is defined within this class. It is used by the neuron group model.
 */
class neuron_t : public core::entity_element {

    using dendrite_type_array = std::array<float,static_cast<int>(dendrite_type_t::TYPE_COUNT)>;

    const neuron_params_t params;

    // state
    std::vector<dendrite_t> dendrites;
    float                   neuron_activity;
    float                   max_activity;
    dendrite_type_array     dendrite_type_activity;
    std::size_t             input_count;
    std::size_t             branch_interval;
    learning_window_t       activity_learning_window;
    [[maybe_unused]] float  energy;
    fast_rng_t              rgen;

public:
    // Entity element interface
    ELEMENT_TYPE(neuron_t, "neuron")
    ELEMENT_CHILDREN_1(dendrites, dendrite_t)

    explicit neuron_t(const cfg_store &cs, const std::string &type, int seed_delta = 0, int seed_offset = 0);

    // input initialization
    void init_input(const std::unordered_map<std::string,sim::output_channel_t> &input_map);

    // id of the neuron within the neuron group
    std::size_t id;

    // param read access
    [[nodiscard]] const neuron_params_t& get_params() const { return params; }
    [[nodiscard]] const std::vector<dendrite_t>& get_dendrites() const { return dendrites; }

    // core processing functions
    void set_inp_func(sim::output_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func);
    float get_response();
    void  adapt(float weight);

    // runtime parameterization
    void set_branch_interval(std::size_t interval)                     { branch_interval          = interval; }
    void set_activity_learning_window(const learning_window_t &window) { activity_learning_window = window;   }

    [[nodiscard]] std::size_t       get_branch_interval()          const { return branch_interval;          }
    [[nodiscard]] const learning_window_t& get_activity_learning_window() const { return activity_learning_window; }

    // introspection support
    [[nodiscard]] dendrite_t::seg_id_t get_representation_count() const;
    [[nodiscard]] const dendrite_t&    get_dendrite(std::size_t idx) const;
    [[nodiscard]] std::size_t          get_dendrite_count() const;
    [[nodiscard]] std::size_t          get_synapse_count() const;

    // State persistence
    void save_state(logging::state_writer& writer) const;
    void load_state(logging::state_reader& reader);
};

}

#endif //HD_NGM2_NEURON_H
