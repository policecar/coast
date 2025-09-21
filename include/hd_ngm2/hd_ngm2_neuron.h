//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_NEURON_H
#define HD_NGM2_NEURON_H

#include <vector>
#include <array>
#include <span>
#include <functional>
#include <random>

#include "hd_ngm2_dendrite.h"
#include "hd_ngm2_tools.h"
#include "io_buffer.h"

namespace ngm2 {

class neuron_t {

public:
    using learning_window_t = std::pair<sigmoid_shape_t,sigmoid_shape_t>;

    struct params_t {
        std::vector<dendrite_t::params_t> dendrite_params;
        std::size_t                       default_branch_interval; // suggestion: 5000
        learning_window_t                 default_activity_learning_window; // suggestion: [sn:0.6 / tp:0.33],[sn:0.6 / tp:0.66]
    };

private:
    using dendrite_type_array = std::array<float,static_cast<int>(dendrite_t::type_t::TYPE_COUNT)>;

    const params_t params;

    // state
    std::vector<dendrite_t> dendrites;
    float                   neuron_activity;
    dendrite_type_array     dendrite_type_activity;
    std::size_t             input_count;
    std::size_t             branch_interval;
    learning_window_t       activity_learning_window;
    float                   energy;
    std::mt19937            rgen;

public:
    explicit neuron_t(params_t _params);

    // id of the neuron within the neuron group
    std::size_t id;

    // param read access
    [[nodiscard]] const params_t& get_params() const { return params; }

    // core processing functions
    void set_inp_func(partial_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func);
    float get_response();
    void  adapt(float weight);

    // runtime parameterization
    void set_branch_interval(std::size_t interval)                     { branch_interval          = interval; }
    void set_activity_learning_window(const learning_window_t &window) { activity_learning_window = window;   }

    [[nodiscard]] std::size_t       get_branch_interval()          const { return branch_interval;          }
    [[nodiscard]] learning_window_t get_activity_learning_window() const { return activity_learning_window; }

    // introspection support
    [[nodiscard]] dendrite_t::seg_id_t get_representation_count() const;
    [[nodiscard]] const dendrite_t&    get_dendrite(std::size_t idx) const;
    [[nodiscard]] std::size_t          get_dendrite_count() const;
    [[nodiscard]] std::size_t          get_synapse_count() const;

};

}

#endif //HD_NGM2_NEURON_H
