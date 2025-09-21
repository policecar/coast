//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_NEURON_GROUP_H
#define HD_NGM2_NEURON_GROUP_H

#include <vector>
#include <functional>
#include <span>
#include <random>

#include "io_entity.h"
#include "io_buffer.h"
#include "hd_ngm2_neuron.h"

namespace ngm2 {

class neuron_group_t : public sim::io_entity {

public:
    struct params_t {
        partial_id_t                    id;
        std::vector<neuron_t::params_t> neuron_params;
        float                           default_local_inhibition_strength; //suggestion: 10.0
        float                           default_common_learning_rate;      //suggestion: 0.01
        sigmoid_shape_t                 default_weight_filter;             //suggestion: sn:0.5 / tp:0.5
    };

private:
    const params_t params;

    // derived params
    std::vector<std::size_t> inp_ids;

    // state
    std::vector<neuron_t> neurons;
    std::function<std::span<float>()> output_mem {};

    float           local_inhibition_strength;
    float           common_learning_rate;
    sigmoid_shape_t weight_filter;

    std::mt19937 rgen;

    // debug
    float last_max {};
    float avg_max {};
public:
    explicit neuron_group_t(params_t  _params);

    // read-only parameter access
    [[nodiscard]] const params_t& get_params() const { return params; }

    // core functionality / io_entity interface
    void set_outp_func(std::function<std::span<float>()> outp_func) override;
    void set_inp_func(partial_id_t id, const std::function<sim::io_buffer::inp_buf_t()> &inp_func) override;
    void process() override;

    [[nodiscard]] std::size_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override;

    [[nodiscard]] std::string status_str() const override;

    // runtime parameterization
    void set_local_inhibition_strength(const float strength) { local_inhibition_strength = strength; }
    void set_common_learning_rate(const float rate)          { common_learning_rate      = rate;     }
    void set_weight_filter(const sigmoid_shape_t filter)     { weight_filter             = filter;   }

    [[nodiscard]] float& get_local_inhibition_strength() { return local_inhibition_strength; }
    [[nodiscard]] float get_common_learning_rate()      const { return common_learning_rate;      }
    [[nodiscard]] sigmoid_shape_t get_weight_filter()   const { return weight_filter;             }

    // introspection support
    [[nodiscard]] const neuron_t&      get_neuron(std::size_t idx)    const;
    [[nodiscard]] std::size_t          get_neuron_count()             const;
    [[nodiscard]] dendrite_t::seg_id_t get_max_representation_count() const;
    [[nodiscard]] std::size_t          get_representation_count()     const;
    [[nodiscard]] std::size_t          get_synapse_count()            const;
    [[nodiscard]] float                get_max_mismatch()             const;
    [[nodiscard]] float                get_avg_mismatch()             const;
    [[nodiscard]] float                get_max_acc_theta()            const;
    [[nodiscard]] float                get_avg_acc_theta()            const;

};

}

#endif //HD_NGM2_NEURON_GROUP_H
