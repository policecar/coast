//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_DENDRITE_H
#define HD_NGM2_DENDRITE_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <random>
#include <span>
#include <tuple>
#include <functional>
#include <set>

#include "io_buffer.h"


namespace ngm2 {

using partial_id_t   = std::size_t;

class dendrite_t {

public:

    enum class type_t {
        apical,
        proximal,
        TYPE_COUNT
    };

    struct params_t {
        type_t      type;
        std::size_t input_size;
        std::set<partial_id_t> input_ids;
        float       permanence_threshold;
        uint8_t     max_branch_level;
        int         rnd_seed;
        float       default_primary_learning_rate;   // suggestion: 0.01
        float       default_secondary_learning_rate; // suggestion: 0.0001
        float       default_mismatch_smoothing;      // suggestion: 0.001
        float       default_accumulated_theta_thres; // suggestion: 2.0
        float       default_min_mismatch_deviation;  // suggestion: 1.0
        float       default_min_mismatch_percentage; // suggestion: 0.05
    };

    using syn_tuple_t     = std::tuple<float, float, float, uint16_t, uint8_t>;
    using syn_tuple_ref_t = std::tuple<float&,float&,float&,uint16_t&,uint8_t&>;

    using seg_id_t = uint16_t;

    struct synapses_t {
        std::vector<float>    permanence;
        std::vector<float>    mismatch;
        std::vector<float>    adapt_history;
        std::vector<seg_id_t> segment_idx;
        std::vector<uint8_t>  input_inc;

        void reserve(std::size_t size);
        void resize(std::size_t size);
        [[nodiscard]] std::size_t size() const;

        syn_tuple_ref_t operator[](std::size_t idx);
        syn_tuple_t     operator[](std::size_t idx) const;
    };

private:
    const params_t params;

    // derived params
    const seg_id_t    max_segment_idx;

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
    float              last_max_inp;

    // helper structures
    std::unordered_map<
        partial_id_t,
        std::function<sim::io_buffer::inp_buf_t()>
    > input_mem;

    std::mt19937 rgen;

    // helper functions
    static constexpr seg_id_t calc_max_segment_idx(seg_id_t max_branch_level);
public:
    explicit dendrite_t(params_t _params);

    // param read access
    const params_t& get_params() const { return params; }

    // the core processing functions
    void set_inp_func(partial_id_t id, std::function<sim::io_buffer::inp_buf_t()> inp_func);
    float get_response();
    void  adapt_synapses(float max_activity, float weight);
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

};

}

#endif //HD_NGM2_DENDRITE_H
