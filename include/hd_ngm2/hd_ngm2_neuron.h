//
// Created by jk on 26.08.25.
//

#ifndef HD_NGM2_NEURON_H
#define HD_NGM2_NEURON_H

#include <vector>
#include <array>
#include <functional>
#include <string_view>

#include "core/rng/fast_rng.h"
#include "hd_ngm2/hd_ngm2_dendrite.h"
#include "sim_core/io_buffer.h"
#include "hd_ngm2/hd_ngm2_cfg.h"
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
    [[maybe_unused]] float  energy;
    fast_rng_t              rgen;

public:
    // Entity element interface
    ELEMENT_TYPE(neuron_t, "neuron")
    ELEMENT_CHILDREN_1(dendrites, dendrite_t)
    void perform_observations(observe::observe_accessor& acc) override;

    // stream_index: this neuron's index in its group; parent_seed: the group's
    // effective seed (random_seed + seed_offset). Both feed ngm2::stream_seed.
    explicit neuron_t(const cfg_store &cs, const std::string &type, int stream_index = 0, seed_t parent_seed = {});

    // input initialization. `who` names this neuron in an error (e.g.
    // "ngm2 neuron group 1: neuron 5"). See dendrite_t::init_input for what
    // `input_map`'s channel_source_t entries carry.
    void init_input(const channel_source_map_t &input_map, std::string_view who);

    // id of the neuron within the neuron group
    std::size_t id;

    // param read access
    [[nodiscard]] const neuron_params_t& get_params() const { return params; }
    [[nodiscard]] const std::vector<dendrite_t>& get_dendrites() const { return dendrites; }

    // core processing functions
    void set_input_buffer(sim::output_id_t id, sim::io_buffer& buf);
    float get_response(const input_stats_map_t& stats);
    // Decay the running activity max and fold in this step's activity. Learning
    // state: the group calls it only on learning steps (not frozen, group output
    // not summing to 0; SPEC.md M.4.15), after get_response and before adapt.
    void  update_max_activity();
    // Adapt to this step's input, gated by the activity learning `window` (the
    // group's tracked window, ngm/window_tracking.h).
    void  adapt(float weight, const input_stats_map_t& stats, const learning_window_t& window);
    // This step's activity (before group inhibition), as read by adapt().
    [[nodiscard]] float get_activity() const { return neuron_activity; }

    // runtime parameterization
    void set_branch_interval(std::size_t interval)                     { branch_interval          = interval; }

    [[nodiscard]] std::size_t       get_branch_interval()          const { return branch_interval;          }

    // introspection support
    [[nodiscard]] std::size_t          get_representation_count() const;
    [[nodiscard]] const dendrite_t&    get_dendrite(std::size_t idx) const;
    [[nodiscard]] std::size_t          get_dendrite_count() const;
    [[nodiscard]] std::size_t          get_synapse_count() const;

    // State persistence. `who` names this neuron in a load error (e.g.
    // "ngm2 neuron group 3: neuron 5"); a checkpoint that does not fit throws.
    void save_state(logging::state_writer& writer) const;
    void load_state(logging::state_reader& reader, std::string_view who);

    // Appends this neuron's own settings (not its dendrites') to its group's
    // configuration record (M.7.8).
    void append_config(logging::config_record& rec, const std::string& prefix) const;
};

}

#endif //HD_NGM2_NEURON_H
