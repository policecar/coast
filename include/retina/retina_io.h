//
// Retina node: reads one image buffer (a dataset node's output) and emits one
// polarity (ON or OFF) of its centre-surround response plus a uniform noise
// floor, the "null" signal. Two nodes, one per polarity, reading the same image
// model the ON and OFF pathways.
//
// Stateless apart from the noise RNG, which is saved with the simulation state
// together with a configuration record of params (checked on load, M.7.8).
// The output size is fixed by the config (width x height x out_channels) because
// sim_builder may bind a consumer before this node is initialised; init_io()
// checks the config against the actual input size.
//

#ifndef COAST_RETINA_IO_H
#define COAST_RETINA_IO_H

#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/entity_element_macros.h"
#include "core/rng/fast_rng.h"
#include "logging/config_record.h"
#include "retina/retina.h"
#include "sim_core/io_entity.h"

namespace retina {

// Checkpoint configuration record (SPEC.md M.7.8): on load the configuration
// wins, so a resume under changed parameters is reported, not continued blindly.
// One entry per retina_param_t field (noise_range as .lo/.hi; the seed as
// effective_seed, random state). unit.retina checks it covers every JSON key, so
// a parameter added later cannot drop out of the resume check unnoticed.
[[nodiscard]] logging::config_record config_record(const retina_param_t& p);

class retina_io : public sim::io_entity {
    const retina_param_t params_;
    contrast_filter_t    filter_;
    sim::output_id_t     id_;

    std::vector<std::size_t> inp_ids_;   // exactly one after init_io()
    sim::io_buffer*          input_buf_  {nullptr};
    sim::io_buffer*          output_buf_ {nullptr};
    std::vector<float>       d_;         // signed response, output_size(params_)

    fast_rng_t                            rgen_;
    std::uniform_real_distribution<float> noise_;

public:
    // Validates params (throws std::invalid_argument).
    retina_io(const retina_param_t& params, sim::output_id_t output_id);

    ELEMENT_TYPE(retina_io, "retina")

    // Binds the single input channel. Throws std::runtime_error unless
    // input_map has exactly one entry whose size equals width*height*channels.
    void init_io(const std::unordered_map<std::string, sim::output_channel_t>& input_map);

    void set_input_buffer(sim::output_id_t id, sim::io_buffer& buf) override;
    void set_output_buffer(sim::io_buffer& buf) override { output_buf_ = &buf; }

    sim::step_result process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override { return id_; }
    [[nodiscard]] std::size_t get_outp_size() const override { return output_size(params_); }
    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return inp_ids_; }

    [[nodiscard]] std::string status_str() const override;

    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override {
        return params_.random_seed;
    }

    [[nodiscard]] const retina_param_t& params() const { return params_; }
};

}

#endif // COAST_RETINA_IO_H
