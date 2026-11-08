//
// dSprites dataset IO entity
//

#ifndef DSPRITES_IO_H
#define DSPRITES_IO_H

#include <functional>
#include <span>
#include <string>
#include <vector>

#include "core/rng/fast_rng.h"
#include "sim_core/io_entity.h"
#include "io/dsprites_cfg.h"
#include "io/sample_sequencer.h"
#include "core/entity_element_macros.h"

namespace hdf5 { class executor; }

namespace dsprites {

class dsprites_io : public sim::io_entity {
public:
    // Entity element interface
    ELEMENT_TYPE(dsprites_io, "dsprites_input")
    void perform_observations(observe::observe_accessor& acc) override;

private:
    static constexpr std::size_t image_dim = 64;
    static constexpr std::size_t image_pixels = image_dim * image_dim;

    std::vector<float> images;
    std::size_t image_count = 0;

    sim::io_buffer* output_buf {nullptr};

    const dsprites_param_t params;
    sim::output_id_t id;

    dataset::sample_sequencer seq_;

    // Snapshot of seq_.current_index() taken in process() before seq_.advance(),
    // read back in perform_observations (post-process pass runs after process()).
    std::uint64_t observed_index_ {0};

    fast_rng_t rgen;
    std::uniform_real_distribution<float> dis;

public:
    /// Loads the dSprites image set from the HDF5 file at construction time.
    /// The libhdf5 work runs on the executor's worker thread, so the
    /// constructor blocks on a single submit_sync that opens the file,
    /// reads the dataset and closes everything before returning.
    dsprites_io(const dsprites_param_t& params, sim::output_id_t output_id,
                hdf5::executor& h5_exec);

    void set_output_buffer(sim::io_buffer&) override;

    sim::step_result process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    // State persistence
    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override {
        return params.random_seed;
    }

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { image_dim, image_dim, 1 };
    }
    [[nodiscard]] std::size_t sample_count() const {
        return image_count;
    }
    [[nodiscard]] std::span<const float> get_sample(std::size_t idx) const {
        if (idx >= image_count) idx = image_count - 1;
        return std::span(images.data() + idx * image_pixels, image_pixels);
    }
};

} // namespace sim

#endif // DSPRITES_IO_H
