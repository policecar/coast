//
// 3D Chairs dataset IO entity
//

#ifndef CHAIRS3D_IO_H
#define CHAIRS3D_IO_H

#include <functional>
#include <span>
#include <string>
#include <vector>

#include "core/rng/fast_rng.h"
#include "sim_core/io_entity.h"
#include "io/chairs3d_cfg.h"
#include "core/entity_element_macros.h"

namespace chairs3d {

class chairs3d_io : public sim::io_entity {
public:
    // Entity element interface
    ELEMENT_TYPE(chairs3d_io, "chairs3d_input")
    void perform_observations(observe::observe_accessor& acc) override;

private:
    std::vector<float> images;
    std::vector<std::string> image_paths;  // For lazy loading if needed
    std::size_t image_count = 0;

    int image_width;
    int image_height;
    std::size_t image_pixels;  // width * height * 3

    sim::io_buffer* output_buf {nullptr};

    const chairs3d_param_t params;
    sim::output_id_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;
    std::uint64_t observed_index_{0};  // snapshot of cur_idx taken before it advances; read in perform_observations

    std::size_t ci_cnt;
    int change_interval;

    fast_rng_t rgen;
    std::uniform_real_distribution<float> dis;

public:
    explicit chairs3d_io(const chairs3d_param_t& params, sim::output_id_t output_id);

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
        return { image_width, image_height, 3 };
    }
    [[nodiscard]] std::size_t sample_count() const {
        return image_count;
    }
    [[nodiscard]] std::span<const float> get_sample(std::size_t idx) const {
        if (idx >= image_count) idx = image_count - 1;
        return std::span(images.data() + idx * image_pixels, image_pixels);
    }
};

} // namespace chairs3d

#endif // CHAIRS3D_IO_H
