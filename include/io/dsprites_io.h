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
#include "io_entity.h"
#include "dsprites_cfg.h"
#include "core/entity_element_macros.h"

namespace dsprites {

class dsprites_io : public sim::io_entity {
public:
    // Entity element interface
    ELEMENT_TYPE(dsprites_io, "dsprites_input")

private:
    static constexpr std::size_t image_dim = 64;
    static constexpr std::size_t image_pixels = image_dim * image_dim;

    std::vector<float> images;
    std::size_t image_count = 0;

    std::function<std::span<float>()> output_mem {};

    const dsprites_param_t params;
    sim::output_id_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;

    std::size_t ci_cnt;
    int change_interval;

    fast_rng_t rgen;
    std::uniform_real_distribution<float> dis;
    std::uniform_int_distribution<std::size_t> idx_dis;

public:
    explicit dsprites_io(const dsprites_param_t& params, sim::output_id_t output_id);

    void set_outp_func(std::function<std::span<float>()>) override;

    void process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    // State persistence
    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    [[nodiscard]] int& get_change_interval() { return change_interval; }

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { image_dim, image_dim, 1 };
    }
    [[nodiscard]] std::size_t sample_count() const {
        return image_count;
    }
    [[nodiscard]] std::size_t current_epoch() const {
        return cur_epoch;
    }
    [[nodiscard]] std::size_t current_index() const {
        return cur_idx;
    }
    [[nodiscard]] std::span<const float> get_sample(std::size_t idx) const {
        if (idx >= image_count) idx = image_count - 1;
        return std::span(images.data() + idx * image_pixels, image_pixels);
    }
};

} // namespace sim

#endif // DSPRITES_IO_H
