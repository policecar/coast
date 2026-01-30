//
// 3D Chairs dataset IO entity
//

#ifndef CHAIRS3D_IO_H
#define CHAIRS3D_IO_H

#include <functional>
#include <span>
#include <string>
#include <random>
#include <vector>

#include "io_entity.h"
#include "chairs3d_cfg.h"

namespace chairs3d {

class chairs3d_io : public sim::io_entity {
    std::vector<float> images;
    std::vector<std::string> image_paths;  // For lazy loading if needed
    std::size_t image_count = 0;

    int image_width;
    int image_height;
    std::size_t image_pixels;  // width * height * 3

    std::function<std::span<float>()> output_mem {};

    const chairs3d_param_t params;
    sim::output_id_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;

    std::size_t ci_cnt;
    int change_interval;

    std::mt19937 rgen;
    std::uniform_real_distribution<float> dis;

public:
    explicit chairs3d_io(const chairs3d_param_t& params, sim::output_id_t output_id);

    void set_outp_func(std::function<std::span<float>()>) override;

    void process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    [[nodiscard]] int& get_change_interval() { return change_interval; }

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { image_width, image_height, 3 };
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

} // namespace chairs3d

#endif // CHAIRS3D_IO_H
