//
// CIFAR-10 dataset IO entity
//

#ifndef CIFAR10_IO_H
#define CIFAR10_IO_H

#include <functional>
#include <span>
#include <string>
#include <random>
#include <vector>

#include "io_entity.h"
#include "cifar10_cfg.h"

namespace cifar10 {

class cifar10_io : public sim::io_entity {
    static constexpr std::size_t image_dim = 32;
    static constexpr std::size_t image_channels = 3;
    static constexpr std::size_t image_pixels = image_dim * image_dim * image_channels;

    std::vector<float> images;
    std::vector<uint8_t> labels;
    std::size_t image_count = 0;

    std::function<std::span<float>()> output_mem {};

    const cifar10_param_t params;
    sim::output_id_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;

    std::size_t ci_cnt;
    int change_interval;

    std::mt19937 rgen;
    std::uniform_real_distribution<float> dis;

public:
    explicit cifar10_io(const cifar10_param_t& params, sim::output_id_t output_id);

    void set_outp_func(std::function<std::span<float>()>) override;

    void process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    [[nodiscard]] int& get_change_interval() { return change_interval; }

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { image_dim, image_dim, image_channels };
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

} // namespace cifar10

#endif // CIFAR10_IO_H
