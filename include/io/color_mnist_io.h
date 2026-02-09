//
// Color MNIST IO - MNIST digits randomly colored red, green, or blue
//

#ifndef COLOR_MNIST_IO_H
#define COLOR_MNIST_IO_H

#include <functional>
#include <span>
#include <string>

#include "core/rng/fast_rng.h"
#include "io_entity.h"
#include "mnist_db.h"
#include "color_mnist_cfg.h"
#include "core/entity_element_macros.h"

namespace color_mnist {

class color_mnist_io : public sim::io_entity, public mnist::mnist_db {
public:
    // Entity element interface
    ELEMENT_TYPE(color_mnist_io, "color_mnist_input")

private:
    const color_mnist_param_t params;

    std::function<std::span<float>()> output_mem {};

    sim::output_id_t id;

    std::size_t cur_epoch;
    std::size_t cur_idx;

    std::size_t ci_cnt;
    int change_interval;

    fast_rng_t rgen;
    std::uniform_real_distribution<float> noise_dis;
    std::uniform_int_distribution<int> color_dis;  // 0=red, 1=green, 2=blue

    // Store current color for visualization
    int current_color = 0;

public:
    explicit color_mnist_io(const color_mnist_param_t &params, sim::output_id_t output_id);

    void set_outp_func(std::function<std::span<float>()>) override;

    void process() override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    [[nodiscard]] std::string status_str() const override;

    [[nodiscard]] int get_change_interval() { return change_interval; }

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { 28, 28, 3 };  // RGB output
    }
    [[nodiscard]] std::size_t sample_count() const {
        return get_image_cnt();
    }
    [[nodiscard]] std::size_t current_epoch() const {
        return cur_epoch;
    }
    [[nodiscard]] std::size_t current_index() const {
        return cur_idx;
    }

    // For color mnist, we need to return the colored version
    // This is used by dataset_browser - returns last processed colored image
    [[nodiscard]] std::span<const float> get_colored_sample() const;

private:
    // Buffer to store the colored image for get_colored_sample
    mutable std::vector<float> colored_buffer;
};

} // color_mnist

#endif //COLOR_MNIST_IO_H
