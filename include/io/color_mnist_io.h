//
// Color MNIST IO - MNIST digits randomly colored red, green, or blue
//

#ifndef COLOR_MNIST_IO_H
#define COLOR_MNIST_IO_H

#include <span>
#include <string>
#include <vector>

#include "io/idx_io_base.h"
#include "io/color_mnist_cfg.h"
#include "core/entity_element_macros.h"

namespace color_mnist {

class color_mnist_io : public dataset::idx_io_base {
public:
    // Entity element interface
    ELEMENT_TYPE(color_mnist_io, "color_mnist_input")

private:
    const color_mnist_param_t params;

    std::uniform_int_distribution<int> color_dis;  // 0=red, 1=green, 2=blue

    // Store current color for visualization
    int current_color = 0;

public:
    explicit color_mnist_io(const color_mnist_param_t &params, sim::output_id_t output_id);

    sim::step_result process() override;

    // Output size is 28*28*3 (RGB), not get_image_size() (grayscale).
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::string status_str() const override;

    // Persists seq_, current_color, and rng (superset of the base save_state).
    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    // dataset_io interface
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { 28, 28, 3 };  // RGB output
    }
    [[nodiscard]] std::size_t sample_count() const {
        return get_image_cnt();
    }

    // For color mnist, we need to return the colored version
    // This is used by dataset_browser - returns last processed colored image
    [[nodiscard]] std::span<const float> get_colored_sample() const;

private:
    // Buffer to store the colored image for get_colored_sample
    mutable std::vector<float> colored_buffer;
};

} // namespace color_mnist

#endif //COLOR_MNIST_IO_H
