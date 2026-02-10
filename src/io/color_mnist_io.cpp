//
// Color MNIST IO implementation
//

#include "color_mnist_io.h"
#include "logging/state_io.h"

namespace color_mnist {

color_mnist_io::color_mnist_io(const color_mnist_param_t &params, sim::output_id_t output_id) :
    mnist_db(params.image_file, params.label_file),
    params(params),
    id(output_id),
    cur_epoch(0),
    cur_idx(0),
    ci_cnt(0),
    change_interval(params.change_interval),
    rgen(params.random_seed),
    noise_dis(params.noise_range.first, params.noise_range.second),
    color_dis(0, 2),  // 0=red, 1=green, 2=blue
    current_color(0)
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }
    // Initialize colored buffer
    colored_buffer.resize(28 * 28 * 3, 0.0f);
}

void color_mnist_io::set_outp_func(std::function<std::span<float>()> function)
{
    output_mem = std::move(function);
}

void color_mnist_io::process()
{
    if (!output_mem) {
        throw std::runtime_error("color_mnist_io::process() called before set_outp_func()");
    }

    auto outp = output_mem();
    const std::size_t rgb_size = 28 * 28 * 3;

    if (outp.size() != rgb_size) {
        throw std::runtime_error("Output span size mismatch: expected " +
            std::to_string(rgb_size) + ", got " + std::to_string(outp.size()));
    }

    // Clear output to black
    std::ranges::fill(outp, 0.0f);

    if (change_interval > 0) {
        auto gray_img = get_norm_image(cur_idx);

        // Pick a random color for this sample
        current_color = color_dis(rgen);

        // Copy grayscale values to the selected color channel
        // Output format is interleaved RGB: [R0,G0,B0, R1,G1,B1, ...]
        for (std::size_t i = 0; i < gray_img.size(); ++i) {
            float val = gray_img[i];
            // Add noise
            val = std::clamp(val + noise_dis(rgen), 0.0f, 1.0f);

            // Set only the selected channel
            outp[i * 3 + current_color] = val;
        }
    }

    // Advance to next sample if interval reached
    if (change_interval > 0 && ++ci_cnt >= static_cast<std::size_t>(change_interval)) {
        ci_cnt = 0;
        if (++cur_idx >= get_image_cnt()) {
            cur_idx = 0;
            ++cur_epoch;
        }
    }
}

sim::output_id_t color_mnist_io::get_outp_id() const
{
    return id;
}

std::size_t color_mnist_io::get_outp_size() const
{
    return 28 * 28 * 3;  // RGB
}

std::string color_mnist_io::status_str() const
{
    static const char* color_names[] = { "Red", "Green", "Blue" };
    std::string status { "Color MNIST IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    status += " | color: " + std::string(color_names[current_color]);
    return status;
}

std::span<const float> color_mnist_io::get_colored_sample() const
{
    // Generate a colored version of the current sample for preview
    auto gray_img = get_norm_image(cur_idx);

    colored_buffer.assign(28 * 28 * 3, 0.0f);

    for (std::size_t i = 0; i < gray_img.size(); ++i) {
        colored_buffer[i * 3 + current_color] = gray_img[i];
    }

    return colored_buffer;
}

void color_mnist_io::save_state(logging::state_writer& writer) const
{
    writer.write("cur_epoch", static_cast<std::uint64_t>(cur_epoch));
    writer.write("cur_idx", static_cast<std::uint64_t>(cur_idx));
    writer.write("ci_cnt", static_cast<std::uint64_t>(ci_cnt));
}

void color_mnist_io::load_state(logging::state_reader& reader)
{
    cur_epoch = static_cast<std::size_t>(reader.read_uint64("cur_epoch", cur_epoch));
    cur_idx = static_cast<std::size_t>(reader.read_uint64("cur_idx", cur_idx));
    ci_cnt = static_cast<std::size_t>(reader.read_uint64("ci_cnt", ci_cnt));
}

} // color_mnist
