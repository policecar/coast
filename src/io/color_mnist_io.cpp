//
// Color MNIST IO implementation
//

#include "io/color_mnist_io.h"
#include "core/require.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"

#include <format>
#include <stdexcept>

namespace color_mnist {

namespace {
// The RGB output layout is fixed at 28x28x3; process() writes outp[i*3 + c] for
// every grayscale pixel i, so the file's images must be exactly 28x28 (#200).
constexpr std::size_t gray_size = 28 * 28;
constexpr std::size_t rgb_size  = gray_size * 3;
}

color_mnist_io::color_mnist_io(const color_mnist_param_t &params, sim::output_id_t output_id) :
    dataset::idx_io_base(params.image_file, params.label_file, output_id, params),
    params(params),
    color_dis(0, 2),  // 0=red, 1=green, 2=blue
    current_color(0)
{
    if (get_image_size() != gray_size) {
        throw std::runtime_error("color_mnist: images in " + params.image_file +
                                 " have " + std::to_string(get_image_size()) +
                                 " pixels, expected 28x28 = " +
                                 std::to_string(gray_size));
    }
    colored_buffer.resize(rgb_size, 0.0f);
}

std::size_t color_mnist_io::get_outp_size() const
{
    return rgb_size;
}

sim::step_result color_mnist_io::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);

    auto outp = output_buf->cur_write_buffer();

    COAST_REQUIRE(outp.size() == rgb_size, core::error_kind::contract_violation);

    // Clear output to black
    std::ranges::fill(outp, 0.0f);

    if (seq_.change_interval() > 0) {
        auto gray_img = get_norm_image(seq_.current_index());

        // Pick a random color ONCE per presentation (first step of each
        // change_interval window). Previously redrawn every step, which flickered
        // the color within a held presentation -- harmless when change_interval==1
        // (one sample per step) but it destroys color-identity for held
        // presentations (change_interval>1), forcing any consumer to learn color
        // invariance and decorrelating the per-presentation color from the sample.
        // The hold counter now lives in the sequencer (was the loader's ci_cnt).
        if (seq_.is_interval_start()) {
            current_color = color_dis(rgen);
        }

        // Copy grayscale values to the selected color channel
        // Output format is interleaved RGB: [R0,G0,B0, R1,G1,B1, ...]
        for (std::size_t i = 0; i < gray_img.size(); ++i) {
            float val = gray_img[i];
            // Add noise
            val = std::clamp(val + dis(rgen), 0.0f, 1.0f);

            // Set only the selected channel
            outp[i * 3 + current_color] = val;
        }
    }

    // Record which sample is presented this step. This does NOT line up with
    // population/out at equal step_index: IO buffers are double-buffered, so a
    // consumer reads this value one step later -- one step per buffered hop on
    // the feedforward path, and recurrent (attractor) groups have no fixed
    // offset at all. cur_idx is std::size_t (no observe dtype specialisation) -> cast.
    observed_index_ = static_cast<std::uint64_t>(seq_.current_index());  // snapshot before seq_ advances; read in idx_io_base::perform_observations

    // Advance to next sample if interval reached
    seq_.advance();
    return {};
}

std::string color_mnist_io::status_str() const
{
    static const char* color_names[] = { "Red", "Green", "Blue" };
    std::string status { "Color MNIST IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(seq_.current_epoch());
    status += " | idx: " + std::to_string(seq_.current_index());
    status += " | color: " + std::string(color_names[current_color]);
    return status;
}

std::span<const float> color_mnist_io::get_colored_sample() const
{
    // Generate a colored version of the current sample for preview
    auto gray_img = get_norm_image(seq_.current_index());

    colored_buffer.assign(rgb_size, 0.0f);

    for (std::size_t i = 0; i < gray_img.size(); ++i) {
        colored_buffer[i * 3 + current_color] = gray_img[i];
    }

    return colored_buffer;
}

void color_mnist_io::save_state(logging::state_writer& writer) const
{
    dataset::save_sequenced_state(params, writer, seq_, rgen);
    writer.write("current_color", static_cast<std::int32_t>(current_color));
}

void color_mnist_io::load_state(logging::state_reader& reader)
{
    if (!dataset::restores_sequence(reader, params.run, std::format("color_mnist {}", get_outp_id()))) {
        return;
    }
    // current_color picks the RGB channel (outp[i*3 + c]) until the next
    // interval start redraws it, so a value outside [0, 2] would write past the
    // pixel (#238). Checked before the sequencer is restored, so a bad colour is
    // refused before anything is assigned.
    const std::int32_t c = reader.read_int32("current_color", current_color);
    if (c < 0 || c > 2) {
        throw std::runtime_error(
            "color_mnist: corrupt checkpoint -- current_color " + std::to_string(c) +
            " is not 0, 1 or 2 (red, green, blue).");
    }
    seq_.load_state(reader);
    logging::read_rng(reader, "rng", rgen, "dataset");
    current_color = c;
}

} // namespace color_mnist
