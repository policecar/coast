//
// 3D Chairs dataset IO entity implementation
// Uses stb_image for PNG loading
//

#include <algorithm>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <ranges>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

#include "io/chairs3d_io.h"
#include "core/require.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"

namespace chairs3d {

chairs3d_io::chairs3d_io(const chairs3d_param_t& params, sim::output_id_t output_id) :
    image_width(params.image_width),
    image_height(params.image_height),
    image_pixels(static_cast<std::size_t>(params.image_width) *
                 static_cast<std::size_t>(params.image_height) * 3),
    params(params),
    id(output_id),
    cur_epoch(0),
    cur_idx(0),
    ci_cnt(0),
    change_interval(params.change_interval),
    rgen(std::to_underlying(params.random_seed)),
    dis(params.noise_range.first, params.noise_range.second)
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }

    namespace fs = std::filesystem;

    // Scan directory recursively for PNG files
    if (!fs::exists(params.data_dir)) {
        throw std::runtime_error("3D Chairs data directory does not exist: " + params.data_dir);
    }

    std::vector<fs::path> png_files;
    for (const auto& entry : fs::recursive_directory_iterator(params.data_dir)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                png_files.push_back(entry.path());
            }
        }
    }

    if (png_files.empty()) {
        throw std::runtime_error("No image files found in 3D Chairs data directory");
    }

    // Sort for reproducibility
    std::sort(png_files.begin(), png_files.end());

    image_count = png_files.size();
    images.resize(image_count * image_pixels);

    // Load and resize all images
    for (std::size_t i = 0; i < image_count; ++i) {
        int w, h, channels;
        unsigned char* data = stbi_load(png_files[i].string().c_str(), &w, &h, &channels, 3);
        if (!data) {
            throw std::runtime_error("Failed to load image: " + png_files[i].string());
        }

        // Resize if necessary
        std::vector<unsigned char> resized;
        unsigned char* src_data = data;

        if (w != image_width || h != image_height) {
            resized.resize(static_cast<std::size_t>(image_width) *
                           static_cast<std::size_t>(image_height) * 3);
            stbir_resize_uint8_linear(
                data, w, h, 0,
                resized.data(), image_width, image_height, 0,
                STBIR_RGB
            );
            src_data = resized.data();
        }

        // Convert to interleaved float RGB
        const std::size_t offset = i * image_pixels;
        for (std::size_t p = 0; p < image_pixels; ++p) {
            images[offset + p] = static_cast<float>(src_data[p]) / 255.0f;
        }

        stbi_image_free(data);
    }
}

void chairs3d_io::set_output_buffer(sim::io_buffer& buf)
{
    output_buf = &buf;
}

sim::step_result chairs3d_io::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);
    auto outp = output_buf->cur_write_buffer();
    COAST_REQUIRE(outp.size() == image_pixels, core::error_kind::contract_violation);

    if (change_interval > 0) {
        COAST_REQUIRE((cur_idx + 1) * image_pixels <= images.size(),
                      core::error_kind::contract_violation);
        auto img = std::span(images.begin() + cur_idx * image_pixels, image_pixels);
        std::ranges::copy(img, outp.begin());
    } else {
        std::ranges::fill(outp, 0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen), 0.0f, 1.0f);
    }

    // Record which sample is presented this step. This does NOT line up with
    // population/out at equal step_index: IO buffers are double-buffered, so a
    // consumer reads this value one step later -- one step per buffered hop on
    // the feedforward path, and recurrent (attractor) groups have no fixed
    // offset at all. cur_idx is std::size_t (no observe dtype specialisation) -> cast.
    observed_index_ = static_cast<std::uint64_t>(cur_idx);  // snapshot before cur_idx advances; read in perform_observations

    if (change_interval > 0 && ++ci_cnt >= static_cast<std::size_t>(change_interval)) {
        ci_cnt = 0;
        if (++cur_idx >= image_count) {
            cur_idx = 0;
            ++cur_epoch;
        }
    }

    return {};
}

sim::output_id_t chairs3d_io::get_outp_id() const
{
    return id;
}

std::size_t chairs3d_io::get_outp_size() const
{
    return image_pixels;
}

std::string chairs3d_io::status_str() const
{
    std::string status { "3D Chairs IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    status += " / " + std::to_string(image_count);
    return status;
}

void chairs3d_io::save_state(logging::state_writer& writer) const
{
    writer.write("cur_epoch", static_cast<std::uint64_t>(cur_epoch));
    writer.write("cur_idx", static_cast<std::uint64_t>(cur_idx));
    writer.write("ci_cnt", static_cast<std::uint64_t>(ci_cnt));
    logging::write_rng(writer, "rng", rgen);
    dataset::save_run_binding(writer, params.run);
}

void chairs3d_io::load_state(logging::state_reader& reader)
{
    if (!dataset::restores_sequence(reader, params.run, std::format("chairs3d {}", get_outp_id())))
        return;
    cur_epoch = static_cast<std::size_t>(reader.read_uint64("cur_epoch", cur_epoch));
    cur_idx = static_cast<std::size_t>(reader.read_uint64("cur_idx", cur_idx));
    ci_cnt = static_cast<std::size_t>(reader.read_uint64("ci_cnt", ci_cnt));
    // The saved cur_idx reflects the training dataset's length; wrap it into the
    // loaded dataset so loading onto a smaller set can't read out of range.
    cur_idx = (image_count == 0) ? 0 : (cur_idx % image_count);

    logging::read_rng(reader, "rng", rgen, std::format("3D Chairs IO {}", get_outp_id()));
}

void chairs3d_io::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("input"))
        acc.observe(*this, "input", "index", observed_index_);
}

} // namespace chairs3d
