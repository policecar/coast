//
// CIFAR-100 dataset IO entity implementation
//

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <ranges>

#include "cifar100_io.h"

namespace cifar100 {

namespace {

// CIFAR-100 binary format: 2 label bytes (coarse + fine) + 3072 image bytes per sample
// Image bytes are stored as: 1024 R + 1024 G + 1024 B (planar format)
constexpr std::size_t bytes_per_sample = 2 + 32 * 32 * 3;

void load_batch_file(const std::string& filepath,
                     std::vector<float>& images,
                     std::vector<uint8_t>& coarse_labels,
                     std::vector<uint8_t>& fine_labels,
                     std::size_t expected_count)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CIFAR-100 batch file: " + filepath);
    }

    std::vector<uint8_t> buffer(bytes_per_sample);
    constexpr std::size_t pixels = 32 * 32;
    constexpr std::size_t pixel_floats = pixels * 3;

    for (std::size_t i = 0; i < expected_count; ++i) {
        if (!file.read(reinterpret_cast<char*>(buffer.data()), bytes_per_sample)) {
            break;
        }

        // First byte is coarse label, second is fine label
        coarse_labels[i] = buffer[0];
        fine_labels[i] = buffer[1];

        // Convert from planar (RRR...GGG...BBB) to interleaved (RGBRGBRGB)
        const std::size_t img_offset = i * pixel_floats;

        for (std::size_t p = 0; p < pixels; ++p) {
            // R channel at buffer[2 + p]
            // G channel at buffer[2 + pixels + p]
            // B channel at buffer[2 + 2*pixels + p]
            images[img_offset + p * 3 + 0] = static_cast<float>(buffer[2 + p]) / 255.0f;
            images[img_offset + p * 3 + 1] = static_cast<float>(buffer[2 + pixels + p]) / 255.0f;
            images[img_offset + p * 3 + 2] = static_cast<float>(buffer[2 + 2 * pixels + p]) / 255.0f;
        }
    }
}

} // namespace

cifar100_io::cifar100_io(const cifar100_param_t& params, sim::output_id_t output_id) :
    params(params),
    id(output_id),
    cur_epoch(0),
    cur_idx(0),
    ci_cnt(0),
    change_interval(params.change_interval),
    rgen(params.random_seed),
    dis(params.noise_range.first, params.noise_range.second)
{
    if (params.noise_range.first > params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }

    namespace fs = std::filesystem;

    if (params.use_test_set) {
        // Test set: 10000 samples
        image_count = 10000;
        std::string test_file = (fs::path(params.data_dir) / "test.bin").string();
        images.resize(image_count * image_pixels);
        coarse_labels.resize(image_count);
        fine_labels.resize(image_count);
        load_batch_file(test_file, images, coarse_labels, fine_labels, image_count);
    } else {
        // Training set: 50000 samples
        image_count = 50000;
        std::string train_file = (fs::path(params.data_dir) / "train.bin").string();
        images.resize(image_count * image_pixels);
        coarse_labels.resize(image_count);
        fine_labels.resize(image_count);
        load_batch_file(train_file, images, coarse_labels, fine_labels, image_count);
    }
}

void cifar100_io::set_outp_func(std::function<std::span<float>()> function)
{
    output_mem = std::move(function);
}

void cifar100_io::process()
{
    if (!output_mem) {
        throw std::runtime_error("cifar100_io::process() called before set_outp_func()");
    }
    auto outp = output_mem();
    if (outp.size() != image_pixels) {
        throw std::runtime_error("Output span size mismatch: expected " +
            std::to_string(image_pixels) + ", got " + std::to_string(outp.size()));
    }

    if (change_interval > 0) {
        auto img = std::span(images.begin() + cur_idx * image_pixels, image_pixels);
        std::ranges::copy(img, outp.begin());
    } else {
        std::ranges::fill(outp, 0.0f);
    }

    for (auto &ov : outp) {
        ov = std::clamp(ov + dis(rgen), 0.0f, 1.0f);
    }

    if (change_interval > 0 && ++ci_cnt >= static_cast<std::size_t>(change_interval)) {
        ci_cnt = 0;
        if (++cur_idx >= image_count) {
            cur_idx = 0;
            ++cur_epoch;
        }
    }
}

sim::output_id_t cifar100_io::get_outp_id() const
{
    return id;
}

std::size_t cifar100_io::get_outp_size() const
{
    return image_pixels;
}

std::string cifar100_io::status_str() const
{
    std::string status { "CIFAR-100 IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    status += " / " + std::to_string(image_count);
    return status;
}

} // namespace cifar100
