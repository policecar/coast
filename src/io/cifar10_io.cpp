//
// CIFAR-10 dataset IO entity implementation
//

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <ranges>

#include "cifar10_io.h"

namespace cifar10 {

namespace {

// CIFAR-10 binary format: 1 label byte + 3072 image bytes per sample
// Image bytes are stored as: 1024 R + 1024 G + 1024 B (planar format)
constexpr std::size_t bytes_per_sample = 1 + 32 * 32 * 3;
constexpr std::size_t samples_per_batch = 10000;

void load_batch_file(const std::string& filepath,
                     std::vector<float>& images,
                     std::vector<uint8_t>& labels,
                     std::size_t& offset)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CIFAR-10 batch file: " + filepath);
    }

    std::vector<uint8_t> buffer(bytes_per_sample);

    for (std::size_t i = 0; i < samples_per_batch; ++i) {
        if (!file.read(reinterpret_cast<char*>(buffer.data()), bytes_per_sample)) {
            break;
        }

        // First byte is label
        labels[offset + i] = buffer[0];

        // Convert from planar (RRR...GGG...BBB) to interleaved (RGBRGBRGB)
        const std::size_t pixels = 32 * 32;
        const std::size_t img_offset = (offset + i) * pixels * 3;

        for (std::size_t p = 0; p < pixels; ++p) {
            // R channel at buffer[1 + p]
            // G channel at buffer[1 + pixels + p]
            // B channel at buffer[1 + 2*pixels + p]
            images[img_offset + p * 3 + 0] = static_cast<float>(buffer[1 + p]) / 255.0f;
            images[img_offset + p * 3 + 1] = static_cast<float>(buffer[1 + pixels + p]) / 255.0f;
            images[img_offset + p * 3 + 2] = static_cast<float>(buffer[1 + 2 * pixels + p]) / 255.0f;
        }
    }
}

} // namespace

cifar10_io::cifar10_io(const cifar10_param_t& params, sim::output_id_t output_id) :
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
        // Load test batch
        image_count = samples_per_batch;
        images.resize(image_count * image_pixels);
        labels.resize(image_count);

        std::string test_file = (fs::path(params.data_dir) / "test_batch.bin").string();
        std::size_t offset = 0;
        load_batch_file(test_file, images, labels, offset);
    } else {
        // Load training batches (5 batches)
        image_count = 5 * samples_per_batch;
        images.resize(image_count * image_pixels);
        labels.resize(image_count);

        for (int batch = 1; batch <= 5; ++batch) {
            std::string batch_file = (fs::path(params.data_dir) /
                ("data_batch_" + std::to_string(batch) + ".bin")).string();
            std::size_t offset = (batch - 1) * samples_per_batch;
            load_batch_file(batch_file, images, labels, offset);
        }
    }
}

void cifar10_io::set_outp_func(std::function<std::span<float>()> function)
{
    output_mem = std::move(function);
}

void cifar10_io::process()
{
    if (!output_mem) {
        throw std::runtime_error("cifar10_io::process() called before set_outp_func()");
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

sim::output_id_t cifar10_io::get_outp_id() const
{
    return id;
}

std::size_t cifar10_io::get_outp_size() const
{
    return image_pixels;
}

std::string cifar10_io::status_str() const
{
    std::string status { "CIFAR-10 IO" };
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(cur_epoch);
    status += " | idx: " + std::to_string(cur_idx);
    status += " / " + std::to_string(image_count);
    return status;
}

} // namespace cifar10
