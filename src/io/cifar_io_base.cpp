//
// Shared implementation for CIFAR binary-batch dataset loaders.
// See include/io/cifar_io_base.h for design notes.
//

#include "io/cifar_io_base.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"
#include "core/require.h"

#include <format>
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace dataset {

cifar_io_base::cifar_io_base(sim::output_id_t output_id,
                             const sequence_params_t& seq_params,
                             const char* name)
    : id(output_id),
      // sample_count=0 because image_count is only known after file loading.
      // Subclass ctors must call rebuild_sequencer() once loading is complete;
      // the index_manifest (if any) is parsed there, against the real count.
      seq_(0, static_cast<int>(seq_params.change_interval), seq_params.shuffle,
           std::to_underlying(seq_params.random_seed)),
      rgen(std::to_underlying(seq_params.random_seed)),
      dis(static_cast<float>(seq_params.noise_range.first),
          static_cast<float>(seq_params.noise_range.second)),
      seq_params_(seq_params),
      name_(name)
{
    if (seq_params.noise_range.first > seq_params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }
    // Surface a bad manifest path before the subclass's multi-second batch read;
    // the manifest is parsed against the real image_count in rebuild_sequencer().
    ensure_manifest_readable(seq_params_.index_manifest);
}

void cifar_io_base::rebuild_sequencer(std::size_t count)
{
    image_count = count;
    seq_ = build_sequencer(seq_params_, count);
}

void cifar_io_base::load_batch_file(const std::string& path,
                                    std::size_t label_bytes,
                                    std::size_t first, std::size_t count)
{
    constexpr std::size_t pixels = image_dim * image_dim;
    const std::size_t bytes_per_sample = label_bytes + pixels * image_channels;
    if ((first + count) * image_pixels > images.size()) {
        throw std::logic_error("cifar_io_base::load_batch_file: images not sized "
                               "for samples [" + std::to_string(first) + ", " +
                               std::to_string(first + count) + ")");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Cannot open ") + name_ +
                                 " batch file: " + path);
    }

    std::vector<std::uint8_t> buffer(bytes_per_sample);
    for (std::size_t i = 0; i < count; ++i) {
        if (!file.read(reinterpret_cast<char*>(buffer.data()),
                       static_cast<std::streamsize>(bytes_per_sample))) {
            throw std::runtime_error(std::string(name_) + " batch file " + path +
                                     " is truncated: holds " + std::to_string(i) +
                                     " complete samples, expected " +
                                     std::to_string(count));
        }
        // Planar (RRR...GGG...BBB) to interleaved (RGBRGB...).
        const std::uint8_t* planes = buffer.data() + label_bytes;
        float* img = images.data() + (first + i) * image_pixels;
        for (std::size_t p = 0; p < pixels; ++p) {
            for (std::size_t c = 0; c < image_channels; ++c) {
                img[p * image_channels + c] =
                    static_cast<float>(planes[c * pixels + p]) / 255.0f;
            }
        }
    }
    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error(std::string(name_) + " batch file " + path +
                                 " holds more than the expected " +
                                 std::to_string(count) + " samples");
    }
}

sim::step_result cifar_io_base::process()
{
    COAST_REQUIRE(output_buf != nullptr, core::error_kind::contract_violation);
    auto outp = output_buf->cur_write_buffer();
    COAST_REQUIRE(outp.size() == image_pixels, core::error_kind::contract_violation);

    if (seq_.change_interval() > 0) {
        const std::size_t cur_idx = seq_.current_index();
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
    observed_index_ = static_cast<std::uint64_t>(seq_.current_index());  // snapshot before seq_ advances; read in perform_observations

    seq_.advance();

    return {};
}

std::string cifar_io_base::status_str() const
{
    std::string status = std::string(name_) + " IO";
    status += " | id: " + std::to_string(get_outp_id());
    status += " | epoch: " + std::to_string(seq_.current_epoch());
    status += " | idx: " + std::to_string(seq_.current_index());
    status += " / " + std::to_string(image_count);
    return status;
}

void cifar_io_base::set_output_buffer(sim::io_buffer& buf)
{
    output_buf = &buf;
}

sim::output_id_t cifar_io_base::get_outp_id() const
{
    return id;
}

std::size_t cifar_io_base::get_outp_size() const
{
    return image_pixels;
}

void cifar_io_base::save_state(logging::state_writer& writer) const
{
    save_sequenced_state(seq_params_, writer, seq_, rgen);
}

void cifar_io_base::load_state(logging::state_reader& reader)
{
    (void)restore_sequenced_state(seq_params_, reader, seq_, rgen,
                                  std::format("dataset {}", get_outp_id()));
}

void cifar_io_base::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("input"))
        acc.observe(*this, "input", "index", observed_index_);
}

} // namespace dataset
