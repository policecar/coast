//
// Shared base for CIFAR binary-batch dataset loaders (cifar10, cifar100).
//
// Both loaders share the image layout (32x32x3 interleaved RGB floats), the
// batch-file format (label bytes, then 1024 R + 1024 G + 1024 B bytes per
// sample), process(), state I/O and the dataset accessors. A subclass only lists
// its batch files and its label byte count (cifar10: 1; cifar100: 2, coarse +
// fine) in its constructor.
//
// @decision DEC-003: The sequencer is initially built with sample_count=0 and rebuilt
// after file loading because the total image count is only known post-load. Any
// index_manifest is parsed at rebuild time (against the real count); its readability
// is checked up front so a bad path fails before the multi-second batch read.
//

#ifndef COAST_CIFAR_IO_BASE_H
#define COAST_CIFAR_IO_BASE_H

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "core/rng/fast_rng.h"
#include "sim_core/io_entity.h"
#include "io/sample_sequencer.h"
#include "io/sequence_params.h"

namespace observe { class observe_accessor; }

namespace dataset {

// Shared base for the two CIFAR loaders. Subclass constructors call
// load_batch_file() for each batch file, then rebuild_sequencer().
class cifar_io_base : public sim::io_entity {
protected:
    static constexpr std::size_t image_dim      = 32;
    static constexpr std::size_t image_channels = 3;
    static constexpr std::size_t image_pixels   = image_dim * image_dim * image_channels;

    std::vector<float>   images;
    std::size_t          image_count = 0;

    sim::io_buffer*  output_buf {nullptr};
    sim::output_id_t id;
    dataset::sample_sequencer seq_;
    fast_rng_t rgen;
    std::uniform_real_distribution<float> dis;

    // The loader's sequencing params, retained so rebuild_sequencer() can rebuild
    // seq_ from the real image count and save/load_state reach the run binding
    // (dataset_run.h).
    sequence_params_t seq_params_;

    // Snapshot of seq_.current_index() taken in process() before seq_.advance(),
    // read back in perform_observations (the post-process pass runs after
    // process(), by which point seq_ has already advanced).
    std::uint64_t observed_index_ {0};

    // Shown in status_str(), e.g. "CIFAR-10".
    const char* name_;

    // Construct common state. The sequencer starts at sample_count=0 (image_count
    // is not yet known); call rebuild_sequencer() after loading the batch files
    // to set the real count and parse any index_manifest. Throws on inverted
    // noise_range or an unreadable index_manifest.
    cifar_io_base(sim::output_id_t output_id,
                  const sequence_params_t& seq_params,
                  const char* name);

    // Read one batch file of exactly `count` samples into images starting at
    // sample `first`; images must already be sized for it. The `label_bytes`
    // leading each record are skipped (nothing reads labels, see #211). Throws
    // if the file cannot be opened, is short (e.g. an interrupted download) or
    // holds more than `count` samples.
    void load_batch_file(const std::string& path, std::size_t label_bytes,
                         std::size_t first, std::size_t count);

    // Rebuild the sequencer with the real image count, reusing the sequencing
    // params captured at construction. Must be called at the end of each concrete
    // subclass constructor, once the image count is known.
    void rebuild_sequencer(std::size_t count);

public:
    // io_entity implementation — shared by both CIFAR loaders.
    sim::step_result process() override;

    [[nodiscard]] std::string status_str() const override;

    void set_output_buffer(sim::io_buffer& buf) override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;

    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    // Persists {seq_, rng, identity record} — identical between cifar10 and
    // cifar100. load_state restores only when dataset::restores_sequence says so.
    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override {
        return seq_params_.random_seed;
    }

    // Emit input/index for the observe post-process pass.
    void perform_observations(observe::observe_accessor& acc) override;

    // dataset_io interface (see gui_core/dataset_access.cpp)
    [[nodiscard]] std::tuple<int,int,int> get_image_info() const {
        return { static_cast<int>(image_dim),
                 static_cast<int>(image_dim),
                 static_cast<int>(image_channels) };
    }
    [[nodiscard]] std::size_t sample_count() const { return image_count; }
    [[nodiscard]] std::span<const float> get_sample(std::size_t idx) const {
        if (idx >= image_count) idx = image_count - 1;
        return std::span(images.data() + idx * image_pixels, image_pixels);
    }
};

} // namespace dataset

#endif // COAST_CIFAR_IO_BASE_H
