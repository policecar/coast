//
// Shared base for IDX-format dataset loaders (mnist, emnist, kmnist, color_mnist).
//
// Each family keeps its own concrete class (mnist::mnist_io, emnist::emnist_io,
// kmnist::kmnist_io, color_mnist::color_mnist_io) with its existing ELEMENT_TYPE
// and namespace. This base eliminates the identical member declarations and the
// five method bodies (set_output_buffer, get_outp_id, get_outp_size, save_state,
// load_state) that were copy-pasted verbatim across all four loaders.
//
// @decision DEC-001: process() and status_str() are NOT implemented here.
// @rationale: color_mnist::process() uses a different output layout (3-channel) and RNG draw
// pattern. Shared logic would require a virtual hook, adding complexity. Since
// process() already differs in observable ways, each subclass implements it directly.
//
// @decision DEC-002: get_outp_size() defaults to get_image_size() (correct for the
// grayscale trio). color_mnist_io overrides it to return 28*28*3 (RGB output).
//
// @decision DEC-003: save_state/load_state write {seq_, rng} only — correct for
// mnist/emnist/kmnist. color_mnist_io overrides both to also persist current_color.
// load_state restores only when the load continues the run that wrote the
// checkpoint (same run_id, dataset_run.h); otherwise the sequencer cursor and noise
// RNG are left as they are.
//

#ifndef COAST_IDX_IO_BASE_H
#define COAST_IDX_IO_BASE_H

#include <random>
#include <string>

#include "core/rng/fast_rng.h"
#include "sim_core/io_entity.h"
#include "io/mnist_db.h"
#include "io/sample_sequencer.h"
#include "io/sequence_params.h"

namespace observe { class observe_accessor; }

namespace dataset {

// Base class shared by the four IDX-format loaders.
// Inherits both io_entity (the entity interface) and mnist::mnist_db (IDX file reader).
// Subclasses add ELEMENT_TYPE, their concrete params member, and implement process()
// and status_str().
class idx_io_base : public sim::io_entity, public mnist::mnist_db {
protected:
    sim::io_buffer*  output_buf {nullptr};
    sim::output_id_t id;
    dataset::sample_sequencer seq_;
    fast_rng_t rgen;
    std::uniform_real_distribution<float> dis;

    // The loader's sequencing params, retained so save/load_state reach the run
    // binding (dataset_run.h). Subclass param structs inherit from
    // sequence_params_t, so they pass their params directly.
    sequence_params_t seq_params_;

    // Snapshot of seq_.current_index() taken in each subclass's process() before
    // seq_.advance(), read back in perform_observations (the post-process pass
    // runs after process(), by which point seq_ has already advanced).
    std::uint64_t observed_index_ {0};

    // Construct the shared state. image_file / label_file go to mnist_db;
    // seq_params drives the sequencer and the input-noise distribution. Throws
    // on inverted noise_range or a missing/invalid index_manifest.
    idx_io_base(const std::string& image_file,
                const std::string& label_file,
                sim::output_id_t   output_id,
                const sequence_params_t& seq_params);

public:
    // io_entity implementation — shared by all four IDX loaders.
    void set_output_buffer(sim::io_buffer& buf) override;

    [[nodiscard]] sim::output_id_t get_outp_id() const override;

    // Returns get_image_size() — correct for the three grayscale loaders (28x28x1).
    // color_mnist_io overrides this to return 28*28*3.
    [[nodiscard]] std::size_t get_outp_size() const override;

    [[nodiscard]] std::span<const std::size_t> get_inp_ids() const override { return {}; }

    // Persists {seq_, rng} — the shared state for mnist/emnist/kmnist.
    // color_mnist_io overrides both to additionally save current_color.
    void save_state(logging::state_writer& writer) const override;
    void load_state(logging::state_reader& reader) override;

    [[nodiscard]] std::optional<seed_t> effective_seed() const override {
        return seq_params_.random_seed;
    }

    // Emit input/index for the observe post-process pass; shared by all four
    // IDX loaders (each snapshots observed_index_ in its own process()).
    void perform_observations(observe::observe_accessor& acc) override;
};

} // namespace dataset

#endif // COAST_IDX_IO_BASE_H
