//
// Shared implementation for IDX-format dataset loaders.
// See include/io/idx_io_base.h for design notes.
//

#include "io/idx_io_base.h"
#include "logging/state_io.h"
#include "logging/rng_state_io.h"
#include "observe/observe_accessor.h"

#include <format>
#include <stdexcept>

namespace dataset {

idx_io_base::idx_io_base(const std::string& image_file,
                         const std::string& label_file,
                         sim::output_id_t   output_id,
                         const sequence_params_t& seq_params)
    : mnist::mnist_db(image_file, label_file),
      id(output_id),
      seq_(build_sequencer(seq_params, get_image_cnt())),
      rgen(std::to_underlying(seq_params.random_seed)),
      dis(static_cast<float>(seq_params.noise_range.first),
          static_cast<float>(seq_params.noise_range.second)),
      seq_params_(seq_params)
{
    if (seq_params.noise_range.first > seq_params.noise_range.second) {
        throw std::runtime_error("noise_range[0] must be <= noise_range[1]");
    }
}

void idx_io_base::set_output_buffer(sim::io_buffer& buf)
{
    output_buf = &buf;
}

sim::output_id_t idx_io_base::get_outp_id() const
{
    return id;
}

std::size_t idx_io_base::get_outp_size() const
{
    return get_image_size();
}

void idx_io_base::save_state(logging::state_writer& writer) const
{
    save_sequenced_state(seq_params_, writer, seq_, rgen);
}

void idx_io_base::load_state(logging::state_reader& reader)
{
    (void)restore_sequenced_state(seq_params_, reader, seq_, rgen,
                                  std::format("dataset {}", get_outp_id()));
}

void idx_io_base::perform_observations(observe::observe_accessor& acc)
{
    if (acc.group_active("input"))
        acc.observe(*this, "input", "index", observed_index_);
}

} // namespace dataset
