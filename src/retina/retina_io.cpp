//
// Retina node implementation
//

#include "retina/retina_io.h"

#include <algorithm>
#include <format>
#include <stdexcept>

#include "core/require.h"
#include "logging/config_record.h"
#include "logging/rng_state_io.h"
#include "logging/state_io.h"

namespace retina {

logging::config_record config_record(const retina_param_t& p)
{
    return {
        { "width",           std::int64_t{ p.width } },
        { "height",          std::int64_t{ p.height } },
        { "channels",        std::int64_t{ p.channels } },
        { "luminance",       p.luminance },
        { "polarity",        std::string(to_string(p.polarity)) },
        { "center_sigma",    p.center_sigma },
        { "surround_sigma",  p.surround_sigma },
        { "surround_weight", p.surround_weight },
        { "gain",            p.gain },
        { "normalize",       p.normalize },
        { "norm_sigma",      p.norm_sigma },
        { "noise_range.lo",  p.noise_range.first },
        { "noise_range.hi",  p.noise_range.second },
        { "effective_seed",  std::uint64_t{ std::to_underlying(p.random_seed) }, /*random_state=*/true },
    };
}

retina_io::retina_io(const retina_param_t& params, sim::output_id_t output_id) :
    params_(params),
    filter_(params),   // validates
    id_(output_id),
    d_(output_size(params)),
    rgen_(std::to_underlying(params.random_seed)),
    noise_(static_cast<float>(params.noise_range.first), static_cast<float>(params.noise_range.second))
{
}

void retina_io::init_io(const std::unordered_map<std::string, sim::output_channel_t>& input_map)
{
    init_structure();
    if (input_map.size() != 1)
        throw std::runtime_error("retina: needs exactly one input, got " + std::to_string(input_map.size()));
    const auto& [src_id, src_size] = input_map.begin()->second;
    if (src_size != input_size(params_))
        throw std::runtime_error("retina: input size " + std::to_string(src_size) + " does not match width*height*channels = "
                                 + std::to_string(input_size(params_)));
    inp_ids_ = {src_id};
}

void retina_io::set_input_buffer(sim::output_id_t id, sim::io_buffer& buf)
{
    if (inp_ids_.empty() || id != inp_ids_.front())
        throw std::runtime_error("retina: set_input_buffer for unknown input id " + std::to_string(id));
    input_buf_ = &buf;
}

sim::step_result retina_io::process()
{
    COAST_REQUIRE(input_buf_ != nullptr && output_buf_ != nullptr, core::error_kind::contract_violation);
    const auto in  = input_buf_->cur_read_buffer();
    auto       out = output_buf_->cur_write_buffer();
    COAST_REQUIRE(in.size() == input_size(params_) && out.size() == d_.size(), core::error_kind::contract_violation);

    filter_.apply(in, d_);
    polarize(d_, params_.polarity, out);
    for (auto& v : out)
        v = std::min(v + noise_(rgen_), 1.0f);
    return {};
}

std::string retina_io::status_str() const
{
    return "Retina " + std::string(to_string(params_.polarity))
         + " | id: " + std::to_string(id_)
         + " | k: " + std::to_string(params_.surround_weight);
}

void retina_io::save_state(logging::state_writer& writer) const
{
    logging::write_rng(writer, "rng", rgen_);
    logging::write_config_record(writer, config_record(params_));
}

void retina_io::load_state(logging::state_reader& reader)
{
    const auto who = std::format("retina node {}", id_);
    logging::check_config_record(reader, config_record(params_), who);
    // The retina format has always carried its rng group, so a missing one is a
    // corrupt checkpoint (state_checks.h), not a pre-field one; read_rng throws
    // for an incomplete one.
    if (!logging::read_rng(reader, "rng", rgen_, who))
        throw std::runtime_error(std::format("{}: corrupt checkpoint — rng missing", who));
}

}
