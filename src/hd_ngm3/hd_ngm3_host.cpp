#include "hd_ngm3/hd_ngm3_host.h"

#include "hd_ngm3/hd_ngm3_layout.h"
#include "logging/state_checks.h"
#include "logging/state_io.h"

#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>

namespace ngm3 {

void validate_params(const neuron_group_params_t &p, const group_caps_t &caps)
{
    const std::string m(caps.model);
    if (p.neuron_cnt < 2)
        throw std::runtime_error(m + ": neuron_cnt must be >= 2 (a neuron group needs at least two "
                                 "neurons to compete); got " + std::to_string(p.neuron_cnt));
    if (p.proximal_cnt < 1)
        throw std::runtime_error(m + ": proximal_cnt must be >= 1");
    // MAX_SEG_CNT=32 covers max_seg_id up to 31 -> branch_depth up to 4 (the CPU
    // segment scratch and the WGSL shared buffer alike).
    if (p.max_branch_level < 1 || p.max_branch_level > 4)
        throw std::runtime_error(m + ": max_branch_level must be in [1,4]");
    // A 0 divisor in the per-step branch gate ((++input_cnt) % branch_interval)
    // is UB, not a "branching disabled" mode.
    if (p.branch_interval == 0)
        throw std::runtime_error(m + ": branch_interval must be >= 1");
    if (caps.branch_interval_u32 && p.branch_interval > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(m + ": branch_interval must fit in 32 bits; got "
                                 + std::to_string(p.branch_interval));
    // Below 0.01 the initial Poisson mean int(100*tau) is 0 (a precondition
    // violation); at 1 or above no synapse can ever connect. Same range as ngm25.
    if (!(p.permanence_threshold >= 0.01f && p.permanence_threshold < 1.0f))
        throw std::runtime_error(m + ": permanence_threshold must be in [0.01, 1.0); got "
                                 + std::to_string(p.permanence_threshold));
    // Both feed std::uniform_real_distribution<float>(a, b) on the CPU, which
    // requires a <= b and a finite b - a; the GPU refuses the same configs so the
    // two backends agree.
    if (!(p.stochastic_win_thres >= 0.0f && p.stochastic_win_thres <= 1.0f))
        throw std::runtime_error(m + ": stochastic_win_thres must be in [0, 1]; got "
                                 + std::to_string(p.stochastic_win_thres));
    if (const auto [lo, hi] = p.neuron_noise_range; !(lo <= hi && std::isfinite(hi - lo)))
        throw std::runtime_error(m + ": neuron_noise_range must have first <= second and a width "
                                 "second - first that is finite in float; got ["
                                 + std::to_string(lo) + ", " + std::to_string(hi) + "]");
}

void bank_inputs_t::bind(sim::output_id_t id, sim::io_buffer &buf)
{
    for (std::size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == id) bufs[i] = &buf;
}

logging::channel_layout bank_inputs_t::layout() const
{
    logging::channel_layout l;
    for (std::size_t i = 0; i < names.size(); ++i) l.push_back({ names[i], sizes[i] });
    return l;
}

input_routing_t route_inputs(
    const neuron_group_params_t                                  &p,
    const std::unordered_map<std::string, sim::output_channel_t> &input_map,
    const group_caps_t                                           &caps,
    sim::output_id_t                                              group_id)
{
    const std::string m(caps.model);
    input_routing_t out;

    auto add = [&](bank_inputs_t &b, const std::string &name,
                   sim::output_id_t channel_id, std::size_t channel_size) {
        b.offsets.push_back(b.width());
        b.names.push_back(name);
        b.ids  .push_back(channel_id);
        b.sizes.push_back(channel_size);
        b.bufs .push_back(nullptr);
        out.inp_ids.push_back(channel_id);
    };

    for (const auto &route : p.inputs) {
        // The channel record (M.7.5) cannot store an empty name, so every save
        // would fail: refuse it at configuration time.
        if (route.channel_name.empty())
            throw std::runtime_error(std::format(
                "{} neuron group {}: init_io: a {} input route has an empty channel name",
                m, group_id, route.bank == bank_t::apical ? "apical" : "proximal"));
        const auto it = input_map.find(route.channel_name);
        if (it == input_map.cend())
            throw std::runtime_error(m + "::init_io: no source channel '" + route.channel_name + "'");
        const auto [channel_id, channel_size] = it->second;
        if (route.bank == bank_t::apical) {
            if (p.apical_cnt == 0)
                throw std::runtime_error(m + "::init_io: input '" + route.channel_name
                                         + "' routed to apical bank but apical_cnt == 0");
            add(out.apical, route.channel_name, channel_id, channel_size);
        } else {
            add(out.proximal, route.channel_name, channel_id, channel_size);
        }
    }

    if (out.proximal.width() == 0)
        throw std::runtime_error(m + "::init_io: proximal bank has zero total input width");
    // Apical dendrites with no apical input would leave the apical bank empty:
    // the CPU combine step dereferenced a null buffer and the GPU silently output
    // noise only. ngm25 rejects the same config (SPEC.md M.0.6 c).
    if (p.apical_cnt > 0 && out.apical.width() == 0)
        throw std::runtime_error(m + "::init_io: apical_cnt > 0 but no input routed to the apical bank");
    return out;
}

void write_checkpoint_shape(logging::state_writer &w, const checkpoint_shape_t &s)
{
    w.write("neuron_cnt",          static_cast<std::uint64_t>(s.params.neuron_cnt));
    w.write("apical_cnt",          static_cast<std::uint64_t>(s.params.apical_cnt));
    w.write("proximal_cnt",        static_cast<std::uint64_t>(s.params.proximal_cnt));
    w.write("max_branch_level",    static_cast<std::uint64_t>(s.params.max_branch_level));
    w.write("apical_input_size",   static_cast<std::uint64_t>(s.apical_input_size));
    w.write("proximal_input_size", static_cast<std::uint64_t>(s.proximal_input_size));
    logging::write_channel_record(w, "apical_channels",   s.apical.layout());
    logging::write_channel_record(w, "proximal_channels", s.proximal.layout());
}

void check_checkpoint_shape(logging::state_reader &r, const checkpoint_shape_t &s,
                            const std::string &who)
{
    logging::check_shape(r, who, {
        { "neuron_cnt",          s.params.neuron_cnt       },
        { "apical_cnt",          s.params.apical_cnt       },
        { "proximal_cnt",        s.params.proximal_cnt     },
        { "max_branch_level",    s.params.max_branch_level },
        { "apical_input_size",   s.apical_input_size       },
        { "proximal_input_size", s.proximal_input_size     },
    });
    if (r.format_version() >= 4) {
        logging::check_channel_record(r, "apical_channels",   s.apical.layout(),   who);
        logging::check_channel_record(r, "proximal_channels", s.proximal.layout(), who);
    } else {
        r.note_unrecorded_channels(who);
    }
}

std::runtime_error corrupt_checkpoint(const std::string &who, std::string_view what)
{
    return std::runtime_error(std::format("{}: corrupt checkpoint — {}", who, what));
}

staged_bank_t stage_bank(logging::state_reader &r, std::string_view bank_name,
                         const neuron_group_params_t &p, std::size_t d_cnt,
                         std::size_t input_size, const std::string &who)
{
    staged_bank_t b;
    if (d_cnt == 0 || input_size == 0) return b;
    // The shape matched, so this bank was saved: it must be there, and full
    // size in the layout the compute relies on (check_bank_layout checks both).
    if (!r.enter_group(bank_name))
        throw corrupt_checkpoint(who, std::format("{} bank missing", bank_name));
    b.perm  = r.read_float_vec("permanences");
    b.seg   = r.read_uint8_vec("segment_ids");
    b.adapt = r.read_float_vec("adapt_history");
    b.mm    = r.read_float_vec("mismatch");
    r.leave_group();
    std::string why;
    if (!check_bank_layout(b.perm, b.seg, b.adapt, b.mm, p.neuron_cnt, d_cnt,
                           p.max_branch_level, input_size, &why))
        throw corrupt_checkpoint(who, std::format("{} bank: {}", bank_name, why));
    b.present = true;
    return b;
}

std::optional<std::vector<float>>
read_neuron_mx_act(logging::state_reader &r, std::size_t neuron_cnt, const std::string &who)
{
    if (!r.has_field("neuron_mx_act")) return std::nullopt;
    auto mx = r.read_float_vec("neuron_mx_act");
    if (mx.size() != neuron_cnt)
        throw corrupt_checkpoint(who, std::format("neuron_mx_act holds {} values, expected {}",
                                                  mx.size(), neuron_cnt));
    return mx;
}

} // namespace ngm3
