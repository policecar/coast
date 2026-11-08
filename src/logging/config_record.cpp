//
// Configuration record of a checkpoint — implementation (SPEC.md M.7.8), and
// the channel record beside it (SPEC.md M.7.5).
//

#include "logging/config_record.h"
#include "logging/state_io.h"
#include "diag/diag.h"

#include <format>
#include <set>
#include <span>
#include <stdexcept>

namespace logging {

namespace {

constexpr std::string_view k_group = "config";

template <class Variant>
std::string to_text(const Variant& v)
{
    return std::visit([](const auto& x) { return std::format("{}", x); }, v);
}

// The recorded value of `e`, read as the type of its current value; nullopt if
// the record has no such entry.
std::optional<config_value> read_entry(state_reader& r, const config_entry& e)
{
    return std::visit(
        [&](const auto& cur) -> std::optional<config_value> {
            using T = std::decay_t<decltype(cur)>;
            if constexpr (std::is_same_v<T, std::string>) {
                auto s = r.read_attr(e.name);
                if (s.empty()) return std::nullopt;
                return config_value{ std::move(s) };
            } else {
                if (!r.has_field(e.name)) return std::nullopt;
                if constexpr (std::is_same_v<T, float>)         return config_value{ r.read_float(e.name) };
                if constexpr (std::is_same_v<T, double>)        return config_value{ r.read_double(e.name) };
                if constexpr (std::is_same_v<T, std::int64_t>)  return config_value{ r.read_int64(e.name) };
                if constexpr (std::is_same_v<T, std::uint64_t>) return config_value{ r.read_uint64(e.name) };
                if constexpr (std::is_same_v<T, bool>)          return config_value{ r.read_bool(e.name) };
            }
        },
        e.value);
}

} // namespace

void write_config_record(state_writer& w, const config_record& rec)
{
    w.begin_group(k_group);
    for (const auto& e : rec)
        std::visit(
            [&](const auto& x) {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, std::string>) w.write_attr(e.name, x);
                else                                          w.write(e.name, x);
            },
            e.value);
    w.end_group();
}

std::optional<std::vector<std::string>>
diff_config_record(state_reader& r, const config_record& current)
{
    if (!r.enter_group(k_group)) return std::nullopt;
    std::vector<std::string> diffs;
    for (const auto& e : current) {
        const auto rec = read_entry(r, e);
        if (rec && *rec == e.value) continue;
        std::string line = std::format("{} {} -> {}", e.name,
                                       rec ? to_text(*rec) : std::string("(not recorded)"),
                                       to_text(e.value));
        if (e.random_state) line += " (not applied: the random state comes from the checkpoint)";
        diffs.push_back(std::move(line));
    }
    // Entries only the record has: settings the current configuration does not
    // record, e.g. an instance that differed from its cfg when saved and does not now.
    std::set<std::string_view> names;
    for (const auto& e : current) names.insert(e.name);
    for (const auto& [name, value] : r.read_scalars())
        if (!names.contains(name))
            diffs.push_back(std::format("{} {} -> (not in the current configuration)", name, to_text(value)));
    r.leave_group();
    return diffs;
}

void check_config_record(state_reader& r, const config_record& current, std::string_view who)
{
    const auto diffs = diff_config_record(r, current);
    if (!diffs) {
        r.note_unrecorded_config(who);
        return;
    }
    if (diffs->empty()) return;
    std::string list;
    for (const auto& d : *diffs) list += (list.empty() ? "" : "; ") + d;
    COAST_WARN("checkpoint", "{}: the checkpoint was saved with other settings; the current "
                             "configuration applies: {}", who, list);
}

namespace {

// "[c0 (20), c1 (28)]"
std::string layout_text(const channel_layout& layout)
{
    std::string s;
    for (const auto& c : layout) s += std::format("{}{} ({})", s.empty() ? "" : ", ", c.name, c.width);
    return "[" + s + "]";
}

std::string join(const std::vector<std::string>& v)
{
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
    return s;
}

} // namespace

void write_channel_record(state_writer& w, std::string_view name, const channel_layout& layout)
{
    std::vector<std::uint64_t> widths;
    for (const auto& c : layout) {
        if (c.name.empty())
            throw std::invalid_argument(std::format("channel record {}: a channel has an empty name", name));
        widths.push_back(c.width);
    }
    w.begin_group(name);
    w.write("count", static_cast<std::uint64_t>(layout.size()));
    if (!widths.empty()) w.write("widths", std::span<const std::uint64_t>(widths));
    for (std::size_t i = 0; i < layout.size(); ++i) w.write_attr(std::to_string(i), layout[i].name);
    w.end_group();
}

void check_channel_record(state_reader& r, std::string_view name, const channel_layout& current,
                          std::string_view who)
{
    auto corrupt = [&](std::string_view what) {
        return std::runtime_error(std::format("{}: corrupt checkpoint — {} {}", who, name, what));
    };
    // Read it whole, leave the group, then judge: every throw leaves the reader
    // where the caller had it. The reads themselves can throw too (a field of
    // the wrong rank, e.g. "widths" written as a group), so they are wrapped to
    // leave the group on the way out, same as the judging below.
    if (!r.enter_group(name)) throw corrupt("missing");
    bool                        has_count = false;
    std::uint64_t               count     = 0;
    std::vector<std::uint64_t>  widths;
    channel_layout              saved;
    try {
        has_count = r.has_field("count");
        count     = r.read_uint64("count", 0);
        if (count > 0 && r.has_field("widths")) widths = r.read_uint64_vec("widths");
        for (std::size_t i = 0; i < widths.size() && i < count; ++i)
            saved.push_back({ r.read_attr(std::to_string(i)), widths[i] });
    } catch (...) {
        r.leave_group();
        throw;
    }
    r.leave_group();

    if (!has_count) throw corrupt("has no count");
    if (widths.size() != count)
        throw corrupt(std::format("records {} channels but {} widths", count, widths.size()));
    for (std::size_t i = 0; i < saved.size(); ++i)
        if (saved[i].name.empty()) throw corrupt(std::format("has no name for channel {}", i));

    if (saved != current)
        throw std::runtime_error(std::format("{}: checkpoint does not fit this group — {} {} (checkpoint) "
                                             "vs {} (configured)", who, name, layout_text(saved), layout_text(current)));
}

void report_unrecorded_configs(state_reader& r)
{
    const auto missing = r.take_unrecorded_configs();
    if (missing.empty()) return;
    COAST_INFO("checkpoint", "the checkpoint has no configuration record ({}); settings were "
                             "not compared", join(missing));
}

void report_unrecorded_channels(state_reader& r)
{
    const auto unrecorded = r.take_unrecorded_channels();
    if (unrecorded.empty()) return;
    COAST_INFO("checkpoint", "the checkpoint does not record the channel order ({}); it was not "
                             "checked against the configuration", join(unrecorded));
}

} // namespace logging
