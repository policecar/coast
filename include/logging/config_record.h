//
// Configuration record of a checkpoint (SPEC.md M.7.8).
//
// On load_state the configuration wins: a group's settings always come from the
// configuration it was built with, never from the checkpoint. So that a changed
// configuration is not continued blindly, each group writes the resolved
// parameters it was built with into a `config` subgroup of its state, and on load
// compares that record with its current parameters. Every difference is reported
// in one warning per group, naming each setting with its recorded and current
// value. Seeds are state, not settings: the checkpoint's random state wins, so a
// differing seed is reported as not applied.
//
// A checkpoint without a record (written before records existed) loads with a
// single notice. Each group without a record is noted on the reader it was loaded
// from (state_reader::note_unrecorded_config), and the load that owns the reader
// reports them all at once after its entity loop (report_unrecorded_configs).
//
// Channel record (SPEC.md M.7.5): the channels a group's synapses are laid out
// by, in order. Unlike a setting, a channel record that differs refuses the load.
//

#ifndef COAST_CONFIG_RECORD_H
#define COAST_CONFIG_RECORD_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace logging {

class state_writer;
class state_reader;

// float is kept apart from double so a float setting prints as written (0.0001,
// not its double widening). Strings must be non-empty: an empty string reads back
// as "not recorded".
using config_value = std::variant<float, double, std::int64_t, std::uint64_t, bool, std::string>;

struct config_entry {
    std::string  name;
    config_value value;
    bool         random_state = false;   // a seed: the checkpoint's random state wins
};

using config_record = std::vector<config_entry>;

// Writes `rec` as the `config` subgroup of the current group.
void write_config_record(state_writer& w, const config_record& rec);

// One line per entry of `current` whose recorded value differs or is not recorded,
// e.g. "common_learning_rate 0.0001 -> 0.001", then one per entry only the record
// has, e.g. "neuron[n]#5.branch_interval 7000 -> (not in the current configuration)".
// nullopt if the current group has no `config` subgroup.
[[nodiscard]] std::optional<std::vector<std::string>>
diff_config_record(state_reader& r, const config_record& current);

// Compares and reports: one warning listing every difference, nothing when the
// record matches. When there is no record, `who` is noted on `r` for
// report_unrecorded_configs instead. `who` names the group in the message.
void check_config_record(state_reader& r, const config_record& current, std::string_view who);

// One input channel of a synapse layout: its configured name and its width.
struct channel_entry {
    std::string   name;
    std::uint64_t width = 0;

    friend bool operator==(const channel_entry&, const channel_entry&) = default;
};
// A synapse layout's channels in declared order.
using channel_layout = std::vector<channel_entry>;

// Writes `layout` as subgroup `name` of the current group: `count`, `widths`
// (absent when count is 0) and one string attribute per channel, "0", "1", ...,
// holding its name. Throws std::invalid_argument for an empty name, which would
// read back as missing.
void write_channel_record(state_writer& w, std::string_view name, const channel_layout& layout);

// Reads subgroup `name` of the current group of `r` and throws std::runtime_error
//   "<who>: checkpoint does not fit this group — <name> [c0 (20), c1 (28)] (checkpoint) vs [c1 (28), c0 (20)] (configured)"
// when it records another layout than `current`, or
//   "<who>: corrupt checkpoint — <name> missing" (or malformed)
// when it is absent or incomplete. Callers run it on format version 4 and later
// only, and note older checkpoints with state_reader::note_unrecorded_channels.
void check_channel_record(state_reader& r, std::string_view name, const channel_layout& current,
                          std::string_view who);

// Takes the groups noted on `r` as having no channel record and reports them in
// one notice; nothing if there are none. A load calls it once, after its entity
// loop, beside report_unrecorded_configs.
void report_unrecorded_channels(state_reader& r);

// Takes the groups noted on `r` as having no record and reports them in one
// notice; nothing if there are none. A load calls it once, after its entity loop.
void report_unrecorded_configs(state_reader& r);

} // namespace logging

#endif // COAST_CONFIG_RECORD_H
