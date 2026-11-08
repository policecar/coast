//
// Hard checks a group's load_state runs on a checkpoint before it adopts any of
// it. Unlike the configuration record (config_record.h), which compares settings
// and warns, these throw: a checkpoint that does not fit the group cannot be
// loaded into it, and continuing from the constructed state would make the run
// claim a resume it never did.
//
// The rule for the payload, once the shape matches:
//   - A field absent from the checkpoint is a pre-field checkpoint, written
//     before the format gained that field: the group keeps its constructed
//     value. This holds only for fields added after checkpoints already existed
//     (e.g. ngm3's neuron_mx_act and the rng streams, both added 2026-06-21
//     in dd12721f and 98122325).
//   - A field present but of the wrong size, or absent although the format has
//     always carried it, is a corrupt checkpoint: load_state throws
//     "<group>: corrupt checkpoint — <what>".
//

#ifndef COAST_STATE_CHECKS_H
#define COAST_STATE_CHECKS_H

#include <cstdint>
#include <string_view>
#include <vector>

namespace logging {

class state_reader;

// One structural field of a group's state: its name and the value this group
// was built with. The checkpoint's value is read by check_shape.
struct shape_field {
    std::string_view name;
    std::uint64_t    built;
};

// Reads each field from the current group of `r` and throws std::runtime_error
//   "<who>: checkpoint does not fit this group — neuron_cnt 5 (checkpoint) vs 3 (configured), dend_count missing"
// listing every field that differs or is absent. Returns when all match. Takes
// a vector (not an initializer_list) so a caller whose field set depends on
// the checkpoint's format version can build it and still check everything in
// one call, reporting every mismatch together.
void check_shape(state_reader& r, std::string_view who, const std::vector<shape_field>& fields);

// Takes the groups noted on `r` as having no recorded dendrite types
// (state_reader::note_unchecked_dendrite_types) and reports them in one notice;
// nothing if there are none. A load calls it once, after its entity loop, beside
// report_unrecorded_channels.
void report_unchecked_dendrite_types(state_reader& r);

} // namespace logging

#endif // COAST_STATE_CHECKS_H
