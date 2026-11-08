//
// Hard checks on a checkpoint before load_state adopts it — implementation.
//

#include "logging/state_checks.h"
#include "logging/state_io.h"
#include "diag/diag.h"

#include <format>
#include <stdexcept>
#include <string>

namespace logging {

void check_shape(state_reader& r, std::string_view who, const std::vector<shape_field>& fields)
{
    std::string diffs;
    for (const auto& f : fields) {
        std::string d;
        if (!r.has_field(f.name)) {
            d = std::format("{} missing", f.name);
        } else if (const auto saved = r.read_uint64(f.name, 0); saved != f.built) {
            d = std::format("{} {} (checkpoint) vs {} (configured)", f.name, saved, f.built);
        } else {
            continue;
        }
        diffs += diffs.empty() ? d : ", " + d;
    }
    if (!diffs.empty())
        throw std::runtime_error(std::format("{}: checkpoint does not fit this group — {}", who, diffs));
}

void report_unchecked_dendrite_types(state_reader& r)
{
    const auto unchecked = r.take_unchecked_dendrite_types();
    if (unchecked.empty()) return;
    std::string list;
    for (const auto& u : unchecked) list += (list.empty() ? "" : ", ") + u;
    COAST_INFO("checkpoint", "the checkpoint does not record the dendrite types (apical/proximal) ({}); "
                             "they were not checked against the configuration", list);
}

} // namespace logging
