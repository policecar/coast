//
// Apply a typed observe::observe_cfg_t to the registry.
//
// The schema validation is now handled by nlohmann/json itself in the
// from_json overloads (observe_cfg.h) — by the time we reach this code,
// the cfg struct is well-formed. The remaining job is purely to seed the
// registry: set the output dir, create each named group, and copy the
// ranges + combine mode.
//

#include "observe/observe_registry.h"
#include "observe/observe_group.h"

#include <filesystem>

namespace observe {

void apply_config(observe_registry& reg, const observe_cfg_t& cfg) {
    // output_dir is set by attach() to the caller-resolved directory (config's
    // observe leaf under config_base, or an absolute override); apply_config no
    // longer re-derives it from cfg.output_dir — doing so would discard the
    // resolution. apply_config's job is purely to seed the groups below.

    // Seed each named group. get_or_create_group returns a reference under
    // the registry mutex; we may mutate ranges + combine here because
    // apply_config runs from attach() before any sim step has executed.
    for (const auto& [name, gcfg] : cfg.groups) {
        observe_group_t& g = reg.get_or_create_group(name);
        g.ranges  = gcfg.ranges;
        g.combine = gcfg.combine;
    }
}

} // namespace observe
