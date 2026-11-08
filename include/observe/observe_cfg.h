//
// Typed configuration schema for the OBSERVE system.
//
// Mirrors the JSON `"observe"` section. Held as a first-class field of
// `loader::global_cfg_t` so it round-trips through the config editor's
// Save / Save As path alongside datasets, ngm2, sim, and vis_hints.
//

#ifndef COAST_OBSERVE_CFG_H
#define COAST_OBSERVE_CFG_H

#include "observe/observe_trigger.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace observe {

// ---- JSON bindings for the trigger primitives -----------------------------
//
// Schema:
//   step_range_t   <->  [start, stop]              (2-element JSON array)
//   combine_mode_t <->  "or" | "and"               (JSON string, case-insensitive)
//
// Templated on the nlohmann basic_json instantiation so they apply to both
// `nlohmann::json` (std::map-backed) and `nlohmann::ordered_json`
// (ordered_map-backed) — `loader::global_cfg_t` uses the latter for stable
// key ordering on save.
//
// Placed in `namespace observe` so ADL finds them when nlohmann serialises a
// step_range_t or combine_mode_t (e.g. inside the group_cfg_t / observe_cfg_t
// macros below).

template <typename BasicJsonType>
inline void to_json(BasicJsonType& j, const step_range_t& r) {
    j = BasicJsonType::array({r.start_step, r.stop_step});
}
template <typename BasicJsonType>
inline void from_json(const BasicJsonType& j, step_range_t& r) {
    if (!j.is_array() || j.size() != 2) {
        throw std::runtime_error(
            "observe: step_range_t expects a 2-element [start, stop] array");
    }
    r.start_step = j.at(0).template get<std::uint64_t>();
    r.stop_step  = j.at(1).template get<std::uint64_t>();
}

template <typename BasicJsonType>
inline void to_json(BasicJsonType& j, const combine_mode_t& m) {
    j = (m == combine_mode_t::AND_) ? "and" : "or";
}
template <typename BasicJsonType>
inline void from_json(const BasicJsonType& j, combine_mode_t& m) {
    auto s = j.template get<std::string>();
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if      (s == "and") m = combine_mode_t::AND_;
    else if (s == "or")  m = combine_mode_t::OR_;
    else {
        throw std::runtime_error(
            "observe: combine_mode_t expects \"or\" or \"and\" (got '" + s + "')");
    }
}

// ---- Group-level config ---------------------------------------------------

struct group_cfg_t {
    std::vector<step_range_t> ranges;
    combine_mode_t            combine = combine_mode_t::OR_;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(group_cfg_t,
    ranges,
    combine)

// ---- Top-level observe section --------------------------------------------

struct observe_cfg_t {
    // Per-config output directory LEAF. Empty ("" = "defer to the app default")
    // means the app_config's paths.observe_dir leaf applies, resolved under
    // config_base (coast::paths::app_config / coast::paths::resolve_config_output).
    // A non-empty value wins; an absolute value bypasses config_base entirely.
    std::string                          output_dir;
    std::map<std::string, group_cfg_t>   groups;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(observe_cfg_t,
    output_dir,
    groups)

} // namespace observe

#endif // COAST_OBSERVE_CFG_H
