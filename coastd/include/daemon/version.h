#ifndef COAST_COASTD_VERSION_H
#define COAST_COASTD_VERSION_H

#include <string_view>

namespace coastd {

// The daemon's own version string. Reported by `coastd --version`, stamped into
// each job's provenance, and served in machine_info. Bump on wire/behaviour
// changes that a client might need to detect.
inline constexpr std::string_view version = "0.1.0-dev";

} // namespace coastd

#endif // COAST_COASTD_VERSION_H
