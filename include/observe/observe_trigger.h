//
// Created for the COAST observe system
//

#ifndef COAST_OBSERVE_TRIGGER_H
#define COAST_OBSERVE_TRIGGER_H

#include <cstdint>

namespace observe {

/// Programmatic trigger modes invoked through OBSERVE_TRIGGER.
enum class trigger_mode_t {
    NONE,
    START,
    STOP,
    SINGLE_CAPTURE
};

/// How multiple trigger sources for a group combine.
///
/// The trailing underscores on the enumerators avoid collisions with the
/// `<iso646.h>` keyword-like macros (`and`, `or`) and the C++ alternative
/// tokens, which would otherwise make `combine_mode_t::OR` and ::AND
/// ill-formed in TUs that include `<ciso646>` (still legal in C++23).
enum class combine_mode_t {
    OR_,
    AND_
};

/// Inclusive step range used by config-driven triggers.
/// Both endpoints are included in the active interval.
struct step_range_t {
    std::uint64_t start_step;
    std::uint64_t stop_step;
};

} // namespace observe

#endif // COAST_OBSERVE_TRIGGER_H
