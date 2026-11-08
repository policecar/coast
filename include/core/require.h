//
// COAST_REQUIRE — always-compiled hot-path contract check.
//
// See design/error_handling_plan.md. Unlike assert() (a no-op under NDEBUG, which
// this project always defines in Release builds), COAST_REQUIRE fires in every
// build. It is the in-place replacement for hot-path asserts: it states the
// assumption at the site that depends on it and reports a violation through the
// value-based error channel instead of being silently compiled out.
//

#ifndef COAST_REQUIRE_H
#define COAST_REQUIRE_H

#include <expected>

#include "core/step_error.h"

/// COAST_REQUIRE(cond, kind): if `cond` is false, return std::unexpected from the
/// *enclosing function*. Therefore only usable inside a function returning
/// std::expected<…, core::step_error>. The success path is a single
/// predicted-not-taken branch; the step_error is built only on the cold branch.
///
/// `entity_id` is left as no_entity here and stamped by env::process() during the
/// post-fan-out reduction (it knows which entity it dispatched). The name is
/// COAST_-prefixed deliberately: a bare REQUIRE would collide with doctest's
/// REQUIRE macro in the integration tests, which include both.
#define COAST_REQUIRE(cond, kind)                                      \
    do {                                                               \
        if (!(cond)) [[unlikely]]                                      \
            return std::unexpected(::core::step_error{                 \
                (kind), ::core::step_error::no_entity, COAST_SITE});   \
    } while (0)

#endif  // COAST_REQUIRE_H
