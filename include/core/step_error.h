//
// Hot-path error payload for COAST's value-based (std::expected) error channel.
//
// See design/error_handling_plan.md. The hot per-step path reports errors as
// std::expected<void, core::step_error> rather than by throwing, so a violated
// invariant surfaces as a value the controller can halt on cleanly instead of an
// exception unwinding out of a TBB worker / the sim thread.
//

#ifndef COAST_STEP_ERROR_H
#define COAST_STEP_ERROR_H

#include <cstddef>
#include <cstdint>
#include <exception>

namespace core {

/// Category of a hot-path error reported through std::expected<…, step_error>.
enum class error_kind : std::uint8_t {
    contract_violation,  ///< a hot-path invariant (a former assert) was violated
    non_finite,          ///< a computed value was NaN or inf
    device_fault,        ///< a GPU device-scoped fault (not tied to one entity)
    partial_load,        ///< a runtime state load failed after applying part of the
                         ///< checkpoint (not a hot-path error; halts the same way)
    exception,           ///< an exception escaped an entity's process() or a step
                         ///< hook; its message is logged where it was caught
};

/// Hot-path error payload. Constructed only on the cold (failure) branch and
/// returned via std::expected<void, step_error> out of process()/swap_io().
///
/// Deliberately lean — the step index is NOT stored here; the controller stamps
/// it at the halt boundary (it owns the step counter). `entity_id` is stamped by
/// env::process() during the post-fan-out reduction; device-scoped faults that do
/// not map to a single entity use `no_entity`.
struct step_error {
    static constexpr std::size_t no_entity = static_cast<std::size_t>(-1);

    error_kind  kind      = error_kind::contract_violation;
    std::size_t entity_id = no_entity;
    const char* site      = "";  ///< COAST_SITE literal ("file:line"), static storage
};

/// Human-readable name for a kind (diagnostics only).
[[nodiscard]] constexpr const char* to_string(error_kind k) noexcept {
    switch (k) {
        case error_kind::contract_violation: return "contract_violation";
        case error_kind::non_finite:         return "non_finite";
        case error_kind::device_fault:       return "device_fault";
        case error_kind::partial_load:       return "partial_load";
        case error_kind::exception:          return "exception";
    }
    return "unknown";
}

/// The message of the exception being handled. Call only inside a catch handler:
/// it rethrows the current exception to read it. The pointer stays valid until
/// that handler exits.
[[nodiscard]] inline const char* current_exception_what() noexcept {
    try {
        throw;
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "unknown exception (not a std::exception)";
    }
}

}  // namespace core

// COAST_SITE expands to a "file:line" string literal at the point of use (static
// storage, zero runtime cost). The double-macro indirection forces __LINE__ to be
// expanded to its numeric value before stringization.
#define COAST_DETAIL_STRINGIFY2(x) #x
#define COAST_DETAIL_STRINGIFY(x)  COAST_DETAIL_STRINGIFY2(x)
#define COAST_SITE (__FILE__ ":" COAST_DETAIL_STRINGIFY(__LINE__))

#endif  // COAST_STEP_ERROR_H
