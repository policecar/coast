//
// Created for the COAST probe system
//

#ifndef COAST_PROBE_OBSERVER_H
#define COAST_PROBE_OBSERVER_H

#include <string_view>
#include <cstdint>

namespace probe {

/// Base interface for probe observers that visualize probe values.
class probe_observer {
public:
    virtual ~probe_observer() = default;

    /// Receive a new timestamped value (called during update_observers from GUI thread).
    /// @param step The simulation step when this value was recorded
    /// @param value The probe value
    virtual void push(std::uint64_t step, double value) = 0;

    /// Render the observer UI (called during GUI rendering).
    virtual void render() = 0;

    /// Get the observer type name for display.
    [[nodiscard]] virtual std::string_view name() const = 0;
};

} // namespace probe

#endif // COAST_PROBE_OBSERVER_H
