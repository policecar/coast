//
// Created for the COAST probe system
//

#ifndef COAST_PROBE_OBSERVER_H
#define COAST_PROBE_OBSERVER_H

#include <string_view>

namespace probe {

/// Base interface for probe observers that visualize probe values.
class probe_observer {
public:
    virtual ~probe_observer() = default;

    /// Receive a new value (called during update_observers from GUI thread).
    virtual void push(double value) = 0;

    /// Render the observer UI (called during GUI rendering).
    virtual void render() = 0;

    /// Get the observer type name for display.
    [[nodiscard]] virtual std::string_view name() const = 0;
};

} // namespace probe

#endif // COAST_PROBE_OBSERVER_H
