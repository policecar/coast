//
// Created for the COAST probe system
//

#ifndef COAST_SCALAR_OBSERVER_H
#define COAST_SCALAR_OBSERVER_H

#include "probe/probe_observer.h"
#include <limits>
#include <cstddef>

namespace probe {

/// Observer that displays current value with running statistics.
class scalar_observer : public probe_observer {
    double current_{0.0};
    double min_{std::numeric_limits<double>::max()};
    double max_{std::numeric_limits<double>::lowest()};
    double sum_{0.0};
    std::size_t count_{0};

public:
    void push(double value) override;
    void render() override;
    [[nodiscard]] std::string_view name() const override { return "Scalar"; }

    void reset();

    [[nodiscard]] double current() const { return current_; }
    [[nodiscard]] double min() const { return min_; }
    [[nodiscard]] double max() const { return max_; }
    [[nodiscard]] double mean() const { return count_ > 0 ? sum_ / static_cast<double>(count_) : 0.0; }
    [[nodiscard]] std::size_t count() const { return count_; }
};

} // namespace probe

#endif // COAST_SCALAR_OBSERVER_H
