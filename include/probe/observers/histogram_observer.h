//
// Created for the COAST probe system
//

#ifndef COAST_HISTOGRAM_OBSERVER_H
#define COAST_HISTOGRAM_OBSERVER_H

#include "probe/probe_observer.h"
#include <vector>
#include <cstddef>

namespace probe {

/// Observer that displays value distribution as a histogram.
class histogram_observer : public probe_observer {
    std::vector<std::size_t> bins_;
    std::size_t num_bins_;
    double min_val_;
    double max_val_;
    double bin_width_;
    std::size_t total_count_{0};

    // Auto-ranging mode
    bool auto_range_{true};
    std::vector<double> pending_values_;
    static constexpr std::size_t AUTO_RANGE_SAMPLES = 100;

public:
    explicit histogram_observer(std::size_t num_bins = 50);
    histogram_observer(std::size_t num_bins, double min_val, double max_val);

    void push(std::uint64_t step, double value) override;
    void render() override;
    [[nodiscard]] std::string_view name() const override { return "Histogram"; }

    void clear();
    void set_range(double min_val, double max_val);

    [[nodiscard]] std::size_t total_count() const { return total_count_; }
    [[nodiscard]] double min_value() const { return min_val_; }
    [[nodiscard]] double max_value() const { return max_val_; }

private:
    void finalize_range();
    void add_to_histogram(double value);
};

} // namespace probe

#endif // COAST_HISTOGRAM_OBSERVER_H
