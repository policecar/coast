//
// Created for the COAST probe system
//

#ifndef COAST_TIME_SERIES_OBSERVER_H
#define COAST_TIME_SERIES_OBSERVER_H

#include "probe/probe_observer.h"
#include <vector>
#include <cstddef>

namespace probe {

/// Observer that displays values over time as a line chart.
class time_series_observer : public probe_observer {
    std::vector<double> buffer_;
    std::size_t head_{0};
    std::size_t count_{0};
    std::size_t capacity_;

    // Optional exponential moving average
    bool use_ema_{false};
    float ema_alpha_{0.1f};
    double ema_value_{0.0};
    std::vector<double> ema_buffer_;

public:
    explicit time_series_observer(std::size_t capacity = 1000);

    void push(double value) override;
    void render() override;
    [[nodiscard]] std::string_view name() const override { return "Time Series"; }

    void set_ema(bool enable, float alpha = 0.1f);
    void clear();

    [[nodiscard]] std::size_t capacity() const { return capacity_; }
    [[nodiscard]] std::size_t size() const { return count_; }
};

} // namespace probe

#endif // COAST_TIME_SERIES_OBSERVER_H
