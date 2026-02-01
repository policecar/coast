//
// Created for the COAST probe system
//

#include "probe/observers/time_series_observer.h"
#include <imgui.h>
#include <implot.h>

namespace probe {

time_series_observer::time_series_observer(std::size_t capacity)
    : buffer_(capacity, 0.0)
    , capacity_(capacity)
    , ema_buffer_(capacity, 0.0)
{}

void time_series_observer::push(double value) {
    buffer_[head_] = value;

    if (use_ema_) {
        if (count_ == 0) {
            ema_value_ = value;  // Initialize EMA with first value
        } else {
            ema_value_ = ema_alpha_ * value + (1.0 - ema_alpha_) * ema_value_;
        }
        ema_buffer_[head_] = ema_value_;
    }

    head_ = (head_ + 1) % capacity_;
    if (count_ < capacity_) ++count_;
}

void time_series_observer::render() {
    // EMA toggle
    if (ImGui::Checkbox("Show EMA", &use_ema_)) {
        if (use_ema_ && count_ > 0) {
            // Recalculate EMA for existing data
            ema_value_ = buffer_[(head_ + capacity_ - count_) % capacity_];
            for (std::size_t i = 0; i < count_; ++i) {
                std::size_t idx = (head_ + capacity_ - count_ + i) % capacity_;
                ema_value_ = ema_alpha_ * buffer_[idx] + (1.0 - ema_alpha_) * ema_value_;
                ema_buffer_[idx] = ema_value_;
            }
        }
    }

    if (use_ema_) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        if (ImGui::SliderFloat("Alpha", &ema_alpha_, 0.01f, 1.0f)) {
            // Recalculate EMA with new alpha
            if (count_ > 0) {
                ema_value_ = buffer_[(head_ + capacity_ - count_) % capacity_];
                for (std::size_t i = 0; i < count_; ++i) {
                    std::size_t idx = (head_ + capacity_ - count_ + i) % capacity_;
                    ema_value_ = ema_alpha_ * buffer_[idx] + (1.0 - ema_alpha_) * ema_value_;
                    ema_buffer_[idx] = ema_value_;
                }
            }
        }
    }

    if (ImGui::Button("Clear")) {
        clear();
    }

    // Plot
    if (count_ > 0 && ImPlot::BeginPlot("##timeseries", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Sample", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        // We need to handle the circular buffer for plotting
        // ImPlot expects contiguous data, so we'll plot in two parts if wrapped
        std::size_t start = (head_ + capacity_ - count_) % capacity_;

        if (start + count_ <= capacity_) {
            // No wrap - single segment
            ImPlot::PlotLine("Value", buffer_.data() + start, static_cast<int>(count_));
            if (use_ema_) {
                ImPlot::PlotLine("EMA", ema_buffer_.data() + start, static_cast<int>(count_));
            }
        } else {
            // Wrapped - need to copy to temporary buffer
            std::vector<double> temp(count_);
            std::vector<double> temp_ema(count_);
            for (std::size_t i = 0; i < count_; ++i) {
                std::size_t idx = (start + i) % capacity_;
                temp[i] = buffer_[idx];
                if (use_ema_) {
                    temp_ema[i] = ema_buffer_[idx];
                }
            }
            ImPlot::PlotLine("Value", temp.data(), static_cast<int>(count_));
            if (use_ema_) {
                ImPlot::PlotLine("EMA", temp_ema.data(), static_cast<int>(count_));
            }
        }

        ImPlot::EndPlot();
    }
}

void time_series_observer::set_ema(bool enable, float alpha) {
    use_ema_ = enable;
    ema_alpha_ = alpha;
}

void time_series_observer::clear() {
    head_ = 0;
    count_ = 0;
    ema_value_ = 0.0;
}

} // namespace probe
