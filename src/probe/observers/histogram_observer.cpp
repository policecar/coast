//
// Created for the COAST probe system
//

#include "probe/observers/histogram_observer.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>

namespace probe {

histogram_observer::histogram_observer(std::size_t num_bins)
    : bins_(num_bins, 0)
    , num_bins_(num_bins)
    , min_val_(0.0)
    , max_val_(1.0)
    , bin_width_(1.0 / static_cast<double>(num_bins))
    , auto_range_(true)
{}

histogram_observer::histogram_observer(std::size_t num_bins, double min_val, double max_val)
    : bins_(num_bins, 0)
    , num_bins_(num_bins)
    , min_val_(min_val)
    , max_val_(max_val)
    , bin_width_((max_val - min_val) / static_cast<double>(num_bins))
    , auto_range_(false)
{}

void histogram_observer::push(double value) {
    if (auto_range_) {
        pending_values_.push_back(value);
        if (pending_values_.size() >= AUTO_RANGE_SAMPLES) {
            finalize_range();
        }
        return;
    }

    add_to_histogram(value);
}

void histogram_observer::add_to_histogram(double value) {
    if (value < min_val_ || value > max_val_) {
        // Clamp to range
        value = std::clamp(value, min_val_, max_val_);
    }

    std::size_t bin = static_cast<std::size_t>((value - min_val_) / bin_width_);
    bin = std::min(bin, num_bins_ - 1);
    ++bins_[bin];
    ++total_count_;
}

void histogram_observer::finalize_range() {
    if (pending_values_.empty()) return;

    auto [min_it, max_it] = std::minmax_element(pending_values_.begin(),
                                                 pending_values_.end());
    double margin = (*max_it - *min_it) * 0.1;
    if (margin < 1e-10) margin = 0.5;  // Handle case where all values are the same

    min_val_ = *min_it - margin;
    max_val_ = *max_it + margin;
    bin_width_ = (max_val_ - min_val_) / static_cast<double>(num_bins_);
    auto_range_ = false;

    // Add all pending values to histogram
    for (double v : pending_values_) {
        add_to_histogram(v);
    }
    pending_values_.clear();
    pending_values_.shrink_to_fit();
}

void histogram_observer::render() {
    ImGui::Text("Count: %zu", total_count_);

    if (auto_range_) {
        ImGui::Text("Collecting samples for auto-range: %zu/%zu",
                    pending_values_.size(), AUTO_RANGE_SAMPLES);
    } else {
        ImGui::Text("Range: [%.4f, %.4f]", min_val_, max_val_);
    }

    if (ImGui::Button("Clear")) {
        clear();
    }

    // Plot histogram
    if (total_count_ > 0 && !auto_range_ && ImPlot::BeginPlot("##histogram", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Value", "Count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        // Convert bins to double for plotting
        std::vector<double> bin_centers(num_bins_);
        std::vector<double> bin_counts(num_bins_);
        for (std::size_t i = 0; i < num_bins_; ++i) {
            bin_centers[i] = min_val_ + (static_cast<double>(i) + 0.5) * bin_width_;
            bin_counts[i] = static_cast<double>(bins_[i]);
        }

        ImPlot::PlotBars("##bars", bin_centers.data(), bin_counts.data(),
                         static_cast<int>(num_bins_), bin_width_ * 0.9);

        ImPlot::EndPlot();
    }
}

void histogram_observer::clear() {
    std::fill(bins_.begin(), bins_.end(), 0);
    total_count_ = 0;
    auto_range_ = true;
    pending_values_.clear();
}

void histogram_observer::set_range(double min_val, double max_val) {
    min_val_ = min_val;
    max_val_ = max_val;
    bin_width_ = (max_val - min_val) / static_cast<double>(num_bins_);
    auto_range_ = false;
    std::fill(bins_.begin(), bins_.end(), 0);
    total_count_ = 0;
}

} // namespace probe
