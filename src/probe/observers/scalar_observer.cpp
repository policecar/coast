//
// Created for the COAST probe system
//

#include "probe/observers/scalar_observer.h"
#include <imgui.h>
#include <algorithm>

namespace probe {

void scalar_observer::push(std::uint64_t step, double value) {
    last_step_ = step;
    current_ = value;
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
    sum_ += value;
    ++count_;
}

void scalar_observer::render() {
    ImGui::Text("Current: %.6f", current_);
    ImGui::Text("Min:     %.6f", min_);
    ImGui::Text("Max:     %.6f", max_);
    ImGui::Text("Mean:    %.6f", mean());
    ImGui::Text("Count:   %zu", count_);

    if (ImGui::Button("Reset")) {
        reset();
    }
}

void scalar_observer::reset() {
    current_ = 0.0;
    min_ = std::numeric_limits<double>::max();
    max_ = std::numeric_limits<double>::lowest();
    sum_ = 0.0;
    count_ = 0;
}

} // namespace probe
