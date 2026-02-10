//
// CSV File Observer implementation
//

#include "probe/observers/csv_file_observer.h"
#include <imgui.h>
#include <filesystem>
#include <iomanip>

namespace probe {

csv_file_observer::csv_file_observer(const std::string& file_path)
    : file_path_(file_path)
{
    // Check if file exists and prompt would have been shown already
    // Just try to open for writing (truncate)
    file_.open(file_path_, std::ios::out | std::ios::trunc);

    if (file_.is_open()) {
        // Write CSV header
        file_ << "step,value\n";
        file_.flush();
    } else {
        error_message_ = "Failed to open file for writing";
    }
}

csv_file_observer::~csv_file_observer() {
    if (file_.is_open()) {
        file_.close();
    }
}

void csv_file_observer::push(std::uint64_t step, double value) {
    if (!file_.is_open()) return;

    // Write step and value
    file_ << step << ","
          << std::setprecision(9) << value << "\n";

    ++sample_count_;

    // Flush periodically to ensure data is written
    if (sample_count_ % 100 == 0) {
        file_.flush();
    }
}

void csv_file_observer::render() {
    // Show file path (just filename, not full path for brevity)
    std::filesystem::path p(file_path_);
    ImGui::Text("File: %s", p.filename().string().c_str());

    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", file_path_.c_str());
    }

    // Show status
    if (file_.is_open()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Recording");
        ImGui::SameLine();
        ImGui::Text("| Samples: %lu", static_cast<unsigned long>(sample_count_));
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s",
                          error_message_.c_str());
    }
}

} // namespace probe
