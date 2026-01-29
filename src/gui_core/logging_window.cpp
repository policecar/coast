#include "logging_window.h"
#include "imgui.h"

namespace coast {

void logging_window::render()
{
    if (!is_visible) return;

    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(200, 200), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Logging", &is_visible)) {
        ImGui::TextDisabled("(Placeholder - future implementation)");
    }
    ImGui::End();
}

} // namespace coast
