#include "buffer_browser.h"
#include "sim_instance.h"
#include "imgui.h"

namespace coast {

buffer_browser::buffer_browser(sim_instance& sim)
    : sim(sim)
{
}

buffer_browser::~buffer_browser()
{
    free_resources();
}

void buffer_browser::render()
{
    if (!is_visible) return;

    render_browser_window();
}

void buffer_browser::free_resources()
{
    // Future: free ring buffer visualization resources
}

void buffer_browser::render_browser_window()
{
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(150, 150), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("IO Buffers", &is_visible)) {
        ImGui::TextDisabled("(Placeholder - future implementation)");
    }
    ImGui::End();
}

} // namespace coast
