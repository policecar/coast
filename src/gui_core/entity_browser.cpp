#include "entity_browser.h"
#include "sim_instance.h"
#include "imgui.h"

namespace coast {

entity_browser::entity_browser(sim_instance& sim)
    : sim(sim)
{
}

entity_browser::~entity_browser()
{
    free_resources();
}

void entity_browser::render()
{
    if (!is_visible) return;

    render_browser_window();
}

void entity_browser::free_resources()
{
    // Future: free visualization resources
}

void entity_browser::render_browser_window()
{
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Sim Entities", &is_visible)) {
        ImGui::TextDisabled("(Placeholder - future implementation)");
    }
    ImGui::End();
}

} // namespace coast
