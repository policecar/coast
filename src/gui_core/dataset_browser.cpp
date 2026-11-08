#include "gui_core/dataset_browser.h"
#include "sim_core/sim_instance.h"
#include "imgui.h"

namespace coast {

dataset_browser::dataset_browser(sim_instance& sim)
    : sim(sim)
{
    rebuild_dataset_list();
}

dataset_browser::~dataset_browser()
{
    free_resources();
}

void dataset_browser::render()
{
    if (!is_visible) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Dataset Browser", &is_visible)) {
        if (datasets.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No datasets loaded in current simulation.");
        } else {
            ImGui::SetNextItemWidth(200);
            const char* preview = (selected_dataset_idx >= 0)
                ? datasets[selected_dataset_idx].name.c_str() : "Select...";
            if (ImGui::BeginCombo("Dataset", preview)) {
                for (int i = 0; i < static_cast<int>(datasets.size()); ++i) {
                    bool is_selected = (selected_dataset_idx == i);
                    if (ImGui::Selectable(datasets[i].name.c_str(), is_selected)) {
                        if (selected_dataset_idx != i) {
                            selected_dataset_idx = i;
                            grid.free_resources();  // re-texture for the new dataset
                        }
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            if (selected_dataset_idx >= 0) {
                const auto& ds = datasets[selected_dataset_idx];
                ImGui::SameLine();
                ImGui::Text("| %dx%d", ds.width, ds.height);
                ImGui::SameLine();
                ImGui::Text("| %d ch", ds.channels);
                ImGui::SameLine();
                ImGui::Text("| %zu samples", ds.sample_count);

                ImGui::Separator();

                grid.render(ds, nullptr);  // view-only
            }
        }
    }
    ImGui::End();
}

void dataset_browser::free_resources()
{
    grid.free_resources();
}

void dataset_browser::rebuild_dataset_list()
{
    datasets = enumerate_datasets(sim.get_sim_env(), sim.get_builder());
    selected_dataset_idx = datasets.empty() ? -1 : 0;  // auto-select first
}

} // namespace coast
