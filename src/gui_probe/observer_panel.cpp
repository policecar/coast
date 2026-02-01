//
// Created for the COAST probe system
//

#include "gui_probe/observer_panel.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "probe/entity_probe_set.h"
#include "probe/probe_instance.h"
#include "probe/probe_observer.h"
#include <imgui.h>
#include <vector>

namespace gui_probe {

observer_panel::observer_panel(sim::env* env)
    : env_(env)
{}

void observer_panel::update() {
    if (!env_) return;

    // Drain probe values to observers for all entities
    for (auto& entity : env_->iterate_entities()) {
        entity.probes().update_observers();
    }
}

void observer_panel::render() {
    if (!visible_ || !env_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Probe Observers", &visible_)) {
        ImGui::End();
        return;
    }

    // Collect all active probes with observers
    std::vector<probe::probe_instance*> active_probes;
    for (auto& entity : env_->iterate_entities()) {
        auto instances = entity.probes().all_instances();
        for (auto* inst : instances) {
            if (inst->active.load(std::memory_order_relaxed) && !inst->observers.empty()) {
                active_probes.push_back(inst);
            }
        }
    }

    if (active_probes.empty()) {
        ImGui::TextDisabled("No active probes with observers.");
        ImGui::TextDisabled("Use the Probe Browser to activate probes.");
        ImGui::End();
        return;
    }

    // Render each active probe's observers
    for (auto* probe : active_probes) {
        ImGui::PushID(probe);

        bool open = ImGui::CollapsingHeader(probe->full_path.c_str(),
                                            ImGuiTreeNodeFlags_DefaultOpen);

        if (open) {
            ImGui::Indent();

            // Show decimation and status
            ImGui::TextDisabled("Decimation: %ux | Observers: %zu",
                               probe->decimation.load(),
                               probe->observers.size());

            // Render each observer
            std::size_t obs_idx = 0;
            std::vector<std::size_t> to_remove;

            for (auto& observer : probe->observers) {
                ImGui::PushID(static_cast<int>(obs_idx));

                // Observer header with remove button
                bool obs_open = ImGui::TreeNodeEx(observer->name().data(),
                                                  ImGuiTreeNodeFlags_DefaultOpen);

                ImGui::SameLine();
                if (ImGui::SmallButton("X")) {
                    to_remove.push_back(obs_idx);
                }

                if (obs_open) {
                    observer->render();
                    ImGui::TreePop();
                }

                ImGui::PopID();
                ++obs_idx;
            }

            // Remove marked observers (in reverse order to preserve indices)
            for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) {
                probe->observers.erase(probe->observers.begin() +
                                       static_cast<std::ptrdiff_t>(*it));
            }

            // If no observers left, deactivate probe
            if (probe->observers.empty()) {
                probe->deactivate();
            }

            ImGui::Unindent();
        }

        ImGui::PopID();
    }

    ImGui::End();
}

} // namespace gui_probe
