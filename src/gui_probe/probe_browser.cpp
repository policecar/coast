//
// Created for the COAST probe system
//

#include "gui_probe/probe_browser.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "core/structure_registry.h"
#include "probe/entity_probe_set.h"
#include "probe/probe_instance.h"
#include "probe/observers/scalar_observer.h"
#include "probe/observers/time_series_observer.h"
#include "probe/observers/histogram_observer.h"
#include <imgui.h>

namespace gui_probe {

probe_browser::probe_browser(sim::env* env)
    : env_(env)
{}

void probe_browser::render() {
    if (!visible_ || !env_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Probe Browser", &visible_)) {
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("Expand entities to discover probes. Click a probe to activate.");
    ImGui::Separator();

    // Iterate over all entities
    for (auto& entity : env_->iterate_entities()) {
        render_entity(&entity);
    }

    ImGui::End();

    // Render activation popup if needed
    render_activation_popup();
}

void probe_browser::render_entity(sim::io_entity* entity) {
    if (!entity) return;

    // Get entity type name
    const char* type_name = entity->type_id().name();
    std::size_t entity_id = entity->get_outp_id();

    // Create tree node for entity
    ImGui::PushID(static_cast<int>(entity_id));
    bool open = ImGui::TreeNode("", "%s [%zu]", type_name, entity_id);

    if (open) {
        auto& probes = entity->probes();
        auto* structure_root = entity->structure().root();

        if (structure_root) {
            render_structure_tree(structure_root, probes);
        } else {
            ImGui::TextDisabled("(structure not initialized)");
        }

        ImGui::TreePop();
    }
    ImGui::PopID();
}

void probe_browser::render_structure_tree(core::structure_entry* entry,
                                           probe::entity_probe_set& probes) {
    if (!entry) return;

    // Render probe points for this entry
    for (const auto& probe_name : entry->probe_names) {
        render_probe_point(entry, probe_name, probes);
    }

    // Render children
    for (auto& child : entry->children) {
        ImGui::PushID(static_cast<int>(child->index));

        bool has_probes = !child->probe_names.empty() || !child->children.empty();
        if (has_probes) {
            bool open = ImGui::TreeNode("", "%s[%zu]", child->type.name(), child->index);
            if (open) {
                render_structure_tree(child.get(), probes);
                ImGui::TreePop();
            }
        }

        ImGui::PopID();
    }
}

void probe_browser::render_probe_point(core::structure_entry* entry,
                                        const std::string& probe_name,
                                        probe::entity_probe_set& probes) {
    // Find or create probe instance (for discovery)
    auto* instance = probes.activate(entry->element, probe_name, 1);
    if (instance) {
        // Immediately deactivate if it was just created - we only want discovery
        if (instance->observers.empty() && !instance->active.load()) {
            // Already inactive, just discovered
        }
    }

    // Get all instances to find this one
    auto all_instances = probes.all_instances();
    probe::probe_instance* found = nullptr;
    for (auto* inst : all_instances) {
        if (inst->element == entry->element && inst->probe_name == probe_name) {
            found = inst;
            break;
        }
    }

    if (!found) return;

    bool is_active = found->active.load(std::memory_order_relaxed);
    std::size_t observer_count = found->observers.size();

    // Display probe with status
    ImGui::PushID(probe_name.c_str());

    if (is_active) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 1.0f, 0.2f, 1.0f));
    }

    bool clicked = ImGui::Selectable(probe_name.c_str(), selected_probe_ == found);

    if (is_active) {
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu obs, %ux)", observer_count, found->decimation.load());
    }

    if (clicked) {
        selected_probe_ = found;
        decimation_ = static_cast<int>(found->decimation.load());
        ImGui::OpenPopup("ProbeActivation");
    }

    ImGui::PopID();
}

void probe_browser::render_activation_popup() {
    if (ImGui::BeginPopup("ProbeActivation")) {
        if (!selected_probe_) {
            ImGui::EndPopup();
            return;
        }

        ImGui::Text("Probe: %s", selected_probe_->full_path.c_str());
        ImGui::Separator();

        bool is_active = selected_probe_->active.load();

        // Decimation control
        ImGui::SetNextItemWidth(150);
        if (ImGui::InputInt("Decimation", &decimation_)) {
            if (decimation_ < 1) decimation_ = 1;
            probe::entity_probe_set::set_decimation(selected_probe_,
                                                     static_cast<std::uint32_t>(decimation_));
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(1 = every sample)");

        ImGui::Separator();

        // Add observer buttons
        ImGui::Text("Add Observer:");

        if (ImGui::Button("Scalar")) {
            selected_probe_->add_observer(std::make_unique<probe::scalar_observer>());
            selected_probe_->active.store(true, std::memory_order_release);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();

        if (ImGui::Button("Time Series")) {
            selected_probe_->add_observer(std::make_unique<probe::time_series_observer>());
            selected_probe_->active.store(true, std::memory_order_release);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();

        if (ImGui::Button("Histogram")) {
            selected_probe_->add_observer(std::make_unique<probe::histogram_observer>());
            selected_probe_->active.store(true, std::memory_order_release);
            ImGui::CloseCurrentPopup();
        }

        ImGui::Separator();

        // Deactivate button
        if (is_active) {
            if (ImGui::Button("Deactivate")) {
                selected_probe_->deactivate();
                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::EndPopup();
    }
}

} // namespace gui_probe
