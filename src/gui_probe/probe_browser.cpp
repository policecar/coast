//
// Created for the COAST probe system
//

#include "gui_probe/probe_browser.h"
#include "gui_core/imgui_scaling.h"
#include "sim_core/sim_env.h"
#include "sim_core/io_entity.h"
#include "core/structure_registry.h"
#include "core/probe_name_registry.h"
#include "probe/entity_probe_set.h"
#include "probe/probe_instance.h"
#include "probe/observers/scalar_observer.h"
#include "probe/observers/time_series_observer.h"
#include "probe/observers/histogram_observer.h"
#include <imgui.h>
#include <algorithm>
#include <set>

using coast::scaled;

namespace gui_probe {

probe_browser::probe_browser(sim::env* env)
    : env_(env)
{}

void probe_browser::set_env(sim::env* env) {
    env_ = env;
    // Clear observer windows when environment changes
    observer_windows_.clear();
    pending_probe_ = nullptr;
}

void probe_browser::update() {
    if (!env_) return;

    // Drain probe values to observers for all entities
    for (auto& entity : env_->iterate_entities()) {
        entity.probes().update_observers();
    }
}

void probe_browser::render() {
    // Always render observer windows, even if browser is hidden
    render_observer_windows();
    cleanup_closed_windows();

    if (!visible_ || !env_) return;

    ImGui::SetNextWindowSize(ImVec2(450, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Probe Browser", &visible_)) {
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("Check to activate. Set decimation. Click [+] to add observer.");
    ImGui::Separator();

    // Iterate over all entities
    for (auto& entity : env_->iterate_entities()) {
        render_entity(&entity);
    }

    ImGui::End();

    // Render observer type popup
    render_observer_popup();
}

void probe_browser::render_entity(sim::io_entity* entity) {
    if (!entity) return;

    const char* type_name = entity->type_id().name();
    std::size_t entity_id = entity->get_outp_id();

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

    // Collect probe names from both sources:
    // 1. Static declarations (ELEMENT_PROBES macro) - stored in entry->probe_names
    // 2. Runtime registrations (PROBE macro) - stored in probe_name_registry
    std::set<std::string> all_probe_names;

    // Add statically declared probes
    for (const auto& name : entry->probe_names) {
        all_probe_names.insert(name);
    }

    // Add dynamically registered probes from the registry
    if (entry->element) {
        auto registered = core::probe_name_registry::get_probes(entry->element->type_id().info());
        all_probe_names.merge(registered);
    }

    // Render probe points for this entry
    for (const auto& probe_name : all_probe_names) {
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
    auto* instance = probes.discover(entry->element, probe_name);
    if (!instance) return;

    bool is_active = instance->active.load(std::memory_order_relaxed);
    std::size_t observer_count = instance->observers.size();

    ImGui::PushID(probe_name.c_str());

    // Checkbox for activation
    bool checkbox_value = is_active;
    if (ImGui::Checkbox("##active", &checkbox_value)) {
        if (checkbox_value && !is_active) {
            instance->activate();
        } else if (!checkbox_value && is_active) {
            instance->deactivate();
            // Remove observer windows for this probe
            observer_windows_.erase(
                std::remove_if(observer_windows_.begin(), observer_windows_.end(),
                    [instance](const observer_window& w) { return w.probe == instance; }),
                observer_windows_.end());
        }
    }

    ImGui::SameLine();

    // Probe name
    if (is_active) {
        ImGui::Text("%s", probe_name.c_str());
    } else {
        ImGui::TextDisabled("%s", probe_name.c_str());
    }

    // Only show controls for active probes
    if (is_active) {
        // Decimation control with label
        ImGui::SameLine();
        ImGui::SetNextItemWidth(scaled(70));
        ImGui::TextDisabled("decimation");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(scaled(50));
        int dec = static_cast<int>(instance->decimation.load());
        if (ImGui::InputInt("##dec", &dec, 0, 0)) {
            if (dec < 1) dec = 1;
            probe::entity_probe_set::set_decimation(instance, static_cast<std::uint32_t>(dec));
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Decimation ratio (1:N means record every Nth sample)");
        }

        // Add observer button with label
        ImGui::SameLine();
        if (ImGui::SmallButton("+ Obs")) {
            pending_probe_ = instance;
            show_observer_popup_ = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Add observer window");
        }

        // Show observer count
        if (observer_count > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu)", observer_count);
        }
    }

    ImGui::PopID();
}

void probe_browser::render_observer_popup() {
    if (show_observer_popup_) {
        ImGui::OpenPopup("AddObserver");
        show_observer_popup_ = false;
    }

    if (ImGui::BeginPopup("AddObserver")) {
        if (!pending_probe_) {
            ImGui::EndPopup();
            return;
        }

        ImGui::Text("Add Observer:");
        ImGui::Separator();

        if (ImGui::Selectable("Scalar")) {
            pending_probe_->add_observer(std::make_unique<probe::scalar_observer>());
            add_observer_window(pending_probe_, pending_probe_->observers.size() - 1);
            pending_probe_ = nullptr;
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::Selectable("Time Series")) {
            pending_probe_->add_observer(std::make_unique<probe::time_series_observer>());
            add_observer_window(pending_probe_, pending_probe_->observers.size() - 1);
            pending_probe_ = nullptr;
            ImGui::CloseCurrentPopup();
        }

        if (ImGui::Selectable("Histogram")) {
            pending_probe_->add_observer(std::make_unique<probe::histogram_observer>());
            add_observer_window(pending_probe_, pending_probe_->observers.size() - 1);
            pending_probe_ = nullptr;
            ImGui::CloseCurrentPopup();
        }

        // Log to File option (only shown if callback is set)
        if (log_probe_callback_) {
            ImGui::Separator();
            if (ImGui::Selectable("Log to File")) {
                log_probe_callback_(pending_probe_);
                pending_probe_ = nullptr;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Add this probe to the active file logger.\nEnable logging in Introspection > Logging...");
            }
        }

        ImGui::EndPopup();
    }
}

void probe_browser::add_observer_window(probe::probe_instance* probe, std::size_t obs_index) {
    observer_window win;
    win.probe = probe;
    win.observer_index = obs_index;

    auto& observer = probe->observers[obs_index];
    std::string_view obs_name = observer->name();

    win.window_title = probe->full_path + " - " +
                       std::string(obs_name) +
                       "##" + std::to_string(next_window_id_++);
    win.open = true;

    // Set initial size based on observer type (scaled for HiDPI)
    if (obs_name == "Scalar") {
        win.initial_size = ImVec2(scaled(250), scaled(150));
    } else if (obs_name == "Time Series") {
        win.initial_size = ImVec2(scaled(500), scaled(200));
    } else if (obs_name == "Histogram") {
        win.initial_size = ImVec2(scaled(450), scaled(225));
    } else {
        win.initial_size = ImVec2(scaled(300), scaled(200));
    }

    observer_windows_.push_back(win);
}

void probe_browser::render_observer_windows() {
    for (auto& win : observer_windows_) {
        if (!win.open) continue;
        if (!win.probe) continue;
        if (win.observer_index >= win.probe->observers.size()) {
            win.open = false;
            continue;
        }

        auto& observer = win.probe->observers[win.observer_index];

        // Set initial size only once (use Appearing to ensure it's applied)
        if (!win.size_set) {
            ImGui::SetNextWindowSize(win.initial_size, ImGuiCond_Appearing);
            win.size_set = true;
        }

        if (ImGui::Begin(win.window_title.c_str(), &win.open)) {
            observer->render();
        }
        ImGui::End();
    }
}

void probe_browser::cleanup_closed_windows() {
    // Remove closed windows and their observers
    for (auto it = observer_windows_.begin(); it != observer_windows_.end(); ) {
        if (!it->open) {
            // Remove the observer from the probe
            if (it->probe && it->observer_index < it->probe->observers.size()) {
                it->probe->observers.erase(
                    it->probe->observers.begin() +
                    static_cast<std::ptrdiff_t>(it->observer_index));

                // Update indices for remaining windows pointing to same probe
                for (auto& other : observer_windows_) {
                    if (other.probe == it->probe && other.observer_index > it->observer_index) {
                        --other.observer_index;
                    }
                }

                // If no observers left, deactivate the probe
                if (it->probe->observers.empty()) {
                    it->probe->deactivate();
                }
            }
            it = observer_windows_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace gui_probe
