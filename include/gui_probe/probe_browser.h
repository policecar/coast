//
// Created for the COAST probe system
//

#ifndef COAST_PROBE_BROWSER_H
#define COAST_PROBE_BROWSER_H

#include <string>
#include <vector>
#include <memory>
#include <imgui.h>
#include <imfilebrowser.h>

namespace sim {
class env;
struct io_entity;
}

namespace core {
struct structure_entry;
}

namespace probe {
struct probe_instance;
class entity_probe_set;
class probe_observer;
}

namespace gui_probe {

/// Represents an observer window that can be independently positioned.
struct observer_window {
    probe::probe_instance* probe;
    std::size_t observer_index;  // Index into probe->observers
    std::string window_title;
    bool open{true};
    bool size_set{false};        // Whether initial size has been applied
    ImVec2 initial_size{300, 200};
};

/// Tree view browser for discovering and activating probes.
/// Also manages individual observer windows.
class probe_browser {
    sim::env* env_{nullptr};
    bool visible_{false};

    // Pending probe for observer type selection popup
    probe::probe_instance* pending_probe_{nullptr};
    bool show_observer_popup_{false};

    // Individual observer windows
    std::vector<observer_window> observer_windows_;
    int next_window_id_{0};

    // File browser for CSV file observer
    ImGui::FileBrowser csv_file_browser_{ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir};
    probe::probe_instance* csv_pending_probe_{nullptr};

public:
    probe_browser() = default;
    explicit probe_browser(sim::env* env);

    void set_env(sim::env* env);

    /// Render the probe browser and all observer windows.
    void render();

    /// Update observers (drain values from queues).
    void update();

    void show() { visible_ = true; }
    void hide() { visible_ = false; }
    void toggle() { visible_ = !visible_; }
    [[nodiscard]] bool is_visible() const { return visible_; }

private:
    void render_entity(sim::io_entity* entity);
    void render_structure_tree(core::structure_entry* entry, probe::entity_probe_set& probes);
    void render_probe_point(core::structure_entry* entry, const std::string& probe_name,
                            probe::entity_probe_set& probes);
    void render_observer_popup();
    void render_observer_windows();
    void add_observer_window(probe::probe_instance* probe, std::size_t obs_index);
    void cleanup_closed_windows();
};

} // namespace gui_probe

#endif // COAST_PROBE_BROWSER_H
