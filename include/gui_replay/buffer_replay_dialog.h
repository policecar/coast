//
// Buffer Replay Dialog - play back recorded buffer activity
//

#ifndef COAST_BUFFER_REPLAY_DIALOG_H
#define COAST_BUFFER_REPLAY_DIALOG_H

#include "buffer_log_reader.h"
#include "replay_node_renderer.h"
#include "gui_layout/layout_graph.h"
#include "gui_layout/force_layout.h"
#include "gui_layout/render/graph_canvas.h"

#include "imgui.h"
#include "imfilebrowser.h"

#include <memory>
#include <chrono>
#include <filesystem>

namespace coast::replay {

/// Playback state
enum class playback_state {
    stopped,
    playing,
    paused
};

/// Buffer replay dialog with force-directed graph visualization
class buffer_replay_dialog {
public:
    buffer_replay_dialog();
    ~buffer_replay_dialog();

    buffer_replay_dialog(const buffer_replay_dialog&) = delete;
    buffer_replay_dialog& operator=(const buffer_replay_dialog&) = delete;

    /// Render the dialog
    /// @param open Pointer to bool controlling visibility
    void render(bool* open);

    /// Check if a file is loaded
    [[nodiscard]] bool is_loaded() const { return reader_.is_open(); }

    /// Get the loaded file path
    [[nodiscard]] std::filesystem::path loaded_path() const {
        return reader_.path();
    }

    /// Open a file directly
    bool open_file(const std::filesystem::path& path);

private:
    // HDF5 reader
    buffer_log_reader reader_;

    // Graph layout
    layout::layout_graph graph_;
    std::unique_ptr<layout::force_layout> layout_;
    layout::render::graph_canvas canvas_;
    replay_node_renderer node_renderer_;

    // File browser
    ImGui::FileBrowser file_browser_;
    bool show_file_browser_ = false;

    // Playback state
    playback_state state_ = playback_state::stopped;
    std::size_t current_frame_ = 0;
    float playback_speed_ = 30.0f;  // Frames per second
    std::chrono::steady_clock::time_point last_frame_time_;

    // Configuration
    bool auto_layout_ = true;
    bool show_config_panel_ = true;
    bool show_info_panel_ = true;
    bool needs_fit_to_view_ = false;  // Flag to fit graph after first render

    // Internal methods
    void render_menu_bar();
    void render_toolbar();
    void render_graph_view();
    void render_config_panel();
    void render_info_panel();
    void render_timeline();
    void render_file_browser();

    void build_graph_from_topology();
    void update_frame_data();
    void step_playback();
    void go_to_frame(std::size_t frame);

    void on_file_loaded();
    void reset_state();
};

} // namespace coast::replay

#endif // COAST_BUFFER_REPLAY_DIALOG_H
