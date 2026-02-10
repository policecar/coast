//
// Buffer Replay Dialog - play back recorded buffer activity
//

#include "gui_replay/buffer_replay_dialog.h"

#include <cstdio>
#include <cmath>
#include <memory>

namespace coast::replay {

buffer_replay_dialog::buffer_replay_dialog() {
    // Set up file browser
    file_browser_.SetTitle("Open Buffer Log");
    file_browser_.SetTypeFilters({".h5", ".hdf5"});

    // Set up canvas
    canvas_.set_node_renderer(&node_renderer_);
    canvas_.set_zoom(1.0f);
    canvas_.set_smooth_animation(true);

    // Configure edge style
    layout::render::edge_style edge_style;
    edge_style.color = IM_COL32(100, 150, 200, 180);
    edge_style.thickness = 2.0f;
    edge_style.show_direction = true;
    edge_style.arrow_size = 10.0f;
    canvas_.set_edge_style(edge_style);
}

buffer_replay_dialog::~buffer_replay_dialog() = default;

void buffer_replay_dialog::render(bool* open) {
    if (!*open) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);

    ImGuiWindowFlags flags = ImGuiWindowFlags_MenuBar;
    if (!ImGui::Begin("Buffer Replay", open, flags)) {
        ImGui::End();
        return;
    }

    render_menu_bar();
    render_toolbar();

    // Main content area
    float panel_width = show_config_panel_ ? 250.0f : 0.0f;
    float timeline_height = 60.0f;

    // Left side: graph view
    ImVec2 available = ImGui::GetContentRegionAvail();
    float graph_width = available.x - panel_width;
    float graph_height = available.y - timeline_height;

    ImGui::BeginChild("GraphArea", ImVec2(graph_width, graph_height), ImGuiChildFlags_None);
    render_graph_view();
    ImGui::EndChild();

    // Right side: config panel
    if (show_config_panel_) {
        ImGui::SameLine();
        ImGui::BeginChild("ConfigPanel", ImVec2(panel_width - 8, graph_height), ImGuiChildFlags_Border);
        render_config_panel();
        if (show_info_panel_) {
            ImGui::Separator();
            render_info_panel();
        }
        ImGui::EndChild();
    }

    // Bottom: timeline
    render_timeline();

    // File browser dialog
    render_file_browser();

    // Handle playback
    if (state_ == playback_state::playing) {
        step_playback();
    }

    // Run layout simulation
    if (layout_ && auto_layout_) {
        layout_->step();
    }

    ImGui::End();
}

bool buffer_replay_dialog::open_file(const std::filesystem::path& path) {
    reset_state();

    if (!reader_.open(path)) {
        return false;
    }

    on_file_loaded();
    return true;
}

void buffer_replay_dialog::render_menu_bar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open...", "Ctrl+O")) {
                show_file_browser_ = true;
                file_browser_.Open();
            }
            if (ImGui::MenuItem("Close", nullptr, false, reader_.is_open())) {
                reset_state();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Config Panel", nullptr, &show_config_panel_);
            ImGui::MenuItem("Info Panel", nullptr, &show_info_panel_);
            ImGui::Separator();
            if (ImGui::MenuItem("Fit Graph", "F")) {
                canvas_.fit_graph_in_view();
            }
            if (ImGui::MenuItem("Center Graph", "C")) {
                canvas_.center_on_graph();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Playback")) {
            if (ImGui::MenuItem("Play/Pause", "Space")) {
                if (state_ == playback_state::playing) {
                    state_ = playback_state::paused;
                } else {
                    state_ = playback_state::playing;
                    last_frame_time_ = std::chrono::steady_clock::now();
                }
            }
            if (ImGui::MenuItem("Stop", "S")) {
                state_ = playback_state::stopped;
                current_frame_ = 0;
                update_frame_data();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Previous Frame", "Left")) {
                if (current_frame_ > 0) {
                    go_to_frame(current_frame_ - 1);
                }
            }
            if (ImGui::MenuItem("Next Frame", "Right")) {
                if (reader_.is_open() && current_frame_ < reader_.frame_count() - 1) {
                    go_to_frame(current_frame_ + 1);
                }
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

void buffer_replay_dialog::render_toolbar() {
    // Playback controls
    bool can_play = reader_.is_open() && reader_.frame_count() > 0;

    ImGui::BeginDisabled(!can_play);

    // Rewind button
    if (ImGui::Button("|<##rewind")) {
        go_to_frame(0);
        state_ = playback_state::paused;
    }
    ImGui::SameLine();

    // Previous frame
    if (ImGui::Button("<##prev")) {
        if (current_frame_ > 0) {
            go_to_frame(current_frame_ - 1);
        }
        state_ = playback_state::paused;
    }
    ImGui::SameLine();

    // Play/Pause button
    if (state_ == playback_state::playing) {
        if (ImGui::Button("||##pause")) {
            state_ = playback_state::paused;
        }
    } else {
        if (ImGui::Button(">##play")) {
            state_ = playback_state::playing;
            last_frame_time_ = std::chrono::steady_clock::now();
        }
    }
    ImGui::SameLine();

    // Next frame
    if (ImGui::Button(">##next")) {
        if (reader_.is_open() && current_frame_ < reader_.frame_count() - 1) {
            go_to_frame(current_frame_ + 1);
        }
        state_ = playback_state::paused;
    }
    ImGui::SameLine();

    // Fast forward
    if (ImGui::Button(">|##end")) {
        if (reader_.is_open()) {
            go_to_frame(reader_.frame_count() - 1);
        }
        state_ = playback_state::paused;
    }

    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    // Speed control
    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("Speed", &playback_speed_, 1.0f, 120.0f, "%.0f fps");

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    // Frame info
    if (reader_.is_open()) {
        ImGui::Text("Frame: %zu / %zu", current_frame_ + 1, reader_.frame_count());
        ImGui::SameLine();
        ImGui::Text("| Step: %lu", static_cast<unsigned long>(reader_.get_step(current_frame_)));
        ImGui::SameLine();
        ImGui::Text("| Time: %.2fs", reader_.get_timestamp(current_frame_));
    } else {
        ImGui::TextDisabled("No file loaded");
    }
}

void buffer_replay_dialog::render_graph_view() {
    if (!reader_.is_open()) {
        ImVec2 available = ImGui::GetContentRegionAvail();
        ImVec2 center = ImVec2(
            ImGui::GetCursorScreenPos().x + available.x * 0.5f,
            ImGui::GetCursorScreenPos().y + available.y * 0.5f
        );

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        const char* text = "Open a buffer_log file to begin";
        ImVec2 text_size = ImGui::CalcTextSize(text);
        draw_list->AddText(
            ImVec2(center.x - text_size.x * 0.5f, center.y - text_size.y * 0.5f),
            IM_COL32(150, 150, 150, 200),
            text
        );
        return;
    }

    canvas_.set_graph(&graph_);
    canvas_.render("##graph_canvas");

    // Fit graph to view after first render (when canvas size is known)
    if (needs_fit_to_view_) {
        canvas_.fit_graph_in_view();
        needs_fit_to_view_ = false;
    }
}

void buffer_replay_dialog::render_config_panel() {
    ImGui::Text("Visualization");
    ImGui::Separator();

    // Vis mode
    static const char* mode_names[] = {"2D Image", "Stats Color", "Delta Flow"};
    int mode = static_cast<int>(node_renderer_.global_config().mode);
    if (ImGui::Combo("Mode", &mode, mode_names, 3)) {
        vis_config config = node_renderer_.global_config();
        config.mode = static_cast<vis_mode>(mode);
        node_renderer_.set_global_config(config);
        node_renderer_.apply_global_config_to_nodes(graph_);
    }

    // Colormap
    static const char* colormap_names[] = {"Grayscale", "Green", "Viridis", "Plasma", "Inferno", "Cool"};
    vis_config config = node_renderer_.global_config();
    int cm = static_cast<int>(config.colormap);
    if (ImGui::Combo("Colormap", &cm, colormap_names, 6)) {
        config.colormap = static_cast<colormap_t>(cm);
        node_renderer_.set_global_config(config);
        node_renderer_.apply_global_config_to_nodes(graph_);
    }

    // Auto normalize
    if (ImGui::Checkbox("Auto Normalize", &config.auto_normalize)) {
        node_renderer_.set_global_config(config);
        node_renderer_.apply_global_config_to_nodes(graph_);
    }

    // Manual range
    if (!config.auto_normalize) {
        ImGui::BeginDisabled(config.auto_normalize);
        if (ImGui::DragFloatRange2("Range", &config.manual_min, &config.manual_max, 0.01f)) {
            node_renderer_.set_global_config(config);
            node_renderer_.apply_global_config_to_nodes(graph_);
        }
        ImGui::EndDisabled();
    }

    // Note: Per-buffer dimensions are configured in the Info panel when a buffer is selected

    // Layout controls
    ImGui::Separator();
    ImGui::Text("Layout");
    ImGui::Separator();

    ImGui::Checkbox("Auto Layout", &auto_layout_);

    if (layout_) {
        auto params = layout_->params();

        if (ImGui::SliderFloat("Repulsion", &params.repulsion, 100.0f, 20000.0f)) {
            layout_->set_params(params);
        }
        if (ImGui::SliderFloat("Attraction", &params.attraction, 0.001f, 0.1f)) {
            layout_->set_params(params);
        }
        if (ImGui::SliderFloat("Damping", &params.damping, 0.5f, 0.99f)) {
            layout_->set_params(params);
        }
    }

    if (ImGui::Button("Reset Layout")) {
        build_graph_from_topology();
    }
    ImGui::SameLine();
    if (ImGui::Button("Settle")) {
        if (layout_) {
            layout_->settle(0.1f, 500);
        }
    }
}

void buffer_replay_dialog::render_info_panel() {
    ImGui::Text("File Info");
    ImGui::Separator();

    if (reader_.is_open()) {
        const auto& meta = reader_.metadata();
        ImGui::Text("Frames: %zu", reader_.frame_count());
        ImGui::Text("Buffers: %zu", reader_.buffer_count());
        ImGui::Text("Schema: %s", meta.schema_version.c_str());
        ImGui::TextWrapped("Created: %s", meta.created.c_str());

        // Selected node info
        auto selected = canvas_.selected_node();
        if (selected.has_value() && graph_.has_node(*selected)) {
            ImGui::Separator();
            ImGui::Text("Selected Buffer");
            ImGui::Separator();

            auto& node = graph_.node(*selected);
            if (node.user_data.has_value()) {
                try {
                    auto ptr = std::any_cast<std::shared_ptr<buffer_node_data>>(node.user_data);
                    buffer_node_data* data = ptr.get();
                    if (data) {
                        ImGui::Text("Name: %s", data->buffer_name.c_str());
                        ImGui::Text("ID: %zu", data->buffer_id);
                        ImGui::Text("Size: %zu elements", data->buffer_size);

                        // Per-buffer image dimensions
                        if (node_renderer_.global_config().mode == vis_mode::image_2d) {
                            ImGui::Separator();
                            ImGui::Text("Image Dimensions");

                            bool dims_changed = false;
                            ImGui::SetNextItemWidth(80);
                            if (ImGui::InputInt("Width##buf", &data->config.width, 1, 10)) {
                                data->config.width = std::max(1, data->config.width);
                                dims_changed = true;
                            }
                            ImGui::SetNextItemWidth(80);
                            if (ImGui::InputInt("Height##buf", &data->config.height, 1, 10)) {
                                data->config.height = std::max(1, data->config.height);
                                dims_changed = true;
                            }

                            // Show computed pixel count
                            int pixels = data->config.width * data->config.height;
                            if (pixels < static_cast<int>(data->buffer_size)) {
                                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                                    "Shows %d of %zu", pixels, data->buffer_size);
                            } else if (pixels > static_cast<int>(data->buffer_size)) {
                                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                                    "Padded: %d > %zu", pixels, data->buffer_size);
                            } else {
                                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                                    "Exact fit: %d", pixels);
                            }

                            // Auto-fit button
                            if (ImGui::Button("Auto-fit##buf")) {
                                auto size = static_cast<int>(std::sqrt(static_cast<double>(data->buffer_size)));
                                if (size * size == static_cast<int>(data->buffer_size)) {
                                    data->config.width = size;
                                    data->config.height = size;
                                } else {
                                    // Try common aspect ratios
                                    data->config.width = static_cast<int>(data->buffer_size);
                                    data->config.height = 1;
                                    // Check for 28x28, 32x32, 64x64 etc.
                                    for (int w : {28, 32, 64, 96, 128}) {
                                        if (data->buffer_size % w == 0) {
                                            data->config.width = w;
                                            data->config.height = static_cast<int>(data->buffer_size / w);
                                            break;
                                        }
                                    }
                                }
                                dims_changed = true;
                            }

                            if (dims_changed && data->data_valid) {
                                // Regenerate texture with new dimensions
                                node_renderer_.update_node_data(node, data->current_data, data->current_stats);
                            }
                        }

                        ImGui::Separator();
                        ImGui::Text("Current Stats:");
                        ImGui::Text("  Sum: %.4f", data->current_stats.sum);
                        ImGui::Text("  Avg: %.4f", data->current_stats.avg);
                        ImGui::Text("  Min: %.4f", data->current_stats.min_val);
                        ImGui::Text("  Max: %.4f", data->current_stats.max_val);
                        ImGui::Text("  NSE: %.4f", data->current_stats.nse);
                    }
                } catch (...) {}
            }
        }
    } else {
        ImGui::TextDisabled("No file loaded");
    }
}

void buffer_replay_dialog::render_timeline() {
    ImGui::Separator();

    if (!reader_.is_open() || reader_.frame_count() == 0) {
        ImGui::BeginDisabled();
        float dummy = 0.0f;
        ImGui::SliderFloat("##timeline", &dummy, 0.0f, 1.0f);
        ImGui::EndDisabled();
        return;
    }

    // Frame slider
    int frame = static_cast<int>(current_frame_);
    int max_frame = static_cast<int>(reader_.frame_count() - 1);

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::SliderInt("##timeline", &frame, 0, max_frame, "Frame %d")) {
        go_to_frame(static_cast<std::size_t>(frame));
    }
}

void buffer_replay_dialog::render_file_browser() {
    file_browser_.Display();

    if (file_browser_.HasSelected()) {
        std::filesystem::path selected = file_browser_.GetSelected();
        file_browser_.ClearSelected();
        open_file(selected);
    }
}

void buffer_replay_dialog::build_graph_from_topology() {
    graph_.clear();
    layout_.reset();

    if (!reader_.is_open()) {
        return;
    }

    const auto& topology = reader_.topology();

    // Create nodes for each buffer
    std::unordered_map<std::size_t, layout::node_id_t> buffer_to_node;

    // Distribute nodes in a grid initially, centered around origin
    int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(topology.buffers.size()))));
    int rows = static_cast<int>((topology.buffers.size() + cols - 1) / cols);
    float spacing = 150.0f;

    // Calculate offset to center the grid
    float grid_width = static_cast<float>(cols - 1) * spacing;
    float grid_height = static_cast<float>(rows - 1) * spacing;
    float offset_x = -grid_width * 0.5f;
    float offset_y = -grid_height * 0.5f;

    for (std::size_t i = 0; i < topology.buffers.size(); ++i) {
        const auto& buf = topology.buffers[i];

        int row = static_cast<int>(i) / cols;
        int col = static_cast<int>(i) % cols;
        float x = static_cast<float>(col) * spacing + offset_x;
        float y = static_cast<float>(row) * spacing + offset_y;

        // Create node
        layout::node_t node;
        node.x = x;
        node.y = y;
        node.radius = 40.0f;

        // Create buffer data
        buffer_node_data data;
        data.buffer_id = buf.id;
        data.buffer_name = buf.name;
        data.buffer_size = buf.size;
        data.config = node_renderer_.global_config();

        // Use vis_hints from the file, but validate/correct them
        // Old files may have incorrect dimensions (e.g., 32x96 instead of 32x32x3 for RGB)
        bool hints_valid = buf.vis_hint.width > 0 && buf.vis_hint.height > 0;
        std::size_t hint_pixels = hints_valid ?
            static_cast<std::size_t>(buf.vis_hint.width * buf.vis_hint.height) : 0;

        // Check if hints match the data correctly
        if (hints_valid && hint_pixels == buf.size) {
            // Grayscale: dimensions match buffer size exactly
            data.config.width = buf.vis_hint.width;
            data.config.height = buf.vis_hint.height;
            data.config.num_channels = 1;
        } else if (hints_valid && hint_pixels * 3 == buf.size) {
            // RGB: dimensions * 3 match buffer size
            data.config.width = buf.vis_hint.width;
            data.config.height = buf.vis_hint.height;
            data.config.num_channels = 3;
        } else if (hints_valid && buf.vis_hint.num_channels == 3 &&
                   hint_pixels * 3 == buf.size) {
            // Explicit RGB from hints
            data.config.width = buf.vis_hint.width;
            data.config.height = buf.vis_hint.height;
            data.config.num_channels = 3;
        } else {
            // Hints don't match - re-detect from buffer size
            // First try RGB detection
            bool detected = false;
            if (buf.size % 3 == 0) {
                auto rgb_pixels = buf.size / 3;
                auto sqrt_rgb = static_cast<int>(std::sqrt(static_cast<double>(rgb_pixels)));
                if (static_cast<std::size_t>(sqrt_rgb * sqrt_rgb) == rgb_pixels && sqrt_rgb >= 16) {
                    data.config.width = sqrt_rgb;
                    data.config.height = sqrt_rgb;
                    data.config.num_channels = 3;
                    detected = true;
                }
            }

            // Then try grayscale detection
            if (!detected) {
                auto sqrt_size = static_cast<int>(std::sqrt(static_cast<double>(buf.size)));
                if (static_cast<std::size_t>(sqrt_size * sqrt_size) == buf.size) {
                    data.config.width = sqrt_size;
                    data.config.height = sqrt_size;
                    data.config.num_channels = 1;
                } else {
                    // Try common widths
                    data.config.width = std::min(28, static_cast<int>(buf.size));
                    data.config.height = static_cast<int>((buf.size + data.config.width - 1) / data.config.width);
                    data.config.num_channels = 1;
                }
            }
        }

        node.user_data = std::make_shared<buffer_node_data>(std::move(data));

        auto node_id = graph_.add_node(node);
        buffer_to_node[buf.id] = node_id;
    }

    // Create edges based on connectivity
    for (const auto& buf : topology.buffers) {
        auto to_it = buffer_to_node.find(buf.id);
        if (to_it == buffer_to_node.end()) continue;

        for (auto input_id : buf.input_ids) {
            auto from_it = buffer_to_node.find(input_id);
            if (from_it != buffer_to_node.end()) {
                graph_.add_edge(from_it->second, to_it->second, 1.0f, 120.0f);
            }
        }
    }

    // Set up force layout
    layout_ = std::make_unique<layout::force_layout>(graph_);
    layout::force_params params;
    params.repulsion = 5000.0f;
    params.attraction = 0.02f;
    params.damping = 0.85f;
    params.gravity = 0.001f;
    auto bounds = graph_.bounds();
    params.gravity_center_x = (bounds[0] + bounds[2]) * 0.5f;
    params.gravity_center_y = (bounds[1] + bounds[3]) * 0.5f;
    layout_->set_params(params);

    // Center the view
    canvas_.fit_graph_in_view();
}

void buffer_replay_dialog::update_frame_data() {
    if (!reader_.is_open() || graph_.node_count() == 0) {
        return;
    }

    // Update each node's data from the current frame
    for (auto& node : graph_.nodes()) {
        if (!node.user_data.has_value()) continue;

        buffer_node_data* data = nullptr;
        try {
            auto ptr = std::any_cast<std::shared_ptr<buffer_node_data>>(node.user_data);
            data = ptr.get();
        } catch (...) {
            continue;
        }

        if (!data) continue;

        // Read frame data
        std::vector<float> frame_data;
        if (reader_.read_frame_data(data->buffer_id, current_frame_, frame_data)) {
            // Get stats
            auto stats = reader_.read_frame_stats(data->buffer_id, current_frame_);
            frame_stats fs;
            if (stats) {
                fs = *stats;
            } else {
                // Compute stats manually if not available
                if (!frame_data.empty()) {
                    fs.sum = 0.0f;
                    fs.min_val = frame_data[0];
                    fs.max_val = frame_data[0];
                    for (float v : frame_data) {
                        fs.sum += v;
                        fs.min_val = std::min(fs.min_val, v);
                        fs.max_val = std::max(fs.max_val, v);
                    }
                    fs.avg = fs.sum / static_cast<float>(frame_data.size());
                }
            }

            node_renderer_.update_node_data(node, frame_data, fs);
        }
    }
}

void buffer_replay_dialog::step_playback() {
    if (!reader_.is_open() || state_ != playback_state::playing) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - last_frame_time_).count();
    float frame_duration = 1.0f / playback_speed_;

    if (elapsed >= frame_duration) {
        last_frame_time_ = now;

        if (current_frame_ < reader_.frame_count() - 1) {
            ++current_frame_;
            update_frame_data();
        } else {
            // End of file
            state_ = playback_state::paused;
        }
    }
}

void buffer_replay_dialog::go_to_frame(std::size_t frame) {
    if (!reader_.is_open()) return;

    current_frame_ = std::min(frame, reader_.frame_count() - 1);
    update_frame_data();
}

void buffer_replay_dialog::on_file_loaded() {
    build_graph_from_topology();
    current_frame_ = 0;
    state_ = playback_state::paused;
    update_frame_data();

    // Let the layout settle a bit
    if (layout_) {
        layout_->settle(0.5f, 100);
    }

    // Set flag to fit graph on next render (when canvas size is known)
    needs_fit_to_view_ = true;
}

void buffer_replay_dialog::reset_state() {
    reader_.close();
    graph_.clear();
    layout_.reset();
    current_frame_ = 0;
    state_ = playback_state::stopped;
    needs_fit_to_view_ = false;
}

} // namespace coast::replay
