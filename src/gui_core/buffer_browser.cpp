#include "gui_core/buffer_browser.h"
#include "sim_core/sim_instance.h"
#include "tools/sim_builder.h"
#include "sim_core/sim_env.h"
#include "gui_vis/vis_registry.h"
#include "sim_core/io_buffer.h"
#include "imgui.h"
#include "gui_core/imgui_scaling.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace coast {

buffer_browser::buffer_browser(sim_instance& sim)
    : sim(sim)
{
}

buffer_browser::~buffer_browser()
{
    // Remove the stats hook while the sim (and its env) is still alive. The
    // owning gui_manager destroys this browser before the sim instance it points
    // at (set_sim_instance resets the browser windows before swapping `sim`, and
    // run_gui calls set_sim_instance(nullptr) before sim.reset()), so this is
    // safe. Removal is synchronous (#235), so the hook cannot fire after it;
    // and it captures the shared_ptr slot, never this object, regardless.
    if (stats_hook_registered_) {
        sim.get_sim_env().remove_post_swap_hook(stats_hook_id_);
        stats_hook_registered_ = false;
    }
    free_resources();
}

std::vector<buffer_browser::buffer_stat> buffer_browser::compute_buffer_stats(sim::env& env)
{
    std::vector<buffer_stat> out;
    auto& buffers = env.get_io_buffers();
    out.reserve(buffers.size());

    for (auto& [id, buffer] : buffers) {
        const std::size_t sz = buffer->size();

        // Read the coherent read page. cur_read_buffer() exposes it for CPU /
        // mirrored variants; io_buffer_gpu (GPU-only, no mirror) returns an empty
        // span, so pull it on demand with read_back_sync — safe here because this
        // runs on the sim thread that owns the wgpu queue.
        auto read_span = buffer->cur_read_buffer();
        const float* data = nullptr;
        std::size_t  n    = 0;
        std::vector<float> tmp;
        if (!read_span.empty()) {
            data = read_span.data();
            n    = read_span.size();
        } else if (sz > 0) {
            tmp.resize(sz);
            if (buffer->read_back_sync(tmp)) {
                data = tmp.data();
                n    = sz;
            }
        }

        float sum = 0.0f, min_val = 0.0f, max_val = 0.0f;
        if (n > 0) {
            min_val = max_val = data[0];
            for (std::size_t i = 0; i < n; ++i) {
                const float v = data[i];
                sum += v;
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
        const float avg = (n > 0) ? sum / static_cast<float>(n) : 0.0f;

        out.push_back({id, sz, sum, avg, min_val, max_val});
    }
    return out;
}

void buffer_browser::sync_stats_hook()
{
    auto* env = &sim.get_sim_env();

    if (is_visible && !stats_hook_registered_) {
        // Register a post_swap hook that publishes buffer stats on demand. It
        // captures the shared publish slot (by value) and the env pointer, never
        // `this`, so its lifetime is decoupled from this browser.
        auto pub = stats_pub_;
        stats_hook_id_ = env->set_post_swap_hook([pub, env]() {
            pub->produce_if_wanted([env]() { return compute_buffer_stats(*env); });
        });
        stats_hook_registered_ = true;
    } else if (!is_visible && stats_hook_registered_) {
        env->remove_post_swap_hook(stats_hook_id_);
        stats_hook_registered_ = false;
    }
}

void buffer_browser::render()
{
    // Register/remove the sim-thread stats publisher to follow window visibility
    // (every frame, so closing the window stops the work).
    sync_stats_hook();

    // Always update active visualizations, even when browser window is closed
    // This ensures the GUI thread processes sync requests from the sim thread
    update_visualizations();

    // Always render visualization windows (they're separate from the browser)
    render_visualization_windows();

    // Only render the browser table/details panel when visible
    if (is_visible) {
        render_browser_window();
    }
}

void buffer_browser::free_resources()
{
    // Free all visualization resources
    for (auto& [key, vis] : active_visualizations) {
        if (vis) {
            vis->free_resources();
        }
    }
    active_visualizations.clear();
}

void buffer_browser::rebuild_cache()
{
    cached_rows.clear();

    auto& sim_env = sim.get_sim_env();
    const auto& builder = sim.get_builder();

    // Build map from buffer ID to entity name (output_of). This is immutable
    // structure (entity ids + builder labels are fixed after construction), so
    // reading it on the GUI thread is race-free.
    std::unordered_map<std::size_t, std::string> output_of_map;
    for (auto& entity : sim_env.iterate_entities()) {
        const auto outp_id = entity.get_outp_id();
        const auto* node = builder.get_node(outp_id);
        if (node) {
            output_of_map[outp_id] = node->label;
        }
    }

    // Buffer stats come from the sim thread's published snapshot (computed at
    // post_swap), not a live read — this is the data race the redesign removes.
    // Requesting the snapshot here also drives the demand-publish handshake: it
    // signals the sim thread to recompute on its next boundary.
    auto stats = stats_pub_->consume();
    cached_rows.reserve(stats.size());
    for (const auto& s : stats) {
        std::string output_of;
        if (auto it = output_of_map.find(s.id); it != output_of_map.end()) {
            output_of = it->second;
        }
        cached_rows.push_back({
            s.id,
            output_of,
            s.size,
            s.sum,
            s.avg,
            s.min_val,
            s.max_val
        });
    }

    cache_dirty = false;
}

std::string buffer_browser::make_vis_key(const std::string& vis_id, std::size_t buffer_id)
{
    return vis_id + ":" + std::to_string(buffer_id);
}

void buffer_browser::update_visualizations()
{
    for (auto& [key, vis] : active_visualizations) {
        if (vis && vis->is_visible()) {
            vis->update();
        }
    }
}

void buffer_browser::render_browser_window()
{
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(150, 150), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("IO Buffers", &is_visible)) {
        // Rebuild cache if needed (refresh each frame to show live stats)
        // For performance, could add a refresh button instead
        rebuild_cache();

        // Filter bar
        ImGui::SetNextItemWidth(scaled(200));
        ImGui::InputTextWithHint("##filter", "Filter...", filter_text, sizeof(filter_text));
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu buffers)", cached_rows.size());

        ImGui::Separator();

        // Calculate available height for table
        float available_height = ImGui::GetContentRegionAvail().y;
        float table_height = detail_panel_open ? available_height * 0.6f : available_height;

        // Table flags
        ImGuiTableFlags table_flags =
            ImGuiTableFlags_Resizable |
            ImGuiTableFlags_RowBg |
            ImGuiTableFlags_BordersOuter |
            ImGuiTableFlags_BordersV |
            ImGuiTableFlags_ScrollY;

        if (ImGui::BeginTable("buffers_table", 6, table_flags, ImVec2(0, table_height))) {
            // Setup columns
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, scaled(50));
            ImGui::TableSetupColumn("Output of", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, scaled(70));
            ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed, scaled(80));
            ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, scaled(80));
            ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, scaled(80));
            ImGui::TableHeadersRow();

            // Track which row is selected
            const buffer_row* selected_row = nullptr;

            for (const auto& row : cached_rows) {
                // Apply filter (substring match on output_of or ID)
                if (filter_text[0] != '\0') {
                    bool matches = false;
                    if (row.output_of.find(filter_text) != std::string::npos) matches = true;
                    if (std::to_string(row.id).find(filter_text) != std::string::npos) matches = true;
                    if (!matches) continue;
                }

                ImGui::TableNextRow();

                bool is_selected = (row.id == selected_buffer_id);
                if (is_selected) {
                    selected_row = &row;
                }

                // ID column with selectable
                ImGui::TableNextColumn();
                char label[32];
                snprintf(label, sizeof(label), "%zu", row.id);
                if (ImGui::Selectable(label, is_selected, ImGuiSelectableFlags_SpanAllColumns)) {
                    selected_buffer_id = row.id;
                    detail_panel_open = true;
                }

                // Output of
                ImGui::TableNextColumn();
                if (row.output_of.empty()) {
                    ImGui::TextDisabled("-");
                } else {
                    ImGui::Text("%s", row.output_of.c_str());
                }

                // Size
                ImGui::TableNextColumn();
                ImGui::Text("%zu", row.size);

                // Avg
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", row.avg);

                // Min
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", row.min_val);

                // Max
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", row.max_val);
            }

            ImGui::EndTable();

            // Render detail panel if a buffer is selected
            if (selected_row && detail_panel_open) {
                render_detail_panel(*selected_row);
            }
        }
    }
    ImGui::End();
}

void buffer_browser::render_visualization_windows()
{
    // Render visualization windows (separate from browser window)
    std::vector<std::string> to_remove;
    for (auto& [key, vis] : active_visualizations) {
        if (vis && vis->is_visible()) {
            if (!vis->render_imgui()) {
                // Window was closed
                vis->set_visible(false);
                to_remove.push_back(key);
            }
        }
    }
    // Clean up closed visualizations
    for (const auto& key : to_remove) {
        if (auto it = active_visualizations.find(key); it != active_visualizations.end()) {
            it->second->free_resources();
            active_visualizations.erase(it);
        }
    }
}

void buffer_browser::render_detail_panel(const buffer_row& selected)
{
    ImGui::Separator();

    // Collapsible header for detail panel
    ImGui::SetNextItemOpen(detail_panel_open, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Buffer Details", &detail_panel_open)) {
        ImGui::Indent();

        // Buffer info
        ImGui::Text("ID: %zu", selected.id);
        ImGui::Text("Size: %zu", selected.size);
        ImGui::Text("Sum: %.3f", selected.sum);
        ImGui::Text("Avg: %.3f", selected.avg);
        ImGui::Text("Min: %.3f", selected.min_val);
        ImGui::Text("Max: %.3f", selected.max_val);

        ImGui::Separator();

        // Available visualizations
        ImGui::Text("Visualizations:");

        auto& registry = vis::get_vis_registry();
        auto available_vis = registry.get_buffer_visualizations();

        if (available_vis.empty()) {
            ImGui::TextDisabled("No visualizations available");
        } else {
            for (const auto* info : available_vis) {
                std::string key = make_vis_key(info->id, selected.id);
                bool is_active = active_visualizations.contains(key);

                if (ImGui::Checkbox(info->display_name.c_str(), &is_active)) {
                    if (is_active) {
                        // Create new visualization
                        auto buffer_opt = sim.get_sim_env().get_io_buffer(selected.id);
                        if (buffer_opt) {
                            std::string window_label = "Buffer " + std::to_string(selected.id) +
                                                       " - " + info->display_name;
                            auto vis_instance = registry.create_buffer_vis(
                                info->id,
                                **buffer_opt,
                                sim.get_sim_env(),
                                sim.get_config().vis_hints,
                                window_label
                            );
                            if (vis_instance) {
                                active_visualizations[key] = std::move(vis_instance);
                            }
                        }
                    } else {
                        // Remove visualization
                        if (auto it = active_visualizations.find(key); it != active_visualizations.end()) {
                            it->second->free_resources();
                            active_visualizations.erase(it);
                        }
                    }
                }

                // Tooltip with description
                if (ImGui::IsItemHovered() && !info->description.empty()) {
                    ImGui::SetTooltip("%s", info->description.c_str());
                }
            }
        }

        ImGui::Unindent();
    }
}

} // namespace coast
