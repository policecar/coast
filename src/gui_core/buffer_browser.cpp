#include "buffer_browser.h"
#include "sim_instance.h"
#include "sim_builder.h"
#include "vis_registry.h"
#include "io_buffer.h"
#include "imgui.h"

#include <algorithm>
#include <unordered_map>

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

    // Update all active visualizations
    update_visualizations();

    render_browser_window();
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
    auto& buffers = sim_env.get_io_buffers();
    const auto& builder = sim.get_builder();

    // Build map from buffer ID to entity name (output_of)
    std::unordered_map<std::size_t, std::string> output_of_map;
    for (auto& entity : sim_env.iterate_entities()) {
        const auto outp_id = entity.get_outp_id();
        const auto* node = builder.get_node(outp_id);
        if (node) {
            output_of_map[outp_id] = node->label;
        }
    }

    for (auto& [id, buffer] : buffers) {
        auto read_buf = buffer.cur_read_buffer();

        // Calculate stats
        float sum = 0.0f, min_val = 0.0f, max_val = 0.0f;
        if (!read_buf.empty()) {
            min_val = read_buf[0];
            max_val = read_buf[0];
            for (float v : read_buf) {
                sum += v;
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
        float avg = read_buf.empty() ? 0.0f : sum / static_cast<float>(read_buf.size());

        // Look up entity name
        std::string output_of;
        if (auto it = output_of_map.find(id); it != output_of_map.end()) {
            output_of = it->second;
        }

        cached_rows.push_back({
            id,
            output_of,
            buffer.size(),
            sum,
            avg,
            min_val,
            max_val
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
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 50.0f);
            ImGui::TableSetupColumn("Output of", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableHeadersRow();

            // Track which row is selected
            const buffer_row* selected_row = nullptr;

            for (const auto& row : cached_rows) {
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

    // Render visualization windows (outside main window)
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
