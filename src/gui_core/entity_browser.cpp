#include "entity_browser.h"
#include "sim_instance.h"
#include "sim_builder.h"
#include "vis_registry.h"
#include "imgui.h"
#include "imgui_scaling.h"

#include <algorithm>

namespace coast {

namespace {

const char* entity_type_name(sim::io_entity_type_t type)
{
    switch (type) {
        case sim::io_entity_type_t::ngm2_neuron_group: return "neuron_group";
        case sim::io_entity_type_t::mnist_input:       return "mnist_input";
        case sim::io_entity_type_t::dsprites_input:    return "dsprites_input";
        default:                                       return "unknown";
    }
}

} // anonymous namespace

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

    // Update all active visualizations
    update_visualizations();

    render_browser_window();
}

void entity_browser::free_resources()
{
    // Free all visualization resources
    for (auto& [key, vis] : active_visualizations) {
        if (vis) {
            vis->free_resources();
        }
    }
    active_visualizations.clear();
}

void entity_browser::rebuild_cache()
{
    cached_rows.clear();

    auto& sim_env = sim.get_sim_env();
    const auto& builder = sim.get_builder();

    for (auto& entity : sim_env.iterate_entities()) {
        const auto id = entity.get_outp_id();
        const auto* node = builder.get_node(id);

        cached_rows.push_back({
            id,
            node ? node->label : "",
            node ? node->entity_type : sim::io_entity_type_t::ENTITY_COUNT,
            node ? node->entity_cfg : "",
            entity.get_outp_size(),
            entity.get_inp_ids().size()
        });
    }

    cache_dirty = false;
}

std::string entity_browser::make_vis_key(const std::string& vis_id, std::size_t entity_id)
{
    return vis_id + ":" + std::to_string(entity_id);
}

void entity_browser::update_visualizations()
{
    for (auto& [key, vis] : active_visualizations) {
        if (vis && vis->is_visible()) {
            vis->update();
        }
    }
}

void entity_browser::render_browser_window()
{
    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Sim Entities", &is_visible)) {
        // Rebuild cache if needed
        if (cache_dirty) {
            rebuild_cache();
        }

        // Filter bar
        ImGui::SetNextItemWidth(scaled(200));
        ImGui::InputTextWithHint("##filter", "Filter...", filter_text, sizeof(filter_text));
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu entities)", cached_rows.size());

        ImGui::Separator();

        // Calculate available height for table (leave room for detail panel)
        float available_height = ImGui::GetContentRegionAvail().y;
        float table_height = detail_panel_open ? available_height * splitter_ratio : available_height;

        // Table flags
        ImGuiTableFlags table_flags =
            ImGuiTableFlags_Resizable |
            ImGuiTableFlags_Reorderable |
            ImGuiTableFlags_Hideable |
            ImGuiTableFlags_Sortable |
            ImGuiTableFlags_RowBg |
            ImGuiTableFlags_BordersOuter |
            ImGuiTableFlags_BordersV |
            ImGuiTableFlags_ScrollY;

        if (ImGui::BeginTable("entities_table", 6, table_flags, ImVec2(0, table_height))) {
            // Setup columns
            ImGui::TableSetupScrollFreeze(0, 1); // Freeze header row
            ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, scaled(50));
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, scaled(120));
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, scaled(100));
            ImGui::TableSetupColumn("Config", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Out Size", ImGuiTableColumnFlags_WidthFixed, scaled(70));
            ImGui::TableSetupColumn("Inputs", ImGuiTableColumnFlags_WidthFixed, scaled(50));
            ImGui::TableHeadersRow();

            // Handle sorting
            if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
                if (sort_specs->SpecsDirty && sort_specs->SpecsCount > 0) {
                    const auto& spec = sort_specs->Specs[0];
                    bool ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);

                    std::sort(cached_rows.begin(), cached_rows.end(),
                        [&](const entity_row& a, const entity_row& b) {
                            int cmp = 0;
                            switch (spec.ColumnIndex) {
                                case 0: // ID
                                    cmp = (a.id < b.id) ? -1 : (a.id > b.id) ? 1 : 0;
                                    break;
                                case 1: // Name
                                    cmp = a.name.compare(b.name);
                                    break;
                                case 2: // Type
                                    cmp = static_cast<int>(a.type) - static_cast<int>(b.type);
                                    break;
                                case 3: // Config
                                    cmp = a.config.compare(b.config);
                                    break;
                                case 4: // Output Size
                                    cmp = (a.output_size < b.output_size) ? -1 : (a.output_size > b.output_size) ? 1 : 0;
                                    break;
                                case 5: // Inputs
                                    cmp = (a.input_count < b.input_count) ? -1 : (a.input_count > b.input_count) ? 1 : 0;
                                    break;
                            }
                            return ascending ? (cmp < 0) : (cmp > 0);
                        });

                    sort_specs->SpecsDirty = false;
                }
            }

            // Track which row is selected for detail panel
            const entity_row* selected_row = nullptr;

            // Populate rows from cache
            for (const auto& row : cached_rows) {
                // Apply filter (simple substring match on name/type/config)
                if (filter_text[0] != '\0') {
                    bool matches = false;
                    if (row.name.find(filter_text) != std::string::npos) matches = true;
                    if (std::string(entity_type_name(row.type)).find(filter_text) != std::string::npos) matches = true;
                    if (row.config.find(filter_text) != std::string::npos) matches = true;
                    if (!matches) continue;
                }

                ImGui::TableNextRow();

                // Check if this row is selected
                bool is_selected = (row.id == selected_entity_id);
                if (is_selected) {
                    selected_row = &row;
                }

                // ID column with selectable
                ImGui::TableNextColumn();
                char label[32];
                snprintf(label, sizeof(label), "%zu", row.id);
                if (ImGui::Selectable(label, is_selected, ImGuiSelectableFlags_SpanAllColumns)) {
                    selected_entity_id = row.id;
                    detail_panel_open = true;
                }

                // Name
                ImGui::TableNextColumn();
                if (!row.name.empty()) {
                    ImGui::TextUnformatted(row.name.c_str());
                } else {
                    ImGui::TextDisabled("(unknown)");
                }

                // Type
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(entity_type_name(row.type));

                // Config
                ImGui::TableNextColumn();
                if (!row.config.empty()) {
                    ImGui::TextUnformatted(row.config.c_str());
                } else {
                    ImGui::TextDisabled("-");
                }

                // Output Size
                ImGui::TableNextColumn();
                ImGui::Text("%zu", row.output_size);

                // Input Count
                ImGui::TableNextColumn();
                ImGui::Text("%zu", row.input_count);
            }

            ImGui::EndTable();

            // Render splitter and detail panel if an entity is selected
            if (selected_row && detail_panel_open) {
                // Draggable splitter
                ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_Separator));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_SeparatorHovered));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_SeparatorActive));

                float splitter_height = 4.0f;
                ImGui::Button("##splitter", ImVec2(-1, splitter_height));

                if (ImGui::IsItemActive()) {
                    float delta = ImGui::GetIO().MouseDelta.y;
                    if (delta != 0.0f) {
                        splitter_ratio += delta / available_height;
                        splitter_ratio = std::clamp(splitter_ratio, SPLITTER_MIN_RATIO, SPLITTER_MAX_RATIO);
                    }
                }

                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                }

                ImGui::PopStyleColor(3);

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

void entity_browser::render_detail_panel(const entity_row& selected)
{
    ImGui::Separator();

    // Collapsible header for detail panel
    ImGui::SetNextItemOpen(detail_panel_open, ImGuiCond_Once);
    if (ImGui::CollapsingHeader("Entity Details", &detail_panel_open)) {
        ImGui::Indent();

        // Entity info
        ImGui::Text("ID: %zu", selected.id);
        ImGui::Text("Name: %s", selected.name.empty() ? "(unknown)" : selected.name.c_str());
        ImGui::Text("Type: %s", entity_type_name(selected.type));
        ImGui::Text("Output Size: %zu", selected.output_size);
        ImGui::Text("Input Count: %zu", selected.input_count);

        ImGui::Separator();

        // Available visualizations
        ImGui::Text("Visualizations:");

        auto& registry = vis::get_vis_registry();
        auto available_vis = registry.get_entity_visualizations(selected.type);

        if (available_vis.empty()) {
            ImGui::TextDisabled("No visualizations available for this entity type");
        } else {
            for (const auto* info : available_vis) {
                std::string key = make_vis_key(info->id, selected.id);
                bool is_active = active_visualizations.contains(key);

                if (ImGui::Checkbox(info->display_name.c_str(), &is_active)) {
                    if (is_active) {
                        // Create new visualization
                        auto* entity = sim.get_sim_env().get_entity(selected.id);
                        if (entity) {
                            std::string window_label = selected.name + " - " + info->display_name;
                            auto vis_instance = registry.create_entity_vis(
                                info->id,
                                *entity,
                                sim.get_sim_env(),
                                sim.get_config().vis_hints,
                                selected.config,  // hints_key: entity_cfg for looking up visualization hints
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
