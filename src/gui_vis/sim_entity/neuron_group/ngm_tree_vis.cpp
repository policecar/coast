//
// Created by jk on 12.08.25.
//
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <unordered_map>

#include "inc_on_exit.h"
#include "sim_entity/neuron_group/ngm_tree_vis.h"
#include "vis_registry.h"
#include "sim_env.h"
#include "imgui.h"
#include "imgui_scaling.h"

namespace coast {

void ngm_tree_vis::get_pixel_data(
    int width, int height,
    uint16_t leaf_idx,
    std::size_t max_idx,
    const ngm2::dendrite_t::synapses_t &synapses,
    std::vector<color_t> &output
)
{
    const std::size_t buffer_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    output.resize(buffer_size);

    // Fill with black (background) initially
    std::fill(output.begin(), output.end(), color_t{0, 0, 0, 255});

    tmp_tree.clear();
    tmp_tree.resize(max_idx+1);

    tmp_tree[leaf_idx] = 1;
    while (leaf_idx > 1) {
        leaf_idx /= 2;
        tmp_tree[leaf_idx] = 1;
    }

    std::size_t pix = 0;
    std::size_t syn_cnt = synapses.size();
    for (std::size_t i = 0; i < syn_cnt && pix < buffer_size; ++i)
        if (tmp_tree[synapses.segment_idx[i]]) {
            float normalized = std::clamp(synapses.permanence[i], 0.0f, 1.0f);
            output[pix] = apply_colormap(colormap_, normalized);
            ++pix;
        }
}

gl_texture ngm_tree_vis::get_texture(int width, int height, const color_t *data)
{
    return gl_texture(width, height, data);
}

ngm_tree_vis::~ngm_tree_vis()
{
}

void ngm_tree_vis::create_model(fbgd &vis, const ngm2::neuron_group_t &ng)
{
    nodes.clear();
    neuron_lu.clear();
    branch_start_lu.clear();
    branch_lu.clear();

    // determine size of nodes-array
    std::size_t nodes_size = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        neuron_lu.push_back(nodes_size);
        branch_start_lu.push_back(branch_lu.size());
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            std::size_t segment_cnt = cur_branch.get_max_segment_idx() + 1;
            branch_lu.push_back(nodes_size);
            nodes_size += segment_cnt;
        }
    }
    nodes.resize(nodes_size);

    // Store region size for rendering
    region_size_ = vis.get_region_size();

    // fill nodes
    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    const float grid_spacing = region_size_ / static_cast<float>(grid_cnt+2);
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        float xp = static_cast<float>(neuron_idx % grid_cnt) * grid_spacing + grid_spacing * 1.5f;
        float yp = static_cast<float>(neuron_idx / grid_cnt) * grid_spacing + grid_spacing * 1.5f;

        // create root node (fixed in grid position)
        std::size_t root_node_idx = neuron_lu[neuron_idx];
        nodes[root_node_idx] = {
            neuron_idx,     //std::size_t neuron_idx
            0,              //std::size_t branch_idx
            0,              //uint16_t    segment_idx
            vis.add_node({xp,yp}, true, 1.0f, root_node_idx), //fbgd::node_id_t node_id - fixed
            1,               //uint8_t used
            {},
            std::vector<color_t>(),
            0
        };
        // go over all branches and create nodes where necessary
        std::size_t branch_base_idx = branch_start_lu[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            auto leaf_mask = cur_branch.get_leaf_mask();
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu[branch_base_idx + branch_idx];
            const auto &synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();

            // First pass: mark segments that need to exist (including ancestors)
            std::vector<uint8_t> segments_needed(max_segment + 1, 0);
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                uint16_t seg = synapses.segment_idx[i];
                while (seg >= 1 && segments_needed[seg] == 0) {
                    segments_needed[seg] = 1;
                    seg /= 2;
                }
            }

            // Second pass: create nodes for all needed segments (ancestors first)
            for (uint16_t seg = 1; seg <= max_segment; ++seg) {
                if (segments_needed[seg] == 0) continue;
                if (segment_base + seg >= nodes.size()) continue;

                auto &cur_node = nodes[segment_base + seg];
                if (cur_node.used == 0) {
                    gl_texture tex;
                    std::vector<color_t> pdata;
                    if (seg < leaf_mask.size() && leaf_mask[seg]) {
                        get_pixel_data(params_.rep_width, params_.rep_height, seg, max_segment, synapses, pdata);
                        tex = get_texture(params_.rep_width, params_.rep_height, pdata.data());
                    }
                    cur_node = {
                        neuron_idx,
                        branch_idx,
                        seg,
                        vis.add_node({xp + rdis(rgen) * (grid_spacing / 3), yp + rdis(rgen) * (grid_spacing / 3)}, false, 1.0f, segment_base + seg),
                        1,
                        std::move(tex),
                        std::move(pdata),
                        0
                    };
                }
            }

            // Third pass: count synapses per segment
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                if (segment_base + synapses.segment_idx[i] < nodes.size()) {
                    nodes[segment_base + synapses.segment_idx[i]].synapse_count++;
                }
            }

            // Fourth pass: create edges (parents guaranteed to exist now)
            const std::size_t max_idx = cur_branch.get_max_segment_idx();
            uint32_t cur_start = (max_idx + 1) / 2;
            while (cur_start > 1) {
                for (std::size_t seg_idx = cur_start; seg_idx < cur_start*2 && seg_idx <= max_idx; ++seg_idx) {
                    if (segment_base + seg_idx >= nodes.size()) continue;
                    if (nodes[segment_base + seg_idx].used == 1) {
                        auto &parent = nodes[segment_base + seg_idx/2];
                        if (parent.used > 0) {
                            vis.add_edge(parent.node_id, nodes[segment_base + seg_idx].node_id);
                        }
                    }
                }
                cur_start /= 2;
            }
            // Connect segment 1 to root if it exists
            if (segment_base + 1 < nodes.size() && nodes[segment_base + 1].used > 0) {
                vis.add_edge(nodes[root_node_idx].node_id, nodes[segment_base + 1].node_id);
            }
        }

    }
}

void ngm_tree_vis::update_model(fbgd &vis, const ngm2::neuron_group_t &ng)
{
    for (auto &node : nodes)
        node.synapse_count = 0;

    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        std::size_t root_node_idx = neuron_lu[neuron_idx];

        std::size_t branch_base_idx = branch_start_lu[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            auto leaf_mask = cur_branch.get_leaf_mask();
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu[branch_base_idx + branch_idx];
            const auto &synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();

            // Track new nodes that need edges, and their spawn positions
            std::vector<uint16_t> new_nodes_needing_edges;
            std::unordered_map<uint16_t, std::array<float, 2>> new_node_positions;

            // Calculate spawn offset based on region size (similar to create_model's grid_spacing/3)
            const float spawn_offset = region_size_ * 0.05f;

            // Get root position once (root node should always exist and be in lookup)
            const auto &root_pos = vis.get_node_pos(nodes[root_node_idx].node_id);

            // Helper to get position of a segment (checking new nodes first, then fbgd)
            auto get_seg_pos = [&](uint16_t seg) -> std::array<float, 2> {
                if (seg == 0) return root_pos;
                auto it = new_node_positions.find(seg);
                if (it != new_node_positions.end()) {
                    return it->second;
                }
                if (segment_base + seg < nodes.size() && nodes[segment_base + seg].used > 0) {
                    return vis.get_node_pos(nodes[segment_base + seg].node_id);
                }
                return root_pos;  // Fallback
            };

            // Single pass: count synapses and create nodes on-demand
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                uint16_t seg = synapses.segment_idx[i];
                if (segment_base + seg >= nodes.size()) continue;

                auto &cur_node = nodes[segment_base + seg];

                // Only count synapse once per segment per update
                cur_node.synapse_count++;

                // Create node if it doesn't exist
                if (cur_node.used == 0) {
                    // Ensure ancestors exist first (walk up tree from root to seg)
                    // Build path from seg to root
                    std::vector<uint16_t> path;
                    uint16_t ancestor = seg;
                    while (ancestor >= 1) {
                        path.push_back(ancestor);
                        if (ancestor == 1) break;
                        ancestor /= 2;
                    }

                    // Process from root towards seg (reverse order)
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        uint16_t s = *it;
                        if (segment_base + s >= nodes.size()) continue;
                        auto &node = nodes[segment_base + s];
                        if (node.used != 0) continue;

                        // Get parent position
                        uint16_t parent_seg = (s == 1) ? 0 : s / 2;
                        auto parent_pos = get_seg_pos(parent_seg);
                        float nx = parent_pos[0] + rdis(rgen) * spawn_offset;
                        float ny = parent_pos[1] + rdis(rgen) * spawn_offset;

                        // Store position for children to use
                        new_node_positions[s] = {nx, ny};

                        // Create texture only for leaf nodes
                        gl_texture tex;
                        std::vector<color_t> pdata;
                        if (s == seg && s < leaf_mask.size() && leaf_mask[s]) {
                            get_pixel_data(params_.rep_width, params_.rep_height, s, max_segment, synapses, pdata);
                            tex = get_texture(params_.rep_width, params_.rep_height, pdata.data());
                        }

                        node = {
                            neuron_idx, branch_idx, s,
                            vis.add_node({nx, ny}, false, 1.0f, segment_base + s),
                            1, std::move(tex), std::move(pdata), 0
                        };
                        new_nodes_needing_edges.push_back(s);
                    }
                }
            }

            // Add edges for new nodes (sorted by segment to ensure parents before children)
            std::sort(new_nodes_needing_edges.begin(), new_nodes_needing_edges.end());
            for (uint16_t seg : new_nodes_needing_edges) {
                if (segment_base + seg >= nodes.size()) continue;
                auto &node = nodes[segment_base + seg];
                if (seg == 1) {
                    vis.add_edge(nodes[root_node_idx].node_id, node.node_id);
                } else if (nodes[segment_base + seg/2].used > 0) {
                    vis.add_edge(nodes[segment_base + seg/2].node_id, node.node_id);
                }
            }
        }
    }
}

void ngm_tree_vis::update_representations(const ngm2::neuron_group_t &ng)
{
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        // go over all branches and create nodes where necessary
        std::size_t branch_base_idx = branch_start_lu[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t &cur_branch = cur_neuron.get_dendrite(branch_idx);
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu[branch_base_idx + branch_idx];
            const auto &synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                auto &cur_node = nodes[segment_base + synapses.segment_idx[i]];
                if (cur_node.pixel_data.empty() == false)
                {
                    get_pixel_data(params_.rep_width, params_.rep_height, synapses.segment_idx[i], max_segment, synapses, cur_node.pixel_data);
                    cur_node.texture.update(cur_node.pixel_data);
                }
            }
        }
    }
}

void ngm_tree_vis::update_repulsion_radii(fbgd &vis)
{
    if (neuron_lu.empty()) return;

    // Calculate total synapse count per neuron
    std::size_t neuron_cnt = neuron_lu.size();
    std::vector<uint32_t> neuron_synapse_counts(neuron_cnt, 0);

    // Sum up synapse counts for all nodes belonging to each neuron
    for (const auto& node : nodes) {
        if (node.used > 0 && node.neuron_idx < neuron_cnt) {
            neuron_synapse_counts[node.neuron_idx] += node.synapse_count;
        }
    }

    // Find max for normalization
    uint32_t max_synapses = 1;
    for (auto count : neuron_synapse_counts) {
        max_synapses = std::max(max_synapses, count);
    }

    // Update repulsion radius for each root node
    // Base radius is 1.0, scaled up to 3.0 based on tree size
    const float base_radius = 1.0f;
    const float max_radius = 3.0f;
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        std::size_t root_node_idx = neuron_lu[neuron_idx];
        if (root_node_idx < nodes.size() && nodes[root_node_idx].used > 0) {
            float ratio = static_cast<float>(neuron_synapse_counts[neuron_idx]) /
                          static_cast<float>(max_synapses);
            float radius = base_radius + (max_radius - base_radius) * ratio;
            vis.set_node_repulsion_radius(nodes[root_node_idx].node_id, radius);
        }
    }
}

void ngm_tree_vis::update_root_positions(fbgd &vis, const ngm2::neuron_group_t &ng)
{
    (void)vis;  // Unused after simplification

    if (neuron_lu.empty()) return;

    std::size_t neuron_cnt = neuron_lu.size();
    std::size_t ng_neuron_cnt = ng.get_neuron_count();

    // Safety check
    if (neuron_cnt > ng_neuron_cnt) {
        neuron_cnt = ng_neuron_cnt;
    }
    if (neuron_cnt == 0) return;

    // Use fixed grid spacing based on neuron count
    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    if (grid_cnt == 0) grid_cnt = 1;

    // Cell size fills the region, spacing_multiplier allows expansion beyond region
    // At multiplier 1.0, grid exactly fills the region
    // Higher values spread neurons further apart (may need to zoom out)
    const float base_cell_size = region_size_ / static_cast<float>(grid_cnt);
    const float cell_size = base_cell_size * spacing_multiplier_;
    const float margin = cell_size * 0.5f;

    // Store target positions for each neuron (animation happens in animate_tree_positions)
    neuron_target_positions_.resize(neuron_cnt);

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu.size()) break;
        std::size_t root_node_idx = neuron_lu[neuron_idx];
        if (root_node_idx >= nodes.size() || nodes[root_node_idx].used == 0) continue;

        std::size_t col = neuron_idx % grid_cnt;
        std::size_t row = neuron_idx / grid_cnt;

        // Calculate ideal position at center of cell (uniform grid)
        float target_x = margin + (static_cast<float>(col) + 0.5f) * cell_size;
        float target_y = margin + (static_cast<float>(row) + 0.5f) * cell_size;

        neuron_target_positions_[neuron_idx] = {target_x, target_y};
    }
}

void ngm_tree_vis::animate_tree_positions(fbgd &vis, float lerp_factor)
{
    if (neuron_lu.empty() || neuron_target_positions_.empty()) return;

    std::size_t neuron_cnt = std::min(neuron_lu.size(), neuron_target_positions_.size());

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        std::size_t root_node_idx = neuron_lu[neuron_idx];
        if (root_node_idx >= nodes.size() || nodes[root_node_idx].used == 0) continue;

        // Get current root position and target
        const auto& current_pos = vis.get_node_pos(nodes[root_node_idx].node_id);
        const auto& target = neuron_target_positions_[neuron_idx];

        // Calculate lerped delta
        float dx = (target[0] - current_pos[0]) * lerp_factor;
        float dy = (target[1] - current_pos[1]) * lerp_factor;

        // Skip if movement is negligible
        if (std::abs(dx) < 0.0001f && std::abs(dy) < 0.0001f) continue;

        // Move all nodes belonging to this neuron by the same delta
        for (const auto& node : nodes) {
            if (node.used == 0 || node.neuron_idx != neuron_idx) continue;
            vis.translate_node(node.node_id, {dx, dy});
        }
    }
}

void ngm_tree_vis::free_resources()
{
    for (auto &node : nodes){
        node.texture.release();
        node.pixel_data.clear();
    }
}

bool ngm_tree_vis::render_imgui(const char* label,
                           const std::vector<fbgd::node> &vis_nodes,
                           const std::vector<fbgd::edge> &vis_edges,
                           const std::vector<uint32_t> &vis_node_lu)
{
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin(label, &open);

    // Early exit if data is not ready
    if (vis_nodes.empty() || vis_node_lu.empty() || nodes.empty()) {
        ImGui::TextDisabled("Waiting for data...");
        ImGui::End();
        return open;
    }

    // Toolbar with zoom, reset, colormap, and hover option
    ImGui::Text("Zoom: %.0f%%", zoom_level_ * 100.0f);
    ImGui::SameLine();
    if (ImGui::Button("Reset View")) {
        target_zoom_ = 1.0f;
        target_pan_x_ = 0.0f;
        target_pan_y_ = 0.0f;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(scaled(100));
    if (ImGui::BeginCombo("##colormap", colormap_name(colormap_))) {
        for (int i = 0; i < static_cast<int>(colormap_t::COUNT); ++i) {
            auto cm = static_cast<colormap_t>(i);
            bool is_selected = (colormap_ == cm);
            if (ImGui::Selectable(colormap_name(cm), is_selected)) {
                if (colormap_ != cm) {
                    colormap_ = cm;
                    colormap_dirty_ = true;
                }
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show on hover", &show_on_hover_);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(scaled(100));
    if (ImGui::SliderFloat("Spacing", &spacing_multiplier_, 0.5f, 3.0f, "%.1fx")) {
        spacing_dirty_ = true;
    }

    ImGui::Separator();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();

    // Draw canvas background
    draw_list->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        IM_COL32(params_.bg_r, params_.bg_g, params_.bg_b, params_.bg_a));

    // Add invisible button to capture mouse input on canvas
    ImGui::InvisibleButton("canvas", canvas_size, ImGuiButtonFlags_MouseButtonLeft);
    bool canvas_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    bool is_hovered = ImGui::IsItemHovered();
    bool is_active = ImGui::IsItemActive();
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 mouse_pos = io.MousePos;

    // Smooth interpolation for zoom/pan
    constexpr float lerp_speed = 15.0f;
    float lerp_t = std::min(1.0f, lerp_speed * io.DeltaTime);

    // Handle scroll wheel zoom (only when hovered)
    if (is_hovered && io.MouseWheel != 0.0f) {
        float mouse_canvas_x = mouse_pos.x - canvas_pos.x;
        float mouse_canvas_y = mouse_pos.y - canvas_pos.y;

        // Calculate position in normalized space before zoom
        float norm_x = (mouse_canvas_x + pan_x_) / (canvas_size.x * zoom_level_);
        float norm_y = (mouse_canvas_y + pan_y_) / (canvas_size.y * zoom_level_);

        // Apply zoom (exponential for smoother feel)
        float zoom_factor = std::pow(1.15f, io.MouseWheel);
        target_zoom_ = std::clamp(target_zoom_ * zoom_factor, 0.1f, 10.0f);

        // Adjust target pan to keep the point under the mouse fixed
        target_pan_x_ = norm_x * canvas_size.x * target_zoom_ - mouse_canvas_x;
        target_pan_y_ = norm_y * canvas_size.y * target_zoom_ - mouse_canvas_y;
    }

    // Handle left-click drag for panning (free panning, no clamping)
    if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        target_pan_x_ -= io.MouseDelta.x;
        target_pan_y_ -= io.MouseDelta.y;
        pan_x_ -= io.MouseDelta.x;
        pan_y_ -= io.MouseDelta.y;
    }

    // Smoothly interpolate current values (no clamping - free pan)
    zoom_level_ = zoom_level_ + (target_zoom_ - zoom_level_) * lerp_t;
    pan_x_ = pan_x_ + (target_pan_x_ - pan_x_) * lerp_t;
    pan_y_ = pan_y_ + (target_pan_y_ - pan_y_) * lerp_t;

    // Transform function with zoom and pan
    const float region_size = region_size_;
    auto transform = [&](float x, float y) -> ImVec2 {
        float screen_x = (x / region_size) * canvas_size.x * zoom_level_ - pan_x_;
        float screen_y = (y / region_size) * canvas_size.y * zoom_level_ - pan_y_;
        return ImVec2(canvas_pos.x + screen_x, canvas_pos.y + screen_y);
    };

    // Hover detection for show_on_hover mode
    hovered_neuron_idx_ = SIZE_MAX;
    if (show_on_hover_ && is_hovered && !is_active) {
        float hover_radius = params_.root_node_radius * zoom_level_ * 2.0f;
        for (const auto &vis_node : vis_nodes) {
            auto* node_idx_ptr = std::any_cast<std::size_t>(&vis_node.payload);
            if (!node_idx_ptr || *node_idx_ptr >= nodes.size())
                continue;

            const auto &node = nodes[*node_idx_ptr];
            if (node.segment_idx != 0)
                continue;

            ImVec2 node_pos = transform(vis_node.pos[0], vis_node.pos[1]);
            float dx = mouse_pos.x - node_pos.x;
            float dy = mouse_pos.y - node_pos.y;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq <= hover_radius * hover_radius) {
                hovered_neuron_idx_ = node.neuron_idx;
                break;
            }
        }
    }

    // Handle click detection on root nodes (only when not dragging)
    if (canvas_clicked && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        bool clicked_on_node = false;
        float click_radius = params_.root_node_radius * zoom_level_ * 2.0f;

        for (const auto &vis_node : vis_nodes) {
            auto* node_idx_ptr = std::any_cast<std::size_t>(&vis_node.payload);
            if (!node_idx_ptr || *node_idx_ptr >= nodes.size())
                continue;

            const auto &node = nodes[*node_idx_ptr];
            if (node.segment_idx != 0)
                continue;

            ImVec2 node_pos = transform(vis_node.pos[0], vis_node.pos[1]);
            float dx = mouse_pos.x - node_pos.x;
            float dy = mouse_pos.y - node_pos.y;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq <= click_radius * click_radius) {
                if (selected_neuron_idx_ == node.neuron_idx) {
                    selected_neuron_idx_ = SIZE_MAX;
                } else {
                    selected_neuron_idx_ = node.neuron_idx;
                }
                clicked_on_node = true;
                break;
            }
        }

        if (!clicked_on_node) {
            selected_neuron_idx_ = SIZE_MAX;
        }
    }

    // Determine which neuron's representations to show
    std::size_t display_neuron_idx = selected_neuron_idx_;
    if (display_neuron_idx == SIZE_MAX && show_on_hover_) {
        display_neuron_idx = hovered_neuron_idx_;
    }

    // Clip drawing to canvas area to prevent nodes from overlapping toolbar
    draw_list->PushClipRect(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), true);

    // Precompute colors
    const ImU32 edge_color = IM_COL32(params_.edge_r, params_.edge_g, params_.edge_b, params_.edge_a);
    const ImU32 root_color = IM_COL32(params_.root_r, params_.root_g, params_.root_b, params_.root_a);
    const ImU32 branch_color = IM_COL32(params_.branch_r, params_.branch_g, params_.branch_b, params_.branch_a);
    const float max_synapses = static_cast<float>(params_.rep_width * params_.rep_height);
    const float thickness_range = params_.edge_thickness_max - params_.edge_thickness_min;

    // Draw edges
    for (const auto &edge : vis_edges) {
        // Bounds check
        if (edge.from >= vis_node_lu.size() || edge.to >= vis_node_lu.size())
            continue;
        if (vis_node_lu[edge.from] >= vis_nodes.size() || vis_node_lu[edge.to] >= vis_nodes.size())
            continue;

        const auto &vnode1 = vis_nodes[vis_node_lu[edge.from]];
        const auto &vnode2 = vis_nodes[vis_node_lu[edge.to]];
        if (vnode1.node_id > vnode2.node_id)
            continue;

        ImVec2 pos1 = transform(vnode1.pos[0], vnode1.pos[1]);
        ImVec2 pos2 = transform(vnode2.pos[0], vnode2.pos[1]);

        // Safe payload extraction
        auto* node_idx_ptr = std::any_cast<std::size_t>(&vnode2.payload);
        if (!node_idx_ptr || *node_idx_ptr >= nodes.size())
            continue;

        const auto &node2 = nodes[*node_idx_ptr];
        float sc_ratio = static_cast<float>(node2.synapse_count) / max_synapses;
        float base_thickness = params_.edge_thickness_min + thickness_range * sc_ratio;
        float thickness = base_thickness * zoom_level_;

        draw_list->AddLine(pos1, pos2, edge_color, thickness);
    }

    // Draw nodes
    for (const auto &vis_node : vis_nodes) {
        // Safe payload extraction
        auto* node_idx_ptr = std::any_cast<std::size_t>(&vis_node.payload);
        if (!node_idx_ptr || *node_idx_ptr >= nodes.size())
            continue;

        const auto &node = nodes[*node_idx_ptr];
        ImVec2 pos = transform(vis_node.pos[0], vis_node.pos[1]);

        // Skip nodes outside visible area
        if (pos.x < canvas_pos.x - 50 || pos.x > canvas_pos.x + canvas_size.x + 50 ||
            pos.y < canvas_pos.y - 50 || pos.y > canvas_pos.y + canvas_size.y + 50)
            continue;

        // Only show textures for the displayed neuron's leaf nodes
        bool show_texture = (display_neuron_idx != SIZE_MAX) &&
                           (node.neuron_idx == display_neuron_idx) &&
                           (!node.pixel_data.empty()) &&
                           (node.texture.is_valid());

        if (show_texture) {
            // Scale texture size inversely proportional to representation dimensions
            // so that larger representations (64x64) display at similar visual size as smaller ones (28x28)
            constexpr float reference_size = 28.0f;  // MNIST size as reference
            float max_dim = std::max(static_cast<float>(node.texture.get_width()),
                                     static_cast<float>(node.texture.get_height()));
            float size_scale = reference_size / std::max(max_dim, 1.0f);
            float scale = canvas_size.x * zoom_level_ * size_scale / params_.texture_scale_divisor;
            float w = static_cast<float>(node.texture.get_width()) * scale;
            float h = static_cast<float>(node.texture.get_height()) * scale;
            ImVec2 tex_pos(pos.x - w / 2, pos.y - h / 2);
            draw_list->AddImage(node.texture.as_imgui_texture(),
                tex_pos, ImVec2(tex_pos.x + w, tex_pos.y + h));
        } else {
            float base_radius = node.segment_idx == 0 ? params_.root_node_radius : params_.branch_node_radius;
            float radius = base_radius * zoom_level_;
            ImU32 color = node.segment_idx == 0 ? root_color : branch_color;
            // Highlight selected or hovered neuron's root node
            if (node.segment_idx == 0 &&
                (node.neuron_idx == selected_neuron_idx_ || node.neuron_idx == hovered_neuron_idx_)) {
                // Draw a slightly larger highlight ring
                draw_list->AddCircle(pos, radius + 3.0f * zoom_level_, IM_COL32(255, 255, 255, 200), 0, 2.0f);
            }
            draw_list->AddCircleFilled(pos, radius, color);
        }
    }

    // Restore clipping
    draw_list->PopClipRect();

    ImGui::End();
    return open;
}

// ============================================================================
// vis_instance adapter for registry integration
// ============================================================================

namespace {

/**
 * Synchronization states for thread-safe visualization updates.
 */
enum class sync_state : int {
    idle = 0,       // Normal operation, no update pending
    requested = 1,  // Sim thread requests update, waiting for GUI
    updating = 2    // GUI is performing update, sim still waiting
};

constexpr auto SYNC_TIMEOUT = std::chrono::milliseconds(100);

/**
 * Adapter that wraps ngm_tree_vis to implement vis_instance interface.
 * This adapter manages its own fbgd instance for layout.
 * Provides thread-safe updates synchronized with the simulation thread.
 */
class ngm_tree_vis_instance : public coast::vis::vis_instance {
public:
    ngm_tree_vis_instance(ngm2::neuron_group_t& ng,
                          sim::env& sim_env,
                          const coast::vis_hints_t& hints,
                          const std::string& hints_key,
                          std::string label)
        : ng_(ng)
        , sim_env_(sim_env)
        , label_(std::move(label))
    {
        // Get hints using hints_key (entity_cfg)
        coast::ngm_tree_vis_hint_t hint;
        auto it = hints.ngm2.neuron_groups.find(hints_key);
        if (it != hints.ngm2.neuron_groups.end()) {
            hint = it->second.ngm_tree_vis;
        } else {
            auto def_it = hints.ngm2.neuron_groups.find("default");
            if (def_it != hints.ngm2.neuron_groups.end()) {
                hint = def_it->second.ngm_tree_vis;
            }
        }

        update_interval_ = static_cast<std::size_t>(hint.update_interval);

        // Convert hint to params
        coast::ngm_tree_vis::params_t params;
        params.rep_width = hint.rep_width;
        params.rep_height = hint.rep_height;
        params.bg_r = static_cast<uint8_t>(hint.bg_r);
        params.bg_g = static_cast<uint8_t>(hint.bg_g);
        params.bg_b = static_cast<uint8_t>(hint.bg_b);
        params.bg_a = static_cast<uint8_t>(hint.bg_a);
        params.edge_thickness_min = hint.edge_thickness_min;
        params.edge_thickness_max = hint.edge_thickness_max;
        params.edge_r = static_cast<uint8_t>(hint.edge_r);
        params.edge_g = static_cast<uint8_t>(hint.edge_g);
        params.edge_b = static_cast<uint8_t>(hint.edge_b);
        params.edge_a = static_cast<uint8_t>(hint.edge_a);
        params.texture_scale_divisor = hint.texture_scale_divisor;
        params.root_node_radius = hint.root_node_radius;
        params.branch_node_radius = hint.branch_node_radius;
        params.root_r = static_cast<uint8_t>(hint.root_r);
        params.root_g = static_cast<uint8_t>(hint.root_g);
        params.root_b = static_cast<uint8_t>(hint.root_b);
        params.root_a = static_cast<uint8_t>(hint.root_a);
        params.branch_r = static_cast<uint8_t>(hint.branch_r);
        params.branch_g = static_cast<uint8_t>(hint.branch_g);
        params.branch_b = static_cast<uint8_t>(hint.branch_b);
        params.branch_a = static_cast<uint8_t>(hint.branch_a);

        vis_.set_params(params);

        // Set a larger region size to accommodate growing dendritic trees
        fbgd_.set_region_size(5.0f);

        vis_.create_model(fbgd_, ng_);

        // Initial update to build lookup tables (required before accessing node positions)
        fbgd_.update();

        // Set initial root positions based on tree sizes
        vis_.update_root_positions(fbgd_, ng_);

        // Register post-swap hook for thread-safe synchronization
        hook_id_ = sim_env_.set_post_swap_hook([this]() {
            on_post_swap();
        });
    }

    ~ngm_tree_vis_instance() override
    {
        // Unregister the hook before destruction
        sim_env_.remove_post_swap_hook(hook_id_);

        // Ensure we're not in the middle of an update
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            std::this_thread::yield();
        }
    }

    /**
     * Called from GUI thread every frame.
     * Checks if sim thread requested an update and performs it if so.
     */
    void update() override
    {
        if (!visible_) return;

        // Check if sim thread has requested an update
        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::updating),
                                                 std::memory_order_acq_rel)) {
            // Safe to read from neuron_group
            vis_.update_model(fbgd_, ng_);
            fbgd_.update();  // Rebuild lookup tables before accessing positions
            vis_.update_root_positions(fbgd_, ng_);  // Update spacing based on tree sizes

            // Signal completion
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override
    {
        if (!visible_) return true;

        // Animate tree positions (moves entire trees towards their targets)
        vis_.animate_tree_positions(fbgd_, 0.05f);

        // Update force-based layout every frame for smooth animation
        fbgd_.update();

        // Call fbgd's draw callback to get latest node/edge data
        bool keep_open = true;

        fbgd_.set_draw_func([this, &keep_open](
            const std::vector<coast::fbgd::node>& nodes,
            const std::vector<coast::fbgd::edge>& edges,
            const std::vector<uint32_t>& node_lu)
        {
            keep_open = vis_.render_imgui(label_.c_str(), nodes, edges, node_lu);
        });

        fbgd_.draw();

        // Check if spacing slider changed and update root positions
        if (vis_.is_spacing_dirty()) {
            vis_.clear_spacing_dirty();
            vis_.update_root_positions(fbgd_, ng_);
        }

        return keep_open;
    }

    void free_resources() override
    {
        vis_.free_resources();
    }

    [[nodiscard]] bool is_visible() const override { return visible_; }
    void set_visible(bool v) override { visible_ = v; }

private:
    /**
     * Called from simulation thread via post_swap_hook.
     */
    void on_post_swap()
    {
        inc_on_exit inc{step_count_};  // Increment step_count_ on function exit

        if (update_interval_ == 0) return;
        if (!visible_) return;

        // Check if colormap changed (needs immediate update)
        bool colormap_changed = vis_.is_colormap_dirty();
        if (colormap_changed) {
            vis_.clear_colormap_dirty();
        }

        // Update if interval elapsed OR colormap changed
        if (step_count_ % update_interval_ != 0 && !colormap_changed) return;

        int expected = static_cast<int>(sync_state::idle);
        if (!sync_state_.compare_exchange_strong(expected,
                                                  static_cast<int>(sync_state::requested),
                                                  std::memory_order_acq_rel)) {
            return;
        }

        auto start = std::chrono::steady_clock::now();
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            if (std::chrono::steady_clock::now() - start > SYNC_TIMEOUT) {
                expected = static_cast<int>(sync_state::requested);
                if (sync_state_.compare_exchange_strong(expected,
                                                         static_cast<int>(sync_state::idle),
                                                         std::memory_order_acq_rel)) {
                    break;
                }
            }
            std::this_thread::yield();
        }
    }

    ngm2::neuron_group_t& ng_;
    sim::env& sim_env_;
    std::string label_;
    coast::fbgd fbgd_;
    coast::ngm_tree_vis vis_;
    bool visible_ = true;

    // Synchronization state
    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_ = 10000;
    std::size_t step_count_ = 0;
    std::size_t hook_id_ = 0;
};

/**
 * Factory function for creating ngm_tree_vis instances.
 */
std::unique_ptr<coast::vis::vis_instance> create_ngm_tree_vis(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    // The entity must be a neuron_group_t
    auto* ng = dynamic_cast<ngm2::neuron_group_t*>(&entity);
    if (!ng) {
        return nullptr;
    }
    return std::make_unique<ngm_tree_vis_instance>(*ng, sim_env, hints, hints_key, label);
}

/**
 * Static registrar - registers at program startup.
 */
struct ngm_tree_vis_registrar {
    ngm_tree_vis_registrar()
    {
        coast::vis::vis_info info {
            "ngm_tree_vis",
            "Neuron Tree",
            "Hierarchical tree view showing dendrite structure with force-directed layout",
            coast::vis::vis_target::sim_entity,
            { sim::io_entity_type_t::ngm2_neuron_group }
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm_tree_vis);
    }
};

static ngm_tree_vis_registrar registrar_instance;

} // anonymous namespace

} // coast
