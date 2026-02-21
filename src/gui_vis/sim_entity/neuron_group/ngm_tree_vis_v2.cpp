//
// Neuron tree visualization using gui_layout architecture
//
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <unordered_map>

#include "sim_entity/neuron_group/ngm_tree_vis_v2.h"
#include "vis_registry.h"
#include "sim_env.h"
#include "inc_on_exit.h"
#include "imgui.h"
#include "imgui_scaling.h"

namespace coast {

// ============================================================================
// tree_node_renderer implementation
// ============================================================================

tree_node_renderer::tree_node_renderer(const ngm_tree_vis_v2_params_t& params)
    : params_(&params)
{
}

void tree_node_renderer::render(ImDrawList* draw_list,
                                 const layout::node_t& node,
                                 ImVec2 screen_pos,
                                 float zoom,
                                 bool selected,
                                 bool hovered) {
    // Extract node data
    const tree_node_data* data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<tree_node_data>>(node.user_data);
            data = ptr.get();
        } catch (...) {
            data = nullptr;
        }
    }

    if (!data || !params_) {
        // Draw placeholder
        float radius = 5.0f * zoom;
        draw_list->AddCircleFilled(screen_pos, radius, IM_COL32(100, 100, 100, 200));
        return;
    }

    const auto& p = *params_;
    bool is_display_neuron = (display_neurons_.count(data->neuron_idx) > 0);

    // Check if this node belongs to selected/hovered neuron and should show texture
    bool show_texture = is_display_neuron &&
                        !data->pixel_data.empty() &&
                        data->texture.is_valid();

    if (show_texture) {
        // Scale texture size inversely proportional to representation dimensions
        constexpr float reference_size = 28.0f;  // MNIST size as reference
        float max_dim = std::max(static_cast<float>(data->texture.get_width()),
                                 static_cast<float>(data->texture.get_height()));
        float size_scale = reference_size / std::max(max_dim, 1.0f);
        float scale = canvas_size_.x * zoom * size_scale / p.texture_scale_divisor;
        float w = static_cast<float>(data->texture.get_width()) * scale;
        float h = static_cast<float>(data->texture.get_height()) * scale;
        ImVec2 tex_pos(screen_pos.x - w / 2.0f, screen_pos.y - h / 2.0f);
        draw_list->AddImage(data->texture.as_imgui_texture(),
            tex_pos, ImVec2(tex_pos.x + w, tex_pos.y + h));
    } else {
        // Draw circle based on node type
        float base_radius = (data->type == tree_node_type::root)
            ? p.root_node_radius : p.branch_node_radius;
        float radius = base_radius * zoom;

        ImU32 color;
        if (data->type == tree_node_type::root) {
            color = IM_COL32(p.root_r, p.root_g, p.root_b, p.root_a);

            // Highlight selected or hovered neuron's root node
            if (selected || hovered) {
                draw_list->AddCircle(screen_pos, radius + 3.0f * zoom,
                    IM_COL32(255, 255, 255, 200), 0, 2.0f);
            }
        } else {
            color = IM_COL32(p.branch_r, p.branch_g, p.branch_b, p.branch_a);
        }

        draw_list->AddCircleFilled(screen_pos, radius, color);
    }
}

ImVec2 tree_node_renderer::get_size(const layout::node_t& node, float zoom) {
    const tree_node_data* data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<tree_node_data>>(node.user_data);
            data = ptr.get();
        } catch (...) {
            data = nullptr;
        }
    }

    if (!data || !params_) {
        float size = 10.0f * zoom;
        return ImVec2(size, size);
    }

    const auto& p = *params_;

    // For texture nodes that are being displayed
    bool is_display_neuron = (display_neurons_.count(data->neuron_idx) > 0);
    if (is_display_neuron && !data->pixel_data.empty() && data->texture.is_valid()) {
        constexpr float reference_size = 28.0f;
        float max_dim = std::max(static_cast<float>(data->texture.get_width()),
                                 static_cast<float>(data->texture.get_height()));
        float size_scale = reference_size / std::max(max_dim, 1.0f);
        float scale = canvas_size_.x * zoom * size_scale / p.texture_scale_divisor;
        float w = static_cast<float>(data->texture.get_width()) * scale;
        float h = static_cast<float>(data->texture.get_height()) * scale;
        return ImVec2(w, h);
    }

    // For circle nodes
    float base_radius = (data->type == tree_node_type::root)
        ? p.root_node_radius : p.branch_node_radius;
    float diameter = base_radius * 2.0f * zoom;
    return ImVec2(diameter, diameter);
}

// ============================================================================
// ngm_tree_vis_v2 implementation
// ============================================================================

ngm_tree_vis_v2::ngm_tree_vis_v2()
    : region_size_(500.0f)  // Use larger coordinate space for graph_canvas
    , renderer_(params_)
{
    renderer_.set_params(params_);
    canvas_.set_node_renderer(&renderer_);
    canvas_.set_graph(&graph_);
    canvas_.set_dragging_enabled(false);  // Disable node dragging (force layout handles positioning)
    canvas_.set_smooth_animation(false);  // Disable smooth animation (fixes zoom-towards-mouse)
    setup_edge_renderer();
}

ngm_tree_vis_v2::ngm_tree_vis_v2(const ngm_tree_vis_v2_params_t& params)
    : params_(params)
    , region_size_(500.0f)  // Use larger coordinate space for graph_canvas
    , renderer_(params_)
{
    renderer_.set_params(params_);
    canvas_.set_node_renderer(&renderer_);
    canvas_.set_graph(&graph_);
    canvas_.set_dragging_enabled(false);  // Disable node dragging (force layout handles positioning)
    setup_edge_renderer();
}

ngm_tree_vis_v2::~ngm_tree_vis_v2() {
    free_resources();
}

void ngm_tree_vis_v2::setup_edge_renderer() {
    canvas_.set_edge_renderer([this](ImDrawList* draw_list,
                                      const layout::edge_t& edge,
                                      ImVec2 from_screen,
                                      ImVec2 to_screen,
                                      float zoom) {
        // Get synapse count from target node
        uint32_t synapse_count = 0;
        if (graph_.has_node(edge.to)) {
            const auto& to_node = graph_.node(edge.to);
            if (to_node.user_data.has_value()) {
                try {
                    auto ptr = std::any_cast<std::shared_ptr<tree_node_data>>(to_node.user_data);
                    if (ptr) {
                        synapse_count = ptr->synapse_count;
                    }
                } catch (...) {}
            }
        }

        // Scale thickness based on synapse count
        float max_synapses = static_cast<float>(params_.rep_width * params_.rep_height);
        float sc_ratio = static_cast<float>(synapse_count) / max_synapses;
        float thickness_range = params_.edge_thickness_max - params_.edge_thickness_min;
        float base_thickness = params_.edge_thickness_min + thickness_range * sc_ratio;
        float thickness = base_thickness * zoom;

        ImU32 edge_color = IM_COL32(params_.edge_r, params_.edge_g,
                                    params_.edge_b, params_.edge_a);
        draw_list->AddLine(from_screen, to_screen, edge_color, thickness);
    });
}

void ngm_tree_vis_v2::get_pixel_data(
    int width, int height,
    uint16_t leaf_idx,
    std::size_t max_idx,
    const ngm2::dendrite_t::synapses_t& synapses,
    std::vector<color_t>& output)
{
    const std::size_t buffer_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    output.resize(buffer_size);

    // Fill with black initially
    std::fill(output.begin(), output.end(), color_t{0, 0, 0, 255});

    tmp_tree_.clear();
    tmp_tree_.resize(max_idx + 1);

    tmp_tree_[leaf_idx] = 1;
    while (leaf_idx > 1) {
        leaf_idx /= 2;
        tmp_tree_[leaf_idx] = 1;
    }

    std::size_t pix = 0;
    std::size_t syn_cnt = synapses.size();

    if (params_.num_channels == 3) {
        // RGB mode
        for (std::size_t i = 0; i + 2 < syn_cnt && pix < buffer_size; i += 3) {
            if (tmp_tree_[synapses.segment_idx[i]]) {
                output[pix] = color_t{
                    static_cast<uint8_t>(std::clamp(synapses.permanence[i + 0] * 255.0f, 0.0f, 255.0f)),
                    static_cast<uint8_t>(std::clamp(synapses.permanence[i + 1] * 255.0f, 0.0f, 255.0f)),
                    static_cast<uint8_t>(std::clamp(synapses.permanence[i + 2] * 255.0f, 0.0f, 255.0f)),
                    255
                };
                ++pix;
            }
        }
    } else {
        // Grayscale with colormap
        for (std::size_t i = 0; i < syn_cnt && pix < buffer_size; ++i) {
            if (tmp_tree_[synapses.segment_idx[i]]) {
                float normalized = std::clamp(synapses.permanence[i], 0.0f, 1.0f);
                output[pix] = apply_colormap(colormap_, normalized);
                ++pix;
            }
        }
    }
}

tree_node_data* ngm_tree_vis_v2::get_node_data(layout::node_id_t id) {
    if (!graph_.has_node(id)) return nullptr;
    auto& node = graph_.node(id);
    if (!node.user_data.has_value()) return nullptr;
    try {
        auto ptr = std::any_cast<std::shared_ptr<tree_node_data>>(node.user_data);
        return ptr.get();
    } catch (...) {
        return nullptr;
    }
}

void ngm_tree_vis_v2::create_model(const ngm2::neuron_group_t& ng) {
    graph_.clear();
    nodes_.clear();
    neuron_lu_.clear();
    branch_start_lu_.clear();
    branch_lu_.clear();

    // Determine size of nodes array
    std::size_t nodes_size = 0;
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        const ngm2::neuron_t& cur_neuron = ng.get_neuron(neuron_idx);
        neuron_lu_.push_back(nodes_size);
        branch_start_lu_.push_back(branch_lu_.size());
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t& cur_branch = cur_neuron.get_dendrite(branch_idx);
            std::size_t segment_cnt = cur_branch.get_max_segment_idx() + 1;
            branch_lu_.push_back(nodes_size);
            nodes_size += segment_cnt;
        }
    }
    nodes_.resize(nodes_size);

    // Setup force layout - parameters can be tuned via debug UI
    layout_ = std::make_unique<layout::force_layout>(graph_);
    layout::force_params fp;
    fp.repulsion = repulsion_;
    fp.attraction = attraction_;
    fp.damping = 0.8f;
    fp.gravity = 0.0f;
    fp.max_velocity = 10.0f;
    fp.min_distance = 1.0f;
    layout_->set_params(fp);

    // Create nodes
    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    const float grid_spacing = region_size_ / static_cast<float>(grid_cnt + 2);

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        const ngm2::neuron_t& cur_neuron = ng.get_neuron(neuron_idx);
        float xp = static_cast<float>(neuron_idx % grid_cnt) * grid_spacing + grid_spacing * 1.5f;
        float yp = static_cast<float>(neuron_idx / grid_cnt) * grid_spacing + grid_spacing * 1.5f;

        // Create root node (fixed in grid position)
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        auto root_data = std::make_shared<tree_node_data>();
        root_data->neuron_idx = neuron_idx;
        root_data->branch_idx = 0;
        root_data->segment_idx = 0;
        root_data->type = tree_node_type::root;

        layout::node_t root_node;
        root_node.x = xp;
        root_node.y = yp;
        root_node.fixed = true;
        root_node.radius = params_.root_node_radius;
        root_node.user_data = root_data;
        layout::node_id_t root_id = graph_.add_node(root_node);

        nodes_[root_node_idx] = {root_id, root_data};

        // Go over all branches
        std::size_t branch_base_idx = branch_start_lu_[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t& cur_branch = cur_neuron.get_dendrite(branch_idx);
            auto leaf_mask = cur_branch.get_leaf_mask();
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu_[branch_base_idx + branch_idx];
            const auto& synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();

            // First pass: mark segments that need to exist
            std::vector<uint8_t> segments_needed(max_segment + 1, 0);
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                uint16_t seg = synapses.segment_idx[i];
                while (seg >= 1 && segments_needed[seg] == 0) {
                    segments_needed[seg] = 1;
                    seg /= 2;
                }
            }

            // Second pass: create nodes for all needed segments
            for (uint16_t seg = 1; seg <= max_segment; ++seg) {
                if (segments_needed[seg] == 0) continue;
                if (segment_base + seg >= nodes_.size()) continue;

                auto& cur_node = nodes_[segment_base + seg];
                if (cur_node.node_id == layout::invalid_node_id) {
                    auto seg_data = std::make_shared<tree_node_data>();
                    seg_data->neuron_idx = neuron_idx;
                    seg_data->branch_idx = branch_idx;
                    seg_data->segment_idx = seg;
                    seg_data->type = (seg < leaf_mask.size() && leaf_mask[seg])
                        ? tree_node_type::leaf : tree_node_type::branch;

                    // Create texture for leaf nodes
                    if (seg_data->type == tree_node_type::leaf) {
                        get_pixel_data(params_.rep_width, params_.rep_height,
                            seg, max_segment, synapses, seg_data->pixel_data);
                        seg_data->texture = gl_texture(params_.rep_width, params_.rep_height,
                            seg_data->pixel_data.data());
                    }

                    layout::node_t seg_node;
                    seg_node.x = xp + rdis_(rgen_) * (grid_spacing / 3.0f);
                    seg_node.y = yp + rdis_(rgen_) * (grid_spacing / 3.0f);
                    seg_node.fixed = false;
                    seg_node.radius = params_.branch_node_radius;
                    seg_node.user_data = seg_data;
                    layout::node_id_t seg_id = graph_.add_node(seg_node);

                    cur_node = {seg_id, seg_data};
                }
            }

            // Third pass: count synapses per segment
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                if (segment_base + synapses.segment_idx[i] < nodes_.size()) {
                    auto& n = nodes_[segment_base + synapses.segment_idx[i]];
                    if (n.data) {
                        n.data->synapse_count++;
                        max_synapse_count_ = std::max(max_synapse_count_, n.data->synapse_count);
                    }
                }
            }

            // Fourth pass: create edges
            // Edge ideal length should be small relative to grid spacing for tight trees
            const float edge_length = grid_spacing * 0.15f;
            const std::size_t max_idx = cur_branch.get_max_segment_idx();
            uint32_t cur_start = static_cast<uint32_t>((max_idx + 1) / 2);
            while (cur_start > 1) {
                for (std::size_t seg_idx = cur_start; seg_idx < cur_start * 2 && seg_idx <= max_idx; ++seg_idx) {
                    if (segment_base + seg_idx >= nodes_.size()) continue;
                    auto& child_node = nodes_[segment_base + seg_idx];
                    if (child_node.node_id != layout::invalid_node_id) {
                        auto& parent_node = nodes_[segment_base + seg_idx / 2];
                        if (parent_node.node_id != layout::invalid_node_id) {
                            graph_.add_edge(parent_node.node_id, child_node.node_id,
                                1.0f, edge_length);
                        }
                    }
                }
                cur_start /= 2;
            }
            // Connect segment 1 to root
            if (segment_base + 1 < nodes_.size() &&
                nodes_[segment_base + 1].node_id != layout::invalid_node_id) {
                graph_.add_edge(nodes_[root_node_idx].node_id,
                    nodes_[segment_base + 1].node_id, 1.0f, edge_length);
            }
        }
    }

    // Start animated settling (will run in render loop)
    start_animated_settling(200);
    initial_settling_ = true;  // Mark as initial settling for view reset

    // Note: view will be set on first render when canvas_size_ is known
    needs_fit_view_ = true;
}

void ngm_tree_vis_v2::update_model(const ngm2::neuron_group_t& ng) {
    // Reset synapse counts
    for (auto& node : nodes_) {
        if (node.data) {
            node.data->synapse_count = 0;
        }
    }
    max_synapse_count_ = 1;

    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        const ngm2::neuron_t& cur_neuron = ng.get_neuron(neuron_idx);
        std::size_t root_node_idx = neuron_lu_[neuron_idx];

        std::size_t branch_base_idx = branch_start_lu_[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t& cur_branch = cur_neuron.get_dendrite(branch_idx);
            auto leaf_mask = cur_branch.get_leaf_mask();
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu_[branch_base_idx + branch_idx];
            const auto& synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();

            // Get root position for new nodes
            float root_x = 0.0f, root_y = 0.0f;
            if (graph_.has_node(nodes_[root_node_idx].node_id)) {
                const auto& root = graph_.node(nodes_[root_node_idx].node_id);
                root_x = root.x;
                root_y = root.y;
            }

            const float spawn_offset = region_size_ * 0.05f;

            // Track new nodes that need edges
            std::vector<uint16_t> new_nodes_needing_edges;
            std::unordered_map<uint16_t, std::array<float, 2>> new_node_positions;

            // Helper to get position of a segment
            auto get_seg_pos = [&](uint16_t seg) -> std::array<float, 2> {
                if (seg == 0) return {root_x, root_y};
                auto it = new_node_positions.find(seg);
                if (it != new_node_positions.end()) {
                    return it->second;
                }
                if (segment_base + seg < nodes_.size()) {
                    auto& n = nodes_[segment_base + seg];
                    if (n.node_id != layout::invalid_node_id && graph_.has_node(n.node_id)) {
                        const auto& node = graph_.node(n.node_id);
                        return {node.x, node.y};
                    }
                }
                return {root_x, root_y};
            };

            // Process synapses
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                uint16_t seg = synapses.segment_idx[i];
                if (segment_base + seg >= nodes_.size()) continue;

                auto& cur_node = nodes_[segment_base + seg];

                // Count synapse
                if (cur_node.data) {
                    cur_node.data->synapse_count++;
                    max_synapse_count_ = std::max(max_synapse_count_, cur_node.data->synapse_count);
                }

                // Create node if it doesn't exist
                if (cur_node.node_id == layout::invalid_node_id) {
                    // Build path from seg to root
                    std::vector<uint16_t> path;
                    uint16_t ancestor = seg;
                    while (ancestor >= 1) {
                        path.push_back(ancestor);
                        if (ancestor == 1) break;
                        ancestor /= 2;
                    }

                    // Process from root towards seg
                    for (auto it = path.rbegin(); it != path.rend(); ++it) {
                        uint16_t s = *it;
                        if (segment_base + s >= nodes_.size()) continue;
                        auto& node = nodes_[segment_base + s];
                        if (node.node_id != layout::invalid_node_id) continue;

                        // Get parent position
                        uint16_t parent_seg = (s == 1) ? 0 : s / 2;
                        auto parent_pos = get_seg_pos(parent_seg);
                        float nx = parent_pos[0] + rdis_(rgen_) * spawn_offset;
                        float ny = parent_pos[1] + rdis_(rgen_) * spawn_offset;

                        new_node_positions[s] = {nx, ny};

                        // Create node data
                        auto seg_data = std::make_shared<tree_node_data>();
                        seg_data->neuron_idx = neuron_idx;
                        seg_data->branch_idx = branch_idx;
                        seg_data->segment_idx = s;
                        seg_data->type = (s < leaf_mask.size() && leaf_mask[s])
                            ? tree_node_type::leaf : tree_node_type::branch;

                        // Create texture for leaf nodes
                        if (s == seg && seg_data->type == tree_node_type::leaf) {
                            get_pixel_data(params_.rep_width, params_.rep_height,
                                s, max_segment, synapses, seg_data->pixel_data);
                            seg_data->texture = gl_texture(params_.rep_width, params_.rep_height,
                                seg_data->pixel_data.data());
                        }

                        layout::node_t seg_node;
                        seg_node.x = nx;
                        seg_node.y = ny;
                        seg_node.fixed = false;
                        seg_node.radius = params_.branch_node_radius;
                        seg_node.user_data = seg_data;
                        layout::node_id_t seg_id = graph_.add_node(seg_node);

                        node = {seg_id, seg_data};
                        new_nodes_needing_edges.push_back(s);
                    }
                }
            }

            // Add edges for new nodes
            auto grid_cnt_update = static_cast<std::size_t>(
                std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
            float grid_spacing_update = region_size_ / static_cast<float>(grid_cnt_update + 2);
            float edge_length_update = grid_spacing_update * 0.15f;
            std::sort(new_nodes_needing_edges.begin(), new_nodes_needing_edges.end());
            for (uint16_t seg : new_nodes_needing_edges) {
                if (segment_base + seg >= nodes_.size()) continue;
                auto& node = nodes_[segment_base + seg];
                if (seg == 1) {
                    graph_.add_edge(nodes_[root_node_idx].node_id, node.node_id,
                        1.0f, edge_length_update);
                } else if (segment_base + seg / 2 < nodes_.size() &&
                           nodes_[segment_base + seg / 2].node_id != layout::invalid_node_id) {
                    graph_.add_edge(nodes_[segment_base + seg / 2].node_id, node.node_id,
                        1.0f, edge_length_update);
                }
            }
        }
    }

    // Start animated settling for new nodes
    // Note: Don't call update_tree_spacing() here - new nodes haven't settled yet
    // so tree radii would appear smaller, causing spacing to collapse.
    // Trees keep their established positions from initial settling.
    start_animated_settling(50);
}

void ngm_tree_vis_v2::update_representations(const ngm2::neuron_group_t& ng) {
    std::size_t neuron_cnt = ng.get_neuron_count();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        const ngm2::neuron_t& cur_neuron = ng.get_neuron(neuron_idx);
        std::size_t branch_base_idx = branch_start_lu_[neuron_idx];
        std::size_t branch_cnt = cur_neuron.get_dendrite_count();
        for (std::size_t branch_idx = 0; branch_idx < branch_cnt; ++branch_idx) {
            const ngm2::dendrite_t& cur_branch = cur_neuron.get_dendrite(branch_idx);
            std::size_t max_segment = cur_branch.get_max_segment_idx();
            std::size_t segment_base = branch_lu_[branch_base_idx + branch_idx];
            const auto& synapses = cur_branch.get_synapses();
            std::size_t syn_cnt = synapses.size();
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                auto& cur_node = nodes_[segment_base + synapses.segment_idx[i]];
                if (cur_node.data && !cur_node.data->pixel_data.empty()) {
                    get_pixel_data(params_.rep_width, params_.rep_height,
                        synapses.segment_idx[i], max_segment, synapses,
                        cur_node.data->pixel_data);
                    cur_node.data->texture.update(cur_node.data->pixel_data);
                }
            }
        }
    }
}

void ngm_tree_vis_v2::update_root_positions(const ngm2::neuron_group_t& ng) {
    if (neuron_lu_.empty()) return;

    std::size_t neuron_cnt = neuron_lu_.size();
    std::size_t ng_neuron_cnt = ng.get_neuron_count();

    if (neuron_cnt > ng_neuron_cnt) {
        neuron_cnt = ng_neuron_cnt;
    }
    if (neuron_cnt == 0) return;

    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    if (grid_cnt == 0) grid_cnt = 1;

    const float base_cell_size = region_size_ / static_cast<float>(grid_cnt);
    const float cell_size = base_cell_size * spacing_multiplier_;
    const float margin = cell_size * 0.5f;

    neuron_target_positions_.resize(neuron_cnt);

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu_.size()) break;
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        if (root_node_idx >= nodes_.size() ||
            nodes_[root_node_idx].node_id == layout::invalid_node_id) continue;

        std::size_t col = neuron_idx % grid_cnt;
        std::size_t row = neuron_idx / grid_cnt;

        float target_x = margin + (static_cast<float>(col) + 0.5f) * cell_size;
        float target_y = margin + (static_cast<float>(row) + 0.5f) * cell_size;

        neuron_target_positions_[neuron_idx] = {target_x, target_y};
    }
}

void ngm_tree_vis_v2::animate_tree_positions(float lerp_factor) {
    if (neuron_lu_.empty() || neuron_target_positions_.empty()) return;

    std::size_t neuron_cnt = std::min(neuron_lu_.size(), neuron_target_positions_.size());

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        if (root_node_idx >= nodes_.size() ||
            nodes_[root_node_idx].node_id == layout::invalid_node_id) continue;

        if (!graph_.has_node(nodes_[root_node_idx].node_id)) continue;

        // Get current root position and target
        const auto& root = graph_.node(nodes_[root_node_idx].node_id);
        const auto& target = neuron_target_positions_[neuron_idx];

        float dx = (target[0] - root.x) * lerp_factor;
        float dy = (target[1] - root.y) * lerp_factor;

        if (std::abs(dx) < 0.0001f && std::abs(dy) < 0.0001f) continue;

        // Move all nodes belonging to this neuron by the same delta
        for (const auto& node_info : nodes_) {
            if (node_info.node_id == layout::invalid_node_id) continue;
            if (!node_info.data || node_info.data->neuron_idx != neuron_idx) continue;
            if (!graph_.has_node(node_info.node_id)) continue;

            auto& node = graph_.node(node_info.node_id);
            node.x += dx;
            node.y += dy;
        }
    }
}

void ngm_tree_vis_v2::apply_per_neuron_repulsion() {
    if (neuron_lu_.empty() || repulsion_ <= 0.0f) return;

    const float min_dist = 1.0f;
    // Texture size adds to effective node radius when textures are visible
    const float texture_radius = params_.rep_width * 0.5f;

    // For each neuron, apply repulsion only between nodes of that neuron
    std::size_t neuron_cnt = neuron_lu_.size();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        // Check if this neuron's textures are visible
        bool textures_visible = selected_neurons_.count(neuron_idx) > 0 ||
                                (show_on_hover_ && hovered_neuron_idx_ == neuron_idx);

        // Collect all nodes belonging to this neuron
        std::vector<layout::node_id_t> neuron_nodes;
        for (const auto& node_info : nodes_) {
            if (node_info.node_id == layout::invalid_node_id) continue;
            if (!node_info.data || node_info.data->neuron_idx != neuron_idx) continue;
            neuron_nodes.push_back(node_info.node_id);
        }

        // Apply repulsion between all pairs within this neuron
        for (std::size_t i = 0; i < neuron_nodes.size(); ++i) {
            if (!graph_.has_node(neuron_nodes[i])) continue;
            auto& n1 = graph_.node(neuron_nodes[i]);
            if (n1.fixed) continue;  // Root nodes don't move

            // Get node data for n1
            tree_node_data* data1 = get_node_data(neuron_nodes[i]);
            bool n1_is_leaf = data1 && data1->type == tree_node_type::leaf;

            for (std::size_t j = 0; j < neuron_nodes.size(); ++j) {
                if (i == j) continue;
                if (!graph_.has_node(neuron_nodes[j])) continue;
                const auto& n2 = graph_.node(neuron_nodes[j]);

                // Get node data for n2
                tree_node_data* data2 = get_node_data(neuron_nodes[j]);
                bool n2_is_leaf = data2 && data2->type == tree_node_type::leaf;

                float dx = n1.x - n2.x;
                float dy = n1.y - n2.y;
                float dist_sq = dx * dx + dy * dy;
                float dist = std::sqrt(std::max(dist_sq, min_dist * min_dist));

                // Repulsive force (inverse square)
                float force = repulsion_ / (dist_sq + min_dist);

                // When textures are visible, leaf nodes need more space
                // Apply stronger repulsion if they're closer than their combined radii
                if (textures_visible && (n1_is_leaf || n2_is_leaf)) {
                    float combined_radius = texture_radius * ((n1_is_leaf ? 1.0f : 0.0f) +
                                                               (n2_is_leaf ? 1.0f : 0.0f));
                    if (dist < combined_radius * 2.0f) {
                        // Boost repulsion when nodes overlap or are close
                        float overlap_factor = combined_radius * 2.0f / std::max(dist, 1.0f);
                        force *= overlap_factor;
                    }
                }

                float inv_dist = 1.0f / dist;
                n1.vx += force * dx * inv_dist;
                n1.vy += force * dy * inv_dist;
            }
        }
    }
}

void ngm_tree_vis_v2::settle_step() {
    const float damping = 0.8f;
    const float max_velocity = 10.0f;

    // Apply per-neuron repulsion (not global)
    apply_per_neuron_repulsion();

    // Apply edge attraction
    for (const auto& edge : graph_.edges()) {
        if (!graph_.has_node(edge.from) || !graph_.has_node(edge.to)) continue;

        auto& n1 = graph_.node(edge.from);
        auto& n2 = graph_.node(edge.to);

        float dx = n2.x - n1.x;
        float dy = n2.y - n1.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        if (dist < 1.0f) continue;

        // Spring force towards ideal length
        float delta = dist - edge.ideal_length;
        float force = attraction_ * delta;
        float inv_dist = 1.0f / dist;
        float fx = force * dx * inv_dist;
        float fy = force * dy * inv_dist;

        if (!n1.fixed) {
            n1.vx += fx;
            n1.vy += fy;
        }
        if (!n2.fixed) {
            n2.vx -= fx;
            n2.vy -= fy;
        }
    }

    // Apply velocities and damping
    float max_vel_sq = max_velocity * max_velocity;
    for (auto& n : graph_.nodes()) {
        if (n.fixed) continue;

        // Limit velocity
        float vel_sq = n.vx * n.vx + n.vy * n.vy;
        if (vel_sq > max_vel_sq) {
            float scale = max_velocity / std::sqrt(vel_sq);
            n.vx *= scale;
            n.vy *= scale;
        }

        n.x += n.vx;
        n.y += n.vy;
        n.vx *= damping;
        n.vy *= damping;
    }
}

void ngm_tree_vis_v2::settle_with_per_neuron_forces(int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        settle_step();
    }
}

void ngm_tree_vis_v2::start_animated_settling(int iterations) {
    settling_active_ = true;
    settling_iterations_remaining_ = iterations;
}

void ngm_tree_vis_v2::update_tree_spacing() {
    if (neuron_lu_.empty()) return;

    std::size_t neuron_cnt = neuron_lu_.size();

    // Calculate bounding box radius for each neuron's tree
    std::vector<float> tree_radii(neuron_cnt, 0.0f);

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu_.size()) break;
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        if (root_node_idx >= nodes_.size() ||
            nodes_[root_node_idx].node_id == layout::invalid_node_id) continue;

        if (!graph_.has_node(nodes_[root_node_idx].node_id)) continue;
        const auto& root = graph_.node(nodes_[root_node_idx].node_id);

        // Find max distance from root to any node in this neuron's tree
        float max_radius = 0.0f;
        for (const auto& node_info : nodes_) {
            if (node_info.node_id == layout::invalid_node_id) continue;
            if (!node_info.data || node_info.data->neuron_idx != neuron_idx) continue;
            if (!graph_.has_node(node_info.node_id)) continue;

            const auto& node = graph_.node(node_info.node_id);
            float dx = node.x - root.x;
            float dy = node.y - root.y;
            float radius = std::sqrt(dx * dx + dy * dy);
            max_radius = std::max(max_radius, radius);
        }
        tree_radii[neuron_idx] = max_radius;
    }

    // Calculate grid layout with variable spacing based on tree sizes
    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    if (grid_cnt == 0) grid_cnt = 1;

    // Find max radius for base spacing calculation
    float max_tree_radius = 1.0f;
    for (float r : tree_radii) {
        max_tree_radius = std::max(max_tree_radius, r);
    }

    // Base cell size should accommodate the largest tree with some padding
    float min_cell_size = max_tree_radius * 2.5f * spacing_multiplier_;
    float base_cell_size = region_size_ / static_cast<float>(grid_cnt);
    float cell_size = std::max(base_cell_size, min_cell_size) * spacing_multiplier_;

    // Update target positions
    neuron_target_positions_.resize(neuron_cnt);
    float margin = cell_size * 0.5f;

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu_.size()) break;
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        if (root_node_idx >= nodes_.size() ||
            nodes_[root_node_idx].node_id == layout::invalid_node_id) continue;

        std::size_t col = neuron_idx % grid_cnt;
        std::size_t row = neuron_idx / grid_cnt;

        float target_x = margin + (static_cast<float>(col) + 0.5f) * cell_size;
        float target_y = margin + (static_cast<float>(row) + 0.5f) * cell_size;

        neuron_target_positions_[neuron_idx] = {target_x, target_y};
    }
}

void ngm_tree_vis_v2::free_resources() {
    for (auto& node : nodes_) {
        if (node.data) {
            node.data->texture.release();
            node.data->pixel_data.clear();
        }
    }
}

bool ngm_tree_vis_v2::render_imgui(const char* label) {
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin(label, &open);

    // Toolbar
    ImGui::Text("Zoom: %.0f%%", canvas_.zoom() * 100.0f);
    ImGui::SameLine();
    if (ImGui::Button("Reset View")) {
        canvas_.fit_graph_in_view();
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
        update_tree_spacing();
    }

    ImGui::Separator();

    // Get canvas size for renderer
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    renderer_.set_canvas_size(canvas_size);

    // Fit view when requested (uses actual graph bounds)
    if (needs_fit_view_ && canvas_size.x > 0 && canvas_size.y > 0) {
        needs_fit_view_ = false;
        canvas_.fit_graph_in_view();
    }

    // Run settling - either fast settling after model changes, or continuous slow settling
    if (settling_active_ && settling_iterations_remaining_ > 0) {
        // Fast settling after model updates
        int iters_this_frame = std::min(settling_iterations_remaining_, 5);
        for (int i = 0; i < iters_this_frame; ++i) {
            settle_step();
        }
        settling_iterations_remaining_ -= iters_this_frame;

        // Update tree spacing periodically during initial settling only
        if (initial_settling_ && settling_iterations_remaining_ % 20 == 0) {
            update_tree_spacing();
        }

        if (settling_iterations_remaining_ <= 0) {
            settling_active_ = false;
            if (initial_settling_) {
                update_tree_spacing();  // Final spacing update for initial settling only
                initial_settling_ = false;
            }
        }
    } else {
        // Continuous slow settling - keeps graph "alive" with subtle movement
        settle_step();
    }

    // Track total frames and reset view once after initial stabilization (~2 seconds)
    total_frames_rendered_++;
    if (!initial_view_reset_done_ && total_frames_rendered_ >= 120) {
        needs_fit_view_ = true;
        initial_view_reset_done_ = true;
    }

    // Animate tree positions (moves entire trees towards their grid targets)
    animate_tree_positions(0.05f);

    // Handle selection/hover for neuron display
    auto canvas_selected = canvas_.selected_node();
    auto hovered = canvas_.hovered_node();

    hovered_neuron_idx_ = SIZE_MAX;

    // Handle multi-select: detect if a new root node was clicked
    // canvas_.selected_node() gives us the most recently clicked node
    static std::optional<layout::node_id_t> last_canvas_selected;
    if (canvas_selected.has_value() && canvas_selected != last_canvas_selected) {
        auto* data = get_node_data(canvas_selected.value());
        if (data && data->type == tree_node_type::root) {
            ImGuiIO& io = ImGui::GetIO();
            if (io.KeyCtrl) {
                // Ctrl+click: toggle selection
                if (selected_neurons_.count(data->neuron_idx) > 0) {
                    selected_neurons_.erase(data->neuron_idx);
                } else {
                    selected_neurons_.insert(data->neuron_idx);
                }
            } else {
                // Regular click: replace selection
                selected_neurons_.clear();
                selected_neurons_.insert(data->neuron_idx);
            }
        }
        last_canvas_selected = canvas_selected;
    } else if (!canvas_selected.has_value() && last_canvas_selected.has_value()) {
        // Clicked on empty space without Ctrl - clear selection
        ImGuiIO& io = ImGui::GetIO();
        if (!io.KeyCtrl) {
            selected_neurons_.clear();
        }
        last_canvas_selected = std::nullopt;
    }

    // Handle hover
    if (show_on_hover_ && hovered.has_value()) {
        auto* data = get_node_data(hovered.value());
        if (data && data->type == tree_node_type::root) {
            hovered_neuron_idx_ = data->neuron_idx;
        }
    }

    // Build display set: all selected neurons + hovered neuron (if show_on_hover)
    renderer_.clear_display_neurons();
    for (std::size_t idx : selected_neurons_) {
        renderer_.add_display_neuron(idx);
    }
    if (show_on_hover_ && hovered_neuron_idx_ != SIZE_MAX) {
        renderer_.add_display_neuron(hovered_neuron_idx_);
    }

    // Set edge style
    layout::render::edge_style es;
    es.color = IM_COL32(params_.edge_r, params_.edge_g, params_.edge_b, params_.edge_a);
    es.thickness = params_.edge_thickness_min;
    canvas_.set_edge_style(es);

    // Render canvas with custom background
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    draw_list->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        IM_COL32(params_.bg_r, params_.bg_g, params_.bg_b, params_.bg_a));

    // Render the graph
    canvas_.render("##tree_canvas", canvas_size);

    ImGui::End();
    return open;
}

// ============================================================================
// vis_instance adapter for registry integration
// ============================================================================

namespace {

enum class sync_state : int {
    idle = 0,
    requested = 1,
    updating = 2
};

constexpr auto SYNC_TIMEOUT = std::chrono::milliseconds(100);

class ngm_tree_vis_v2_instance : public coast::vis::vis_instance {
public:
    ngm_tree_vis_v2_instance(ngm2::neuron_group_t& ng,
                              sim::env& sim_env,
                              const coast::vis_hints_t& hints,
                              const std::string& hints_key,
                              std::string label)
        : ng_(ng)
        , sim_env_(sim_env)
        , label_(std::move(label))
    {
        // Get hints
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
        ngm_tree_vis_v2_params_t params;
        params.rep_width = hint.rep_width;
        params.rep_height = hint.rep_height;
        params.num_channels = hint.num_channels;
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
        vis_.create_model(ng_);
        vis_.update_root_positions(ng_);

        hook_id_ = sim_env_.set_post_swap_hook([this]() {
            on_post_swap_signal();
        });
        barrier_hook_id_ = sim_env_.set_post_swap_barrier_hook([this]() {
            on_post_swap_wait();
        });
    }

    ~ngm_tree_vis_v2_instance() override {
        sim_env_.remove_post_swap_hook(hook_id_);
        sim_env_.remove_post_swap_barrier_hook(barrier_hook_id_);
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            std::this_thread::yield();
        }
    }

    void update() override {
        if (!visible_.load(std::memory_order_acquire)) return;

        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::updating),
                                                 std::memory_order_acq_rel)) {
            vis_.update_model(ng_);
            // Note: Don't call update_root_positions() here - it would reset
            // the dynamic spacing calculated by update_tree_spacing() during
            // initial settling. Trees keep their established positions.
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override {
        if (!visible_.load(std::memory_order_acquire)) return true;

        // Check if spacing slider changed
        if (vis_.is_spacing_dirty()) {
            vis_.clear_spacing_dirty();
            vis_.update_tree_spacing();  // Uses tree radii for proper spacing
        }

        // Check if colormap changed
        if (vis_.is_colormap_dirty()) {
            vis_.clear_colormap_dirty();
            vis_.update_representations(ng_);  // Regenerate textures with new colormap
        }

        return vis_.render_imgui(label_.c_str());
    }

    void free_resources() override {
        vis_.free_resources();
    }

    [[nodiscard]] bool is_visible() const override {
        return visible_.load(std::memory_order_acquire);
    }

    void set_visible(bool v) override {
        visible_.store(v, std::memory_order_release);
    }

private:
    void on_post_swap_signal() {
        inc_on_exit inc{step_count_};

        if (update_interval_ == 0) return;
        if (!visible_.load(std::memory_order_acquire)) return;

        bool colormap_changed = vis_.is_colormap_dirty();
        if (colormap_changed) {
            vis_.clear_colormap_dirty();
        }

        if (step_count_ % update_interval_ != 0 && !colormap_changed) return;

        int expected = static_cast<int>(sync_state::idle);
        if (!sync_state_.compare_exchange_strong(expected,
                                                  static_cast<int>(sync_state::requested),
                                                  std::memory_order_acq_rel)) {
            return;
        }

        pending_wait_ = true;
    }

    void on_post_swap_wait() {
        if (!pending_wait_) return;
        pending_wait_ = false;

        auto start = std::chrono::steady_clock::now();
        while (sync_state_.load(std::memory_order_acquire) != static_cast<int>(sync_state::idle)) {
            if (std::chrono::steady_clock::now() - start > SYNC_TIMEOUT) {
                int expected = static_cast<int>(sync_state::requested);
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
    ngm_tree_vis_v2 vis_;
    std::atomic<bool> visible_{true};
    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_ = 10000;
    std::size_t step_count_ = 0;
    std::size_t hook_id_ = 0;
    std::size_t barrier_hook_id_ = 0;
    bool pending_wait_ = false;
};

std::unique_ptr<coast::vis::vis_instance> create_ngm_tree_vis_v2(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm2::neuron_group_t*>(&entity);
    if (!ng) {
        return nullptr;
    }
    return std::make_unique<ngm_tree_vis_v2_instance>(*ng, sim_env, hints, hints_key, label);
}

struct ngm_tree_vis_v2_registrar {
    ngm_tree_vis_v2_registrar() {
        coast::vis::vis_info info{
            "ngm_tree_vis",
            "Neuron Tree",
            "Hierarchical tree view using gui_layout architecture",
            coast::vis::vis_target::sim_entity,
            {sim::io_entity_type_t::ngm2_neuron_group}
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm_tree_vis_v2);
    }
};

static ngm_tree_vis_v2_registrar registrar_instance;

} // anonymous namespace

} // namespace coast
