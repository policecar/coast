//
// ngm3 dendritic tree visualization. See ngm3_tree_vis.h header.
//
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "gui_vis/sim_entity/neuron_group/ngm3_tree_vis.h"
#include "gui_vis/vis_registry.h"
#include "sim_core/sim_env.h"
#include "tools/inc_on_exit.h"
#include "hd_ngm25/hd_ngm25_neuron_group.h"   // ngm2.5 reuses this renderer (duck-typed)
#ifdef COAST_WITH_GPU
#include "hd_ngm3/hd_ngm3_gpu_neuron_group.h"
#include "hd_ngm25/hd_ngm25_gpu_neuron_group.h"
#endif
#include "imgui.h"
#include "gui_core/imgui_scaling.h"

namespace coast {

namespace {
// A segment is a true leaf iff it is occupied and neither child (2*seg,
// 2*seg+1) is occupied. in_S marks occupied segment ids (index 0 unused;
// indices [1..max_seg_id] valid). A split promotes a segment's rows to its
// children, so an occupied segment with an occupied child is an internal
// branch point, not a leaf — only the terminal segments carry a representation.
bool ngm3_is_leaf_segment(const std::vector<uint8_t>& in_S,
                          uint16_t seg, uint16_t max_seg_id) {
    if (seg < 1 || seg > max_seg_id || !in_S[seg]) return false;
    const uint32_t left = 2u * static_cast<uint32_t>(seg);
    const uint32_t right = left + 1u;
    if (left <= max_seg_id && in_S[left]) return false;
    if (right <= max_seg_id && in_S[right]) return false;
    return true;
}
} // namespace

// ============================================================================
// ngm3_tree_node_renderer implementation
// ============================================================================

ngm3_tree_node_renderer::ngm3_tree_node_renderer(const ngm3_tree_vis_params_t& params)
    : params_(&params)
{
}

void ngm3_tree_node_renderer::render(ImDrawList* draw_list,
                                      const layout::node_t& node,
                                      ImVec2 screen_pos,
                                      float zoom,
                                      bool selected,
                                      bool hovered) {
    const ngm3_tree_node_data* data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<ngm3_tree_node_data>>(node.user_data);
            data = ptr.get();
        } catch (...) {
            data = nullptr;
        }
    }

    if (!data || !params_) {
        float radius = 5.0f * zoom;
        draw_list->AddCircleFilled(screen_pos, radius, IM_COL32(100, 100, 100, 200));
        return;
    }

    const auto& p = *params_;
    bool is_display_neuron = (display_neurons_.count(data->neuron_idx) > 0);

    bool show_texture = is_display_neuron &&
                        data->type == ngm3_tree_node_type::leaf &&
                        !data->pixel_data.empty() &&
                        data->texture.is_valid();

    if (show_texture) {
        constexpr float reference_size = 28.0f;
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
        float base_radius = (data->type == ngm3_tree_node_type::root)
            ? p.root_node_radius : p.branch_node_radius;
        float radius = base_radius * zoom;

        ImU32 color;
        if (data->type == ngm3_tree_node_type::root) {
            color = IM_COL32(p.root_r, p.root_g, p.root_b, p.root_a);
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

ImVec2 ngm3_tree_node_renderer::get_size(const layout::node_t& node, float zoom) {
    const ngm3_tree_node_data* data = nullptr;
    if (node.user_data.has_value()) {
        try {
            auto ptr = std::any_cast<std::shared_ptr<ngm3_tree_node_data>>(node.user_data);
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

    bool is_display_neuron = (display_neurons_.count(data->neuron_idx) > 0);
    if (is_display_neuron && data->type == ngm3_tree_node_type::leaf &&
        !data->pixel_data.empty() && data->texture.is_valid()) {
        constexpr float reference_size = 28.0f;
        float max_dim = std::max(static_cast<float>(data->texture.get_width()),
                                 static_cast<float>(data->texture.get_height()));
        float size_scale = reference_size / std::max(max_dim, 1.0f);
        float scale = canvas_size_.x * zoom * size_scale / p.texture_scale_divisor;
        float w = static_cast<float>(data->texture.get_width()) * scale;
        float h = static_cast<float>(data->texture.get_height()) * scale;
        return ImVec2(w, h);
    }

    float base_radius = (data->type == ngm3_tree_node_type::root)
        ? p.root_node_radius : p.branch_node_radius;
    float diameter = base_radius * 2.0f * zoom;
    return ImVec2(diameter, diameter);
}

// ============================================================================
// ngm3_tree_vis implementation
// ============================================================================

ngm3_tree_vis::ngm3_tree_vis()
    : region_size_(500.0f)
    , renderer_(params_)
{
    renderer_.set_params(params_);
    canvas_.set_node_renderer(&renderer_);
    canvas_.set_graph(&graph_);
    canvas_.set_dragging_enabled(false);
    canvas_.set_smooth_animation(false);
    setup_edge_renderer();
}

ngm3_tree_vis::ngm3_tree_vis(const ngm3_tree_vis_params_t& params)
    : params_(params)
    , region_size_(500.0f)
    , renderer_(params_)
{
    renderer_.set_params(params_);
    canvas_.set_node_renderer(&renderer_);
    canvas_.set_graph(&graph_);
    canvas_.set_dragging_enabled(false);
    setup_edge_renderer();
}

ngm3_tree_vis::~ngm3_tree_vis() {
    free_resources();
}

void ngm3_tree_vis::setup_edge_renderer() {
    canvas_.set_edge_renderer([this](ImDrawList* draw_list,
                                      const layout::edge_t& edge,
                                      ImVec2 from_screen,
                                      ImVec2 to_screen,
                                      float zoom) {
        // Lookup target node to fetch bank + cell_count for color/thickness
        uint32_t cell_count = 0;
        ngm3_tree_bank bank = ngm3_tree_bank::proximal;
        if (graph_.has_node(edge.to)) {
            const auto& to_node = graph_.node(edge.to);
            if (to_node.user_data.has_value()) {
                try {
                    auto ptr = std::any_cast<std::shared_ptr<ngm3_tree_node_data>>(to_node.user_data);
                    if (ptr) {
                        cell_count = ptr->cell_count;
                        bank = ptr->bank;
                    }
                } catch (...) {}
            }
        }

        float max_cells = static_cast<float>(params_.rep_width * params_.rep_height);
        if (max_cells <= 0.0f) max_cells = 1.0f;
        float sc_ratio = std::min(1.0f, static_cast<float>(cell_count) / max_cells);
        float thickness_range = params_.edge_thickness_max - params_.edge_thickness_min;
        float base_thickness = params_.edge_thickness_min + thickness_range * sc_ratio;
        float thickness = base_thickness * zoom;

        ImU32 edge_color;
        if (bank == ngm3_tree_bank::apical) {
            edge_color = IM_COL32(params_.apical_edge_r, params_.apical_edge_g,
                                  params_.apical_edge_b, params_.apical_edge_a);
        } else {
            edge_color = IM_COL32(params_.edge_r, params_.edge_g,
                                  params_.edge_b, params_.edge_a);
        }
        draw_list->AddLine(from_screen, to_screen, edge_color, thickness);
    });
}

// ---------------------------------------------------------------------------
// Dendrite scan + leaf-rep helpers (ngm3-specific)
// ---------------------------------------------------------------------------

template<class Group>
void ngm3_tree_vis::scan_dendrite(const Group& ng,
                                  std::size_t neuron_idx,
                                  ngm3_tree_bank bank,
                                  std::size_t bank_dendrite_idx,
                                  dendrite_scan_t& out) const
{
    const std::size_t rep_cnt = cached_rep_cnt_;
    const std::size_t max_seg = cached_max_seg_id_;
    out.in_S.assign(max_seg + 1, 0);
    out.in_T.assign(max_seg + 1, 0);
    out.cell_count.assign(max_seg + 1, 0);

    for (std::size_t r = 0; r < rep_cnt; ++r) {
        std::span<const uint8_t> sids = (bank == ngm3_tree_bank::apical)
            ? ng.get_apical_segment_ids(neuron_idx, bank_dendrite_idx, r)
            : ng.get_proximal_segment_ids(neuron_idx, bank_dendrite_idx, r);
        for (uint8_t s : sids) {
            if (s == 0 || s > max_seg) continue;
            out.in_S[s] = 1;
            ++out.cell_count[s];
        }
    }

    // Parent closure: every member of S has all ancestors in T.
    for (std::size_t s = 1; s <= max_seg; ++s) {
        if (!out.in_S[s]) continue;
        std::size_t a = s;
        while (a >= 1) {
            if (out.in_T[a]) break;
            out.in_T[a] = 1;
            if (a == 1) break;
            a /= 2;
        }
    }
}

std::size_t ngm3_tree_vis::rep_for_leaf(uint8_t seg_id) const
{
    if (seg_id < 1) return 0;
    // depth of seg_id in 0..branch_depth
    int d = 0;
    uint8_t x = seg_id;
    while (x > 1) { x >>= 1; ++d; }
    int shift = static_cast<int>(cached_branch_depth_) - d;
    if (shift < 0) shift = 0;
    std::size_t base = (static_cast<std::size_t>(seg_id) - (std::size_t{1} << d));
    return base << shift;
}

template<class Group>
void ngm3_tree_vis::get_pixel_data(const Group& ng,
                                   std::size_t neuron_idx,
                                   ngm3_tree_bank bank,
                                   std::size_t bank_dendrite_idx,
                                   uint8_t leaf_seg_id,
                                   std::vector<color_t>& output)
{
    const std::size_t buffer_size = static_cast<std::size_t>(params_.rep_width)
                                  * static_cast<std::size_t>(params_.rep_height);
    output.resize(buffer_size);
    std::fill(output.begin(), output.end(), color_t{0, 0, 0, 255});

    std::size_t rep_idx = rep_for_leaf(leaf_seg_id);
    std::span<const float> row = (bank == ngm3_tree_bank::apical)
        ? ng.get_apical_representation(neuron_idx, bank_dendrite_idx, rep_idx)
        : ng.get_proximal_representation(neuron_idx, bank_dendrite_idx, rep_idx);

    std::size_t pix = 0;
    if (params_.num_channels == 3) {
        for (std::size_t i = 0; i + 2 < row.size() && pix < buffer_size; i += 3) {
            output[pix] = color_t{
                static_cast<uint8_t>(std::clamp(row[i + 0] * 255.0f, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(row[i + 1] * 255.0f, 0.0f, 255.0f)),
                static_cast<uint8_t>(std::clamp(row[i + 2] * 255.0f, 0.0f, 255.0f)),
                255
            };
            ++pix;
        }
    } else {
        for (std::size_t i = 0; i < row.size() && pix < buffer_size; ++i) {
            float normalized = std::clamp(row[i], 0.0f, 1.0f);
            output[pix] = apply_colormap(colormap_, normalized);
            ++pix;
        }
    }
}

ngm3_tree_node_data* ngm3_tree_vis::get_node_data(layout::node_id_t id) {
    if (!graph_.has_node(id)) return nullptr;
    auto& node = graph_.node(id);
    if (!node.user_data.has_value()) return nullptr;
    try {
        auto ptr = std::any_cast<std::shared_ptr<ngm3_tree_node_data>>(node.user_data);
        return ptr.get();
    } catch (...) {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// create_model
// ---------------------------------------------------------------------------

template<class Group>
void ngm3_tree_vis::create_model(const Group& ng) {
    graph_.clear();
    nodes_.clear();
    neuron_lu_.clear();
    dendrite_start_lu_.clear();
    dendrite_lu_.clear();

    cached_apical_cnt_   = ng.get_apical_count();
    cached_proximal_cnt_ = ng.get_proximal_count();
    cached_rep_cnt_      = ng.get_rep_count();
    cached_branch_depth_ = ng.get_branch_depth();
    cached_max_seg_id_   = ng.get_max_segment_id();

    const std::size_t neuron_cnt   = ng.get_neuron_count();
    const std::size_t dendrite_cnt = cached_apical_cnt_ + cached_proximal_cnt_;
    const std::size_t segs_per_dendrite = static_cast<std::size_t>(cached_max_seg_id_) + 1;

    // Layout of nodes_: per neuron — 1 root node, then dendrite_cnt
    // contiguous blocks of (max_seg_id+1) slots (slot 0 unused, slots
    // [1..max_seg_id] indexed by seg_id).
    std::size_t nodes_size = 0;
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        neuron_lu_.push_back(nodes_size);
        ++nodes_size;  // root
        dendrite_start_lu_.push_back(dendrite_lu_.size());
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            dendrite_lu_.push_back(nodes_size);
            nodes_size += segs_per_dendrite;
        }
    }
    nodes_.resize(nodes_size);

    layout_ = std::make_unique<layout::force_layout>(graph_);
    layout::force_params fp;
    fp.repulsion = repulsion_;
    fp.attraction = attraction_;
    fp.damping = 0.8f;
    fp.gravity = 0.0f;
    fp.max_velocity = 10.0f;
    fp.min_distance = 1.0f;
    layout_->set_params(fp);

    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    if (grid_cnt == 0) grid_cnt = 1;
    const float grid_spacing = region_size_ / static_cast<float>(grid_cnt + 2);
    const float edge_length = grid_spacing * 0.15f;

    dendrite_scan_t scan;

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        float xp = static_cast<float>(neuron_idx % grid_cnt) * grid_spacing + grid_spacing * 1.5f;
        float yp = static_cast<float>(neuron_idx / grid_cnt) * grid_spacing + grid_spacing * 1.5f;

        // Root node
        const std::size_t root_node_idx = neuron_lu_[neuron_idx];
        auto root_data = std::make_shared<ngm3_tree_node_data>();
        root_data->neuron_idx = neuron_idx;
        root_data->dendrite_idx = 0;
        root_data->bank = ngm3_tree_bank::none;
        root_data->segment_idx = 0;
        root_data->type = ngm3_tree_node_type::root;

        layout::node_t root_node;
        root_node.x = xp;
        root_node.y = yp;
        root_node.fixed = true;
        root_node.radius = params_.root_node_radius;
        root_node.user_data = root_data;
        layout::node_id_t root_id = graph_.add_node(root_node);

        nodes_[root_node_idx] = {root_id, root_data};

        // Each dendrite of this neuron
        const std::size_t dendrite_base_idx = dendrite_start_lu_[neuron_idx];
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            const ngm3_tree_bank bank = (d < cached_apical_cnt_)
                ? ngm3_tree_bank::apical : ngm3_tree_bank::proximal;
            const std::size_t bank_dendrite_idx = (bank == ngm3_tree_bank::apical)
                ? d : d - cached_apical_cnt_;

            scan_dendrite(ng, neuron_idx, bank, bank_dendrite_idx, scan);
            const std::size_t segment_base = dendrite_lu_[dendrite_base_idx + d];

            for (uint16_t seg = 1; seg <= cached_max_seg_id_; ++seg) {
                if (!scan.in_T[seg]) continue;
                if (segment_base + seg >= nodes_.size()) continue;

                auto& cur_node = nodes_[segment_base + seg];
                if (cur_node.node_id != layout::invalid_node_id) continue;

                auto seg_data = std::make_shared<ngm3_tree_node_data>();
                seg_data->neuron_idx = neuron_idx;
                seg_data->dendrite_idx = d;
                seg_data->bank = bank;
                seg_data->segment_idx = seg;
                seg_data->cell_count = scan.cell_count[seg];
                seg_data->type = ngm3_is_leaf_segment(scan.in_S, seg, cached_max_seg_id_)
                    ? ngm3_tree_node_type::leaf : ngm3_tree_node_type::branch;
                max_cell_count_ = std::max(max_cell_count_, seg_data->cell_count);

                if (seg_data->type == ngm3_tree_node_type::leaf) {
                    get_pixel_data(ng, neuron_idx, bank, bank_dendrite_idx,
                                   static_cast<uint8_t>(seg), seg_data->pixel_data);
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

            // Edges: parent → child for every (seg/2 → seg) where both exist,
            // and root → seg 1 to attach the dendrite to the neuron root.
            for (uint16_t seg = 2; seg <= cached_max_seg_id_; ++seg) {
                if (segment_base + seg >= nodes_.size()) continue;
                auto& child_node = nodes_[segment_base + seg];
                if (child_node.node_id == layout::invalid_node_id) continue;
                auto& parent_node = nodes_[segment_base + (seg / 2)];
                if (parent_node.node_id == layout::invalid_node_id) continue;
                graph_.add_edge(parent_node.node_id, child_node.node_id, 1.0f, edge_length);
            }
            if (segment_base + 1 < nodes_.size() &&
                nodes_[segment_base + 1].node_id != layout::invalid_node_id) {
                graph_.add_edge(nodes_[root_node_idx].node_id,
                                nodes_[segment_base + 1].node_id, 1.0f, edge_length);
            }
        }
    }

    start_animated_settling(200);
    initial_settling_ = true;
    needs_fit_view_ = true;
}

// ---------------------------------------------------------------------------
// update_model — re-scan dendrites; add nodes for newly-appearing seg_ids.
// ---------------------------------------------------------------------------

template<class Group>
void ngm3_tree_vis::update_model(const Group& ng) {
    if (cached_max_seg_id_ == 0) return;
    const std::size_t neuron_cnt   = ng.get_neuron_count();
    const std::size_t dendrite_cnt = cached_apical_cnt_ + cached_proximal_cnt_;

    dendrite_scan_t scan;

    auto grid_cnt_update = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    if (grid_cnt_update == 0) grid_cnt_update = 1;
    float grid_spacing_update = region_size_ / static_cast<float>(grid_cnt_update + 2);
    float edge_length_update = grid_spacing_update * 0.15f;

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu_.size()) break;
        std::size_t root_node_idx = neuron_lu_[neuron_idx];

        float root_x = 0.0f, root_y = 0.0f;
        if (graph_.has_node(nodes_[root_node_idx].node_id)) {
            const auto& root = graph_.node(nodes_[root_node_idx].node_id);
            root_x = root.x;
            root_y = root.y;
        }
        const float spawn_offset = region_size_ * 0.05f;

        const std::size_t dendrite_base_idx = dendrite_start_lu_[neuron_idx];
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            const ngm3_tree_bank bank = (d < cached_apical_cnt_)
                ? ngm3_tree_bank::apical : ngm3_tree_bank::proximal;
            const std::size_t bank_dendrite_idx = (bank == ngm3_tree_bank::apical)
                ? d : d - cached_apical_cnt_;

            scan_dendrite(ng, neuron_idx, bank, bank_dendrite_idx, scan);
            const std::size_t segment_base = dendrite_lu_[dendrite_base_idx + d];

            // Update cell_count + flip leaf→branch if children appeared.
            for (uint16_t seg = 1; seg <= cached_max_seg_id_; ++seg) {
                if (segment_base + seg >= nodes_.size()) continue;
                auto& n_info = nodes_[segment_base + seg];
                if (n_info.node_id == layout::invalid_node_id) continue;
                if (!n_info.data) continue;
                n_info.data->cell_count = scan.cell_count[seg];
                max_cell_count_ = std::max(max_cell_count_, n_info.data->cell_count);
                if (ngm3_is_leaf_segment(scan.in_S, seg, cached_max_seg_id_)) {
                    if (n_info.data->type != ngm3_tree_node_type::leaf) {
                        n_info.data->type = ngm3_tree_node_type::leaf;
                        // Lazily build texture if needed
                        if (n_info.data->pixel_data.empty()) {
                            get_pixel_data(ng, neuron_idx, bank, bank_dendrite_idx,
                                           static_cast<uint8_t>(seg), n_info.data->pixel_data);
                            n_info.data->texture = gl_texture(params_.rep_width, params_.rep_height,
                                                              n_info.data->pixel_data.data());
                        }
                    }
                } else {
                    if (n_info.data->type == ngm3_tree_node_type::leaf) {
                        // Was a leaf, now internal (branched). Free texture.
                        n_info.data->texture.release();
                        n_info.data->pixel_data.clear();
                        n_info.data->type = ngm3_tree_node_type::branch;
                    }
                }
            }

            // Add nodes for any seg in T that doesn't yet have one.
            std::vector<uint16_t> new_segs;
            for (uint16_t seg = 1; seg <= cached_max_seg_id_; ++seg) {
                if (!scan.in_T[seg]) continue;
                if (segment_base + seg >= nodes_.size()) continue;
                auto& cur = nodes_[segment_base + seg];
                if (cur.node_id != layout::invalid_node_id) continue;

                // Position near parent
                float px = root_x, py = root_y;
                if (seg > 1) {
                    auto& parent = nodes_[segment_base + (seg / 2)];
                    if (parent.node_id != layout::invalid_node_id &&
                        graph_.has_node(parent.node_id)) {
                        const auto& pn = graph_.node(parent.node_id);
                        px = pn.x;
                        py = pn.y;
                    }
                }
                float nx = px + rdis_(rgen_) * spawn_offset;
                float ny = py + rdis_(rgen_) * spawn_offset;

                auto seg_data = std::make_shared<ngm3_tree_node_data>();
                seg_data->neuron_idx = neuron_idx;
                seg_data->dendrite_idx = d;
                seg_data->bank = bank;
                seg_data->segment_idx = seg;
                seg_data->cell_count = scan.cell_count[seg];
                seg_data->type = ngm3_is_leaf_segment(scan.in_S, seg, cached_max_seg_id_)
                    ? ngm3_tree_node_type::leaf : ngm3_tree_node_type::branch;
                max_cell_count_ = std::max(max_cell_count_, seg_data->cell_count);

                if (seg_data->type == ngm3_tree_node_type::leaf) {
                    get_pixel_data(ng, neuron_idx, bank, bank_dendrite_idx,
                                   static_cast<uint8_t>(seg), seg_data->pixel_data);
                    seg_data->texture = gl_texture(params_.rep_width, params_.rep_height,
                                                   seg_data->pixel_data.data());
                }

                layout::node_t sn;
                sn.x = nx;
                sn.y = ny;
                sn.fixed = false;
                sn.radius = params_.branch_node_radius;
                sn.user_data = seg_data;
                layout::node_id_t sid = graph_.add_node(sn);

                cur = {sid, seg_data};
                new_segs.push_back(seg);
            }

            // Add edges for new nodes
            std::sort(new_segs.begin(), new_segs.end());
            for (uint16_t seg : new_segs) {
                if (segment_base + seg >= nodes_.size()) continue;
                auto& n_info = nodes_[segment_base + seg];
                if (seg == 1) {
                    graph_.add_edge(nodes_[root_node_idx].node_id, n_info.node_id,
                                    1.0f, edge_length_update);
                } else if (segment_base + seg / 2 < nodes_.size() &&
                           nodes_[segment_base + seg / 2].node_id != layout::invalid_node_id) {
                    graph_.add_edge(nodes_[segment_base + seg / 2].node_id, n_info.node_id,
                                    1.0f, edge_length_update);
                }
            }
        }
    }

    start_animated_settling(50);
}

// ---------------------------------------------------------------------------
// update_representations — refresh leaf textures (e.g. colormap change)
// ---------------------------------------------------------------------------

template<class Group>
void ngm3_tree_vis::update_representations(const Group& ng) {
    const std::size_t neuron_cnt   = ng.get_neuron_count();
    const std::size_t dendrite_cnt = cached_apical_cnt_ + cached_proximal_cnt_;
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu_.size()) break;
        // Only refresh textures that are actually shown. Representations render
        // exclusively for the display set (selected + hovered) leaf nodes, so
        // refreshing non-displayed neurons every interval is wasted work. Their
        // textures already exist from create_model/update_model and stay frozen
        // until shown again — invisible while not rendered.
        const bool displayed = selected_neurons_.count(neuron_idx) > 0 ||
                               (show_on_hover_ && hovered_neuron_idx_ == neuron_idx);
        if (!displayed) continue;
        const std::size_t dendrite_base_idx = dendrite_start_lu_[neuron_idx];
        for (std::size_t d = 0; d < dendrite_cnt; ++d) {
            const ngm3_tree_bank bank = (d < cached_apical_cnt_)
                ? ngm3_tree_bank::apical : ngm3_tree_bank::proximal;
            const std::size_t bank_dendrite_idx = (bank == ngm3_tree_bank::apical)
                ? d : d - cached_apical_cnt_;
            const std::size_t segment_base = dendrite_lu_[dendrite_base_idx + d];
            for (uint16_t seg = 1; seg <= cached_max_seg_id_; ++seg) {
                if (segment_base + seg >= nodes_.size()) continue;
                auto& n_info = nodes_[segment_base + seg];
                if (n_info.node_id == layout::invalid_node_id) continue;
                if (!n_info.data || n_info.data->type != ngm3_tree_node_type::leaf) continue;
                get_pixel_data(ng, neuron_idx, bank, bank_dendrite_idx,
                               static_cast<uint8_t>(seg), n_info.data->pixel_data);
                if (n_info.data->texture.is_valid()) {
                    n_info.data->texture.update(n_info.data->pixel_data);
                } else {
                    n_info.data->texture = gl_texture(params_.rep_width, params_.rep_height,
                                                       n_info.data->pixel_data.data());
                }
            }
        }
    }
}

template<class Group>
void ngm3_tree_vis::update_root_positions(const Group& ng) {
    if (neuron_lu_.empty()) return;

    std::size_t neuron_cnt = neuron_lu_.size();
    std::size_t ng_neuron_cnt = ng.get_neuron_count();
    if (neuron_cnt > ng_neuron_cnt) neuron_cnt = ng_neuron_cnt;
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

void ngm3_tree_vis::animate_tree_positions(float lerp_factor) {
    if (neuron_lu_.empty() || neuron_target_positions_.empty()) return;
    std::size_t neuron_cnt = std::min(neuron_lu_.size(), neuron_target_positions_.size());

    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        if (root_node_idx >= nodes_.size() ||
            nodes_[root_node_idx].node_id == layout::invalid_node_id) continue;
        if (!graph_.has_node(nodes_[root_node_idx].node_id)) continue;

        const auto& root = graph_.node(nodes_[root_node_idx].node_id);
        const auto& target = neuron_target_positions_[neuron_idx];

        float dx = (target[0] - root.x) * lerp_factor;
        float dy = (target[1] - root.y) * lerp_factor;
        if (std::abs(dx) < 0.0001f && std::abs(dy) < 0.0001f) continue;

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

void ngm3_tree_vis::apply_per_neuron_repulsion() {
    if (neuron_lu_.empty() || repulsion_ <= 0.0f) return;

    const float min_dist = 1.0f;
    const float texture_radius = static_cast<float>(params_.rep_width) * 0.5f;

    std::size_t neuron_cnt = neuron_lu_.size();
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        bool textures_visible = selected_neurons_.count(neuron_idx) > 0 ||
                                (show_on_hover_ && hovered_neuron_idx_ == neuron_idx);

        std::vector<layout::node_id_t> neuron_nodes;
        for (const auto& node_info : nodes_) {
            if (node_info.node_id == layout::invalid_node_id) continue;
            if (!node_info.data || node_info.data->neuron_idx != neuron_idx) continue;
            neuron_nodes.push_back(node_info.node_id);
        }

        for (std::size_t i = 0; i < neuron_nodes.size(); ++i) {
            if (!graph_.has_node(neuron_nodes[i])) continue;
            auto& n1 = graph_.node(neuron_nodes[i]);
            if (n1.fixed) continue;

            ngm3_tree_node_data* data1 = get_node_data(neuron_nodes[i]);
            bool n1_is_leaf = data1 && data1->type == ngm3_tree_node_type::leaf;

            for (std::size_t j = 0; j < neuron_nodes.size(); ++j) {
                if (i == j) continue;
                if (!graph_.has_node(neuron_nodes[j])) continue;
                const auto& n2 = graph_.node(neuron_nodes[j]);

                ngm3_tree_node_data* data2 = get_node_data(neuron_nodes[j]);
                bool n2_is_leaf = data2 && data2->type == ngm3_tree_node_type::leaf;

                float dx = n1.x - n2.x;
                float dy = n1.y - n2.y;
                float dist_sq = dx * dx + dy * dy;
                float dist = std::sqrt(std::max(dist_sq, min_dist * min_dist));
                float force = repulsion_ / (dist_sq + min_dist);

                if (textures_visible && (n1_is_leaf || n2_is_leaf)) {
                    float combined_radius = texture_radius * ((n1_is_leaf ? 1.0f : 0.0f) +
                                                              (n2_is_leaf ? 1.0f : 0.0f));
                    if (dist < combined_radius * 2.0f) {
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

void ngm3_tree_vis::settle_step() {
    const float damping = 0.8f;
    const float max_velocity = 10.0f;

    apply_per_neuron_repulsion();

    for (const auto& edge : graph_.edges()) {
        if (!graph_.has_node(edge.from) || !graph_.has_node(edge.to)) continue;
        auto& n1 = graph_.node(edge.from);
        auto& n2 = graph_.node(edge.to);
        float dx = n2.x - n1.x;
        float dy = n2.y - n1.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 1.0f) continue;

        float delta = dist - edge.ideal_length;
        float force = attraction_ * delta;
        float inv_dist = 1.0f / dist;
        float fx = force * dx * inv_dist;
        float fy = force * dy * inv_dist;

        if (!n1.fixed) { n1.vx += fx; n1.vy += fy; }
        if (!n2.fixed) { n2.vx -= fx; n2.vy -= fy; }
    }

    float max_vel_sq = max_velocity * max_velocity;
    for (auto& n : graph_.nodes()) {
        if (n.fixed) continue;
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

void ngm3_tree_vis::settle_with_per_neuron_forces(int iterations) {
    for (int iter = 0; iter < iterations; ++iter) settle_step();
}

void ngm3_tree_vis::start_animated_settling(int iterations) {
    settling_active_ = true;
    settling_iterations_remaining_ = iterations;
}

void ngm3_tree_vis::update_tree_spacing() {
    if (neuron_lu_.empty()) return;
    std::size_t neuron_cnt = neuron_lu_.size();

    std::vector<float> tree_radii(neuron_cnt, 0.0f);
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx) {
        if (neuron_idx >= neuron_lu_.size()) break;
        std::size_t root_node_idx = neuron_lu_[neuron_idx];
        if (root_node_idx >= nodes_.size() ||
            nodes_[root_node_idx].node_id == layout::invalid_node_id) continue;
        if (!graph_.has_node(nodes_[root_node_idx].node_id)) continue;
        const auto& root = graph_.node(nodes_[root_node_idx].node_id);
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

    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    if (grid_cnt == 0) grid_cnt = 1;

    float max_tree_radius = 1.0f;
    for (float r : tree_radii) max_tree_radius = std::max(max_tree_radius, r);

    float min_cell_size = max_tree_radius * 2.5f * spacing_multiplier_;
    float base_cell_size = region_size_ / static_cast<float>(grid_cnt);
    float cell_size = std::max(base_cell_size, min_cell_size) * spacing_multiplier_;

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

void ngm3_tree_vis::free_resources() {
    for (auto& node : nodes_) {
        if (node.data) {
            node.data->texture.release();
            node.data->pixel_data.clear();
        }
    }
}

bool ngm3_tree_vis::render_imgui(const char* label) {
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin(label, &open);

    ImGui::Text("Zoom: %.0f%%", canvas_.zoom() * 100.0f);
    ImGui::SameLine();
    if (ImGui::Button("Reset View")) canvas_.fit_graph_in_view();
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
            if (is_selected) ImGui::SetItemDefaultFocus();
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

    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    renderer_.set_canvas_size(canvas_size);

    if (needs_fit_view_ && canvas_size.x > 0 && canvas_size.y > 0) {
        needs_fit_view_ = false;
        canvas_.fit_graph_in_view();
    }

    if (settling_active_ && settling_iterations_remaining_ > 0) {
        int iters_this_frame = std::min(settling_iterations_remaining_, 5);
        for (int i = 0; i < iters_this_frame; ++i) settle_step();
        settling_iterations_remaining_ -= iters_this_frame;

        if (initial_settling_ && settling_iterations_remaining_ % 20 == 0) {
            update_tree_spacing();
        }

        if (settling_iterations_remaining_ <= 0) {
            settling_active_ = false;
            if (initial_settling_) {
                update_tree_spacing();
                initial_settling_ = false;
            }
        }
    } else {
        settle_step();
    }

    total_frames_rendered_++;
    if (!initial_view_reset_done_ && total_frames_rendered_ >= 120) {
        needs_fit_view_ = true;
        initial_view_reset_done_ = true;
    }

    animate_tree_positions(0.05f);

    auto canvas_selected = canvas_.selected_node();
    auto hovered = canvas_.hovered_node();

    hovered_neuron_idx_ = SIZE_MAX;

    static std::optional<layout::node_id_t> last_canvas_selected;
    if (canvas_selected.has_value() && canvas_selected != last_canvas_selected) {
        auto* data = get_node_data(canvas_selected.value());
        if (data && data->type == ngm3_tree_node_type::root) {
            ImGuiIO& io = ImGui::GetIO();
            if (io.KeyCtrl) {
                if (selected_neurons_.count(data->neuron_idx) > 0) {
                    selected_neurons_.erase(data->neuron_idx);
                } else {
                    selected_neurons_.insert(data->neuron_idx);
                }
            } else {
                selected_neurons_.clear();
                selected_neurons_.insert(data->neuron_idx);
            }
        }
        last_canvas_selected = canvas_selected;
    } else if (!canvas_selected.has_value() && last_canvas_selected.has_value()) {
        ImGuiIO& io = ImGui::GetIO();
        if (!io.KeyCtrl) selected_neurons_.clear();
        last_canvas_selected = std::nullopt;
    }

    if (show_on_hover_ && hovered.has_value()) {
        auto* data = get_node_data(hovered.value());
        if (data && data->type == ngm3_tree_node_type::root) {
            hovered_neuron_idx_ = data->neuron_idx;
        }
    }

    renderer_.clear_display_neurons();
    for (std::size_t idx : selected_neurons_) renderer_.add_display_neuron(idx);
    if (show_on_hover_ && hovered_neuron_idx_ != SIZE_MAX) {
        renderer_.add_display_neuron(hovered_neuron_idx_);
    }

    layout::render::edge_style es;
    es.color = IM_COL32(params_.edge_r, params_.edge_g, params_.edge_b, params_.edge_a);
    es.thickness = params_.edge_thickness_min;
    canvas_.set_edge_style(es);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    draw_list->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        IM_COL32(params_.bg_r, params_.bg_g, params_.bg_b, params_.bg_a));

    canvas_.render("##ngm3_tree_canvas", canvas_size);

    ImGui::End();
    return open;
}

// ============================================================================
// vis_instance adapter + registrar
// ============================================================================

namespace {

enum class sync_state : int {
    idle = 0,
    requested = 1,
    updating = 2
};

constexpr auto SYNC_TIMEOUT = std::chrono::milliseconds(100);

template<class Group>
class ngm3_tree_vis_instance : public coast::vis::vis_instance {
    // Whether Group is the GPU group: its data reads require a (barrier-protected)
    // GPU->CPU readback, so initial modeling is deferred from the ctor (no barrier)
    // into the first update() (barrier-protected). The CPU group has no such need.
    static constexpr bool is_gpu_group =
#ifdef COAST_WITH_GPU
        std::is_same_v<Group, ngm3::neuron_group_gpu_t> ||
        std::is_same_v<Group, ngm25::neuron_group_gpu_t>;
#else
        false;
#endif

public:
    ngm3_tree_vis_instance(Group& ng,
                           sim::env& sim_env,
                           const coast::vis_hints_t& hints,
                           const std::string& hints_key,
                           std::string label)
        : ng_(ng)
        , sim_env_(sim_env)
        , label_(std::move(label))
    {
        coast::ngm_tree_vis_hint_t hint;
        auto it = hints.ngm3.neuron_groups.find(hints_key);
        if (it != hints.ngm3.neuron_groups.end()) {
            hint = it->second.ngm_tree_vis;
        } else {
            auto def_it = hints.ngm3.neuron_groups.find("default");
            if (def_it != hints.ngm3.neuron_groups.end()) {
                hint = def_it->second.ngm_tree_vis;
            }
        }

        update_interval_ = static_cast<std::size_t>(hint.update_interval);

        ngm3_tree_vis_params_t params;
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
        params.apical_edge_r = static_cast<uint8_t>(hint.apical_edge_r);
        params.apical_edge_g = static_cast<uint8_t>(hint.apical_edge_g);
        params.apical_edge_b = static_cast<uint8_t>(hint.apical_edge_b);
        params.apical_edge_a = static_cast<uint8_t>(hint.apical_edge_a);
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
        // CPU: build the model now (reads CPU-resident data — benign). GPU: the
        // data lives on the device and reading it needs the post-swap barrier, so
        // defer create_model to the first update() (see modeled_ below).
        if constexpr (!is_gpu_group) {
            vis_.create_model(ng_);
            vis_.update_root_positions(ng_);
            modeled_ = true;
        }

        hook_id_ = sim_env_.set_post_swap_hook([this]() {
            on_post_swap_signal();
        });
        barrier_hook_id_ = sim_env_.set_post_swap_barrier_hook([this]() {
            on_post_swap_wait();
        });
    }

    ~ngm3_tree_vis_instance() override {
        // Removal is synchronous (#235): once both calls return, neither hook is
        // running or will run again, so nothing on the sim thread touches `this`.
        // update() runs on this (GUI) thread, so no update is in flight either; the
        // former spin on sync_state_ == idle is not needed (and could hang if a
        // signal fired before its barrier hook was applied).
        sim_env_.remove_post_swap_hook(hook_id_);
        sim_env_.remove_post_swap_barrier_hook(barrier_hook_id_);
    }

    void update() override {
        if (!visible_.load(std::memory_order_acquire)) return;
        int expected = static_cast<int>(sync_state::requested);
        if (sync_state_.compare_exchange_strong(expected,
                                                 static_cast<int>(sync_state::updating),
                                                 std::memory_order_acq_rel)) {
            // Barrier window: sim thread parked, device idle. Do the (Group-specific)
            // GPU->CPU readback once, then pull. No-op refresh on the CPU group.
            ng_.refresh_vis_mirror();
            if (!modeled_) {
                // First barrier-protected pull for the GPU group: build the model
                // now that the mirror holds real seg_id / permanence data.
                vis_.create_model(ng_);
                vis_.update_root_positions(ng_);
                modeled_ = true;
            } else {
                vis_.update_model(ng_);
                // Refresh leaf textures so representations track learning, not
                // just structural changes. Same update interval as update_model.
                vis_.update_representations(ng_);
            }
            sync_state_.store(static_cast<int>(sync_state::idle), std::memory_order_release);
        }
    }

    bool render_imgui() override {
        if (!visible_.load(std::memory_order_acquire)) return true;
        if (vis_.is_spacing_dirty()) {
            vis_.clear_spacing_dirty();
            vis_.update_tree_spacing();
        }
        if (vis_.is_colormap_dirty()) {
            vis_.clear_colormap_dirty();
            vis_.update_representations(ng_);
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
        if (colormap_changed) vis_.clear_colormap_dirty();
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
            // Give up early while a hook removal waits for this phase (#235):
            // the remover is usually the GUI thread this loop is waiting for.
            if (std::chrono::steady_clock::now() - start > SYNC_TIMEOUT ||
                sim_env_.hook_removal_waiting()) {
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

    Group& ng_;
    sim::env& sim_env_;
    std::string label_;
    ngm3_tree_vis vis_;
    std::atomic<bool> visible_{true};
    std::atomic<int> sync_state_{static_cast<int>(sync_state::idle)};
    std::size_t update_interval_ = 10000;
    std::size_t step_count_ = 0;
    std::size_t hook_id_ = 0;
    std::size_t barrier_hook_id_ = 0;
    bool pending_wait_ = false;
    bool modeled_ = false;   // set once create_model has run (deferred for GPU)
};

std::unique_ptr<coast::vis::vis_instance> create_ngm3_tree_vis(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm3::neuron_group_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_tree_vis_instance<ngm3::neuron_group_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm3_tree_vis_registrar {
    ngm3_tree_vis_registrar() {
        coast::vis::vis_info info{
            "ngm3_tree_vis",
            "Neuron Tree (ngm3)",
            "Hierarchical tree view for ngm3 neuron groups (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            {sim::io_entity_type_t::ngm3_neuron_group}
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm3_tree_vis);
    }
};

static ngm3_tree_vis_registrar registrar_instance;

// ngm2.5 reuses this renderer (same duck-typed accessor contract; hints from the
// shared hints.ngm3 tree in v1).
std::unique_ptr<coast::vis::vis_instance> create_ngm25_tree_vis(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm25::neuron_group_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_tree_vis_instance<ngm25::neuron_group_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm25_tree_vis_registrar {
    ngm25_tree_vis_registrar() {
        coast::vis::vis_info info{
            "ngm25_tree_vis",
            "Neuron Tree (ngm2.5)",
            "Hierarchical tree view for ngm2.5 neuron groups (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            {sim::io_entity_type_t::ngm25_neuron_group}
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm25_tree_vis);
    }
};

static ngm25_tree_vis_registrar ngm25_tree_registrar_instance;

#ifdef COAST_WITH_GPU
// GPU sibling: same tree vis instantiated over the GPU group. The instance defers
// initial modeling to the first barrier-protected update() and performs the
// GPU->CPU readback there (is_gpu_group path); the templated tree methods then
// read the CPU mirror exactly as for the CPU group.
std::unique_ptr<coast::vis::vis_instance> create_ngm3_tree_vis_gpu(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm3::neuron_group_gpu_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_tree_vis_instance<ngm3::neuron_group_gpu_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm3_tree_vis_gpu_registrar {
    ngm3_tree_vis_gpu_registrar() {
        coast::vis::vis_info info{
            "ngm3_tree_vis_gpu",
            "Neuron Tree (ngm3 GPU)",
            "Hierarchical tree view for ngm3_gpu neuron groups (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            {sim::io_entity_type_t::ngm3_neuron_group_gpu}
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm3_tree_vis_gpu);
    }
};

static ngm3_tree_vis_gpu_registrar tree_gpu_registrar_instance;

// ngm2.5 GPU tree vis (duck-typed; is_gpu_group defers the readback to update()).
std::unique_ptr<coast::vis::vis_instance> create_ngm25_tree_vis_gpu(
    sim::io_entity& entity,
    sim::env& sim_env,
    const coast::vis_hints_t& hints,
    const std::string& hints_key,
    const std::string& label)
{
    auto* ng = dynamic_cast<ngm25::neuron_group_gpu_t*>(&entity);
    if (!ng) return nullptr;
    return std::make_unique<ngm3_tree_vis_instance<ngm25::neuron_group_gpu_t>>(
        *ng, sim_env, hints, hints_key, label);
}

struct ngm25_tree_vis_gpu_registrar {
    ngm25_tree_vis_gpu_registrar() {
        coast::vis::vis_info info{
            "ngm25_tree_vis_gpu",
            "Neuron Tree (ngm2.5 GPU)",
            "Hierarchical tree view for ngm2.5_gpu neuron groups (apical + proximal banks)",
            coast::vis::vis_target::sim_entity,
            {sim::io_entity_type_t::ngm25_neuron_group_gpu}
        };
        coast::vis::get_vis_registry().register_entity_vis(info, create_ngm25_tree_vis_gpu);
    }
};

static ngm25_tree_vis_gpu_registrar ngm25_tree_gpu_registrar_instance;
#endif // COAST_WITH_GPU

} // anonymous namespace

} // namespace coast
