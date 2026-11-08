//
// ngm3 counterpart of ngm_tree_vis_v2. Force-directed hierarchical
// tree view of an ngm3::neuron_group_t. One tree per neuron: root
// node is the neuron, first-level children are the dendritic branches
// (apical_cnt + proximal_cnt children per neuron). Edges represent
// dendritic segments; apical vs proximal edges are color-coded.
//
// Leaf textures are sourced directly from one rep row of the
// permanence matrix (no synapse-gather pass — ngm3's redundant-rep
// storage already holds each leaf representation in full).
//

#ifndef COAST_NGM3_TREE_VIS_H
#define COAST_NGM3_TREE_VIS_H

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include "core/rng/fast_rng.h"
#include "hd_ngm3/hd_ngm3_neuron_group.h"
#include "gui_layout/layout_graph.h"
#include "gui_layout/force_layout.h"
#include "gui_layout/render/graph_canvas.h"
#include "gui_layout/render/node_renderer.h"
#include "gui_vis/core/color_types.h"
#include "gui_vis/core/colormap.h"
#include "gui_vis/core/gl_texture.h"

namespace coast {

enum class ngm3_tree_node_type : uint8_t {
    root,    // Neuron root node (orange circle)
    branch,  // Internal branch node (small blue circle)
    leaf     // Leaf segment with synapse texture
};

enum class ngm3_tree_bank : uint8_t {
    none,     // root node
    apical,
    proximal
};

/// User data stored in node_t::user_data via shared_ptr
struct ngm3_tree_node_data {
    std::size_t neuron_idx = 0;
    std::size_t dendrite_idx = 0;   // flat index over (apical | proximal); see bank
    ngm3_tree_bank bank = ngm3_tree_bank::none;
    uint16_t segment_idx = 0;
    ngm3_tree_node_type type = ngm3_tree_node_type::branch;
    uint32_t cell_count = 0;        // count of (rep, syn_idx) cells holding this seg_id
    gl_texture texture;
    std::vector<color_t> pixel_data;
};

/// Rendering parameters for the tree visualization
struct ngm3_tree_vis_params_t {
    // Representation dimensions (for leaf node textures)
    int rep_width  = 28;
    int rep_height = 28;
    int num_channels = 1;  // 1 = grayscale (use colormap), 3 = RGB

    // Background color (RGBA 0-255)
    uint8_t bg_r = 30, bg_g = 30, bg_b = 30, bg_a = 255;

    // Edge appearance
    float edge_thickness_min = 2.0f;
    float edge_thickness_max = 10.0f;
    // Proximal edge color (default same as v2's blue)
    uint8_t edge_r = 0, edge_g = 121, edge_b = 241, edge_a = 255;
    // Apical edge color (distinct hue — greenish by default)
    uint8_t apical_edge_r = 90, apical_edge_g = 200, apical_edge_b = 120, apical_edge_a = 255;

    // Node texture scale (canvas_width / this value)
    float texture_scale_divisor = 1500.0f;

    // Node radii
    float root_node_radius   = 6.0f;
    float branch_node_radius = 3.0f;

    // Root node color (RGBA 0-255)
    uint8_t root_r = 255, root_g = 161, root_b = 0, root_a = 255;

    // Branch node color (RGBA 0-255) - light blue
    uint8_t branch_r = 100, branch_g = 180, branch_b = 255, branch_a = 255;
};

/// Custom node renderer for ngm3 dendritic tree visualization
class ngm3_tree_node_renderer : public layout::render::node_renderer {
public:
    explicit ngm3_tree_node_renderer(const ngm3_tree_vis_params_t& params);

    void render(ImDrawList* draw_list,
                const layout::node_t& node,
                ImVec2 screen_pos,
                float zoom,
                bool selected,
                bool hovered) override;

    ImVec2 get_size(const layout::node_t& node, float zoom) override;

    void set_display_neurons(const std::unordered_set<std::size_t>& indices) { display_neurons_ = indices; }
    void add_display_neuron(std::size_t idx) { display_neurons_.insert(idx); }
    void clear_display_neurons() { display_neurons_.clear(); }
    [[nodiscard]] bool is_display_neuron(std::size_t idx) const { return display_neurons_.count(idx) > 0; }

    void set_params(const ngm3_tree_vis_params_t& params) { params_ = &params; }
    void set_canvas_size(ImVec2 size) { canvas_size_ = size; }

private:
    const ngm3_tree_vis_params_t* params_ = nullptr;
    std::unordered_set<std::size_t> display_neurons_;
    ImVec2 canvas_size_{600, 600};
};

/// Main tree visualization class for ngm3 neuron groups
class ngm3_tree_vis {
public:
    ngm3_tree_vis();
    explicit ngm3_tree_vis(const ngm3_tree_vis_params_t& params);
    ~ngm3_tree_vis();

    void set_params(const ngm3_tree_vis_params_t& params) { params_ = params; }
    [[nodiscard]] const ngm3_tree_vis_params_t& get_params() const { return params_; }

    [[nodiscard]] bool is_colormap_dirty() const { return colormap_dirty_; }
    void clear_colormap_dirty() { colormap_dirty_ = false; }

    [[nodiscard]] bool is_spacing_dirty() const { return spacing_dirty_; }
    void clear_spacing_dirty() { spacing_dirty_ = false; }

    // The four data-pull methods below are member templates on the source group
    // type (Group) so the same tree vis renders both ngm3::neuron_group_t (CPU)
    // and ngm3::neuron_group_gpu_t (GPU) — both expose the same accessor surface.
    // Definitions live in the .cpp and are instantiated for each Group by the
    // registrars at the bottom of that TU. NOTE: none of these call
    // refresh_vis_mirror() — the instance wrapper performs the (barrier-protected)
    // GPU readback exactly once per pull; these only read the refreshed mirror.

    /// Build graph from neuron group hierarchy
    template<class Group> void create_model(const Group& ng);

    /// Incremental update as dendrites grow (re-scans seg_ids, adds new nodes)
    template<class Group> void update_model(const Group& ng);

    /// Regenerate textures with current colormap
    template<class Group> void update_representations(const Group& ng);

    /// Update root node target positions based on tree sizes
    template<class Group> void update_root_positions(const Group& ng);

    /// Calculate bounding box radius for each neuron's tree and update spacing
    void update_tree_spacing();

    /// Release GPU resources
    void free_resources();

    /// Render in ImGui window. Returns false if window was closed.
    bool render_imgui(const char* label);

private:
    ngm3_tree_vis_params_t params_;
    float region_size_ = 500.0f;

    colormap_t colormap_ = colormap_t::viridis;
    bool colormap_dirty_ = false;

    float spacing_multiplier_ = 1.0f;
    bool spacing_dirty_ = false;

    std::unordered_set<std::size_t> selected_neurons_;
    bool show_on_hover_ = false;
    std::size_t hovered_neuron_idx_ = SIZE_MAX;

    std::vector<std::array<float, 2>> neuron_target_positions_;

    layout::layout_graph graph_;
    std::unique_ptr<layout::force_layout> layout_;
    layout::render::graph_canvas canvas_;
    ngm3_tree_node_renderer renderer_;

    // Lookup tables
    struct node_info_t {
        layout::node_id_t node_id = layout::invalid_node_id;
        std::shared_ptr<ngm3_tree_node_data> data;
    };
    std::vector<node_info_t> nodes_;
    std::vector<std::size_t> neuron_lu_;       // neuron_idx -> first node index (root)
    std::vector<std::size_t> dendrite_start_lu_; // neuron_idx -> first dendrite index in dendrite_lu_
    std::vector<std::size_t> dendrite_lu_;     // flat list of segment bases per dendrite

    // Cached structural info (set during create_model)
    std::size_t cached_apical_cnt_ = 0;
    std::size_t cached_proximal_cnt_ = 0;
    std::size_t cached_rep_cnt_ = 0;
    uint8_t     cached_branch_depth_ = 0;
    uint8_t     cached_max_seg_id_ = 0;

    fast_rng_t rgen_{0};
    std::uniform_real_distribution<float> rdis_{-1.0f, 1.0f};

    uint32_t max_cell_count_ = 1;

    bool needs_fit_view_ = false;

    float repulsion_ = 170.0f;
    float attraction_ = 0.05f;

    bool settling_active_ = false;
    int  settling_iterations_remaining_ = 0;
    bool initial_settling_ = false;

    bool initial_view_reset_done_ = false;
    int  total_frames_rendered_ = 0;

    // Build the (S, T, cell_count_per_seg) info for one dendrite of one neuron.
    // S = set of distinct seg_ids present; T = parent-closure of S (includes
    // internal branch points); cell_counts[seg] = number of (rep, syn_idx)
    // cells carrying seg in this dendrite's segment_ids array.
    struct dendrite_scan_t {
        std::vector<uint8_t> in_S;          // size max_seg_id+1
        std::vector<uint8_t> in_T;          // size max_seg_id+1
        std::vector<uint32_t> cell_count;   // size max_seg_id+1
    };
    template<class Group>
    void scan_dendrite(const Group& ng,
                       std::size_t neuron_idx,
                       ngm3_tree_bank bank,
                       std::size_t bank_dendrite_idx,
                       dendrite_scan_t& out) const;

    // For a leaf seg_id n compute the representative rep index.
    // depth-of-n is d = floor(log2(n)); range starts at (n - 2^d) << (depth - d).
    [[nodiscard]] std::size_t rep_for_leaf(uint8_t seg_id) const;

    // Read one rep row directly into pixel_data with current colormap.
    template<class Group>
    void get_pixel_data(const Group& ng,
                        std::size_t neuron_idx,
                        ngm3_tree_bank bank,
                        std::size_t bank_dendrite_idx,
                        uint8_t leaf_seg_id,
                        std::vector<color_t>& output);

    void setup_edge_renderer();
    void animate_tree_positions(float lerp_factor = 0.05f);
    ngm3_tree_node_data* get_node_data(layout::node_id_t id);

    void apply_per_neuron_repulsion();
    void settle_step();
    void settle_with_per_neuron_forces(int iterations);
    void start_animated_settling(int iterations);
};

} // namespace coast

#endif // COAST_NGM3_TREE_VIS_H
