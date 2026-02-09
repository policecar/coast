//
// Created by jk on 11.08.25.
//

#ifndef FBGD_H
#define FBGD_H

#include <cstdint>
#include <array>
#include <vector>
#include <functional>
#include <any>

#include "core/rng/fast_rng.h"

namespace coast {

class fbgd {

    // Configurable parameters
    float region_size_ = 1.0f;
    float cell_size_   = 0.1f;
    float local_size_  = 0.04f;

    static constexpr float attr_force    = 0.1f;
    static constexpr float rep_force     = 0.01f;
    static constexpr float sib_rep_force = 0.000f;
    static constexpr float border_wiggle = 0.0f; //0.005f;
    static constexpr float center_pull   = 0.0f;
    static constexpr float temp_cf       = 0.95f;
    static constexpr float temp_min      = 0.05f;
    static constexpr float it_step       = 0.01f;

    static constexpr std::size_t iterations = 1;

public:
    using node_id_t = uint32_t;
    using cell_id_t = uint32_t;

    struct node {
        node_id_t node_id;
        cell_id_t cell_id;
        std::array<float,2> pos;
        std::array<float,2> target_pos;  // For smooth animation of fixed nodes
        bool fixed;
        float repulsion_radius = 1.0f;  // Multiplier for repulsion force (sphere of influence)
        std::any payload;
    };

    struct edge {
        node_id_t from;
        node_id_t to;
    };

private:
    fast_rng_t rgen {0};
    std::uniform_real_distribution<float> rnd_dis {0.0f, 1.0f};  // Will be scaled by region_size_

    node_id_t next_node_id = 0;

    bool nodes_dirty = true;
    bool edges_dirty = true;

    std::vector<node> nodes;
    std::vector<edge> edges;

    using edge_it = std::vector<edge>::iterator;
    using node_it = std::vector<node>::iterator;

    std::vector<uint32_t> node_look_up;
    std::vector<std::pair<node_it,node_it>> cell_look_up;
    std::vector<std::pair<edge_it,edge_it>> edge_look_up;

    cell_id_t pos_to_cell(const std::array<float,2> &pos) const;

    void rebuild_cell_and_node_look_up();
    void rebuild_edge_look_up();

    std::vector<std::pair<node_it,node_it>> nb_tmp;
    std::vector<edge> sib_tmp;

    using draw_func_t = std::function<
        void(
            const std::vector<node>&,     // nodes
            const std::vector<edge>&,     // edges
            const std::vector<uint32_t>&  // node look up
    )>;

     draw_func_t draw_func = nullptr;

public:
    // Configuration
    void set_region_size(float size);
    void set_cell_size(float size) { cell_size_ = size; }
    void set_local_size(float size) { local_size_ = size; }

    [[nodiscard]] float get_region_size() const { return region_size_; }
    [[nodiscard]] float get_local_size() const { return local_size_; }

    // Node/edge management
    node_id_t add_node(std::any payload = nullptr);
    node_id_t add_node(const std::array<float,2> &pos, bool fixed = true,
                       float repulsion_radius = 1.0f, std::any payload = nullptr);
    void add_edge(node_id_t node_a, node_id_t node_b);

    // Update node properties
    void set_node_repulsion_radius(node_id_t node_id, float radius);
    void set_node_fixed(node_id_t node_id, bool fixed);
    void set_node_target_pos(node_id_t node_id, const std::array<float, 2>& target_pos);
    void translate_node(node_id_t node_id, const std::array<float, 2>& delta);  // Move node by delta
    void animate_fixed_nodes(float lerp_factor);  // Smoothly move fixed nodes towards target

    void update();

    void set_draw_func(draw_func_t df);

    void draw() const;

    const std::array<float,2>& get_node_pos(node_id_t node_id);
};

}

#endif //FBGD_H
