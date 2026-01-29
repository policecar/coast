//
// Created by jk on 11.08.25.
//

#ifndef FBGD_H
#define FBGD_H

#include <cstdint>
#include <array>
#include <vector>
#include <random>
#include <functional>
#include <any>

namespace coast {

class fbgd {

    static constexpr float region_size = 1.0f;
    static constexpr float cell_size   = 0.1f;
    static constexpr float local_size  = 0.04f;

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
        bool fixed;
        std::any payload;
    };

    struct edge {
        node_id_t from;
        node_id_t to;
    };

private:
    std::mt19937 rgen {0};
    std::uniform_real_distribution<float> rnd_dis {0.0, region_size};

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

    static constexpr cell_id_t pos_to_cell(const std::array<float,2> &pos) ;

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
    node_id_t add_node(std::any payload = nullptr);
    node_id_t add_node(const std::array<float,2> &pos, bool fixed = true, std::any payload = nullptr);
    void add_edge(node_id_t node_a, node_id_t node_b);

    void update();

    void set_draw_func(draw_func_t df);

    void draw() const;

    static float get_region_size() { return region_size; }
    static float get_local_size() { return local_size; }

    const std::array<float,2>& get_node_pos(node_id_t node_id);
};

}

#endif //FBGD_H
