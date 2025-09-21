//
// Created by jk on 11.08.25.
//
#include <cmath>
#include <algorithm>
#include <iterator>

#include "fbgd.h"

namespace coast {

constexpr fbgd::cell_id_t fbgd::pos_to_cell(const std::array<float, 2> &pos)
{
    //constexpr float cell_cnt = std::ceil(region_size / cell_size);
    const float cell_cnt = std::ceil(region_size / cell_size);
    const float cell_id =
        std::floor(pos[0] / cell_size) +
        std::floor(pos[1] / cell_size) * cell_cnt;
    return static_cast<cell_id_t>(std::clamp(cell_id,0.0f,cell_cnt*cell_cnt));
}

void fbgd::rebuild_cell_and_node_look_up()
{
    if (!nodes_dirty) return;

    if (nodes.empty()) {
        nodes_dirty = false;
        return;
    }

    std::ranges::sort(
        nodes,
        [](const auto &a, const auto &b) {
            return a.cell_id < b.cell_id;
        }
    );

    //constexpr auto cell_cnt = static_cast<uint32_t>(std::ceil(region_size / cell_size));
    const auto cell_cnt = static_cast<uint32_t>(std::ceil(region_size / cell_size));
    cell_look_up.resize( cell_cnt * cell_cnt);
    std::ranges::fill(cell_look_up, std::make_pair(nodes.end(), nodes.end()) );
    cell_id_t cur_id = nodes[0].cell_id;
    auto cur_start = nodes.begin();
    for (std::size_t idx = 1; idx < nodes.size(); ++idx) {
        if (nodes[idx].cell_id != cur_id) {
            auto cur_end = nodes.begin();
            std::advance(cur_end, idx);
            cell_look_up[cur_id] = std::make_pair(cur_start,cur_end);
            cur_start = cur_end;
            cur_id = nodes[idx].cell_id;
        }
    }
    cell_look_up[cur_id] = std::make_pair(cur_start,nodes.end());
    nodes_dirty = false;

    node_look_up.resize( next_node_id );
    std::size_t node_cnt = nodes.size();
    for (std::size_t idx = 0; idx < node_cnt; ++idx) {
        node_look_up[nodes[idx].node_id] = idx;
    }

}

void fbgd::rebuild_edge_look_up()
{
    if (!edges_dirty) return;

    if (edges.empty()) {
        edges_dirty = false;
        return;
    }

    std::ranges::sort(
        edges,
        [](const auto &a, const auto &b) {
            if (a.from < b.from)
                return true;
            if (a.from > b.from)
                return false;
            return a.to < b.to;
        }
    );

    edge_look_up.resize( next_node_id );
    std::ranges::fill(edge_look_up, std::make_pair(edges.end(),edges.end()));
    node_id_t cur_id = edges[0].from;
    auto cur_start = edges.begin();
    for (std::size_t idx = 1; idx < edges.size(); ++idx) {
        if (edges[idx].from != cur_id) {
            auto cur_end = edges.begin();
            std::advance(cur_end, idx);
            edge_look_up[cur_id] = std::make_pair( cur_start, cur_end );
            cur_start = cur_end;
            cur_id = edges[idx].from;
        }
    }
    edge_look_up[cur_id] = std::make_pair( cur_start, edges.end() );
    edges_dirty = false;
}

fbgd::node_id_t fbgd::add_node(std::any payload)
{
    return add_node( {rnd_dis(rgen),rnd_dis(rgen)}, false, std::move(payload));
}

fbgd::node_id_t fbgd::add_node(const std::array<float, 2> &pos, bool fixed, std::any payload)
{
    node new_node {};
    new_node.node_id = next_node_id++;
    new_node.pos     = pos;
    new_node.cell_id = pos_to_cell(new_node.pos);
    new_node.fixed   = fixed;
    new_node.payload = std::move(payload);

    nodes.push_back(new_node);
    nodes_dirty = true;

    return new_node.node_id;
}

void fbgd::add_edge(node_id_t node_a, node_id_t node_b)
{
    edges.push_back( {node_a, node_b} );
    edges.push_back( {node_b, node_a} );
    edges_dirty = true;
}

void fbgd::update()
{

    const auto node_cnt = nodes.size();
    //constexpr auto cell_cnt = static_cast<int>(std::ceil(region_size / cell_size));
    const auto cell_cnt = static_cast<int>(std::ceil(region_size / cell_size));

    const float cnt_normalizer = static_cast<float>(node_cnt*3) / static_cast<float>(cell_cnt);

    const float w    = region_size;
    const float area = w*w;
    const float c1   = attr_force * area;
    const float c2   = rep_force * area / cnt_normalizer;
    const float c3   = sib_rep_force * area;

    constexpr float min_dist = 0.0000001f;

    float temperature = 1.0f;

    std::array<float,2> node_update {0.0f, 0.0f};
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        rebuild_cell_and_node_look_up();
        rebuild_edge_look_up();
        int cur_cell = std::numeric_limits<int>::max();
        for (std::size_t idx = 0; idx < node_cnt; ++idx) {
            node &cur_node = nodes[idx];
            if (cur_node.fixed)
                continue;
            node_update = {0.0f, 0.0f};
            // get starting indices of all cells we want to look at
            if (cur_cell != cur_node.cell_id) {
                cur_cell = static_cast<int>(cur_node.cell_id);
                nb_tmp.clear();
                int center_x = cur_cell % cell_cnt;
                int center_y = cur_cell / cell_cnt;
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy) {
                        int nb_x = center_x + dx;
                        int nb_y = center_y + dy;

                        if ((nb_x < 0) || (nb_x >= cell_cnt) || (nb_y < 0) || (nb_y >= cell_cnt))
                            continue;

                        nb_tmp.push_back( cell_look_up[nb_x + nb_y * cell_cnt] );
                    }
            }
            // go through all cells to determined repellent forces
            for (const auto &nb_cell : nb_tmp) {
                for (auto nb_it = nb_cell.first; nb_it != nb_cell.second; ++nb_it) {
                    const node &other_node = *nb_it;

                    if (cur_node.node_id == other_node.node_id)
                        continue;

                    const std::array<float,2> delta {
                        cur_node.pos[0] - other_node.pos[0],
                        cur_node.pos[1] - other_node.pos[1]
                    };
                    const float dist = std::sqrt(std::max(min_dist,
                            delta[0] * delta[0] +
                            delta[1] * delta[1]
                        ));

                    float local_push = 1.0f;

                    if (dist < local_size) {
                        local_push = 5000.0f * (1.0f - (dist / local_size));
                    }

                    /*
                    if (dist < region_size / 1000.0f) {
                        node_update[0] += rnd_dis(rgen) * 0.01; //border_wiggle;
                        node_update[1] += rnd_dis(rgen) * 0.01; //border_wiggle;
                    }
                    */


                    node_update[0] += delta[0] * c2 * local_push / dist ;
                    node_update[1] += delta[1] * c2 * local_push / dist ;
                }
            }
            // go through all edges of node to determine attractive forces
            // also create repellent forces of "indirect siblings"
            auto cur_edges = edge_look_up[cur_node.node_id];
            for (auto edge_it = cur_edges.first; edge_it != cur_edges.second; ++edge_it)
            {
                const node &other_node = nodes[node_look_up[edge_it->to]];

                const std::array<float,2> delta {
                    cur_node.pos[0] - other_node.pos[0],
                    cur_node.pos[1] - other_node.pos[1]
                };

                const float dist = std::sqrt(std::max(min_dist,
                        delta[0] * delta[0] +
                        delta[1] * delta[1]
                    ));

                node_update[0] -= delta[0] * c1 / dist;
                node_update[1] -= delta[1] * c1 / dist;

                // get sibling repellents

                sib_tmp.clear();
                auto other_edges = edge_look_up[other_node.node_id];
                std::set_difference(
                    other_edges.first,other_edges.second,
                    cur_edges.first, cur_edges.second,
                    std::back_inserter(sib_tmp),
                    [](const auto &a, const auto &b) {
                        return a.to < b.to;
                    }
                );

                for (auto &sib : sib_tmp) {
                    const std::array<float,2> sib_delta {
                        cur_node.pos[0] - other_node.pos[0],
                        cur_node.pos[1] - other_node.pos[1]
                    };
                    const float sib_dist = std::sqrt(std::max(min_dist,
                            delta[0] * delta[0] +
                            delta[1] * delta[1]
                        ));

                    if (sib_dist > local_size)
                        continue;

                    node_update[0] += sib_delta[0] * c3 / sib_dist ;
                    node_update[1] += sib_delta[1] * c3 / sib_dist ;
                }
            }
            // add center pull
            const std::array<float,2> delta {
                cur_node.pos[0] - w / 2.0f,
                cur_node.pos[1] - w / 2.0f
            };

            const float dist_sq = std::max(min_dist,
                    delta[0] * delta[0] +
                    delta[1] * delta[1]
                );

            node_update[0] -= delta[0] * dist_sq * center_pull;
            node_update[1] -= delta[1] * dist_sq * center_pull;


            // update node pos
            const float nd_length = std::sqrt(node_update[0] * node_update[0] +
                                              node_update[1] * node_update[1]);

            if (!std::isnormal(nd_length))
                continue;

            const float upd_factor = std::min(c1, nd_length * temperature * it_step) / nd_length;

            cur_node.pos[0] += node_update[0] * upd_factor;
            cur_node.pos[1] += node_update[1] * upd_factor;

/*
            if (cur_node.pos[0] < 0.0f) cur_node.pos[0] = 0;//    rnd_dis(rgen) * border_wiggle;
            if (cur_node.pos[0] > w)    cur_node.pos[0] = w;// - rnd_dis(rgen) * border_wiggle;
            if (cur_node.pos[1] < 0.0f) cur_node.pos[1] = 0;//    rnd_dis(rgen) * border_wiggle;
            if (cur_node.pos[1] > w)    cur_node.pos[1] = w;// - rnd_dis(rgen) * border_wiggle;
*/

            cur_node.cell_id = pos_to_cell(cur_node.pos);

            // nodes have moved
            nodes_dirty = true;
        }
        temperature = std::min(temp_cf * temperature, temp_min);
    }

}

void fbgd::set_draw_func(draw_func_t df)
{
    draw_func = std::move(df);
}

void fbgd::draw() const
{
    if (draw_func)
        draw_func(nodes,edges,node_look_up);
}

const std::array<float, 2> & fbgd::get_node_pos(node_id_t node_id)
{
    return nodes[node_look_up[node_id]].pos;
}

}
