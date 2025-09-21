//
// Created by jk on 12.08.25.
//
#include <cstdio>
#include <cassert>
#include <raylib.h>
#include "ngm_vis.h"

#include <algorithm>

namespace coast {

void ngm_vis::get_pixel_data(
    int width, int height,
    uint16_t leaf_idx,
    std::size_t max_idx,
    const ngm2::dendrite_t::synapses_t &synapses,
    std::vector<Color> &output
)
{
    output.resize(width*height);

    tmp_tree.clear();
    tmp_tree.resize(max_idx+1);

    tmp_tree[leaf_idx] = 1;
    while (leaf_idx > 1) {
        leaf_idx /= 2;
        tmp_tree[leaf_idx] = 1;
    }

    int pix = 0;
    std::size_t syn_cnt = synapses.size();
    for (std::size_t i = 0; i < syn_cnt; ++i)
        if (tmp_tree[synapses.segment_idx[i]]) {
            uint8_t value = static_cast<uint8_t>(std::clamp(synapses.permanence[i] * 255.0f, 0.0f, 255.0f));
            output[pix].r = value;
            output[pix].g = value;
            output[pix].b = value;
            output[pix].a = 255;
            ++pix;
        }
}

Texture2D ngm_vis::get_texture(int width, int height, Color *data)
{
    Image texture_img {
        data,
        width,
        height,
        1,
        PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    };
    Texture2D tex = LoadTextureFromImage(texture_img);
    return tex;
}

ngm_vis::~ngm_vis()
{
}

void ngm_vis::create_model(fbgd &vis, const ngm2::neuron_group_t &ng)
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

    // fill nodes
    //group_node = vis.add_node();
    auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    const float grid_spacing = coast::fbgd::get_region_size() / static_cast<float>(grid_cnt+2);
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);
        float xp = static_cast<float>(neuron_idx % grid_cnt) * grid_spacing + grid_spacing * 1.5f;
        float yp = static_cast<float>(neuron_idx / grid_cnt) * grid_spacing + grid_spacing * 1.5f;

        // create root node
        std::size_t root_node_idx = neuron_lu[neuron_idx];
        nodes[root_node_idx] = {
            neuron_idx,     //std::size_t neuron_idx
            0,              //std::size_t branch_idx
            0,              //uint16_t    segment_idx
            vis.add_node({xp,yp}, true, root_node_idx), //fbgd::node_id_t node_id
            1,               //uint8_t used
            {},
            std::vector<Color>(),
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
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                if (nodes[segment_base + synapses.segment_idx[i]].used == 0) {
                    Texture2D tex;
                    std::vector<Color> pdata;
                    if (leaf_mask[synapses.segment_idx[i]]) {
                        get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,pdata);
                        tex = get_texture(28,28,pdata.data());
                    }
                    nodes[segment_base + synapses.segment_idx[i]] = {
                        neuron_idx,          //std::size_t neuron_idx
                        branch_idx,          //std::size_t branch_idx
                        synapses.segment_idx[i], //uint16_t    segment_idx
                        vis.add_node({xp + rdis(rgen) * (grid_spacing / 3), yp + rdis(rgen) * (grid_spacing / 3)},false,segment_base + synapses.segment_idx[i]),      //fbgd::node_id_t node_id
                        1,                    //uint8_t used
                        tex,
                        pdata,
                        0
                    };
                }
                nodes[segment_base + synapses.segment_idx[i]].synapse_count++;
            }
            // create edges
            const std::size_t max_idx = cur_branch.get_max_segment_idx();
            uint32_t cur_start = (max_idx + 1) / 2;
            while (cur_start > 1) {
                for (std::size_t seg_idx = cur_start; seg_idx < cur_start*2; ++seg_idx) {
                    if (nodes[segment_base + seg_idx].used == 1) {
                        vis.add_edge(nodes[segment_base + seg_idx/2].node_id, nodes[segment_base + seg_idx].node_id);
                        assert(nodes[segment_base + seg_idx/2].used == 1);
                    }
                }
                cur_start /= 2;
            }
            vis.add_edge(nodes[root_node_idx].node_id, nodes[segment_base + 1].node_id);
        }
        //vis.add_edge(group_node, nodes[root_node_idx].node_id);

    }
}

void ngm_vis::update_model(fbgd &vis, const ngm2::neuron_group_t &ng)
{
    for (auto &node : nodes)
        node.synapse_count = 0;

    std::size_t neuron_cnt = ng.get_neuron_count();
    //auto grid_cnt = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<float>(neuron_cnt))));
    //const float grid_spacing = ngm::fbgd::get_region_size() / static_cast<float>(grid_cnt);
    for (std::size_t neuron_idx = 0; neuron_idx < neuron_cnt; ++neuron_idx ) {
        const ngm2::neuron_t &cur_neuron = ng.get_neuron(neuron_idx);

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
            for (std::size_t i = 0; i < syn_cnt; ++i) {
                auto &cur_node = nodes[segment_base + synapses.segment_idx[i]];
                if (cur_node.pixel_data.empty() == false)
                {
                    if (leaf_mask[synapses.segment_idx[i]] == 0) {
                        //if (IsTextureValid(cur_node.texture))
                            //UnloadTexture(cur_node.texture);
                        cur_node.pixel_data.clear();
                    } else {
                        get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,cur_node.pixel_data);
                        //if (IsTextureValid(cur_node.texture))
                            UpdateTexture(cur_node.texture,cur_node.pixel_data.data());
                        //else
                        //    cur_node.texture = get_texture(28,28,cur_node.pixel_data.data());

                    }
                } else if (cur_node.used == 0) {
                    auto &vis_pos = vis.get_node_pos(nodes[segment_base + synapses.segment_idx[i]/2].node_id);
                    float vx = vis_pos[0] + rdis(rgen) * 0.0001f;
                    float vy = vis_pos[1] + rdis(rgen) * 0.0001f;
                    Texture2D tex;
                    std::vector<Color> pdata;
                    if (leaf_mask[synapses.segment_idx[i]]) {
                        get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,pdata);
                        tex = get_texture(28,28,pdata.data());
                    }
                    cur_node = {
                        neuron_idx,          //std::size_t neuron_idx
                        branch_idx,          //std::size_t branch_idx
                        synapses.segment_idx[i], //uint16_t    segment_idx
                        vis.add_node({vx,vy},false, segment_base + synapses.segment_idx[i]),      //fbgd::node_id_t node_id
                        2,                    //uint8_t used
                        tex,
                        pdata,
                        0
                    };
                }
                cur_node.synapse_count++;
            }
            // create edges
            const std::size_t max_idx = cur_branch.get_max_segment_idx();
            uint32_t cur_start = (max_idx + 1) / 2;
            while (cur_start > 1) {
                for (std::size_t seg_idx = cur_start; seg_idx < cur_start*2; ++seg_idx) {
                    if (nodes[segment_base + seg_idx].used == 2) {
                        nodes[segment_base + seg_idx].used = 1;
                        vis.add_edge(nodes[segment_base + seg_idx/2].node_id, nodes[segment_base + seg_idx].node_id);
                        assert(nodes[segment_base + seg_idx/2].used > 0);
                    }
                }
                cur_start /= 2;
            }
        }
    }


}

void ngm_vis::update_representations(const ngm2::neuron_group_t &ng)
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
            //for (const auto &synapse : synapses) {
                auto &cur_node = nodes[segment_base + synapses.segment_idx[i]];
                if (cur_node.pixel_data.empty() == false)
                {
                    get_pixel_data(28,28,synapses.segment_idx[i],max_segment,synapses,cur_node.pixel_data);
                    //if (IsTextureValid(cur_node.texture))
                        UpdateTexture(cur_node.texture,cur_node.pixel_data.data());
                    //else
                    //    cur_node.texture = get_texture(28,28,cur_node.pixel_data.data());
                    //UpdateTexture(cur_node.texture,cur_node.pixel_data.data());
                }
            }
        }
    }

}

void ngm_vis::free_resources()
{

    for (auto &node : nodes){
        if (IsTextureValid(node.texture))
            UnloadTexture(node.texture);
        node.pixel_data.clear();
    }

}

void ngm_vis::paint(const std::vector<fbgd::node> &vis_nodes, const std::vector<fbgd::edge> &vis_edges,
                    const std::vector<uint32_t> &vis_node_lu)
{
    // perform basic rendering
    constexpr float draw_scale_y = 1000.0f;
    constexpr float draw_scale_x = 1500.0f;
    for (auto &edge : vis_edges) {
        const auto &vnode1 = vis_nodes[vis_node_lu[edge.from]];
        const auto &vnode2 = vis_nodes[vis_node_lu[edge.to]];
        if (vnode1.node_id > vnode2.node_id)
            continue;
        const Vector2 pos1 {
            vnode1.pos[0] * draw_scale_x,
            vnode1.pos[1] * draw_scale_y
        };
        const Vector2 pos2 {
            vnode2.pos[0] * draw_scale_x,
            vnode2.pos[1] * draw_scale_y
        };
        auto node_idx = std::any_cast<std::size_t>(vnode2.payload);
        auto &node2 = nodes[node_idx];
        float sc_ratio = static_cast<float>(node2.synapse_count) / (28.0f * 28.0f);

        DrawLineEx(pos1, pos2, 2.f + 8.f * sc_ratio, BLUE);
    }

    for (auto &vis_node : vis_nodes) {
        auto node_idx = std::any_cast<std::size_t>(vis_node.payload);
        auto &node = nodes[node_idx];
        if (node.pixel_data.empty() == false) {
            float scale = 1.0f;
            Vector2 pos = {
                vis_node.pos[0] * draw_scale_x - static_cast<float>(node.texture.width) * scale / 2,
                vis_node.pos[1] * draw_scale_y - static_cast<float>(node.texture.height) * scale / 2,
            };
            DrawTextureEx(
                node.texture,
                pos,
                0.0f,
                scale,
                WHITE
            );
        } else {
            DrawCircle(
                static_cast<int>(vis_node.pos[0] * draw_scale_x),
                static_cast<int>(vis_node.pos[1] * draw_scale_y),
                node.segment_idx == 0 ? 6.0f : 3.0f,
                node.segment_idx == 0 ? ORANGE : BLACK
            );
        }
    }
}

} // ngm