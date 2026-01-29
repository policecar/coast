//
// Created by jk on 12.08.25.
//

#ifndef NGM_TREE_VIS_H
#define NGM_TREE_VIS_H

#include <vector>
#include <random>
#include "hd_ngm2.h"
#include "../../layout/fbgd.h"
#include "../../core/color_types.h"
#include "../../core/gl_texture.h"

namespace coast {

class ngm_tree_vis {

    struct node_info_t {
        std::size_t neuron_idx  = 0;
        std::size_t branch_idx  = 0;
        uint16_t    segment_idx = 0;

        fbgd::node_id_t node_id = 0;
        uint8_t used = 0;

        gl_texture texture;
        std::vector<color_t> pixel_data;

        uint32_t synapse_count = 0;
    };

    std::mt19937 rgen {0};
    std::uniform_real_distribution<float> rdis { -1.0, 1.0 };

    std::vector<node_info_t> nodes;
    std::vector<std::size_t> neuron_lu;
    std::vector<std::size_t> branch_start_lu;
    std::vector<std::size_t> branch_lu;

    [[maybe_unused]] fbgd::node_id_t group_node = 0;

    std::vector<uint8_t> tmp_tree;

    void get_pixel_data(int width, int height, uint16_t leaf_idx, std::size_t max_idx, const ngm2::dendrite_t::synapses_t &synapses, std::vector<color_t> &output);
    static gl_texture get_texture(int width, int height, const color_t *data);

public:
    ~ngm_tree_vis();

    void create_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_representations(const ngm2::neuron_group_t &ng);

    void free_resources();

    // Render in ImGui window using ImDrawList
    void render_imgui(
        const char* label,
        const std::vector<fbgd::node>& vis_nodes,
        const std::vector<fbgd::edge>& vis_edges,
        const std::vector<uint32_t>&   vis_node_lu
    );

};

} // coast

#endif //NGM_TREE_VIS_H
