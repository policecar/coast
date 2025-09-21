//
// Created by jk on 12.08.25.
//

#ifndef NGM_VIS_H
#define NGM_VIS_H

#include <vector>
#include <random>
#include "hd_ngm2.h"
#include "fbgd.h"
#include "raylib.h"

namespace coast {

class ngm_vis {

    struct node_info_t {
        std::size_t neuron_idx  = 0;
        std::size_t branch_idx  = 0;
        uint16_t    segment_idx = 0;

        fbgd::node_id_t node_id = 0;
        uint8_t used = 0;

        Texture2D texture {};
        std::vector<Color> pixel_data {};

        uint32_t synapse_count = 0;
    };

    std::mt19937 rgen {0};
    std::uniform_real_distribution<float> rdis { -1.0, 1.0 };

    std::vector<node_info_t> nodes;
    std::vector<std::size_t> neuron_lu;
    std::vector<std::size_t> branch_start_lu;
    std::vector<std::size_t> branch_lu;

    fbgd::node_id_t group_node = 0;

    std::vector<uint8_t> tmp_tree;

    void get_pixel_data(int width, int height, uint16_t leaf_idx, std::size_t max_idx, const ngm2::dendrite_t::synapses_t &synapses, std::vector<Color> &output);
    static Texture2D get_texture(int width, int height, Color *data);

public:
    ~ngm_vis();

    void create_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_model(fbgd &vis, const ngm2::neuron_group_t &ng);
    void update_representations(const ngm2::neuron_group_t &ng);

    void free_resources();

    void paint(
        const std::vector<fbgd::node>& vis_nodes,
        const std::vector<fbgd::edge>& vis_edges,
        const std::vector<uint32_t>&   vis_node_lu
    );

};

} // ngm

#endif //NGM_VIS_H
