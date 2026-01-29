//
// Created by jk on 27.01.26.
//

#ifndef VIS_HINTS_H
#define VIS_HINTS_H

#include <unordered_map>
#include <string>

namespace coast {

struct ngm_flat_vis_hint_t {
    int width  = 1;
    int height = 1;
    int update_interval = 10000;  // Simulation steps between updates (0 = every frame)
};

struct mnist_hint_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {.width = 28, .height = 28, .update_interval = 1000};
};

struct dsprites_hint_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {.width = 64, .height = 64, .update_interval = 1000};
};

struct buffer_vis_hint_t {
    int update_interval = 100;  // Simulation steps between updates (0 = every frame)
};

struct dataset_hints_t {
    std::unordered_map<std::string,mnist_hint_t>    mnist;
    std::unordered_map<std::string,dsprites_hint_t> dsprites;

    dataset_hints_t()
    {
        mnist["default"];
        dsprites["default"];
    }
};

struct neuron_group_hints_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {.width = 50, .height = 1, .update_interval = 10000};
};

struct ngm2_hints_t {
    std::unordered_map<std::string,neuron_group_hints_t> neuron_groups;

    ngm2_hints_t()
    {
        neuron_groups["default"];
    }
};

struct vis_hints_t {
    dataset_hints_t  datasets;
    ngm2_hints_t     ngm2;
    buffer_vis_hint_t buffer_vis;  // Default hints for io_buffer visualizations
};


}


#endif //VIS_HINTS_H
