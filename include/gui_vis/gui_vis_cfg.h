//
// Created by jk on 27.01.26.
//

#ifndef GUI_VIS_CFG_H
#define GUI_VIS_CFG_H

#include <unordered_map>
#include <string>

namespace coast {

struct ngm_flat_vis_hint_t {
    int width  = 1;
    int height = 1;
};

struct mnist_hint_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {28, 28};
};

struct dsprites_hint_t {
    ngm_flat_vis_hint_t ngm_flat_vis = {64, 64};
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
    ngm_flat_vis_hint_t ngm_flat_vis = {50, 1};
};

struct ngm2_hints_t {
    std::unordered_map<std::string,neuron_group_hints_t> neuron_groups;

    ngm2_hints_t()
    {
        neuron_groups["default"];
    }
};

struct vis_hints_t {
    dataset_hints_t datasets;
    ngm2_hints_t    ngm2;
};


}


#endif //GUI_VIS_CFG_H
