//
// Created by jk on 08.09.25.
//

#include <cstdio>

#include "vec_group_vis.h"

#include <utility>

namespace coast {

vec_group_vis::vec_group_vis(const params_t &_params) :
    vector_vis(_params.vec_params),
    params(_params)
{
    if ((params.layout == layout_t::grid) && (params.vec_cnt > params.grid_width * params.grid_height)) {
        std::printf("Warning: Given grid dimensions will be not large enough for number of vectors specified!\n");
    }
}

uint32_t vec_group_vis::get_total_width() const
{
    const uint32_t elem_width  = params.vec_params.rep_width  * params.vec_params.elem_width  + params.margin;
    switch (params.layout) {
        case layout_t::vertical : {
            return elem_width - params.margin;
        }
        case layout_t::horizontal : {
            return params.vec_cnt * elem_width - params.margin;
        }
        case layout_t::grid : {
            return params.grid_width * elem_width - params.margin;
        }
    }
    std::unreachable();
}

uint32_t vec_group_vis::get_total_height() const
{
    const uint32_t elem_height = params.vec_params.rep_height * params.vec_params.elem_height + params.margin;
    switch (params.layout) {
        case layout_t::vertical : {
            return params.vec_cnt * elem_height - params.margin;
        }
        case layout_t::horizontal : {
            return elem_height - params.margin;
        }
        case layout_t::grid : {
            return params.grid_height * elem_height - params.margin;
        }
    }
    std::unreachable();
}

} // coast