//
// Created by jk on 08.09.25.
//

#ifndef VEC_GROUP_VIS_H
#define VEC_GROUP_VIS_H

#include <vector>
#include <span>
#include <cstdint>
#include "vec_vis.h"

namespace coast {

class vec_group_vis {

public:
    enum class layout_t
    {
        vertical,
        horizontal,
        grid
    };

    struct params_t
    {
        uint32_t         vec_cnt;
        vec_vis::param_t vec_params;
        layout_t         layout;
        uint32_t         margin;
        uint32_t         grid_width;
        uint32_t         grid_height;
    };

private:

    vec_vis        vector_vis;
    const params_t params;

public:
    explicit vec_group_vis(const params_t &_params);

    template<uint32_t N>
    void update_vec(
        uint32_t               vec_id,
        std::span<const float> vec_data,
        const gradient<N>     &colors,
        const px_pos_t         out_pos,
        const pixel_data_t    &output
    ){
        assert(vec_id < params.vec_cnt);
        const uint32_t elem_width  = params.vec_params.rep_width  * params.vec_params.elem_width  + params.margin;
        const uint32_t elem_height = params.vec_params.rep_height * params.vec_params.elem_height + params.margin;
        px_pos_t elem_pos {};
        bool pos_ok = true;
        switch (params.layout) {
            case layout_t::vertical : {
                elem_pos = {
                    out_pos.x,
                    out_pos.y + vec_id * elem_height
                };
            } break;
            case layout_t::horizontal : {
                elem_pos = {
                    out_pos.x + vec_id * elem_width,
                    out_pos.y
                };
            } break;
            case layout_t::grid : {
                elem_pos = {
                    out_pos.x + (vec_id % params.grid_width) * elem_width,
                    out_pos.y + (vec_id / params.grid_width) * elem_height
                };
                if (vec_id / params.grid_width >= params.grid_height)
                    pos_ok = false;
            } break;
        }
        if (pos_ok) vector_vis.update<2>(vec_data,colors,elem_pos,output);
    }

    [[nodiscard]] uint32_t get_total_width() const;
    [[nodiscard]] uint32_t get_total_height() const;

};

} // coast

#endif //VEC_GROUP_VIS_H
