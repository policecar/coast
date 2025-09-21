//
// Created by jk on 08.09.25.
//

#ifndef VEC_VIS_H
#define VEC_VIS_H

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>
#include <raylib.h>

#include <gradient.h>
#include <numeric>

namespace coast {

struct pixel_data_t {
    std::span<Color> data;
    uint32_t         width;
    uint32_t         height;
};

struct px_pos_t {
    uint32_t x;
    uint32_t y;
};

class vec_vis {

public:
    struct param_t {
        uint32_t rep_width;
        uint32_t rep_height;
        uint32_t elem_width;
        uint32_t elem_height;
    };

private:
    const param_t params;

public:
    explicit vec_vis(const param_t &_params);

    [[nodiscard]] const param_t& get_params() const;

    template<uint32_t N>
    void update(std::span<const float> vec_data, const gradient<N> &colors, const px_pos_t out_pos, const pixel_data_t &output) {
        assert(vec_data.size() >= params.rep_width * params.rep_height);
        //float max_val = std::reduce(vec_data.begin(),vec_data.end(),0.0f,[](float a, float b){ return std::max(a,b);});
        //if (!std::isnormal(max_val)) max_val = 1.0f;
        for (uint32_t y = 0; y < params.rep_height; ++y)
            for (uint32_t x = 0; x < params.rep_width; ++x) {
                const uint32_t data_idx = y * params.rep_width + x;
                //const Color cur_color = colors[vec_data[data_idx] / max_val];
                const Color cur_color = colors[vec_data[data_idx]];
                for (uint32_t ey = 0; ey < params.elem_height; ++ey)
                    for (uint32_t ex = 0; ex < params.elem_width; ++ex) {
                        const uint32_t out_x = out_pos.x + x * params.elem_width  + ex;
                        const uint32_t out_y = out_pos.y + y * params.elem_height + ey;
                        if ((out_x < output.width) && (out_y < output.height))
                            output.data[out_y * output.width + out_x] = cur_color;
                    }
            }
    }
};

} // coast

#endif //VEC_VIS_H
