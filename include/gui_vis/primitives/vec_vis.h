//
// Created by jk on 08.09.25.
//

#ifndef VEC_VIS_H
#define VEC_VIS_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include "../core/color_types.h"
#include "../core/gradient.h"

namespace coast {

struct pixel_data_t {
    std::span<color_t> data;
    uint32_t           width;
    uint32_t           height;
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
        uint32_t num_channels = 1;  // 1 = grayscale (use colormap), 3 = RGB
    };

private:
    const param_t params;

public:
    explicit vec_vis(const param_t &_params);

    [[nodiscard]] const param_t& get_params() const;

    template<uint32_t N>
    void update(std::span<const float> vec_data, const gradient<N> &colors, const px_pos_t out_pos, const pixel_data_t &output) {
        [[maybe_unused]] const uint32_t num_pixels = params.rep_width * params.rep_height;
        if (params.num_channels == 3) {
            // RGB mode: vec_data contains interleaved R,G,B values
            assert(vec_data.size() >= num_pixels * 3);
            for (uint32_t y = 0; y < params.rep_height; ++y)
                for (uint32_t x = 0; x < params.rep_width; ++x) {
                    const uint32_t pixel_idx = y * params.rep_width + x;
                    const uint32_t data_idx = pixel_idx * 3;
                    // Convert float [0,1] RGB to color_t
                    const color_t cur_color {
                        static_cast<uint8_t>(std::clamp(vec_data[data_idx + 0] * 255.0f, 0.0f, 255.0f)),
                        static_cast<uint8_t>(std::clamp(vec_data[data_idx + 1] * 255.0f, 0.0f, 255.0f)),
                        static_cast<uint8_t>(std::clamp(vec_data[data_idx + 2] * 255.0f, 0.0f, 255.0f)),
                        255
                    };
                    for (uint32_t ey = 0; ey < params.elem_height; ++ey)
                        for (uint32_t ex = 0; ex < params.elem_width; ++ex) {
                            const uint32_t out_x = out_pos.x + x * params.elem_width  + ex;
                            const uint32_t out_y = out_pos.y + y * params.elem_height + ey;
                            if ((out_x < output.width) && (out_y < output.height))
                                output.data[out_y * output.width + out_x] = cur_color;
                        }
                }
        } else {
            // Grayscale mode: use colormap
            assert(vec_data.size() >= num_pixels);
            for (uint32_t y = 0; y < params.rep_height; ++y)
                for (uint32_t x = 0; x < params.rep_width; ++x) {
                    const uint32_t data_idx = y * params.rep_width + x;
                    const color_t cur_color = colors[vec_data[data_idx]];
                    for (uint32_t ey = 0; ey < params.elem_height; ++ey)
                        for (uint32_t ex = 0; ex < params.elem_width; ++ex) {
                            const uint32_t out_x = out_pos.x + x * params.elem_width  + ex;
                            const uint32_t out_y = out_pos.y + y * params.elem_height + ey;
                            if ((out_x < output.width) && (out_y < output.height))
                                output.data[out_y * output.width + out_x] = cur_color;
                        }
                }
        }
    }
};

} // coast

#endif //VEC_VIS_H
