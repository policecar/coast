//
// Created by jk on 08.09.25.
//

#ifndef GRADIENT_H
#define GRADIENT_H

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <array>
#include <raylib.h>

namespace coast {

template<uint8_t N>
    requires (N >= 2)
class gradient {

    const std::array<Color,N> grad_colors;

public:

    explicit gradient(const std::array<Color,N> &gradient_colors) :
        grad_colors(gradient_colors)
    {}

    Color operator[](float grad_pos) const {
        float grad_pos_c = std::clamp(grad_pos, 0.0f, 1.0f);
        const float scaled_idx = grad_pos_c * static_cast<float>(N-1);
        const uint32_t lower_idx = std::clamp(static_cast<uint32_t>(std::floor(scaled_idx)),0u,static_cast<uint32_t>(N-1));
        const uint32_t upper_idx = std::clamp(static_cast<uint32_t>(std::ceil(scaled_idx)),0u,static_cast<uint32_t>(N-1));
        // edge case
        if (lower_idx == upper_idx) {
            return grad_colors[lower_idx];
        }
        // regular case
        Color a = grad_colors[lower_idx];
        Color b = grad_colors[upper_idx];
        const float frac = scaled_idx - static_cast<uint32_t>(scaled_idx);
        return {
            static_cast<uint8_t>(std::lerp(static_cast<float>(a.r),static_cast<float>(b.r),frac)),
            static_cast<uint8_t>(std::lerp(static_cast<float>(a.g),static_cast<float>(b.g),frac)),
            static_cast<uint8_t>(std::lerp(static_cast<float>(a.b),static_cast<float>(b.b),frac)),
            static_cast<uint8_t>(std::lerp(static_cast<float>(a.a),static_cast<float>(b.a),frac))
        };
    }

};

}

#endif //GRADIENT_H
