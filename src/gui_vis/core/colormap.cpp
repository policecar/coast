//
// Created by claude on 29.01.26.
//

#include "core/colormap.h"
#include <algorithm>

namespace coast {

// Colormap names for UI
const char* colormap_name(colormap_t cm)
{
    switch (cm) {
        case colormap_t::grayscale: return "Grayscale";
        case colormap_t::green:     return "Green";
        case colormap_t::viridis:   return "Viridis";
        case colormap_t::plasma:    return "Plasma";
        case colormap_t::inferno:   return "Inferno";
        case colormap_t::cool:      return "Cool";
        default:                    return "Unknown";
    }
}

// Interpolate between two colors
static color_t lerp_color(const color_t& a, const color_t& b, float t)
{
    return {
        static_cast<uint8_t>(a.r + (b.r - a.r) * t),
        static_cast<uint8_t>(a.g + (b.g - a.g) * t),
        static_cast<uint8_t>(a.b + (b.b - a.b) * t),
        255
    };
}

// Apply colormap to normalized value [0,1]
color_t apply_colormap(colormap_t cm, float value)
{
    value = std::clamp(value, 0.0f, 1.0f);

    switch (cm) {
        case colormap_t::grayscale: {
            uint8_t v = static_cast<uint8_t>(value * 255.0f);
            return {v, v, v, 255};
        }
        case colormap_t::green: {
            uint8_t v = static_cast<uint8_t>(value * 255.0f);
            return {static_cast<uint8_t>(v / 2), v, static_cast<uint8_t>(v / 4), 255};
        }
        case colormap_t::viridis: {
            // Simplified viridis: dark purple -> teal -> yellow
            static const color_t stops[] = {
                {68, 1, 84, 255},     // 0.0 - dark purple
                {59, 82, 139, 255},   // 0.25 - blue
                {33, 145, 140, 255},  // 0.5 - teal
                {94, 201, 98, 255},   // 0.75 - green
                {253, 231, 37, 255}   // 1.0 - yellow
            };
            float idx = value * 4.0f;
            int i = std::min(static_cast<int>(idx), 3);
            float t = idx - static_cast<float>(i);
            return lerp_color(stops[i], stops[i + 1], t);
        }
        case colormap_t::plasma: {
            // Simplified plasma: dark blue -> purple -> orange -> yellow
            static const color_t stops[] = {
                {13, 8, 135, 255},    // 0.0 - dark blue
                {126, 3, 168, 255},   // 0.25 - purple
                {204, 71, 120, 255},  // 0.5 - pink
                {248, 149, 64, 255},  // 0.75 - orange
                {240, 249, 33, 255}   // 1.0 - yellow
            };
            float idx = value * 4.0f;
            int i = std::min(static_cast<int>(idx), 3);
            float t = idx - static_cast<float>(i);
            return lerp_color(stops[i], stops[i + 1], t);
        }
        case colormap_t::inferno: {
            // Simplified inferno: black -> purple -> red -> yellow
            static const color_t stops[] = {
                {0, 0, 4, 255},       // 0.0 - black
                {87, 16, 110, 255},   // 0.25 - purple
                {188, 55, 84, 255},   // 0.5 - red
                {249, 142, 9, 255},   // 0.75 - orange
                {252, 255, 164, 255}  // 1.0 - light yellow
            };
            float idx = value * 4.0f;
            int i = std::min(static_cast<int>(idx), 3);
            float t = idx - static_cast<float>(i);
            return lerp_color(stops[i], stops[i + 1], t);
        }
        case colormap_t::cool: {
            // Cool: cyan -> magenta
            static const color_t stops[] = {
                {0, 255, 255, 255},   // 0.0 - cyan
                {255, 0, 255, 255}    // 1.0 - magenta
            };
            return lerp_color(stops[0], stops[1], value);
        }
        default:
            return {static_cast<uint8_t>(value * 255.0f), static_cast<uint8_t>(value * 255.0f), static_cast<uint8_t>(value * 255.0f), 255};
    }
}

} // namespace coast
