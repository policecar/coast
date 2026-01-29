//
// Created by claude on 29.01.26.
//

#ifndef COLORMAP_H
#define COLORMAP_H

#include <cstdint>
#include "color_types.h"

namespace coast {

enum class colormap_t : uint8_t {
    grayscale = 0,
    green,      // Green-tinted
    viridis,
    plasma,
    inferno,
    cool,
    COUNT
};

// Get display name for colormap
const char* colormap_name(colormap_t cm);

// Apply colormap to normalized value [0,1]
color_t apply_colormap(colormap_t cm, float value);

} // namespace coast

#endif //COLORMAP_H
