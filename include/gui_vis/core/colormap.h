//
// Created by claude on 29.01.26.
//

#ifndef COLORMAP_H
#define COLORMAP_H

#include <cstdint>
#include <string_view>
#include "gui_vis/core/color_types.h"

namespace coast {

// The framework's single palette enum. Members mirror ImPlot's *continuous*
// built-in colormaps 1:1 (see 3rd_party/implot/implot.cpp) and in ImPlot's own
// order, so a plot rendered natively by ImPlot and a value colored in software
// via apply_colormap() agree. ImPlot's qualitative palettes (Deep/Dark/Pastel/
// Paired) are intentionally excluded — they are discrete category palettes and
// are meaningless as a value->color ramp.
enum class colormap_t : uint8_t {
    viridis = 0,
    plasma,
    hot,
    cool,
    pink,
    jet,
    twilight,
    rdbu,
    brbg,
    piyg,
    spectral,
    greys,
    COUNT
};

// Get display name for colormap (matches ImPlot's colormap names).
const char* colormap_name(colormap_t cm);

// Parse a display name (as written by colormap_name / stored in a saved plot)
// back to the enum. Unknown names fall back to viridis.
colormap_t colormap_from_name(std::string_view name);

// Apply colormap to normalized value [0,1].
color_t apply_colormap(colormap_t cm, float value);

} // namespace coast

#endif //COLORMAP_H
