//
// Created by claude on 29.01.26.
//

#include "gui_vis/core/colormap.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace coast {

// Colormap display names — identical to ImPlot's built-in names so the enum,
// the software renderer, and ImPlot's native heatmaps all agree.
const char* colormap_name(colormap_t cm)
{
    switch (cm) {
        case colormap_t::viridis:  return "Viridis";
        case colormap_t::plasma:   return "Plasma";
        case colormap_t::hot:      return "Hot";
        case colormap_t::cool:     return "Cool";
        case colormap_t::pink:     return "Pink";
        case colormap_t::jet:      return "Jet";
        case colormap_t::twilight: return "Twilight";
        case colormap_t::rdbu:     return "RdBu";
        case colormap_t::brbg:     return "BrBG";
        case colormap_t::piyg:     return "PiYG";
        case colormap_t::spectral: return "Spectral";
        case colormap_t::greys:    return "Greys";
        default:                   return "Unknown";
    }
}

colormap_t colormap_from_name(std::string_view name)
{
    for (int i = 0; i < static_cast<int>(colormap_t::COUNT); ++i) {
        const auto cm = static_cast<colormap_t>(i);
        if (name == colormap_name(cm)) return cm;
    }
    return colormap_t::viridis;
}

// Exact control-point data transcribed from ImPlot's continuous colormaps
// (3rd_party/implot/implot.cpp). Sampling these in software reproduces the
// colors ImPlot draws for the same colormap, keeping both render paths in sync.
namespace {

using stops_t = std::array<color_t, 11>;

// clang-format off
constexpr stops_t viridis  = {{{68,1,84},{72,36,117},{65,68,135},{53,95,141},{42,120,142},{33,145,140},{34,168,132},{68,191,112},{122,209,81},{189,223,38},{253,231,37}}};
constexpr stops_t plasma   = {{{13,8,135},{65,4,157},{106,0,168},{143,13,164},{177,42,144},{204,71,120},{225,100,98},{242,132,75},{252,166,54},{252,206,37},{240,249,33}}};
constexpr stops_t hot      = {{{64,0,0},{128,0,0},{191,0,0},{255,0,0},{255,64,0},{255,128,0},{255,191,0},{255,255,0},{255,255,85},{255,255,170},{255,255,255}}};
constexpr stops_t cool     = {{{0,255,255},{26,230,255},{51,204,255},{77,179,255},{102,153,255},{128,128,255},{153,102,255},{179,77,255},{204,51,255},{230,26,255},{255,0,255}}};
constexpr stops_t pink     = {{{74,0,0},{123,66,66},{158,93,93},{186,114,114},{198,151,132},{208,180,147},{218,206,161},{228,228,174},{237,237,205},{246,246,231},{255,255,255}}};
constexpr stops_t jet      = {{{0,0,170},{0,0,255},{0,85,255},{0,170,255},{0,255,255},{85,255,170},{170,255,85},{255,255,0},{255,170,0},{255,85,0},{255,0,0}}};
constexpr stops_t twilight = {{{226,217,226},{166,191,202},{109,144,192},{95,88,176},{83,30,124},{47,20,54},{100,25,75},{159,60,80},{192,117,94},{208,179,158},{226,217,226}}};
constexpr stops_t rdbu     = {{{103,0,31},{178,24,43},{214,96,77},{244,165,130},{253,219,199},{247,247,247},{209,229,240},{146,197,222},{67,147,195},{33,102,172},{5,48,97}}};
constexpr stops_t brbg     = {{{84,48,5},{140,81,10},{191,129,45},{223,194,125},{246,232,195},{245,245,245},{199,234,229},{128,205,193},{53,151,143},{1,102,94},{0,60,48}}};
constexpr stops_t piyg     = {{{142,1,82},{197,27,125},{222,119,174},{241,182,218},{253,224,239},{247,247,247},{230,245,208},{184,225,134},{127,188,65},{77,146,33},{39,100,25}}};
constexpr stops_t spectral = {{{158,1,66},{213,62,79},{244,109,67},{253,174,97},{254,224,139},{255,255,191},{230,245,152},{171,221,164},{102,194,165},{50,136,189},{94,79,162}}};
// clang-format on

// Greys is a 2-stop ramp (white -> black) in ImPlot; handled separately below.

color_t lerp_color(const color_t& a, const color_t& b, float t)
{
    return {
        static_cast<uint8_t>(a.r + (b.r - a.r) * t),
        static_cast<uint8_t>(a.g + (b.g - a.g) * t),
        static_cast<uint8_t>(a.b + (b.b - a.b) * t),
        255
    };
}

// Sample an N-stop ramp at value in [0,1] with piecewise-linear interpolation.
color_t sample(const color_t* stops, std::size_t n, float value)
{
    value = std::clamp(value, 0.0f, 1.0f);
    if (n == 1) return stops[0];
    const float pos = value * static_cast<float>(n - 1);
    const auto  i   = std::min(static_cast<std::size_t>(pos), n - 2);
    const float t   = pos - static_cast<float>(i);
    return lerp_color(stops[i], stops[i + 1], t);
}

} // namespace

// Apply colormap to normalized value [0,1].
color_t apply_colormap(colormap_t cm, float value)
{
    switch (cm) {
        case colormap_t::viridis:  return sample(viridis.data(),  viridis.size(),  value);
        case colormap_t::plasma:   return sample(plasma.data(),   plasma.size(),   value);
        case colormap_t::hot:      return sample(hot.data(),      hot.size(),      value);
        case colormap_t::cool:     return sample(cool.data(),     cool.size(),     value);
        case colormap_t::pink:     return sample(pink.data(),     pink.size(),     value);
        case colormap_t::jet:      return sample(jet.data(),      jet.size(),      value);
        case colormap_t::twilight: return sample(twilight.data(), twilight.size(), value);
        case colormap_t::rdbu:     return sample(rdbu.data(),     rdbu.size(),     value);
        case colormap_t::brbg:     return sample(brbg.data(),     brbg.size(),     value);
        case colormap_t::piyg:     return sample(piyg.data(),     piyg.size(),     value);
        case colormap_t::spectral: return sample(spectral.data(), spectral.size(), value);
        case colormap_t::greys: {
            // ImPlot Greys: white (0) -> black (1).
            static constexpr std::array<color_t, 2> greys = {{{255,255,255},{0,0,0}}};
            return sample(greys.data(), greys.size(), value);
        }
        default:
            return {static_cast<uint8_t>(std::clamp(value,0.0f,1.0f) * 255.0f),
                    static_cast<uint8_t>(std::clamp(value,0.0f,1.0f) * 255.0f),
                    static_cast<uint8_t>(std::clamp(value,0.0f,1.0f) * 255.0f), 255};
    }
}

} // namespace coast
