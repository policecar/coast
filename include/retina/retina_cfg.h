//
// Retina node configuration
//

#ifndef COAST_RETINA_CFG_H
#define COAST_RETINA_CFG_H

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "core/seed.h"

namespace retina {

// Which half of the signed centre-surround response a node emits. A pair of
// nodes (one per polarity) reading the same image models the ON and OFF pathways.
enum class polarity_t : std::uint8_t {
    on,    // emits max(d, 0): increments relative to the surround / adapting level
    off,   // emits max(-d, 0): decrements
    TYPE_COUNT
};

// Input layout: width x height pixels, `channels` interleaved values per pixel
// (HWC, as the dataset loaders emit). Output layout: width x height x out_channels,
// out_channels = luminance ? 1 : channels.
//
// Signed response per output plane y (see retina.h):
//   d = gain * (G(center_sigma)*y - k * G(surround_sigma)*y - (1 - k) * mean(y)),
//   k = surround_weight.
// k = 1: pure local contrast (a uniform field gives 0); k = 0: luminance relative
// to the mean of the current image (adaptation to the mean light level).
//
// With normalize on, the gain-free response x of each plane is divided by
// norm_sigma plus its RMS over the whole plane (divisive normalisation):
//   d = gain * x / (norm_sigma + RMS(x)),  x = d above with gain = 1.
// A plane with RMS(x) >> norm_sigma comes out contrast-invariant (twice the
// contrast, the same d); one with RMS(x) << norm_sigma stays near-linear, so a
// faint image is not blown up to full contrast. gain still sets where polarize's
// clamp at 1 starts clipping. Off by default: d is then exactly the formula above.
struct retina_param_t {
    int                      width           = 28;
    int                      height          = 28;
    int                      channels        = 1;
    bool                     luminance       = false;  // RGB -> Y (Rec. 601) before filtering; needs channels == 3
    polarity_t               polarity        = polarity_t::on;
    double                   center_sigma    = 1.0;    // pixels
    double                   surround_sigma  = 3.0;    // pixels
    double                   surround_weight = 1.0;    // k
    double                   gain            = 1.0;
    bool                     normalize       = false;  // divisive normalisation by norm_sigma + RMS (above)
    double                   norm_sigma      = 0.05;   // semi-saturation constant, same units as x
    std::pair<double,double> noise_range     = {0.0, 0.05};  // uniform output noise: the "null" signal
    seed_t                   random_seed     = {};
};

// The config spelling, "on" / "off". The JSON adaptor (json_cfg/retina_json.h)
// keeps its own table for the strict enum; unit.retina checks the two agree.
[[nodiscard]] constexpr std::string_view to_string(polarity_t p)
{
    return p == polarity_t::on ? "on" : "off";
}

using cfg_store = std::unordered_map<std::string, retina_param_t>;

}

#endif // COAST_RETINA_CFG_H
