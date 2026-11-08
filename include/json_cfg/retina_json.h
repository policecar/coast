//
// JSON adaptors for the retina node config
//

#ifndef COAST_RETINA_JSON_H
#define COAST_RETINA_JSON_H

#include <nlohmann/json.hpp>
#include "json_cfg/strict_enum.h"

#include "retina/retina_cfg.h"

namespace retina {

// Strict like every config enum: an unknown polarity string throws at load.
COAST_JSON_STRICT_ENUM(polarity_t, {
    {polarity_t::on,  "on" },
    {polarity_t::off, "off"},
})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(retina_param_t,
    width,
    height,
    channels,
    luminance,
    polarity,
    center_sigma,
    surround_sigma,
    surround_weight,
    gain,
    normalize,
    norm_sigma,
    noise_range,
    random_seed
)

}

#endif // COAST_RETINA_JSON_H
