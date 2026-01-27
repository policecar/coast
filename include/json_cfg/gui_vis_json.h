//
// Created by jk on 27.01.26.
//

#ifndef GUI_VIS_JSON_H
#define GUI_VIS_JSON_H

#include <nlohmann/json.hpp>

#include "gui_vis_cfg.h"

using json = nlohmann::json;

namespace coast {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ngm_flat_vis_hint_t,
    width,
    height
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(mnist_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(dsprites_hint_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(dataset_hints_t,
    mnist,
    dsprites
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(neuron_group_hints_t,
    ngm_flat_vis
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ngm2_hints_t,
    neuron_groups
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(vis_hints_t,
    datasets,
    ngm2
)


}

#endif //GUI_VIS_JSON_H
