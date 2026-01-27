//
// Created by jk on 25.01.26.
//

#ifndef DATASET_JSON_H
#define DATASET_JSON_H

#include <nlohmann/json.hpp>
#include "dataset_cfg.h"

#include "mnist_cfg.h"
#include "dsprites_cfg.h"

using json = nlohmann::json;

/*
// Custom serializer for std::pair<float, float> (used by noise_range)
namespace nlohmann {
template <>
struct adl_serializer<std::pair<float, float>> {
    static void to_json(json& j, const std::pair<float, float>& p) {
        j = json::array({p.first, p.second});
    }
    static void from_json(const json& j, std::pair<float, float>& p) {
        if (j.is_array() && j.size() == 2) {
            p.first = j[0].get<float>();
            p.second = j[1].get<float>();
        }
    }
};
}
*/

namespace mnist {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(mnist_param_t,
    image_file,
    label_file,
    change_interval,
    noise_range,
    random_seed
)

}

namespace dsprites {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(dsprites_param_t,
    hdf5_file,
    change_interval,
    noise_range,
    random_seed,
    randomize_readout
)

}

namespace dataset {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    mnist,
    dsprites
)


}

#endif //DATASET_JSON_H
