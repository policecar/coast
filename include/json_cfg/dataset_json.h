//
// Created by jk on 25.01.26.
//

#ifndef DATASET_JSON_H
#define DATASET_JSON_H

#include <nlohmann/json.hpp>
#include "dataset_cfg.h"

#include "mnist_cfg.h"
#include "dsprites_cfg.h"
#include "emnist_cfg.h"
#include "kmnist_cfg.h"
#include "cifar10_cfg.h"
#include "cifar100_cfg.h"
#include "chairs3d_cfg.h"

using json = nlohmann::ordered_json;

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

namespace emnist {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(emnist_param_t,
    image_file,
    label_file,
    change_interval,
    noise_range,
    random_seed
)

}

namespace kmnist {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(kmnist_param_t,
    image_file,
    label_file,
    change_interval,
    noise_range,
    random_seed
)

}

namespace cifar10 {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cifar10_param_t,
    data_dir,
    change_interval,
    noise_range,
    random_seed,
    use_test_set
)

}

namespace cifar100 {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cifar100_param_t,
    data_dir,
    change_interval,
    noise_range,
    random_seed,
    use_test_set,
    use_coarse_labels
)

}

namespace chairs3d {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(chairs3d_param_t,
    data_dir,
    image_width,
    image_height,
    change_interval,
    noise_range,
    random_seed
)

}

namespace dataset {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(cfg_store,
    mnist,
    dsprites,
    emnist,
    kmnist,
    cifar10,
    cifar100,
    chairs3d
)


}

#endif //DATASET_JSON_H
