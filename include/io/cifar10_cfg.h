//
// CIFAR-10 dataset configuration
//

#ifndef CIFAR10_CFG_H
#define CIFAR10_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace cifar10 {

struct cifar10_param_t : dataset::sequence_params_t {
    std::string data_dir     = "";
    bool        use_test_set = false;
};

}

#endif //CIFAR10_CFG_H
