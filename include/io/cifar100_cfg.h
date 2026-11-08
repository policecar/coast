//
// CIFAR-100 dataset configuration
//

#ifndef CIFAR100_CFG_H
#define CIFAR100_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace cifar100 {

struct cifar100_param_t : dataset::sequence_params_t {
    std::string data_dir          = "";
    bool        use_test_set      = false;
    bool        use_coarse_labels = false;  // Use 20 superclasses instead of 100 fine classes
};

}

#endif //CIFAR100_CFG_H
