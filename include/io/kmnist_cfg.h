//
// KMNIST dataset configuration
//

#ifndef KMNIST_CFG_H
#define KMNIST_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace kmnist {

struct kmnist_param_t : dataset::sequence_params_t {
    std::string image_file = "";
    std::string label_file = "";
};

}

#endif //KMNIST_CFG_H
