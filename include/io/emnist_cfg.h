//
// EMNIST dataset configuration
//

#ifndef EMNIST_CFG_H
#define EMNIST_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace emnist {

struct emnist_param_t : dataset::sequence_params_t {
    std::string image_file = "";
    std::string label_file = "";
};

}

#endif //EMNIST_CFG_H
