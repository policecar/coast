//
// Created by jk on 25.01.26.
//

#ifndef MNIST_CFG_H
#define MNIST_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace mnist {

struct mnist_param_t : dataset::sequence_params_t {
    std::string image_file = "";
    std::string label_file = "";
};

}

#endif //MNIST_CFG_H
