//
// Color MNIST dataset configuration
// MNIST digits randomly colored red, green, or blue
//

#ifndef COLOR_MNIST_CFG_H
#define COLOR_MNIST_CFG_H

#include <string>
#include "io/sequence_params.h"

namespace color_mnist {

struct color_mnist_param_t : dataset::sequence_params_t {
    std::string image_file = "";
    std::string label_file = "";
};

}

#endif //COLOR_MNIST_CFG_H
