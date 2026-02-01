//
// Color MNIST dataset configuration
// MNIST digits randomly colored red, green, or blue
//

#ifndef COLOR_MNIST_CFG_H
#define COLOR_MNIST_CFG_H

#include <string>

namespace color_mnist {

struct color_mnist_param_t {
    std::string            image_file      = "";
    std::string            label_file      = "";
    std::size_t            change_interval = 15;
    std::pair<double,double> noise_range   = {0.0, 0.05};
    int                    random_seed     = 0;
};

}

#endif //COLOR_MNIST_CFG_H
