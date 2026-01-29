//
// Created by jk on 25.01.26.
//

#ifndef MNIST_CFG_H
#define MNIST_CFG_H

#include <string>

namespace mnist {

struct mnist_param_t {
    std::string            image_file      = "";
    std::string            label_file      = "";
    std::size_t            change_interval = 15;
    std::pair<double,double> noise_range   = {0.0, 0.05};
    int                    random_seed     = 0;
};

}

#endif //MNIST_CFG_H
